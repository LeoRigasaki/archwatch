"""Rules introduced after the frozen first-run holdout, with primary-shaped fixtures."""

import copy
import hashlib
import json
from math import prod
from pathlib import Path

import pytest

from archwatch import extract, pipeline, verify, weights
from archwatch.publication import PublicationError, render_values

ROOT = Path(__file__).resolve().parents[1]


def tensor(shape, dtype="BF16"):
    return {"shape": shape, "dtype": dtype}


def test_native_fp4_and_fp8_scales_and_integer_lookup_buffers():
    ts = {
        "layers.0.ffn.experts.0.w1.weight": tensor([6, 4], "I8"),
        "layers.0.ffn.experts.0.w1.scale": tensor([6, 1], "F8_E8M0"),
        "layers.0.ffn.experts.0.w2.qscale_weight": tensor([1], "F32"),
        "ngram.offsets": tensor([3], "I64"),
        "model.layers.0.router.scale": tensor([8]),
    }
    assert weights.count(ts, None, native_expert_bits=4)["total"] == 48 + 8
    assert weights.count(ts, None)["total"] == 24 + 8  # No guessed unpacking.
    assert weights.shape_facts(ts, None, native_expert_bits=4)["expert_input_dim"] == 8


def test_auxiliary_and_mtp_tensors_do_not_override_main_dimensions():
    ts = {
        "cls_net.model.embed_tokens.weight": tensor([2, 8]),
        "cls_net.model.layers.0.mlp.gate_proj.weight": tensor([99, 8]),
        "model.visual.layers.0.mlp.fc1.weight": tensor([77, 8]),
        "model.embed_tokens.weight": tensor([16, 8]),
        "model.layers.0.mlp.gate_proj.weight": tensor([24, 8]),
        "model.layers.0.self_attn.q_proj.weight": tensor([8, 8]),
        "model.layers.1.self_attn.q_proj.weight": tensor([8, 8]),
    }
    f = weights.shape_facts(ts, None, num_layers=1)
    assert (f["vocab_rows"], f["ffn_intermediate"]) == (16, 24)
    assert weights.count(ts, None, num_layers=1)["layer_indices_in_weights"] == [0, 0]
    assert weights.layer_kinds(ts, num_layers=1) == ["attention"]
    assert weights.count(ts, None, num_layers=1)["total"] == sum(prod(t["shape"]) for t in ts.values())


@pytest.mark.parametrize("down,gate", [([4, 8, 3], [4, 6, 8]), ([4, 3, 8], [4, 8, 6])])
def test_fused_expert_orientations_have_same_logical_dimensions(down, gate):
    ts = {"model.layers.0.mlp.experts.down_proj": tensor(down), "model.layers.0.mlp.experts.gate_up_proj": tensor(gate)}
    f = weights.shape_facts(ts, None)
    assert (f["expert_input_dim"], f["expert_intermediate"], f["ffn_gated"]) == (8, 3, True)
    assert weights.count(ts, None)["experts_per_moe_layer"] == [4]


def test_new_attention_kinds_are_read_from_tensor_mechanisms():
    ts = {
        "model.layers.0.linear_attn.A_log": tensor([4]),
        "model.layers.1.self_attn.indexer.index_qk_proj.weight": tensor([4, 8]),
        "model.layers.2.attn.wkv.weight": tensor([4, 8]),
    }
    assert weights.layer_kinds(ts) == ["linear_delta", "qsa", "compressed_attention"]


def test_fused_shared_module_does_not_contradict_an_explicit_logical_expert_count():
    facts = {
        k: (v, "config:" + k)
        for k, v in dict(
            num_layers=2,
            moe_num_experts=4,
            moe_active_experts=1,
            moe_shared_experts=2,
            moe_expert_intermediate=3,
            dense_layers=0,
        ).items()
    }
    checks = verify.verify(facts, dict(total=10, moe_layers=2), dict(shared_experts=1), None, {}, {})
    shared = next(c for c in checks if c["field"] == "moe_shared_experts")
    assert shared["published"] and shared["value"] == 2


def test_dory_gated_mlp_and_recurrence_are_not_a_standard_decoder_graph():
    ts = {
        "backbone.layers.1.mixer.gate_proj.weight": tensor([12, 8]),
        "backbone.layers.1.mixer.up_proj.weight": tensor([12, 8]),
    }
    assert weights.shape_facts(ts, None)["ffn_gated"] is True
    cfg = dict(
        model_type="dory",
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        layer_types=["full_attention", "mlp"],
        n_recurrent_loops=2,
        n_recurrent_layers=2,
        n_input_layers=0,
        n_output_layers=0,
        recurrent_kv_cache_mode="per_loop",
    )
    f = extract.extract(cfg)
    assert f["generation_type"][0] == "latent_recurrent"
    assert f["recurrent_layers"][0] == 2
    with pytest.raises(PublicationError, match="not qualified"):
        render_values({"published": {k: v[0] for k, v in f.items()}})


def test_llama4_alternates_dense_and_fused_moe_layers():
    cfg = dict(
        model_type="llama4_text",
        num_hidden_layers=48,
        hidden_size=5120,
        num_attention_heads=40,
        num_local_experts=128,
        num_experts_per_tok=1,
        intermediate_size=8192,
        intermediate_size_mlp=16384,
        interleave_moe_layer_step=2,
    )
    f = extract.extract(cfg)
    assert f["dense_layers"][0] == 24
    assert f["dense_ffn_intermediate"][0] == 16384
    assert f["moe_expert_intermediate"][0] == 8192


def test_first_run_labels_and_results_are_immutable_evidence():
    result = json.loads((ROOT / "results/newest-first-run/summary.json").read_text())
    assert result["heldout"]["correct"] == 97 and result["heldout"]["total"] == 139
    for name, digest in result["label_hashes"].items():
        assert hashlib.sha256((ROOT / "evals/heldout/newest-2026-11-14" / name).read_bytes()).hexdigest() == digest


def test_diff_exposes_generation_and_shared_cache_changes():
    old = dict(published=dict(generation_type="autoregressive", shared_kv_layers=0))
    new = copy.deepcopy(old)
    new["published"].update(generation_type="block_diffusion", shared_kv_layers=9)
    assert {r["field"] for r in pipeline.diff(new, old)} == {"generation_type", "shared_kv_layers"}


def test_speculative_decoding_heads_are_auxiliary_and_not_active():
    import json
    from pathlib import Path

    from archwatch import weights

    assert weights._component("mtp_dspark.backbone.attention.wq.weight") == "auxiliary"
    assert weights._component("model.mtp.layers.0.mlp.weight") == "auxiliary"
    assert weights._component("model.layers.3.smtp_proj.weight") != "auxiliary"
    rec = json.loads(
        (Path(__file__).resolve().parents[1] / "data/models/mistralai__Mistral-Large-4-1T-A52B.json").read_text()
    )
    # Card: "52 billion including embeddings and output layers".
    assert rec["published"]["params_active"] == 52_354_527_520
