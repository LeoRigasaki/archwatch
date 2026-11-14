"""Small primary-source-shaped fixtures; no Hub access or model code execution."""

from collections import Counter
from xml.etree import ElementTree as ET

import pytest
import yaml

from archwatch import calc, codescan, diagram, extract, verify, weights


def config(**overrides):
    return dict(
        model_type="toy",
        num_hidden_layers=2,
        hidden_size=8,
        vocab_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=24,
        max_position_embeddings=128,
        **overrides,
    )


def tensor(shape):
    return {"dtype": "BF16", "shape": shape}


def record(cfg, tensors):
    facts = extract.extract(cfg)
    checks = verify.verify(
        facts,
        weights.count(tensors, None),
        weights.shape_facts(tensors, None),
        None,
        {},
        {},
        weights.layer_kinds(tensors),
    )
    return dict(
        id="lab/example", published={c["field"]: c["value"] for c in checks if c.get("published")}, checks=checks
    )


def test_mistral_all_layer_window_and_gemma_periodic_pattern():
    cfg = config()
    cfg.update(model_type="mistral", num_hidden_layers=32, sliding_window=4096)
    assert extract.extract(cfg)["layer_types"][0] == ["sliding_attention"] * 32
    cfg["sliding_window"] = None
    assert extract.extract(cfg)["layer_types"][0] == ["full_attention"] * 32
    cfg.update(model_type="gemma3_text", num_hidden_layers=26, sliding_window=512, sliding_window_pattern=6)
    assert Counter(extract.extract(cfg)["layer_types"][0]) == {"sliding_attention": 22, "full_attention": 4}


def test_nested_rope_facts_are_portable_yaml_and_consumed():
    cfg = config()
    cfg.update(rope_theta=500000, rope_scaling={"factor": 8, "rope_type": "llama3"})
    facts, usage = extract.extract_with_usage(cfg)
    assert yaml.safe_load(yaml.safe_dump(facts["rope"][0]))["scaling"]["factor"] == 8
    assert "rope_scaling.factor" in usage["consumed"]
    assert "rope_scaling.rope_type" not in usage["unconsumed"]


@pytest.mark.parametrize(
    "suffix,rows,gated", [("mlp.gate_proj", 24, True), ("mlp.fc1", 24, False), ("mlp.gate_up_proj", 48, True)]
)
def test_qwen_llama_phi_glm_ffn_tensor_layouts(suffix, rows, gated):
    t = {f"model.layers.0.{suffix}.weight": tensor([rows, 8]), "vision.layers.0.mlp.fc1.weight": tensor([99, 11])}
    shape = weights.shape_facts(t, None)
    assert shape["ffn_intermediate"] == 24
    assert shape["ffn_gated"] is gated


def test_deepseek_expert_gate_alias_and_packed_gate_evidence():
    t = {"model.layers.1.mlp.experts.0.gate_proj.weight": tensor([1408, 2048])}
    shape = weights.shape_facts(t, None)
    assert shape["expert_intermediate"] == 1408
    assert shape["expert_input_dim"] == 2048
    t = {"model.layers.0.mlp.experts.0.w1.weight_packed": {"dtype": "U8", "shape": [8, 4]}}
    assert "ffn_gated" not in weights.shape_facts(t, 4)
    t["model.layers.0.mlp.experts.0.w3.weight_packed"] = {"dtype": "U8", "shape": [8, 4]}
    assert weights.shape_facts(t, 4)["ffn_gated"] is True


def test_post_ffn_norm_is_evidence_and_no_rmsnorm_default_is_fabricated():
    assert "norm" not in extract.extract(config())
    assert weights.shape_facts({"model.layers.0.post_feedforward_layernorm.weight": tensor([8])}, None)["sandwich_norm"]


def test_lfm_silu_is_cited_by_static_scan_without_running_remote_code():
    code = "raise RuntimeError('must never execute')\nclass Lfm3MLP:\n def forward(self, x):\n  return self.w2(F.silu(self.w1(x)) * self.w3(x))\n"
    scan = codescan.scan({"modeling_lfm3.py": code})
    assert scan["facts"]["activation"] == ("silu", "code:modeling_lfm3.py#L4")


def nemotron():
    cfg = config()
    cfg.update(
        model_type="nemotron_h",
        num_hidden_layers=3,
        hybrid_override_pattern="M-*",
        mamba_num_heads=2,
        mamba_head_dim=4,
        ssm_state_size=3,
        conv_kernel=4,
        n_groups=1,
        mlp_hidden_act="relu2",
        attention_head_dim=4,
    )
    del cfg["head_dim"]
    tensors = {
        "backbone.embeddings.weight": tensor([16, 8]),
        "backbone.layers.0.mixer.A_log": tensor([2]),
        "backbone.layers.0.mixer.conv1d.weight": tensor([14, 1, 4]),
        "backbone.layers.1.mixer.up_proj.weight": tensor([24, 8]),
        "backbone.layers.1.mixer.down_proj.weight": tensor([8, 24]),
        "backbone.layers.2.mixer.q_proj.weight": tensor([8, 8]),
    }
    return cfg, tensors


def test_nemotron_pattern_tensor_aliases_and_head_dim_provenance():
    cfg, tensors = nemotron()
    facts = extract.extract(cfg)
    assert facts["layer_types"][0] == ["mamba2", "mlp", "full_attention"]
    assert facts["head_dim"] == (4, "config:attention_head_dim")
    assert weights.layer_kinds(tensors) == ["mamba2", "mlp", "attention"]
    rec = record(cfg, tensors)
    assert rec["published"]["mamba_conv_channels"] == 14
    assert rec["published"]["ffn_gated"] is False
    assert rec["published"]["hidden_size"] == 8
    cfg["hybrid_override_pattern"] = "M?*"
    with pytest.raises(ValueError, match="pattern"):
        extract.extract(cfg)


def test_mamba_state_budget_hand_math_and_mlp_zero_state():
    cfg, tensors = nemotron()
    v = record(cfg, tensors)["published"]
    memory = calc.cache_model(v)
    assert memory["grows_per_token_bytes"] == 2 * 1 * 4 * 2
    assert memory["constant_bytes"] == 2 * 4 * 3 * 4 + 14 * 4 * 2
    assert memory["constant_bytes_low"] == 2 * 4 * 3 * 2 + 14 * 4 * 2
    assert next(r for r in memory["breakdown"] if r["layer_type"] == "mlp")["constant_bytes"] == 0
    assert any("not qualified" in note for note in memory["assumptions"])


def test_standalone_and_parallel_graphs_do_not_invent_sequential_gated_ffns():
    cfg, tensors = nemotron()
    svg = diagram.render(record(cfg, tensors))
    assert "One operator per residual block" in svg
    assert "Mamba-2 x 1" in svg and "MLP x 1" in svg
    ET.fromstring(svg)
    cfg = config()
    cfg.update(model_type="phi", hidden_act="gelu_new")
    tensors = {"model.embed_tokens.weight": tensor([16, 8])}
    for i in range(2):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = tensor([8, 8])
        tensors[f"model.layers.{i}.mlp.fc1.weight"] = tensor([24, 8])
    svg = diagram.render(record(cfg, tensors))
    assert "Parallel residual block x 2" in svg and "Ungated FFN" in svg
    assert "Gated FFN" not in svg
    ET.fromstring(svg)
