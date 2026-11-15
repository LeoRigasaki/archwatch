"""Regression tests for the pipeline errors found by the reference audit, and for the audit's own boundaries."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from archwatch import hub, pipeline, weights
from archwatch.extract import extract_with_usage

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def t(shape, dtype="BF16", **extra):
    return dict(dtype=dtype, shape=shape, **extra)


# ---------------------------------------------------------------- counting errors the audit exposed


def test_mxfp4_blocks_hold_two_values_per_byte_and_scales_are_metadata():
    # gpt-oss: experts.*_blocks are U8 with two FP4 values per byte; *_scales are block scales.
    blocks = t([32, 2880, 90, 16], "U8")
    assert weights.logical_elements("model.layers.1.mlp.experts.down_proj_blocks", blocks) == 32 * 2880 * 2880
    assert weights.logical_elements("model.layers.1.mlp.experts.down_proj_scales", t([32, 2880, 90], "U8")) is None


def test_native_fp4_experts_flagged_on_the_model_config():
    # DeepSeek-V4 sets expert_dtype on config.json, not inside quantization_config.
    assert pipeline.native_expert_bits_of({"expert_dtype": "fp4", "quantization_config": {"quant_method": "fp8"}}) == 4
    assert pipeline.native_expert_bits_of({"text_config": {"expert_dtype": "fp4"}}) == 4
    assert pipeline.native_expert_bits_of({"quantization_config": {"expert_dtype": "fp4"}}) == 4
    assert pipeline.native_expert_bits_of({}) is None


def test_consolidated_export_is_not_counted_twice_but_extra_components_are_kept(monkeypatch):
    info = dict(
        sha="a" * 40,
        files=[
            "consolidated.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "streammind_gate.safetensors",
        ],
    )
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    monkeypatch.setattr(hub, "get_json", lambda repo, path, revision=None: index)
    assert hub.weight_shards("x/y", info) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "streammind_gate.safetensors",  # a separate component the index does not list (Mage-VL's gate)
    ]


def test_tensor_parallel_slices_count_every_rank(tmp_path, monkeypatch):
    # grok-2: -TP-000..007 files hold slices of the same tensor names.
    info = dict(
        sha="b" * 40, files=["m-00001-TP-common.safetensors"] + [f"m-00002-TP-00{i}.safetensors" for i in range(8)]
    )
    headers = {"m-00001-TP-common.safetensors": {"embed.weight": t([10, 4])}}
    headers.update({f"m-00002-TP-00{i}.safetensors": {"layers.0.w.weight": t([2, 4])} for i in range(8)})
    monkeypatch.setattr(hub, "CACHE", tmp_path)
    monkeypatch.setattr(hub, "_header", lambda repo, shard, sha: headers[shard])
    merged = hub.safetensors_headers("x/y", info=info)
    assert merged["layers.0.w.weight"]["tp_parts"] == 8
    assert weights.logical_elements("layers.0.w.weight", merged["layers.0.w.weight"]) == 8 * 2 * 4
    assert weights.block_shapes(merged, ["attn"], num_layers=1) is None  # per-rank shapes are not logical


def test_mtp_layers_after_the_decoder_stack_are_auxiliary_not_active():
    # DeepSeek-V3 / GLM: the MTP predictor is stored as layers.<num_layers>.
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": t([4, 4]),
        "model.layers.0.mlp.experts.0.w1.weight": t([4, 4]),
        "model.layers.0.mlp.experts.1.w1.weight": t([4, 4]),
        "model.layers.1.self_attn.q_proj.weight": t([4, 4]),  # MTP layer (num_layers = 1)
        "model.layers.1.mlp.experts.0.w1.weight": t([4, 4]),
    }
    counts = weights.count(tensors, None, num_layers=1)
    assert counts["by_component"]["auxiliary"] == 32
    assert counts["routed_expert_params"] == 32 and counts["total"] == 80
    active = weights.active_params(counts, 2, 1)
    assert active["text_active"] == 80 - 16 - 32  # one unselected expert and the MTP layer excluded
    assert active["with_vision"] == active["text_active"]  # auxiliary heads stay excluded


def test_multimodal_projectors_are_not_text_weights():
    for name in (
        "multi_modal_projector.linear_1.weight",
        "patch_merger.merging_layer.weight",
        "mm_projector.proj.0.weight",
    ):
        assert weights._component(name) == "vision_projector", name


def test_native_output_heads_are_recognized():
    # DeepSeek-native head.weight and Mistral-native output.weight are separate output heads (not tied).
    for name in ("head.weight", "output.weight", "lm_head.weight"):
        assert weights._component(name) == "lm_head"
        facts = weights.shape_facts({name: t([8, 4]), "embed.weight": t([8, 4])}, None)
        assert facts.get("has_lm_head") is True, name


def test_moe_keys_with_moe_prefix_are_read():
    # Step-3.5: moe_num_experts / moe_top_k.
    facts, _ = extract_with_usage(
        dict(hidden_size=8, num_hidden_layers=1, num_attention_heads=1, vocab_size=4, moe_num_experts=288, moe_top_k=8)
    )
    assert facts["moe_num_experts"][0] == 288 and facts["moe_active_experts"][0] == 8
    assert facts["decoder_type"][0] == "MoE"


# ---------------------------------------------------------------- comparison normalisation


def test_layer_mix_counts_leading_terms_and_skips_mtp():
    ref = load_tool("reference_agreement")
    assert ref.parse_layers("4 GQA + 21 Mamba-2 + 17 FFN") == (42, 0)
    assert ref.parse_layers("80 GQA + 1 MTP layer") == (80, 0)
    assert ref.parse_layers("10 MLA + 70 Lightning Attention + 1 MTP layer") == (80, 0)
    assert ref.same_k_shorthand(131072, 128000) and ref.same_k_shorthand(1048576, 1_000_000)
    assert not ref.same_k_shorthand(163840, 128000)


def test_rounding_normalisation_is_capped():
    audit = load_tool("reference_audit")
    assert audit._sig_tolerance("230", 1e9) == 5e9  # 2 significant figures
    assert audit._sig_tolerance("2.5", 1e9) == 0.05e9
    assert audit._sig_tolerance("400", 1e9) == 50e9  # capped at 2.5% of the value where it is applied


# ---------------------------------------------------------------- audit boundaries and outcome


def saved():
    return json.loads((ROOT / "results/reference-audit.json").read_text())


def test_every_difference_has_exactly_one_group_with_evidence():
    audit = load_tool("reference_audit")
    rows = saved()["rows"]
    diffs = [r for r in rows if not r["agree"]]
    assert diffs and all(r["group"] in audit.GROUPS and len(r["evidence"]) > 40 for r in diffs)
    counts = {g: sum(r["group"] == g for r in diffs) for g in audit.GROUPS}
    assert counts["ours_wrong"] == 0 and counts["unresolved"] == 0


def test_his_wrong_entries_cite_a_file_or_key_at_a_pinned_commit():
    for r in saved()["rows"]:
        if not r["agree"] and r["group"] == "his_wrong":
            assert re.search(r"(README\.md|config\.json)", r["evidence"]) and re.search(
                r"\b[0-9a-f]{10}\b", r["evidence"]
            )


def test_reference_values_never_reach_records_site_or_gold():
    source = (ROOT / "tools/reference_audit.py").read_text()
    writes = set(re.findall(r'\(out / "([^"]+)"\)\.write_text', source))
    assert writes == {"reference-audit.json", "reference-audit.md"}
    code = source.split('"""', 2)[2]  # the module docstring describes the boundary; check the code itself
    assert "data/models" not in code
    for path in list((ROOT / "src").rglob("*.py")) + [ROOT / "build_site.py", ROOT / "run.py", ROOT / "watch.py"]:
        text = path.read_text()
        assert "reference_audit" not in text and "reference-audit" not in text, path
    assert "reference-audit" not in (ROOT / "site/index.html").read_text()


def test_audit_workflow_is_weekly_change_gated_and_llm_free():
    import yaml

    flow = yaml.safe_load((ROOT / ".github/workflows/reference-audit.yml").read_text())
    assert flow[True]["schedule"] == [{"cron": "11 4 * * 0"}]
    text = (ROOT / ".github/workflows/reference-audit.yml").read_text()
    assert "steps.changed.outputs.run == 'true'" in text and "GEMINI" not in text
    assert "git diff --quiet -- data/gemini-requests.json" in text and "needs-review" in text


@pytest.mark.parametrize("repo", ["openai/gpt-oss-120b", "deepseek-ai/DeepSeek-V4-Flash", "xai-org/grok-2"])
def test_audited_totals_now_match_the_model_cards(repo):
    row = next(r for r in saved()["rows"] if r["model"] == repo and r["field"] == "params_total")
    assert row["agree"] or row["group"] == "definition"
