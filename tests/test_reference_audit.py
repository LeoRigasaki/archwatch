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
    assert ref.parse_layers("3 GQA + 15 Mamba-2 + 12 FFN") == (42, 0)
    assert ref.parse_layers("80 GQA + 1 MTP layer") == (80, 0)
    assert ref.parse_layers("8 MLA + 40 Lightning Attention + 1 MTP layer") == (80, 0)
    assert ref.same_k_shorthand(131072, 128000) and ref.same_k_shorthand(1048576, 1_000_000)
    assert not ref.same_k_shorthand(163840, 128000)


def test_his_value_becomes_an_accepted_range():
    audit = load_tool("reference_audit")
    lo, hi, _ = audit.his_range("params_active", "30B total, 3B active", None)
    assert (lo, hi) == (2.5e9, 3.5e9)  # card rounding: '3B' covers 2.5-3.5B
    lo, hi, _ = audit.his_range("params_total", "671B total, 37B active", (671e9, 0.5e9))
    assert lo == 671e9 * 0.98 and hi == 671e9 * 1.02  # 2% is wider than the rounding here
    lo, hi, label = audit.his_range("params_active", "400B backbone; 6B active prefill, 12B active decode", None)
    assert lo == 8e9 * 0.98 and hi == 16e9 * 1.02 and "prefill" in label


def test_lookup_tables_are_stored_but_not_active():
    tensors = {
        "layers.0.engram.embed.weight": t([1000, 10]),
        "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight": t([500, 10]),
        "layers.0.attn.wq.weight": t([10, 10]),
        "layers.0.ffn.experts.0.w1.weight": t([10, 10]),
        "layers.0.ffn.experts.1.w1.weight": t([10, 10]),
    }
    counts = weights.count(tensors, None, num_layers=1)
    assert counts["by_component"]["lookup_embedding"] == 15000 and counts["total"] == 15300
    assert weights.active_params(counts, 2, 1)["text_active"] == 15300 - 15000 - 100


def test_fp4_weights_with_block_scales_hold_two_values_per_byte():
    tensors = {
        # NVFP4 (ModelOpt): U8 weight + FP8 scale per 16 values + a scalar second scale
        "backbone.layers.1.mixer.experts.0.up_proj.weight": t([1856, 1344], "U8"),
        "backbone.layers.1.mixer.experts.0.up_proj.weight_scale": t([1856, 168], "F8_E4M3"),
        # MXFP4 (MiMo-V2.6): U8 weight + U8 scale per 32 values
        "model.layers.1.mlp.experts.0.gate_proj.weight": t([2048, 2048], "U8"),
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale": t([2048, 128], "U8"),
        # INT8 with a per-row scale is not 4-bit
        "model.layers.1.mlp.down_proj.weight": t([64, 64], "I8"),
        "model.layers.1.mlp.down_proj.weight_scale": t([64, 1], "F32"),
    }
    weights.annotate_packed_fp4(tensors)
    assert weights.logical_elements("x", tensors["backbone.layers.1.mixer.experts.0.up_proj.weight"]) == 1856 * 2688
    assert weights.logical_elements("x", tensors["model.layers.1.mlp.experts.0.gate_proj.weight"]) == 2048 * 4096
    assert not tensors["model.layers.1.mlp.down_proj.weight"].get("fp4_packed")


def test_more_embedding_and_projector_names():
    assert weights._component("model.word_embeddings.weight") == "embedding"
    assert weights._component("patch_merge_mlp.linear_1.weight") == "vision_projector"


def test_active_excluding_embeddings_is_published():
    counts = {
        "text_total": 100,
        "total": 110,
        "routed_expert_params": 40,
        "by_component": {"embedding": 7, "lm_head": 5, "vision": 10, "moe_routed_experts": 40},
    }
    active = weights.active_params(counts, 4, 1)
    assert active["text_active"] == 70 and active["text_active_non_embedding"] == 58


def test_card_ranges_hold_for_lookup_table_models():
    # Published gallery records: DeepSeek-V4.1-Flash card "8B prefill, 16B decode"; Qwen3.8-Flash-Next "6B activated".
    ds = json.loads((ROOT / "data/models/deepseek-ai__DeepSeek-V4.1-Flash.json").read_text())["published"]
    qw = json.loads((ROOT / "data/models/Qwen__Qwen3.8-Flash-Next.json").read_text())["published"]
    assert 8e9 * 0.98 <= ds["params_active_non_embedding"] <= 16e9 * 1.02
    assert 5.5e9 <= qw["params_active_non_embedding"] <= 6.5e9
    assert ds["params_active"] < 20e9 and qw["params_active"] < 10e9  # lookup tables no longer counted as active


# ---------------------------------------------------------------- audit boundaries and outcome


def saved():
    path = ROOT / "cache" / "reference" / "audit.json"
    if not path.exists():
        pytest.skip("no local audit report (cache/reference/ is built on demand and never committed)")
    return json.loads(path.read_text())


def test_every_difference_has_exactly_one_group_and_definitions_are_recomputed():
    audit = load_tool("reference_audit")
    diffs = [r for r in saved()["rows"] if not r["agree"]]
    assert diffs and all(r["group"] in audit.GROUPS and r["evidence"] for r in diffs)
    assert not [r for r in diffs if r["group"] == "unresolved"]
    for r in diffs:
        if r["group"] == "definition":
            # The evidence states the rule and OUR recomputed number, and that number is inside his accepted range.
            assert "Rule:" in r["evidence"] and "Ours recomputed:" in r["evidence"] and r["recomputed"] is not None
            lo, hi = (
                float(x.replace(",", "")) for x in re.search(r"accepted ([\d,]+)-([\d,]+)", r["evidence"]).groups()
            )
            assert lo <= r["recomputed"] <= hi, r
        if r["group"] == "card_unreproducible":
            assert r["evidence"].startswith("Checked: ")
    assert not [r for r in diffs if r["group"] == "ours_wrong"]  # no confirmed bug left open


def test_his_wrong_entries_cite_a_file_or_key_at_a_pinned_commit():
    for r in saved()["rows"]:
        if not r["agree"] and r["group"] == "his_wrong":
            assert re.search(r"(README\.md|config\.json)", r["evidence"]) and re.search(
                r"\b[0-9a-f]{10}\b", r["evidence"]
            )


def test_reference_values_never_reach_records_site_or_gold():
    source = (ROOT / "tools/reference_audit.py").read_text()
    writes = set(re.findall(r'\(out / "([^"]+)"\)\.write_text', source))
    assert writes == {"audit.json", "audit.md"} and "out = ref.REPORTS" in source
    code = source.split('"""', 2)[2]  # the module docstring describes the boundary; check the code itself
    assert "data/models" not in code
    for path in list((ROOT / "src").rglob("*.py")) + [ROOT / "build_site.py", ROOT / "run.py", ROOT / "watch.py"]:
        text = path.read_text()
        assert "reference_audit" not in text and "reference-audit" not in text, path
    assert "reference-audit" not in (ROOT / "site/index.html").read_text()
    assert not list((ROOT / "results").glob("reference-*"))  # reports live only in the ignored cache


def test_audit_workflow_is_manual_commits_nothing_and_uploads_an_artifact():
    import yaml

    flow = yaml.safe_load((ROOT / ".github/workflows/reference-audit.yml").read_text())
    on = flow[True]
    assert set(on) == {"workflow_dispatch"} and on["workflow_dispatch"]["inputs"]["repo"]["required"] is True
    assert "default" not in on["workflow_dispatch"]["inputs"]["repo"]  # the reference is never named in the repo
    assert flow["permissions"] == {"contents": "read"}
    text = (ROOT / ".github/workflows/reference-audit.yml").read_text()
    assert "create-pull-request" not in text and "actions/upload-artifact" in text and "GEMINI" not in text
    assert "git diff --quiet" in text


@pytest.mark.parametrize("repo", ["openai/gpt-oss-120b", "deepseek-ai/DeepSeek-V3", "xai-org/grok-2"])
def test_audited_totals_now_match_the_model_cards(repo):
    row = next(r for r in saved()["rows"] if r["model"] == repo and r["field"] == "params_total")
    assert row["agree"] or row["group"] == "definition"


def test_step_stacked_experts_and_share_expert_are_recognized():
    assert weights._component("model.layers.3.moe.gate_proj.weight") == "moe_routed_experts"
    assert weights._component("model.layers.3.share_expert.up_proj.weight") == "moe_shared_experts"
    tensors = {
        "model.layers.0.moe.gate_proj.weight": t([8, 4, 2]),
        "model.layers.0.moe.up_proj.weight": t([8, 4, 2]),
        "model.layers.0.moe.down_proj.weight": t([8, 2, 4]),
        "model.layers.0.share_expert.up_proj.weight": t([4, 2]),
    }
    counts = weights.count(tensors, None, num_layers=1)
    assert counts["routed_expert_params"] == 192 and counts["experts_per_moe_layer"] == [8]
    assert weights.active_params(counts, 8, 2)["active"] == 192 // 4 + 8


def test_int32_pack_quantized_experts_hold_eight_int4_values():
    assert weights.logical_elements("m.experts.0.up_proj.weight_packed", t([16, 4], "I32"), 4) == 16 * 4 * 8
    assert weights.logical_elements("m.experts.0.up_proj.weight_shape", t([2], "I32"), 4) is None


def test_inkling_embed_and_unembed_are_embedding_and_head():
    assert weights._component("model.llm.embed.weight") == "embedding"
    assert weights._component("model.llm.unembed.weight") == "lm_head"
    assert weights._component("mtp.2.markov_head.embed.weight") == "auxiliary"  # MTP stays auxiliary
    assert weights._component("visual.patch_embed.weight") == "vision"


# ---------------------------------------------------------------- one fixed rule, card-stated exclusions only


def fake_record(repo="org/Model-X", total=110, active=12, **comp):
    by = dict(embedding=5, lm_head=5, vision=0, auxiliary=0, lookup_embedding=0)
    by.update(comp)
    return dict(
        id=repo,
        source_commit="0" * 40,
        published=dict(params_total=total, params_active=active),
        params=dict(counted_from_weights=dict(by_component=by, tensor_count=10), active={}),
    )


def classify_one(monkeypatch, record, field, his, card=""):
    audit = load_tool("reference_audit")
    monkeypatch.setattr(audit, "_card_text", lambda r: card)
    parsed = None
    if field == "params_total":
        number, unit = audit._his_number(field, his)
        parsed = (float(number) * audit.UNIT[unit], 0.0)
    return audit.close(record, field, his, parsed, None)


def test_no_search_over_exclusions_without_a_card_statement(monkeypatch):
    # Excluding embedding + head would give exactly 100B, but the card does not say so: no definition.
    record = fake_record(total=110e9, active=12e9, embedding=5e9, lm_head=5e9)
    row = classify_one(monkeypatch, record, "params_total", "100B total", card="It has 100B parameters.")
    assert row["group"] == "card_unreproducible" and row["recomputed"] == 110e9 and row["rule"].startswith("default:")
    assert row["evidence"].startswith("Checked: single safetensors file, all 10 tensor headers read")


def test_unread_tensors_are_ours_wrong_not_card_unreproducible(monkeypatch):
    audit = load_tool("reference_audit")
    monkeypatch.setattr(audit, "_card_text", lambda r: "")
    monkeypatch.setattr(audit, "tensors_read", lambda r: dict(complete=False, text="3 tensors missing."))
    row = audit.close(fake_record(total=110e9), "params_total", "100B total", (100e9, 0.0), None)
    assert row["group"] == "ours_wrong" and "3 tensors missing" in row["evidence"]


def test_card_stated_non_embedding_also_drops_the_head(monkeypatch):
    # Published active already excludes the input embedding (12B - 1B); "non-embedding" also drops the 1B head.
    record = fake_record(total=110e9, active=11e9, embedding=1e9, lm_head=1e9)
    card = "Model-X is a MoE model with 10B non-embedding active parameters."
    row = classify_one(monkeypatch, record, "params_active", "110B total, 10B active", card=card)
    assert row["group"] == "definition" and row["recomputed"] == 10e9
    assert "card states: excluding output head" in row["rule"] and row["card_quote"] == [card]


def test_card_footnote_keeping_the_head_matches_the_default(monkeypatch):
    record = fake_record(total=40e9, active=2.5e9, embedding=0.3e9, lm_head=0.3e9)
    card = (
        "| Active parameters per token | Approximately 2.5B, including the output head¹ |\n"
        "¹ The convention includes the output head and excludes the full input embedding matrix."
    )
    row = classify_one(monkeypatch, record, "params_active", "40B total, 2.5B active", card=card)
    assert row["group"] == "definition" and row["recomputed"] == 2.5e9 and "card states" not in row["rule"]


def test_mtp_is_not_subtracted_twice_and_vision_inclusion_is_card_stated(monkeypatch):
    # Published total already leaves MTP out (685B stored = 671B + 14B MTP).
    record = fake_record(total=671e9, active=37e9, auxiliary=14e9, embedding=1e9)
    card = "The total size is 685B, which includes 671B of the Main Model weights and 14B of the MTP Module weights."
    row = classify_one(monkeypatch, record, "params_total", "671B total", card=card)
    assert row["group"] == "definition" and row["recomputed"] == 671e9
    record = fake_record(total=29.7e9, vision=0.5e9)
    row = classify_one(
        monkeypatch, record, "params_total", "29.6B", card="| Parameters | 29.6B (including vision encoder) |"
    )
    assert row["group"] == "definition" and row["recomputed"] == 29.7e9
    row = classify_one(monkeypatch, record, "params_total", "29.6B", card="It has 29.6B parameters.")
    assert row["recomputed"] == 29.2e9  # default: text-only figure, vision removed


def test_a_card_part_of_a_summed_figure_is_not_an_exclusion(monkeypatch):
    audit = load_tool("reference_audit")
    monkeypatch.setattr(
        audit,
        "_card_text",
        lambda r: "Number of Parameters: 125B with 6B activated, plus 51B n-gram embedding and 4B MTP",
    )
    record = fake_record(total=177e9, lookup_embedding=51e9)
    rule, value, quotes = audit.fixed_rule(record, "params_total", "100B core; plus 40B n-gram", his_value=180e9)
    assert value == 177e9 and not quotes


def test_name_rounding_needs_ours_within_half_a_unit(monkeypatch):
    record = fake_record(repo="org/Model-30B-A3B", total=30e9, active=3.3e9)  # 10% off, inside 2.5-3.5B
    row = classify_one(monkeypatch, record, "params_active", "30B total, 3B active")
    assert row["group"] == "name_rounding" and "model name" in row["evidence"]
    record = fake_record(repo="org/Model-30B-A3B", total=30e9, active=3.6e9)  # outside 2.5-3.5B
    assert classify_one(monkeypatch, record, "params_active", "30B total, 3B active")["group"] == "card_unreproducible"
    card = "It has ~428B parameters and ~23B activated parameters."
    row = classify_one(
        monkeypatch, fake_record(total=430e9, active=23.48e9), "params_active", "428B, 23B active", card=card
    )
    assert row["group"] == "name_rounding" and row["card_quote"] == [card]
    row = classify_one(
        monkeypatch, fake_record(total=430e9, active=24e9), "params_active", "428B, 23B active", card=card
    )
    assert row["group"] == "card_unreproducible"
    row = classify_one(
        monkeypatch, fake_record(total=430e9, active=23.1e9), "params_active", "428B, 23B active", card=card
    )
    assert row["group"] == "definition"  # within 2%: plain agreement under the default rule


def test_active_default_excludes_the_input_embedding_and_keeps_the_head():
    untied = {
        "text_total": 100,
        "total": 110,
        "routed_expert_params": 40,
        "by_component": {"embedding": 7, "lm_head": 5, "vision": 10, "moe_routed_experts": 40},
    }
    assert weights.active_params(untied, 4, 1)["active"] == 70 - 7
    tied = dict(untied, by_component={"embedding": 7, "vision": 10, "moe_routed_experts": 40})
    assert weights.active_params(tied, 4, 1)["active"] == 70  # the shared matrix counted once, as the head


def test_mtp_modules_are_published_separately_from_the_total():
    tensors = {
        "model.embed_tokens.weight": t([10, 4]),
        "model.layers.0.mlp.up_proj.weight": t([8, 4]),
        "model.layers.1.mlp.up_proj.weight": t([8, 4]),  # stored after the 1-layer decoder stack: MTP
        "model.mtp.layers.0.proj.weight": t([4, 4]),
    }
    counts = weights.count(tensors, None, num_layers=1)
    assert counts["mtp_params"] == 32 + 16 and counts["headline_total"] == counts["total"] - 48
    rec = json.loads((ROOT / "data/models/zai-org__GLM-5.2.json").read_text())["published"]
    assert rec["params_mtp"] == 9_952_920_576 and rec["params_total"] == 743_377_019_904


def test_card_active_figure_is_recorded_but_does_not_block_publication():
    rec = json.loads((ROOT / "data/models/LiquidAI__LFM2.5-24B-A2B-Base.json").read_text())
    check = next(c for c in rec["checks"] if c["field"] == "params_active")
    assert check["published"] and check["card_reference"]["raw"] == "2.45B"
    assert all(c["evidence"].startswith("weights:") for c in check["candidates"])  # the card is not a gate


# ---------------------------------------------------------------- card figures on the site and as a gate


def test_site_marks_active_that_differs_from_the_card_and_never_calls_it_verified():
    site = load_tool_path(ROOT / "build_site.py", "build_site")
    differs = dict(
        field="params_active",
        status="verified",
        card_reference=dict(value=23e9, raw="~23B", agrees=False),
    )
    assert site.card_status(differs) == ("differs from card", 'card says "~23B"')
    agrees = dict(differs, status="single source", card_reference=dict(value=2.45e9, raw="2.45B", agrees=True))
    assert site.card_status(agrees) == ("single source", None)


def test_card_total_gate_accepts_half_a_percent_or_the_card_precision():
    from archwatch import verify

    ours = 196_956_130_368  # Step-3.5-Flash, MTP reported separately; card "196.81B" (0.07% away)
    card = dict(value=196.81e9, tol=0.005e9, evidence="card:params_total", raw="196.81B")
    row = verify.check("params_total", [dict(value=ours, evidence="weights:sum"), verify.card_total(card)])
    assert row["published"] and row["status"] == "verified"
    off = dict(card, value=ours * 1.01)  # 1% away: still a conflict
    assert (
        verify.check("params_total", [dict(value=ours, evidence="weights:sum"), verify.card_total(off)])["status"]
        == "conflict"
    )
    coarse = dict(value=30e9, tol=0.5e9, evidence="card:params_total", raw="30B")  # card precision wins
    assert verify.card_total(coarse)["tol"] == 0.5e9


def load_tool_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
