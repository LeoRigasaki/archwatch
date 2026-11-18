"""Concept claims are scoped, evidence-backed and never inferred from missing settings."""

import copy
import json
from pathlib import Path

import pytest

from archwatch import concept_extract, concepts, generated
from archwatch.diagram_spec import build_spec
from archwatch.publication import PublicationError

ROOT = Path(__file__).resolve().parents[1]


def record(stem="moonshotai__Kimi-K3.1-Pro"):
    return json.loads((ROOT / "data/models" / (stem + ".json")).read_text())


def keys(values):
    return {hit["key"] for hit in concepts.from_values(values)}


def test_registry_covers_all_requested_names_with_unambiguous_aliases():
    names = [
        "MHA",
        "RMSNorm",
        "GQA",
        "RoPE",
        "QK-Norm",
        "MLA",
        "MoE",
        "MTP",
        "SWA",
        "NoPE",
        "Gated Attention",
        "Hybrid Attention",
        "Looped Transformer",
        "ShortConv",
        "Attention Residuals",
        "DSA",
        "LatentMoE",
        "KV Sharing",
        "PLE",
        "CSA/HCA",
        "mHC",
        "Attention Budgeting",
        "CCA",
        "SiLU",
        "IndexShare",
        "PolyNorm",
        "Qwen Sparse Attention",
        "Gated Residuals",
    ]
    assert len({concepts.canonical(name) for name in names}) == 28
    assert all(concepts.canonical(name) for name in names)
    assert concepts.canonical("MQA") == "gqa"
    registry = concepts.registry()
    assert len(registry) == 32 and sum(i["catalog"] == "requested" for i in registry.values()) == 28
    for item in registry.values():
        assert all(item[field] for field in ("summary", "mechanism", "memory", "verify", "flow", "rule", "scope"))
        assert item["source"]["url"].startswith("https://")


def test_absence_false_zero_and_similar_names_are_not_positive_evidence():
    assert not keys({})
    values = dict(
        layer_type_counts={"full_attention": 4},
        rms_norm_eps=1e-6,
        qk_norm=False,
        num_mtp_layers=0,
        params_mtp=0,
        moe_num_experts=0,
        hyper_connection_multiplier=4,
        hc_sinkhorn_iters=20,
        index_topk=2048,
        activation="situ",
        attention_output_gate=False,
    )
    matches = keys(values)
    assert not matches & {
        "rmsnorm",
        "qk-norm",
        "mtp",
        "moe",
        "mhc",
        "gated-residuals",
        "attention-budgeting",
        "silu",
        "nope",
    }
    assert "nope" not in keys(dict(values, rope={}))
    assert "silu" not in keys(dict(values, activation="relu2", mamba_activation="silu"))


def test_head_relationships_and_local_global_softmax_are_distinct():
    values = dict(
        layer_type_counts={"full_attention": 3, "sliding_attention": 1},
        attention_type="GQA",
        num_attention_heads=8,
        num_kv_heads=1,
        sliding_window=256,
    )
    hits = concepts.from_values(values)
    assert "gqa" in {h["key"] for h in hits} and "MQA subtype" in next(h["scope"] for h in hits if h["key"] == "gqa")
    assert "hybrid-attention" not in keys(values)
    assert "gqa" not in keys(dict(values, num_kv_heads=3))
    assert "gqa" not in keys(dict(values, layer_type_counts={"full_attention_mla": 4}))
    assert "hybrid-attention" in keys(dict(values, layer_type_counts={"full_attention": 3, "kda_linear": 1}))


def test_nested_convolution_scope_preserves_the_decoder_layer_sum():
    rec = record()
    before = copy.deepcopy(rec)
    hit = next(h for h in concepts.for_record(rec) if h["key"] == "short-convolution")
    assert "inside KDA" in hit["scope"] and "adds no decoder layers" in hit["scope"]
    assert rec == before and rec["published"]["num_layers"] == sum(rec["published"]["layer_type_counts"].values())
    assert "short-convolution" not in keys(dict(layer_type_counts={"kda_linear": 2}, linear_attn_conv_kernel=0))


def test_rope_does_not_promote_a_disabled_or_unused_path():
    base = dict(layer_type_counts={"full_attention": 4})
    assert "rope" not in keys(dict(base, rope={"rope_theta": 10000, "partial_rotary_factor": 0}))
    assert "rope" not in keys(dict(base, rope={"sliding_attention": {"rope_theta": 10000}}))
    assert "rope" in keys(dict(base, rope={"rope_theta": 10000, "partial_rotary_factor": 0.5}))


def test_every_match_retains_current_values_status_and_pinned_links():
    rec = record()
    checks = {c["field"]: c for c in rec["checks"]}
    for hit in concepts.for_record(rec):
        assert hit["scope"] and hit["fields"]
        for field in hit["fields"]:
            assert field["value"] == rec["published"][field["field"]] == checks[field["field"]]["value"]
            for source in field["sources"]:
                if source["url"]:
                    assert rec["source_commit"] in source["url"]
    single = next(h for h in concepts.for_record(rec) if h["key"] == "attention-residuals")
    assert single["status"] == "single source"
    tampered = copy.deepcopy(rec)
    next(c for c in tampered["checks"] if c["field"] == "mla_use_nope")["value"] = False
    assert "nope" not in {h["key"] for h in concepts.for_record(tampered)}


def test_held_derivative_and_conflicting_records_cannot_export_concepts():
    for stem in ["Qwen__Qwen3.8-Flash-Next", "LiquidAI__LFM3-1.7B"]:
        with pytest.raises(PublicationError):
            concepts.evidence_spec(record(stem))
    rec = record()
    next(c for c in rec["checks"] if c["field"] == "mla_use_nope")["status"] = "conflict"
    with pytest.raises(PublicationError):
        concepts.evidence_spec(rec)


def test_named_settings_need_complete_schedules_and_dedicated_fields():
    cfg = dict(
        num_hidden_layers=4,
        ple_embed_dim=16,
        ple_layer_ids=[2],
        indexer_types=["full", "shared", "full", "shared"],
        kv_source_layers=[0, 1, 1, 1],
        hc_mult=4,
        hc_sinkhorn_iters=20,
        index_topk=16,
        rms_norm_eps=1e-6,
    )
    facts = concept_extract.config_facts(cfg)
    assert set(facts) == {"per_layer_embeddings", "index_sharing", "kv_sharing"}
    assert facts["kv_sharing"][0]["source_layers"] == [0, 1, 1, 1]
    bad = dict(cfg, ple_layer_ids=[9], indexer_types=["shared"], kv_source_layers=[2, 1, 2, 3])
    assert not concept_extract.config_facts(bad)
    assert not concept_extract.config_facts(dict(num_hidden_layers=4, index_source_layer_ids=[1, 3]))
    explicit = dict(
        num_hidden_layers=2,
        compressed_attention_types=["csa", "hca"],
        residual_type="mhc",
        attention_mechanism="cca",
        attention_head_budget=[8, 4],
        normalization_type="RMSNorm",
    )
    assert set(concept_extract.config_facts(explicit)) == {
        "compressed_attention_types",
        "residual_mechanism",
        "attention_mechanism",
        "attention_budget",
        "normalization_type",
    }


NORM = """
class ExampleRMSNorm:
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(variance + self.eps)
        return self.weight * y
class ExampleDecoderLayer:
    def __init__(self):
        self.input_norm = ExampleRMSNorm()
    def forward(self, x):
        return self.input_norm(x)
"""


def test_norm_detection_requires_equation_wiring_and_returned_result(tmp_path):
    facts = concept_extract.code_facts({"modeling.py": NORM})
    assert facts["normalization_type"][0] == "RMSNorm"
    assert facts["normalization_modules"][0][0]["module"] == "ExampleDecoderLayer.input_norm"
    for bad in [
        NORM.replace("return self.weight * y", "return x"),
        NORM.replace("x.pow(2).mean(-1, keepdim=True)", "(x - x.mean(-1)).pow(2).mean(-1)"),
        NORM.replace("return self.input_norm(x)", "return x"),
        NORM.replace("x * torch.rsqrt", "other * torch.rsqrt"),
        NORM.replace("mean(-1, keepdim=True)", "mean(0, keepdim=True)"),
        NORM.replace("y = x *", "variance = unrelated\n        y = x *"),
        NORM.replace(
            "self.input_norm = ExampleRMSNorm()",
            "self.input_norm = ExampleRMSNorm()\n        self.input_norm = Identity()",
        ),
    ]:
        assert not concept_extract.code_facts({"modeling.py": bad})
    marker = tmp_path / "must-not-execute"
    source = f"open({str(marker)!r}, 'w').write('bad')\n" + NORM
    assert concept_extract.code_facts({"modeling.py": source})
    assert not marker.exists()


def test_versioned_companion_does_not_rewrite_an_approved_diagram():
    rec = record("meta-llama__Llama-3.1-8B")
    before = generated.select(rec)
    drawing = build_spec(rec)
    evidence = concepts.evidence_spec(rec)
    assert evidence["schema_version"] == "archwatch-concept-evidence/1"
    assert evidence["concepts"] and evidence["source_commit"] == rec["source_commit"]
    assert generated.select(rec) == before and build_spec(rec) == drawing


def test_cached_enrichment_is_additive_reproducible_and_conflicts_are_held(tmp_path):
    import importlib.util

    from archwatch.artifacts import save_record
    from archwatch.concept_extract import FIELDS

    spec = importlib.util.spec_from_file_location("enrich_concept_test", ROOT / "tools/enrich_concepts.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    rec = record("LiquidAI__LFM3-1.7B-Base")
    rec["checks"] = [c for c in rec["checks"] if c["field"] not in FIELDS]
    rec["published"] = {k: v for k, v in rec["published"].items() if k not in FIELDS}
    rec["source_files"] = ["config.json", "modeling_example.py"]
    original = copy.deepcopy(rec)
    cfg = dict(num_hidden_layers=24, hidden_size=2048, num_attention_heads=16, intermediate_size=7680)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "modeling_example.py").write_text(NORM)
    enriched, fields = tool.enrich(rec, tmp_path)
    assert set(fields) == {"normalization_type", "normalization_modules"}
    assert {k: v for k, v in enriched["published"].items() if k not in FIELDS} == original["published"]
    assert enriched["analyzed_at"] == original["analyzed_at"] and rec == original
    again, fields = tool.enrich(enriched, tmp_path)
    assert not fields and again == enriched
    save_record(enriched, tmp_path / "models", with_yaml=True)
    snapshot = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "models").iterdir()}
    save_record(again, tmp_path / "models", with_yaml=True)
    assert snapshot == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (tmp_path / "models").iterdir()}
    (tmp_path / "config.json").write_text(json.dumps(dict(cfg, normalization_type="LayerNorm")))
    conflicted, _ = tool.enrich(original, tmp_path)
    assert "normalization_type" not in conflicted["published"]
    assert next(c for c in conflicted["checks"] if c["field"] == "normalization_type")["status"] == "conflict"
    assert "rmsnorm" not in {hit["key"] for hit in concepts.for_record(conflicted)}


def test_concept_downloads_disappear_when_a_model_is_held(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("concept_site_gate", ROOT / "build_site.py")
    site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site)
    monkeypatch.setattr(site, "ROOT", tmp_path)
    rec = record("LiquidAI__LFM3-1.7B-Base")
    folder = tmp_path / "data/models"
    folder.mkdir(parents=True)
    target = folder / "arbitrary.json"
    target.write_text(json.dumps(rec))
    site.main()
    export = tmp_path / "site/data/concepts/arbitrary.json"
    assert export.exists()
    assert "data/concepts/arbitrary.json" in (tmp_path / "site/models/arbitrary.html").read_text()
    next(c for c in rec["checks"] if c["field"] == "layer_pattern_per_layer")["status"] = "single source"
    target.write_text(json.dumps(rec))
    site.main()
    assert not export.exists() and not (tmp_path / "site/models/arbitrary.html").exists()
    assert not (tmp_path / "site/data/models/arbitrary.json").exists()


def test_every_guide_source_is_a_paper_or_the_models_own_files():
    import re

    import yaml

    text = (ROOT / "config/concepts.yaml").read_text() + (ROOT / "config/concept-lessons.yaml").read_text()
    assert not re.search(r"raschka|rasbt", text, re.I)
    publishers = {path.name.split("__")[0] for path in (ROOT / "data/models").glob("*.json")}
    for key, item in yaml.safe_load((ROOT / "config/concepts.yaml").read_text()).items():
        url = item["source"]["url"]
        paper = re.match(r"https://arxiv\.org/abs/\d{4}\.\d{4,5}$", url) or url.endswith(".pdf")
        model = re.match(r"https://huggingface\.co/([\w.-]+)/[\w.-]+$", url)
        assert paper or model, (key, url)
        if model:
            assert model.group(1) in publishers | {"deepseek-ai", "Motif-Technologies"}, (key, url)
    for url in re.findall(r"https?://\S+", text):
        assert "huggingface.co/docs" not in url and "blog" not in url, url


def test_unreviewed_guides_are_marked_draft():
    import yaml

    guides = yaml.safe_load((ROOT / "config/concepts.yaml").read_text())
    assert all(isinstance(item["reviewed"], bool) for item in guides.values())
    site_dir = ROOT / "site/concepts"
    index = (site_dir / "index.html").read_text()
    for key, item in guides.items():
        page = (site_dir / f"{key}.html").read_text()
        card = index.split(f'href="{key}.html"')[1].split("</li>")[0]
        assert ('class="draft"' in page) == (not item["reviewed"]), key
        assert ('class="draft"' in card) == (not item["reviewed"]), key


def test_reviewed_flag_is_required(tmp_path, monkeypatch):
    import yaml

    data = yaml.safe_load((ROOT / "config/concepts.yaml").read_text())
    del data["mha"]["reviewed"]
    path = tmp_path / "concepts.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(concepts, "REGISTRY", path)
    with pytest.raises(ValueError, match="reviewed"):
        concepts.registry()
