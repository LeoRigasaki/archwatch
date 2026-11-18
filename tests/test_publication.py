"""Architecture outputs cannot revive diagnostic or withheld values."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from archwatch import diagram, pipeline
from archwatch.publication import PublicationError, render_cache

ROOT = Path(__file__).resolve().parents[1]


def record():
    rec = json.loads((ROOT / "data/models/LiquidAI__LFM3-1.7B.json").read_text())
    rec["hub_metadata"] = {}  # neutral publication fixture; derivative admission has separate coverage
    return rec


def builder():
    spec = importlib.util.spec_from_file_location("archwatch_site_builder", ROOT / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagram_does_not_read_raw_facts_weights_or_cached_calculations():
    rec = record()
    expected = diagram.render(rec)
    rec["facts"] = {"hidden_size": {"value": 987654321, "evidence": "diagnostic"}}
    rec["params"] = {"counted_from_weights": {"total": 987654321}}
    rec["cache"] = {"grows_per_token_bytes": 987654321}
    assert diagram.render(rec) == expected


@pytest.mark.parametrize("mutation", ["single_source_pattern", "missing_hidden_size", "inconsistent_pattern"])
def test_diagram_rejects_ineligible_inputs(mutation):
    rec = record()
    if mutation == "single_source_pattern":
        next(c for c in rec["checks"] if c["field"] == "layer_pattern_per_layer")["status"] = "single source"
    elif mutation == "missing_hidden_size":
        del rec["published"]["hidden_size"]
    else:
        rec["published"]["layer_types"][0] = "full_attention"
    with pytest.raises(PublicationError):
        diagram.render(rec)


def test_site_uses_published_inputs_and_rejects_unverified_pattern(tmp_path, monkeypatch):
    module = builder()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "data/models"
    folder.mkdir(parents=True)
    rec = record()
    rec.pop("predecessor", None)
    target = folder / "toy.json"
    target.write_text(json.dumps(rec))
    expected = module.model_section("toy", "toy")
    rec["facts"] = {}
    rec["params"] = {}
    rec["cache"] = {}
    target.write_text(json.dumps(rec))
    assert module.model_section("toy", "toy") == expected
    next(c for c in rec["checks"] if c["field"] == "layer_pattern_per_layer")["status"] = "single source"
    target.write_text(json.dumps(rec))
    with pytest.raises(PublicationError, match="Layer pattern"):
        module.model_section("toy", "toy")


def test_diff_does_not_reintroduce_withheld_values():
    new = record()
    old = copy.deepcopy(new)
    old["published"].pop("ffn_intermediate")
    old["facts"]["ffn_intermediate"]["value"] = 987654321
    row = next(row for row in pipeline.diff(new, old) if row["field"] == "ffn_intermediate")
    assert row["before"] is None
    assert row["after"] == new["published"]["ffn_intermediate"]


def test_site_recomputes_field_diff_instead_of_trusting_cached_values(tmp_path, monkeypatch):
    module = builder()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "data/models"
    folder.mkdir(parents=True)
    new = record()
    old = copy.deepcopy(new)
    old["id"] = "lab/predecessor"
    old["published"].pop("ffn_intermediate")
    new["predecessor"] = {"id": old["id"]}
    new["diff_vs_predecessor"] = [{"field": "ffn_intermediate", "before": 987654321, "after": 987654322}]
    (folder / "toy.json").write_text(json.dumps(new))
    (folder / "lab__predecessor.json").write_text(json.dumps(old))
    page = module.model_section("toy", "toy")
    assert "987,654,321" not in page
    assert "987,654,322" not in page


def test_unknown_cache_rule_is_not_a_complete_estimate():
    with pytest.raises(PublicationError, match="unsupported layer"):
        render_cache(
            {
                "layer_type_counts": {"unknown": 1},
                "num_layers": 1,
                "num_attention_heads": 2,
                "head_dim": 8,
                "context_length": 1024,
            }
        )
