"""Offline integration checks for the supplied four-model prototype.

These validate saved artifacts and implemented behavior, not fresh Hub runs or
gold-set accuracy. No model files or weight headers are fetched by this suite.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from archwatch import codescan, diagram, pipeline, verify

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = (
    "moonshotai/Kimi-K3",
    "moonshotai/Kimi-K3.1-Pro",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "LiquidAI/LFM3-1.7B",
)


def record(repo):
    return json.loads((ROOT / "data/models" / (repo.replace("/", "__") + ".json")).read_text())


@pytest.mark.parametrize("repo", MODEL_IDS)
def test_saved_record_publication_gate(repo):
    rec = record(repo)
    assert rec["id"] == repo
    assert len(rec["source_commit"]) == 40
    for check in rec["checks"]:
        assert check["candidates"]
        assert all(candidate["evidence"] for candidate in check["candidates"])
        if check["status"] == "conflict":
            assert check["published"] is False
            assert check["field"] not in rec["published"]
        else:
            assert rec["published"][check["field"]] == check["value"]


def test_ast_scan_cites_lines_without_executing_source():
    source = '''raise RuntimeError("this source must never execute")
class ToyAttention:
    def __init__(self, layer_idx, config):
        self.q_norm = RMSNorm(...)
        self.gate_proj = Linear(...)
        if layer_idx + 1 in config.layers:
            pass
'''
    result = codescan.scan({"modeling_toy.py": source})
    facts = result["facts"]
    assert facts["qk_norm"] == (True, "code:modeling_toy.py#L4")
    assert facts["attention_output_gate"] == (True, "code:modeling_toy.py#L5")
    assert facts["layer_index_base_in_config"][0] == 1
    assert "modeling_toy.py#L6" in facts["layer_index_base_in_config"][1]


def test_disagreeing_context_is_held_for_review():
    result = verify.check("context_length", [
        {"value": 128000, "evidence": "config:max_position_embeddings"},
        {"value": 32768, "evidence": "card:context_length"},
    ])
    assert result["status"] == "conflict"
    assert result["published"] is False
    assert result["confidence"] < verify.PUBLISH_THRESHOLD


@pytest.mark.parametrize("repo", ("moonshotai/Kimi-K3.1-Pro", "LiquidAI/LFM3-1.7B"))
def test_saved_predecessor_field_diff_reproduces(repo):
    rec = record(repo)
    assert pipeline.diff(rec, record(rec["predecessor"]["id"])) == rec["diff_vs_predecessor"]


def test_recorded_dsa_indexer_accounts_for_exact_parameter_delta():
    new, old = record("moonshotai/Kimi-K3.1-Pro"), record("moonshotai/Kimi-K3")
    difference = new["tensor_diff_vs_predecessor"]
    assert difference["added_total"] == 178526208
    assert difference["removed_total"] == 0
    assert sum(group["params"] for group in difference["added"]) == 178526208
    assert all("indexer" in group["pattern"] for group in difference["added"])
    assert new["params"]["counted_from_weights"]["total"] - old["params"]["counted_from_weights"]["total"] == 178526208


@pytest.mark.parametrize("repo", ("moonshotai/Kimi-K3.1-Pro", "LiquidAI/LFM3-1.7B"))
def test_generated_svg_uses_record_layer_count(repo):
    rec = record(repo)
    svg = ET.fromstring(diagram.render(rec))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    cells = [rect for rect in svg.findall("svg:rect", ns) if rect.get("height") == "34"]
    assert len(cells) == rec["published"]["num_layers"]
    assert not svg.findall(".//svg:image", ns)
