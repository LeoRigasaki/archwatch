import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def evaluator():
    spec = importlib.util.spec_from_file_location("gold_evaluator", ROOT / "tools/evaluate_gold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_never_scores_withheld_values_or_stale_revisions(tmp_path):
    gold, records = tmp_path / "gold", tmp_path / "records"
    gold.mkdir()
    records.mkdir()
    label = dict(id="lab/model", source_commit="a" * 40, expected=dict(hidden_size=8, head_dim=4, ffn_gated=True))
    (gold / "toy.yaml").write_text(yaml.safe_dump(label))
    rec = dict(source_commit="a" * 40, published=dict(hidden_size=8, ffn_gated=1), facts=dict(head_dim=dict(value=4)))
    (records / "lab__model.json").write_text(json.dumps(rec))
    report = evaluator().evaluate(gold, records)
    assert report["correct"] == 1 and report["total"] == 3
    assert report["fields"]["head_dim"]["missing"] == 1
    rec["source_commit"] = "b" * 40
    (records / "lab__model.json").write_text(json.dumps(rec))
    assert evaluator().evaluate(gold, records)["correct"] == 0


def test_gold_covers_exactly_one_primary_model_per_core_publisher():
    labels = [yaml.safe_load(p.read_text()) for p in (ROOT / "evals/gold").glob("*.yaml")]
    policy = yaml.safe_load((ROOT / "config/watcher.yaml").read_text())
    assert len(labels) == 10
    assert {label["id"].split("/")[0] for label in labels} == set(policy["core_publishers"])
    for label in labels:
        assert label["label_method"] == "manual_primary_source_inspection"
        assert label["human_reviewed"] is False
        assert label["source_commit"] in label["evidence"]["config"]
        assert label["source_commit"] in label["evidence"]["card"]
        assert label["evidence"]["weight_files"]
        assert label["expected"]["params_total"] > 0
