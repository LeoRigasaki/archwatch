import importlib.util
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def evaluator():
    spec = importlib.util.spec_from_file_location("gold_evaluator", ROOT / "tools/evaluate_gold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_never_scores_withheld_values(tmp_path):
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
    assert {e["reason"] for e in report["models"][0]["errors"].values()} == {"not published", "different value"}


def toy(tmp_path, record_commit, published):
    gold, records = tmp_path / "gold", tmp_path / "records"
    gold.mkdir()
    records.mkdir()
    label = dict(id="lab/model", source_commit="a" * 40, expected=dict(hidden_size=8, num_layers=4, vocab_size=100))
    (gold / "toy.yaml").write_text(yaml.safe_dump(label))
    (records / "lab__model.json").write_text(json.dumps(dict(source_commit=record_commit, published=published)))
    return evaluator().evaluate(gold, records)


def test_not_moved_every_field_must_match(tmp_path):
    ev = evaluator()
    report = toy(tmp_path, "a" * 40, dict(hidden_size=8, num_layers=5, vocab_size=100))
    assert report["correct"] == 2 and report["needs_relabel"] == []
    assert report["models"][0]["revision_moved"] is False
    assert [(f, e["reason"]) for _, f, e in ev.strict_errors(report)] == [("num_layers", "different value")]


def test_moved_same_value_counts_as_correct(tmp_path):
    ev = evaluator()
    report = toy(tmp_path, "b" * 40, dict(hidden_size=8, num_layers=4, vocab_size=100))
    assert report["correct"] == report["total"] == 3
    assert report["models"][0]["revision_moved"] is True and report["models"][0]["revision_matches"] is False
    assert report["needs_relabel"] == [] and ev.strict_errors(report) == []


def test_moved_different_value_is_reported_for_that_field_only(tmp_path):
    ev = evaluator()
    report = toy(tmp_path, "b" * 40, dict(hidden_size=8, num_layers=6, vocab_size=100))
    gold_before = (tmp_path / "gold/toy.yaml").read_text()
    assert report["correct"] == 2 and report["total"] == 3  # the other fields of the model still count
    assert report["needs_relabel"] == [
        dict(id="lab/model", field="num_layers", expected=4, actual=6, label_commit="a" * 40, record_commit="b" * 40)
    ]
    assert ev.strict_errors(report) == []  # excluded from the all-must-match rule, this field only
    assert "| lab/model | `num_layers` | `4` | `6` |" in ev.relabel_markdown(report)
    ev.write_report(report, tmp_path / "out/report")
    assert "changed upstream, needs relabel" in (tmp_path / "out/report.md").read_text()
    assert (tmp_path / "gold/toy.yaml").read_text() == gold_before  # labels are never edited


def test_moved_record_still_fails_on_unpublished_field_or_missing_record(tmp_path):
    ev = evaluator()
    report = toy(tmp_path, "b" * 40, dict(hidden_size=8, num_layers=4))
    assert [(f, e["reason"]) for _, f, e in ev.strict_errors(report)] == [("vocab_size", "not published")]
    (tmp_path / "records/lab__model.json").unlink()
    report = ev.evaluate(tmp_path / "gold", tmp_path / "records")
    assert report["correct"] == 0 and len(ev.strict_errors(report)) == 3


def test_kimi_k3_1_pro_rebuilt_at_tokenizer_only_commit_scores_full(tmp_path):
    """Real pipeline output at 80aff651 (only encoding_k3.py changed upstream) against the label pinned to d9e99b0."""
    ev = evaluator()
    gold, records = tmp_path / "gold", tmp_path / "records"
    gold.mkdir()
    records.mkdir()
    shutil.copy(ROOT / "evals/gold/moonshotai__Kimi-K3.1-Pro.yaml", gold)
    shutil.copy(
        ROOT / "tests/fixtures/records/moonshotai__Kimi-K3.1-Pro@80aff651.json",
        records / "moonshotai__Kimi-K3.1-Pro.json",
    )
    report = ev.evaluate(gold, records)
    model = report["models"][0]
    assert model["revision_moved"] and model["record_commit"].startswith("80aff651")
    assert (report["correct"], report["total"]) == (20, 20)
    assert report["needs_relabel"] == [] and ev.strict_errors(report) == []


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


def test_saved_backfill_preserves_all_independently_labeled_published_fields():
    ev = evaluator()
    report = ev.evaluate(ROOT / "evals/gold", ROOT / "data/models")
    assert report["total"] == 144
    # Fields that changed upstream after a label's pinned commit are listed for relabelling, not hidden; every
    # other labeled field must match.
    assert ev.strict_errors(report) == []
    assert report["correct"] == report["total"] - len(report["needs_relabel"])


def test_daily_opens_the_pr_before_running_tests_and_lists_relabels_in_its_body():
    steps = yaml.safe_load((ROOT / ".github/workflows/daily.yml").read_text())["jobs"]["run"]["steps"]
    names = [s.get("name") or s.get("uses", "") or s.get("run", "") for s in steps]
    gold = next(i for i, s in enumerate(steps) if "evaluate_gold.py" in s.get("run", ""))
    pr = next(i for i, n in enumerate(names) if n.startswith("peter-evans/create-pull-request"))
    tests = next(i for i, s in enumerate(steps) if "pytest" in s.get("run", ""))
    assert gold < pr < tests, names
    assert "--relabel-md" in steps[gold]["run"] and "pr-body.md" in steps[pr]["with"]["body-path"]
    assert steps[tests]["if"].startswith("always()")
    assert "evals/gold" not in steps[gold]["run"]  # reads labels only; never writes them
