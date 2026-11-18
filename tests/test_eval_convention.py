"""Frozen labels are compared under their own counting convention; the labels themselves never change."""

import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def tool():
    spec = importlib.util.spec_from_file_location("evaluate_gold", ROOT / "tools/evaluate_gold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convention_1_adds_mtp_back_to_the_total_exactly():
    ev = tool()
    assert ev.as_convention({"params_total": 100, "params_mtp": 7}, 1)["params_total"] == 107
    assert ev.as_convention({"params_total": 100}, 1)["params_total"] == 100
    assert ev.as_convention({"params_total": 100, "params_mtp": 7}, 2)["params_total"] == 100


def test_frozen_held_out_scores_under_both_conventions():
    ev = tool()
    gold = ROOT / "evals/heldout/newest-2026-11-14"
    v2, v1 = ev.evaluate(gold, ROOT / "data/models"), ev.evaluate(gold, ROOT / "data/models", 1)
    assert (v2["correct"], v2["total"]) == (134, 139) and (v1["correct"], v1["total"]) == (138, 139)
    mtp_misses = [
        m["id"]
        for m in v2["models"]
        if "params_total" in m["errors"] and m["errors"]["params_total"]["reason"] == "different value"
    ]
    assert len(mtp_misses) == 4
    saved = json.loads((ROOT / "results/heldout-current-v1.json").read_text())
    assert saved["correct"] == 138 and saved["convention"] == 1
    conventions = yaml.safe_load((ROOT / "config/counting-conventions.yaml").read_text())
    assert conventions["current"] == 2 and conventions["gold_sets"]["evals/heldout/newest-2026-11-14"] == 1


def test_gold_labels_are_unchanged_since_they_were_frozen():
    # The commit that added the newest held-out labels; nothing in them may differ from it (see docs/COMMIT_MAP.md).
    labels = "evals/heldout/newest-2026-11-14"
    added = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", labels], capture_output=True, text=True, cwd=ROOT
    ).stdout.split()
    if not added:
        import pytest

        pytest.skip("no git history in this checkout")
    out = subprocess.run(["git", "diff", "--stat", added[-1], "--", labels], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and out.stdout.strip() == ""
    touched = subprocess.run(["git", "log", "--format=%H", "--", labels], capture_output=True, text=True, cwd=ROOT)
    assert touched.stdout.split() == [added[-1]]  # no commit ever edited a label after it was frozen


def test_frozen_labels_and_first_run_evidence_match_content_manifest():
    import hashlib

    manifest = json.loads((ROOT / "evals/frozen-evidence.json").read_text())["sha256"]
    assert any(name.startswith("results/newest-first-run/records/") for name in manifest)
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
