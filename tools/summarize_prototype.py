"""Write portable progress/results from saved records and the latest offline tests."""

import json
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from archwatch.publication import PublicationError, render_cache, render_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = [json.loads(path.read_text()) for path in sorted((ROOT / "data/models").glob("*.json"))]
    summaries = []
    for rec in records:
        assert re.fullmatch(r"[0-9a-f]{40}", rec["source_commit"])
        for check in rec["checks"]:
            if check["status"] == "conflict":
                assert check["field"] not in rec["published"]
        try:
            cache = render_cache(render_values(rec))
            rendering_review = None
        except (PublicationError, KeyError, TypeError) as error:
            cache, rendering_review = None, str(error)
        summaries.append(
            {
                "id": rec["id"],
                "source_commit": rec["source_commit"],
                "total_parameters_from_saved_headers": rec["params"]["counted_from_weights"]["total"],
                "field_checks": dict(Counter(c["status"] for c in rec["checks"])),
                "published_fields": len(rec["published"]),
                "review_fields": [c["field"] for c in rec["review_queue"]],
                "code_files": rec["code_scan"]["files"],
                "cache_growth_bytes_per_token": cache["grows_per_token_bytes"] if cache else None,
                "fixed_cache_bytes": cache["constant_bytes"] if cache else None,
                "rendering_review": rendering_review,
            }
        )
    suites = ET.parse(ROOT / "results/test-results.xml").getroot().findall("testsuite")
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    tests["passed"] = tests["tests"] - tests["failures"] - tests["errors"] - tests["skipped"]
    tests["python"] = platform.python_version()
    tests["command"] = "uv run --frozen pytest -q --junitxml=results/test-results.xml"
    tests["scope"] = "Offline unit/integration tests and saved-record checks; no fresh Hub or weight-header rerun."
    total_checks = Counter(c["status"] for rec in records for c in rec["checks"])
    gold_count = len(list((ROOT / "evals/gold").glob("*.yaml")))
    gold_path = ROOT / "results/gold-exact-match.json"
    gold_report = json.loads(gold_path.read_text()) if gold_path.exists() else None
    tensor_deltas = [
        {
            "id": rec["id"],
            "predecessor": rec["predecessor"]["id"],
            "added_total": rec["tensor_diff_vs_predecessor"]["added_total"],
        }
        for rec in records
        if rec.get("predecessor") and rec.get("tensor_diff_vs_predecessor")
    ]
    import_info = json.loads((ROOT / "docs/prototype-import.json").read_text())
    acceptance_path = ROOT / "results/github-workflow-acceptance.json"
    acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
    manual_runs = acceptance.get("manual_runs", [])
    snapshot = {
        "as_of": datetime.now(ZoneInfo("Asia/Karachi")).date().isoformat(),
        "project": "ArchWatch",
        "scope_reference": "docs/PROJECT_SCOPE.md",
        "project_root": ".",
        "package": "src/archwatch",
        "tests_path": "tests",
        "records_path": "data/models",
        "prototype_archive": import_info["archive"],
        "prototype_archive_sha256": import_info["archive_sha256"],
        "stage": "Newest-release first-run evaluation and source-based adaptation implemented; new graph/cache qualification, human review and LLM stages remain open.",
        "models": summaries,
        "record_count": len(records),
        "yaml_record_count": len(list((ROOT / "data/models").glob("*.yaml"))),
        "saved_record_check_totals": dict(total_checks),
        "test_run": tests,
        "tensor_group_additions": tensor_deltas,
        "prototype_pipeline_implemented": True,
        "watcher_implemented": True,
        "watcher_policy": "config/watcher.yaml",
        "watcher_state": "data/seen.json",
        "watcher_live_evidence": "results/watcher-live-validation.json",
        "no_op_acceptance": "results/no-op-acceptance.json",
        "tooling_configured": ["uv.lock", "ruff", "pre-commit", "offline PR CI"],
        "fresh_source_run_evidence": "results/live-validation.json",
        "daily_workflow": {
            "path": ".github/workflows/daily.yml",
            "written": True,
            "executed": bool(manual_runs),
            "inputs": "manual mode: fixed watchlist, discover or process",
            "no_op_runs": "results/no-op-acceptance.json",
            "github_repository_configured": subprocess.run(
                ["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True
            ).returncode
            == 0,
        },
        "formal_gold_set_models": gold_count,
        "gold_label_method": "Assistant manual primary-source inspection; human_reviewed=false",
        "development_gold_evaluation": {k: gold_report[k] for k in ("correct", "total", "exact_match")}
        if gold_report
        else None,
        "backfill_source_run_evidence": "results/backfill-processing.json",
        "newest_backfill_protocol": "docs/NEWEST_BACKFILL.md",
        "first_run_holdout": {
            "models": 7,
            "correct": 97,
            "total": 139,
            "evidence": "results/newest-first-run/heldout-exact-match.json",
        },
        "late_access_baseline": {
            "models": 1,
            "correct": 11,
            "total": 16,
            "evidence": "results/newest-first-run/meta-late-access.json",
        },
        "formal_archwatch_accuracy_measured": False,
        "note": "Saved-record agreement and tests are prototype evidence, not gold-set accuracy or universal model-family coverage.",
    }
    (ROOT / "docs/progress-snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    (ROOT / "results/prototype-validation.json").write_text(
        json.dumps({"models": summaries, "check_totals": dict(total_checks), "tests": tests}, indent=2) + "\n"
    )
    lines = [
        "# Saved prototype results",
        "",
        f"Generated from {len(records)} saved records. This command does not refetch sources. The ten-publisher backfill run is recorded in [backfill-processing.json](backfill-processing.json); original-prototype evidence remains in [live-validation.json](live-validation.json).",
        "",
        "| Model | Parameters in saved header counts | Verified fields | Single-source fields | Conflicts |",
        "|---|---:|---:|---:|---:|",
    ]
    for rec in summaries:
        c = rec["field_checks"]
        lines.append(
            f"| {rec['id']} | {rec['total_parameters_from_saved_headers']:,} | {c.get('verified', 0)} | {c.get('single source', 0)} | {c.get('conflict', 0)} |"
        )
    lines += [
        "",
        f"Offline tests: {tests['passed']} passed; {tests['failures']} failures; {tests['errors']} errors. Python {tests['python']}.",
        "",
        "Tensor-group additions: " + "; ".join(f"{d['id']}: {d['added_total']:,}" for d in tensor_deltas) + ".",
        "",
        f"Manual primary-source gold labels: {gold_count}. [Per-field exact match](gold-exact-match.md) reports the development set; it is not human-reviewed or held-out accuracy. The longer-term 30-model evaluation remains open.",
    ]
    (ROOT / "results/README.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(records)} saved records; {tests['passed']} passing offline tests.")


if __name__ == "__main__":
    main()
