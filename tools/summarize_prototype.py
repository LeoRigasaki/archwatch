"""Write portable progress/results from saved records and the latest offline tests."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = [json.loads(path.read_text()) for path in sorted((ROOT / "data/models").glob("*.json"))]
    summaries = []
    for rec in records:
        assert re.fullmatch(r"[0-9a-f]{40}", rec["source_commit"])
        for check in rec["checks"]:
            if check["status"] == "conflict":
                assert check["field"] not in rec["published"]
        summaries.append({
            "id": rec["id"], "source_commit": rec["source_commit"],
            "total_parameters_from_saved_headers": rec["params"]["counted_from_weights"]["total"],
            "field_checks": dict(Counter(c["status"] for c in rec["checks"])),
            "published_fields": len(rec["published"]), "review_fields": [c["field"] for c in rec["review_queue"]],
            "code_files": rec["code_scan"]["files"],
            "cache_growth_bytes_per_token": rec["cache"]["grows_per_token_bytes"],
            "fixed_cache_bytes": rec["cache"]["constant_bytes"],
        })
    suites = ET.parse(ROOT / "results/test-results.xml").getroot().findall("testsuite")
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    tests["passed"] = tests["tests"] - tests["failures"] - tests["errors"] - tests["skipped"]
    tests["python"] = platform.python_version()
    tests["command"] = ".venv/bin/python -m pytest -q --junitxml=results/test-results.xml"
    tests["scope"] = "Offline unit/integration tests and saved-record checks; no fresh Hub or weight-header rerun."
    total_checks = Counter(c["status"] for rec in records for c in rec["checks"])
    kimi = next(rec for rec in records if rec["id"] == "moonshotai/Kimi-K3.1-Pro")
    import_info = json.loads((ROOT / "docs/prototype-import.json").read_text())
    snapshot = {
        "as_of": datetime.now(ZoneInfo("Asia/Karachi")).date().isoformat(), "project": "ArchWatch", "scope_reference": "docs/PROJECT_SCOPE.md",
        "project_root": ".", "package": "src/archwatch", "tests_path": "tests", "records_path": "data/models",
        "prototype_archive": import_info["archive"], "prototype_archive_sha256": import_info["archive_sha256"],
        "stage": "Implemented deterministic prototype through approximately Day 9, with earlier tooling/watcher gaps and later diff/SVG features already present.",
        "models": summaries, "record_count": len(records), "yaml_record_count": len(list((ROOT / "data/models").glob("*.yaml"))),
        "saved_record_check_totals": dict(total_checks), "test_run": tests,
        "dsa_indexer_parameter_delta": kimi["tensor_diff_vs_predecessor"]["added_total"],
        "prototype_pipeline_implemented": True, "watcher_implemented": False,
        "daily_workflow": {"path": ".github/workflows/daily.yml", "written": True, "executed": False, "inputs": "watchlist.txt", "github_repository_configured": False},
        "formal_gold_set_models": 0, "formal_archwatch_accuracy_measured": False,
        "legacy_research": {"path": "research-reference/gallery-prototype", "usage": "Private research and spot checks only", "production_input": False},
        "deployment_access_evidence": "docs/deployment-access-review.json",
        "note": "Saved-record agreement and tests are prototype evidence, not gold-set accuracy or universal model-family coverage.",
    }
    (ROOT / "docs/progress-snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    (ROOT / "results/prototype-validation.json").write_text(json.dumps({"models": summaries, "check_totals": dict(total_checks), "tests": tests}, indent=2) + "\n")
    lines = ["# Saved prototype results", "", "Generated from the four imported records. This run does not refetch the Hub or tensor headers.", "",
             "| Model | Parameters in saved header counts | Verified fields | Single-source fields | Conflicts |", "|---|---:|---:|---:|---:|"]
    for rec in summaries:
        c = rec["field_checks"]
        lines.append(f"| {rec['id']} | {rec['total_parameters_from_saved_headers']:,} | {c.get('verified', 0)} | {c.get('single source', 0)} | {c.get('conflict', 0)} |")
    lines += ["", f"Offline tests: {tests['passed']} passed; {tests['failures']} failures; {tests['errors']} errors. Python {tests['python']}.", "",
              f"Recorded Kimi DSA-indexer addition: {snapshot['dsa_indexer_parameter_delta']:,} parameters.", "",
              "The LFM2.5-1.2B-Instruct FFN conflict is excluded from its published field dictionary. The independent 30-model gold set is still open."]
    (ROOT / "results/README.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(records)} saved records; {tests['passed']} passing offline tests.")


if __name__ == "__main__":
    main()
