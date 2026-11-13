"""Compare published records to manually curated, revision-pinned gold labels.

This evaluator never creates labels or imports the extraction/calculation pipeline.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def equal(expected, actual):
    if isinstance(expected, bool):
        return type(actual) is bool and expected == actual
    return expected == actual


def evaluate(gold_dir: Path, records_dir: Path):
    fields = defaultdict(lambda: {"correct": 0, "total": 0, "missing": 0})
    models = []
    for path in sorted(gold_dir.glob("*.yaml")):
        label = yaml.safe_load(path.read_text())
        record_path = records_dir / (label["id"].replace("/", "__") + ".json")
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        revision_ok = record.get("source_commit") == label["source_commit"]
        pub = record.get("published", {}) if revision_ok else {}
        errors, correct = {}, 0
        for field, expected in label["expected"].items():
            exists = field in pub
            passed = exists and equal(expected, pub[field])
            fields[field]["total"] += 1
            fields[field]["correct"] += int(passed)
            fields[field]["missing"] += int(not exists)
            correct += int(passed)
            if not passed:
                errors[field] = {
                    "expected": expected,
                    "actual": pub.get(field),
                    "reason": "source revision mismatch/missing record"
                    if not revision_ok
                    else "not published"
                    if not exists
                    else "different value",
                }
        models.append(
            {
                "id": label["id"],
                "source_commit": label["source_commit"],
                "revision_matches": revision_ok,
                "correct": correct,
                "total": len(label["expected"]),
                "errors": errors,
            }
        )
    total = sum(v["total"] for v in fields.values())
    correct = sum(v["correct"] for v in fields.values())
    return {
        "gold_models": len(models),
        "correct": correct,
        "total": total,
        "exact_match": correct / total if total else None,
        "fields": dict(sorted(fields.items())),
        "models": models,
        "scope": "Ten-model development gold set, manually inspected primary sources; not human-reviewed or a held-out accuracy estimate.",
    }


def write_report(report, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Primary-source gold exact match",
        "",
        report["scope"],
        "",
        f"Published fields: {report['correct']}/{report['total']} exact matches across {report['gold_models']} labels.",
        "",
        "| Field | Exact matches | Labeled | Not published |",
        "|---|---:|---:|---:|",
    ]
    lines += [f"| {k} | {v['correct']} | {v['total']} | {v['missing']} |" for k, v in report["fields"].items()]
    lines += ["", "| Model | Exact matches | Labeled | Revision matches |", "|---|---:|---:|---|"]
    lines += [f"| {m['id']} | {m['correct']} | {m['total']} | {m['revision_matches']} |" for m in report["models"]]
    for model in report["models"]:
        if model["errors"]:
            lines += ["", "## " + model["id"], ""]
            for field, error in model["errors"].items():
                lines.append(
                    f"- `{field}`: expected `{error['expected']}`, got `{error['actual']}` ({error['reason']})."
                )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/models")
    parser.add_argument("--gold", type=Path, default=ROOT / "evals/gold")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gold-exact-match")
    args = parser.parse_args()
    report = evaluate(args.gold, args.records)
    write_report(report, args.output)
    print(f"{report['correct']}/{report['total']} published fields match ({report['gold_models']} models)")


if __name__ == "__main__":
    main()
