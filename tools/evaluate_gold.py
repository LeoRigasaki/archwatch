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


def as_convention(pub: dict, version: int) -> dict:
    """Published values mapped back to an older counting convention (config/counting-conventions.yaml), so frozen
    labels are compared like for like without editing them. Version 1 counted MTP modules in the total."""
    out = dict(pub)
    if version == 1 and "params_total" in out and out.get("params_mtp"):
        out["params_total"] = out["params_total"] + out["params_mtp"]
    return out


NEEDS_RELABEL = "changed upstream, needs relabel"


def evaluate(gold_dir: Path, records_dir: Path, convention: int | None = None):
    """Score published values against the labels.

    Not moved (record at the label's pinned commit): every labeled field must match.
    Moved (the record was rebuilt at a later commit of the same repo): fields are still compared by value. A match
    counts as correct; a different value is reported as "changed upstream, needs relabel" for that field only and
    listed in `needs_relabel`. Labels are never edited here. A missing record or an unpublished field always fails.
    """
    fields = defaultdict(lambda: {"correct": 0, "total": 0, "missing": 0})
    models, needs_relabel = [], []
    for path in sorted(gold_dir.glob("*.yaml")):
        label = yaml.safe_load(path.read_text())
        record_path = records_dir / (label["id"].replace("/", "__") + ".json")
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        revision_ok = record.get("source_commit") == label["source_commit"]
        moved = bool(record) and not revision_ok
        pub = record.get("published", {})
        if convention is not None:
            pub = as_convention(pub, convention)
        errors, correct = {}, 0
        for field, expected in label["expected"].items():
            exists = field in pub
            passed = exists and equal(expected, pub[field])
            fields[field]["total"] += 1
            fields[field]["correct"] += int(passed)
            fields[field]["missing"] += int(not exists)
            correct += int(passed)
            if not passed:
                reason = (
                    "missing record"
                    if not record
                    else "not published"
                    if not exists
                    else NEEDS_RELABEL
                    if moved
                    else "different value"
                )
                errors[field] = {"expected": expected, "actual": pub.get(field), "reason": reason}
                if reason == NEEDS_RELABEL:
                    needs_relabel.append(
                        {
                            "id": label["id"],
                            "field": field,
                            "expected": expected,
                            "actual": pub[field],
                            "label_commit": label["source_commit"],
                            "record_commit": record["source_commit"],
                        }
                    )
        models.append(
            {
                "id": label["id"],
                "source_commit": label["source_commit"],
                "record_commit": record.get("source_commit"),
                "revision_matches": revision_ok,
                "revision_moved": moved,
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
        "needs_relabel": needs_relabel,
        "fields": dict(sorted(fields.items())),
        "models": models,
        "scope": "Ten-model development gold set, manually inspected primary sources; not human-reviewed or a held-out accuracy estimate.",
    }


def strict_errors(report) -> list:
    """Every failing field except those reported as changed upstream: these must be zero."""
    return [
        (m["id"], field, e)
        for m in report["models"]
        for field, e in m["errors"].items()
        if e["reason"] != NEEDS_RELABEL
    ]


def relabel_markdown(report) -> str:
    """Markdown list of fields that changed upstream (for the eval report and the daily PR body); empty if none."""
    if not report["needs_relabel"]:
        return ""
    lines = [
        "### Gold labels: changed upstream, needs relabel",
        "",
        "These records were rebuilt at a newer Hub commit than the gold label is pinned to, and the value differs."
        " Labels are not edited automatically; relabel by hand after checking the new files.",
        "",
        "| Model | Field | Gold label | Now published | Label commit | Record commit |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['id']} | `{r['field']}` | `{r['expected']}` | `{r['actual']}` | {r['label_commit'][:8]} | {r['record_commit'][:8]} |"
        for r in report["needs_relabel"]
    ]
    return "\n".join(lines) + "\n"


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
    lines += [
        f"| {m['id']} | {m['correct']} | {m['total']} | "
        + (f"moved to {m['record_commit'][:8]}" if m["revision_moved"] else str(m["revision_matches"]))
        + " |"
        for m in report["models"]
    ]
    if report["needs_relabel"]:
        lines += ["", relabel_markdown(report).rstrip("\n")]
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
    parser.add_argument("--scope", help="Explicit evaluation cohort/protocol description for a separate report")
    parser.add_argument("--convention", type=int, help="map records to this counting-convention version first")
    parser.add_argument("--relabel-md", type=Path, help="also write the needs-relabel list (Markdown) here")
    args = parser.parse_args()
    report = evaluate(args.gold, args.records, args.convention)
    if args.convention is not None:
        report["convention"] = args.convention
    if args.scope:
        report["scope"] = args.scope
    write_report(report, args.output)
    if args.relabel_md:
        args.relabel_md.write_text(relabel_markdown(report))
    print(f"{report['correct']}/{report['total']} published fields match ({report['gold_models']} models)")
    if report["needs_relabel"]:
        print(f"{len(report['needs_relabel'])} field(s) changed upstream, needs relabel")


if __name__ == "__main__":
    main()
