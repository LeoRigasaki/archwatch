"""Carry the open diagram PR's unmerged state into this run before it generates anything.

    uv run --frozen python tools/carry_pending.py --source pending   # the open PR branch, checked out there

Why: create-pull-request rebuilds the PR branch from main on every run. Without this step a run that happens
before the previous PR is merged throws that run's ledger entries, queue state and outputs away (what happened
to the 18, 20 and 23 Nov runs). With it, the new commit extends the open PR instead of replacing it.

Merging rules: the call ledger is the union of both (no entry is ever dropped); queue state takes the newer
entry per model and keeps every run summary; generated files are added when missing and never overwritten.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = "data/gemini-requests.json"
QUEUE = "data/diagram-queue.json"
OUTPUT_DIRS = ("data/generated-diagrams/", "diagrams/generated/")


def merge_ledger(ours: list, theirs: list) -> list:
    seen, merged = set(), []
    for entry in ours + theirs:
        key = json.dumps(entry, sort_keys=True)
        if key not in seen:
            seen.add(key)
            merged.append(entry)
    return sorted(merged, key=lambda e: e.get("time", 0))


def merge_queue(ours: dict, theirs: dict) -> dict:
    models = dict(ours.get("models", {}))
    for repo, entry in theirs.get("models", {}).items():
        mine = models.get(repo)
        if mine is None or entry.get("updated", "") > mine.get("updated", ""):
            models[repo] = entry
    runs, seen = [], set()
    for run in ours.get("runs", []) + theirs.get("runs", []):
        key = json.dumps(run, sort_keys=True)
        if key not in seen:
            seen.add(key)
            runs.append(run)
    runs.sort(key=lambda r: r.get("date", ""))
    return {"models": models, "runs": runs[-50:]}


def carry(source: Path, root: Path = ROOT) -> dict:
    """Merge the state found in `source` (a checkout of the open diagram PR's branch) into `root`."""
    added, ledger_added = [], 0
    theirs = source / LEDGER
    if theirs.exists():
        path = root / LEDGER
        ours = json.loads(path.read_text()) if path.exists() else []
        merged = merge_ledger(ours, json.loads(theirs.read_text()))
        path.write_text(json.dumps(merged, indent=2) + "\n")
        ledger_added = len(merged) - len(ours)
    theirs = source / QUEUE
    if theirs.exists():
        path = root / QUEUE
        ours = json.loads(path.read_text()) if path.exists() else {"models": {}, "runs": []}
        path.write_text(json.dumps(merge_queue(ours, json.loads(theirs.read_text())), indent=2) + "\n")
    for folder in OUTPUT_DIRS:
        for file in sorted((source / folder).rglob("*")) if (source / folder).exists() else []:
            target = root / file.relative_to(source)
            if file.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(file.read_bytes())
                added.append(target.relative_to(root).as_posix())
    return dict(ledger_entries_added=ledger_added, files_added=len(added))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="checkout of the open diagram PR's branch")
    args = parser.parse_args()
    print(json.dumps(carry(args.source.resolve())))


if __name__ == "__main__":
    sys.exit(main())
