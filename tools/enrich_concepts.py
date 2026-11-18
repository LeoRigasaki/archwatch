"""Add optional concept facts from each record's exact cached primary snapshot.

No network access, weight-body download, provider calls or baseline rewrites.
Existing facts and verifier decisions are retained. Conflicting new claims are
held by the ordinary verifier, never silently overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from archwatch import concept_extract, extract, verify
from archwatch.artifacts import save_record
from archwatch.watcher import derivative_relation

ROOT = Path(__file__).resolve().parents[1]


def enrich(record, directory):
    cfg_path = directory / "config.json"
    if not cfg_path.exists():
        raise ValueError("The record has no cached pinned config")
    cfg = json.loads(cfg_path.read_text())
    config_facts = concept_extract.config_facts(cfg)
    files = {
        name: (directory / name.replace("/", "__")).read_text()
        for name in record.get("source_files", [])
        if name.endswith(".py") and (directory / name.replace("/", "__")).exists()
    }
    code_facts = concept_extract.code_facts(files)
    record = json.loads(json.dumps(record))
    changed = []
    checks = {c["field"]: c for c in record["checks"]}
    for field in sorted(set(config_facts) | set(code_facts)):
        candidates = [
            dict(value=value, evidence=evidence)
            for facts in (config_facts, code_facts)
            if field in facts
            for value, evidence in [facts[field]]
        ]
        previous = checks.get(field)
        if previous:
            candidates += previous.get("candidates", [])
        candidates = list({json.dumps(c, sort_keys=True): c for c in candidates}.values())
        decision = verify.check(field, candidates)
        if previous == decision:
            continue
        if previous:
            record["checks"][record["checks"].index(previous)] = decision
        else:
            record["checks"].append(decision)
        if decision.get("published"):
            record["published"][field] = decision["value"]
        else:
            record["published"].pop(field, None)
        record["review_queue"] = [c for c in record.get("review_queue", []) if c["field"] != field]
        if not decision.get("published"):
            record["review_queue"].append(decision)
        if field in config_facts:
            value, evidence = config_facts[field]
            record["facts"][field] = dict(value=value, evidence=evidence)
        if field in code_facts:
            value, evidence = code_facts[field]
            record.setdefault("code_scan", {}).setdefault("facts", {})[field] = dict(value=value, evidence=evidence)
        changed.append(field)
    if changed:
        record["config_coverage"] = extract.extract_with_usage(cfg)[1]
        record["unchecked_notes"] = [
            n for n in record.get("unchecked_notes", []) if not n.startswith("Config keys not consumed")
        ]
        unknown = record["config_coverage"]["unconsumed"]
        if unknown:
            record["unchecked_notes"].append(
                "Config keys not consumed by the extractor (including metadata/defaults): " + ", ".join(unknown)
            )
    return record, changed


def run(root=ROOT):
    changes = {}
    for path in sorted((root / "data/models").glob("*.json")):
        original = json.loads(path.read_text())
        if derivative_relation(original.get("hub_metadata") or {}):
            continue
        directory = root / "cache" / original["id"].replace("/", "__") / original["source_commit"]
        record, fields = enrich(original, directory)
        if not fields:
            continue
        save_record(record, path.parent, with_yaml=path.with_suffix(".yaml").exists())
        changes[record["id"]] = fields
    return changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2))
