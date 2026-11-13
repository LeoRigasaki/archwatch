"""Persist public artifacts without touching unchanged files or analysis dates."""

import json
from pathlib import Path

import yaml


def write_if_changed(path: Path, content: str) -> bool:
    data = content.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return True


def save_record(record: dict, directory: Path, *, with_yaml: bool = False, diagrams: Path | None = None) -> dict:
    name = record["id"].replace("/", "__")
    path = directory / (name + ".json")
    if path.exists():
        saved = json.loads(path.read_text())

        def comparable(value):
            return {k: v for k, v in value.items() if k != "analyzed_at"}

        if comparable(saved) == comparable(record):
            record = saved  # Keep the old timestamp and serialization order, too.
        else:
            write_if_changed(path, json.dumps(record, indent=1, default=str))
    else:
        write_if_changed(path, json.dumps(record, indent=1, default=str))
    if with_yaml:
        slim = {k: record[k] for k in ("id", "source_commit", "hub_created", "license", "analyzed_at", "published")}
        slim["review_queue"] = [{"field": r["field"], "candidates": r["candidates"]} for r in record["review_queue"]]
        slim["unchecked_notes"] = record["unchecked_notes"]
        target = directory / (name + ".yaml")
        if not target.exists() or yaml.safe_load(target.read_text()) != slim:
            write_if_changed(target, yaml.safe_dump(slim, sort_keys=False, allow_unicode=True))
    if diagrams is not None:
        from .diagram import render

        write_if_changed(diagrams / (name + ".svg"), render(record))
    return record
