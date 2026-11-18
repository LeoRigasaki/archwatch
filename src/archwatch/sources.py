"""Resolve provenance strings to primary files at the record's pinned commit."""

import re


def source_url(rec, evidence):
    src, _, detail = evidence.partition(":")
    blob = f"https://huggingface.co/{rec['id']}/blob/{rec['source_commit']}/"
    if src == "config":
        return blob + "config.json"
    if src == "card":
        return blob + "README.md"
    if src == "code":
        match = re.match(r"([\w./-]+\.py)(#L\d+)?", detail)
        return blob + match.group(1) + (match.group(2) or "") if match else None
    if src == "weights":
        files = rec.get("source_files", [])
        if "model.safetensors.index.json" in files:
            return blob + "model.safetensors.index.json"
        weights = sorted(f for f in files if f.endswith(".safetensors"))
        if len(weights) == 1:
            return blob + weights[0]
        return f"https://huggingface.co/{rec['id']}/tree/{rec['source_commit']}"
    return None
