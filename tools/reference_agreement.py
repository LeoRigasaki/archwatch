"""Compare ArchWatch's published fields with an external reference sheet (a YAML file in a GitHub repo).

    REFERENCE_REPO=owner/repo REFERENCE_PATH=file.yml uv run --frozen python tools/reference_agreement.py

Comparison only. The reference is named at run time (environment or arguments), downloaded into the ignored
cache/ folder and never committed. Nothing here writes records, gold labels, diagrams or the site: the report
goes to cache/reference/agreement.{md,json}. Image fields are dropped unread.
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archwatch.publication import PublicationError, render_cache, render_values  # noqa: E402

UNITS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
BYTES = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "KB": 1024, "MB": 1024**2}
DROP = ("image",)  # never read or fetch the reference's images
REPORTS = ROOT / "cache" / "reference"  # ignored: reports that contain reference values never reach git
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
MAX_BYTES = 2_000_000


def reference_config(repo=None, path=None, commit=None, license=None) -> dict:
    """The reference named at run time; nothing about it is stored in the repository."""
    repo = repo or os.environ.get("REFERENCE_REPO")
    path = path or os.environ.get("REFERENCE_PATH")
    if not repo or not path:
        raise SystemExit("Name the reference: REFERENCE_REPO=owner/repo REFERENCE_PATH=file.yml (or --repo/--path)")
    config = dict(repo=repo, path=path, raw_url=RAW_URL, max_bytes=MAX_BYTES)
    config["commit"] = commit or os.environ.get("REFERENCE_COMMIT") or latest_commit(config)
    config["license"] = license or os.environ.get("REFERENCE_LICENSE") or "see the reference repository"
    return config


def latest_commit(config: dict) -> str:
    url = f"https://api.github.com/repos/{config['repo']}/commits?path={config['path']}&per_page=1"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())[0]["sha"]


def fetch(config: dict, cache: Path) -> dict:
    path = cache / "reference" / "source" / config["commit"] / config["path"]
    if not path.exists():
        url = config["raw_url"].format(**config)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(config["max_bytes"] + 1)
        if len(data) > config["max_bytes"]:
            raise ValueError("Reference file exceeds the configured size bound")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    entries = yaml.safe_load(path.read_text())
    return {name: {k: v for k, v in e.items() if k not in DROP} for name, e in entries.items() if isinstance(e, dict)}


# ---------------------------------------------------------------- parsing their fact-sheet strings


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def parse_count(text):
    """'2.8T' -> (2.8e12, tolerance); '512,000' -> (512000, 0)."""
    if text is None:
        return None
    m = re.match(r"\s*([\d,]+(?:\.\d+)?)\s*([KMBT])?\b", str(text))
    if not m:
        return None
    number, unit = m.group(1).replace(",", ""), m.group(2)
    scale = UNITS.get(unit, 1)
    return float(number) * scale, (0.5 * 10 ** -_decimals(number) * scale if unit else 0)


def parse_scale(text):
    """'300B total, 12B active' -> totals; composite '100B core ... plus 40B n-gram embeddings and 2B MTP'
    is summed (scope stated in the raw string, which the report shows next to the value)."""
    out = {}
    if not text:
        return out
    text = str(text)
    head = re.split(r"[;,]", text)[0]
    parts = [parse_count(m) for m in re.findall(r"[\d.]+\s*[KMBT]\b", head)]
    tail = text.split("plus", 1)[1] if "plus" in text else ""
    plus = [parse_count(m) for m in re.findall(r"[\d.]+\s*[KMBT]\b", tail)]
    terms = [p for p in parts + plus if p]
    if terms:
        unique = parts[:1] + [p for p in plus if p] if "+" not in head else terms
        out["params_total"] = (sum(v for v, _ in unique), sum(t for _, t in unique))
    m = re.search(r"([\d.]+\s*[KMBT])\s*active", text)
    if m:
        out["params_active"] = parse_count(m.group(1))
    return out


def parse_layers(text):
    """Decoder layers in the first clause: the leading count of each '+'-separated term.

    '30 sliding-window + 10 global' -> 40; '3 GQA + 15 Mamba-2 + 12 FFN' -> 30 (the 2 in 'Mamba-2' is a name);
    '80 GQA + 1 MTP layer' -> 80 (MTP predictors are not decoder layers); 'N shared x 2 passes' counts N.
    """
    clause = re.sub(r"\([^)]*\)", "", str(text or "").split(";")[0])
    total, found = 0, False
    for term in clause.split("+"):
        if re.search(r"\bMTP\b", term):
            continue
        m = re.match(r"\s*(\d+)\b", term)
        if m:
            total, found = total + int(m.group(1)), True
    return (total, 0) if found else None


def parse_bytes(text):
    m = re.match(r"\s*([\d.,]+)\s*(B|KiB|MiB|GiB|KB|MB)\b", str(text or ""))
    if not m:
        return None
    number = m.group(1).replace(",", "")
    unit = BYTES[m.group(2)]
    return float(number) * unit, 0.5 * 10 ** -_decimals(number) * unit


def same_k_shorthand(ours: int, theirs: float) -> bool:
    """Unit normalisation: '128,000' and '1,000,000' are written shorthand for 128K = 131,072 and 1M = 2^20."""
    for base, unit in ((1000, 1024), (1_000_000, 1024**2)):
        if theirs % base == 0 and ours == theirs / base * unit:
            return True
    return False


def decoder_kind(text):
    """Compare only the dense/sparse axis; 'hybrid' describes token mixing, which the sets label differently."""
    t = str(text or "").lower()
    return "sparse" if ("moe" in t or "sparse" in t) else "dense" if t else None


# ---------------------------------------------------------------- comparison


def ours(record: dict) -> dict:
    p = record["published"]
    out = {k: p.get(k) for k in ("params_total", "params_active", "context_length", "vocab_size", "num_layers")}
    out["decoder_type"] = decoder_kind(p.get("decoder_type"))
    try:
        out["kv_cache_per_token_bf16"] = render_cache(render_values(record))["grows_per_token_bytes"]
    except PublicationError:
        out["kv_cache_per_token_bf16"] = None  # withheld by our publication gate; not compared
    return out


def evidence(record: dict, field: str) -> str:
    if field == "kv_cache_per_token_bf16":
        rows = record.get("cache", {}).get("breakdown", [])
        return "; ".join(r["formula"] for r in rows if r.get("grows_per_token_bytes")) or "calculator"
    check = next((c for c in record["checks"] if c["field"] == field), None)
    if not check:
        return "no publication decision"
    return f"{check['status']}: " + "; ".join(c["evidence"] for c in check["candidates"])


def theirs(entry: dict) -> dict:
    out = parse_scale(entry.get("scale"))
    out["context_length"] = parse_count(entry.get("context_tokens"))
    out["vocab_size"] = parse_count(entry.get("vocab_size"))
    out["num_layers"] = parse_layers(entry.get("layer_mix"))
    out["kv_cache_per_token_bf16"] = parse_bytes(entry.get("kv_cache_per_token_bf16"))
    kind = decoder_kind(entry.get("decoder_type"))
    out["decoder_type"] = (kind, None) if kind else None
    return out


RAW = {
    "params_total": "scale",
    "params_active": "scale",
    "context_length": "context_tokens",
    "vocab_size": "vocab_size",
    "num_layers": "layer_mix",
    "kv_cache_per_token_bf16": "kv_cache_per_token_bf16",
    "decoder_type": "decoder_type",
}
LABEL = {
    "decoder_type": "FFN sparsity (dense / MoE)",
    "kv_cache_per_token_bf16": "growing cache per token (sliding-window and constant states excluded)",
}
CAUSES = {
    "definition": "definition difference (recomputed)",
    "ours_wrong": "confirmed bug (ArchWatch)",
    "card_unreproducible": "card figure not reproducible from the weights",
    "his_wrong": "real error (reference)",
    "name_rounding": "nominal size (model name or '~' figure; ours exact)",
    "unresolved": "unresolved",
}


def compare(records: dict, gallery: dict) -> dict:
    by_repo = {(e.get("config") or {}).get("repo", "").lower(): (name, e) for name, e in gallery.items()}
    rows, missing = [], []
    for repo, record in sorted(records.items()):
        hit = by_repo.get(repo.lower())
        if not hit:
            continue
        name, entry = hit
        mine, other = ours(record), theirs(entry)
        for field, raw in RAW.items():
            if other.get(field) is None or mine.get(field) is None:
                continue
            value, tol = other[field]
            if field == "decoder_type":
                agree = mine[field] == value
            else:
                agree = abs(float(mine[field]) - value) <= tol + 1e-9 * max(1.0, value)
                if not agree and field == "context_length":
                    agree = same_k_shorthand(int(mine[field]), value)
            rows.append(
                dict(
                    model=repo,
                    gallery_name=name,
                    field=field,
                    ours=mine[field],
                    theirs=entry.get(raw),
                    agree=agree,
                    our_evidence=None if agree else evidence(record, field),
                )
            )
    processed = {r.lower() for r in records}
    for name, entry in gallery.items():
        repo = (entry.get("config") or {}).get("repo")
        if not repo or repo.lower() not in processed:
            missing.append(
                dict(gallery_name=name, repo=repo, company=entry.get("company"), date=str(entry.get("date")))
            )
    missing.sort(key=lambda m: m["date"], reverse=True)
    return dict(rows=rows, not_processed=missing)


def attach_causes(result: dict, records: dict, gallery: dict, causes: dict) -> dict:
    """Same rule as the audit: a definition difference only if OUR number, under the fixed rule or an exclusion
    the card states, lands in its accepted range (2% or its own rounding); anything else is ours to fix."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("reference_audit", ROOT / "tools/reference_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    by_repo = {(e.get("config") or {}).get("repo", ""): e for e in gallery.values()}
    for r in result["rows"]:
        if r["agree"]:
            continue
        parsed = theirs(by_repo[r["model"]]).get(r["field"])
        manual = (causes.get(r["model"]) or {}).get(r["field"])
        ours = r["ours"] if isinstance(r["ours"], (int, float)) else None
        closed = audit.close(records[r["model"]], r["field"], str(r["theirs"]), parsed, manual, ours=ours)
        r["cause"], r["cause_why"] = closed["group"], closed["evidence"]
    return result


def report(result: dict, config: dict) -> str:
    rows = result["rows"]
    lines = [
        "# Agreement with an outside reference",
        "",
        f"Reference: [{config['repo']}](https://github.com/{config['repo']}) `{config['path']}` at commit "
        f"`{config['commit'][:10]}` ({config['license']}). **Comparison only**: it is never a "
        "source for ArchWatch's published fields, gold labels or diagrams, and its images are not used.",
        "",
        "Ours: published, verifier-approved record fields (KV cache recomputed from published values, bf16, "
        "growing part per token). Theirs: fact-sheet strings, compared within their displayed rounding.",
        "",
        "## Per-field agreement (models in both sets)",
        "",
        "| Field | Agree | Compared |",
        "|---|---:|---:|",
    ]
    for field in RAW:
        sub = [r for r in rows if r["field"] == field]
        if sub:
            lines.append(f"| {LABEL.get(field, field)} | {sum(r['agree'] for r in sub)} | {len(sub)} |")
    total = len(rows)
    lines += [
        "",
        f"Overall: {sum(r['agree'] for r in rows)} of {total} compared values agree "
        f"across {len({r['model'] for r in rows})} models.",
        "",
        "## Disagreements",
        "",
    ]
    bad = [r for r in rows if not r["agree"]]
    if not bad:
        lines.append("None.")
    else:
        counts = {}
        for r in bad:
            counts[r.get("cause", "unclassified")] = counts.get(r.get("cause", "unclassified"), 0) + 1
        lines.append(
            "Causes: " + ", ".join(f"{CAUSES.get(k, k)} {v}" for k, v in sorted(counts.items())) + ". "
            "A definition difference is accepted only when ArchWatch's own number, under one fixed rule or an "
            "exclusion the model card states, lands within 2% or the reference's own rounding (rule in each row)."
        )
        lines += ["", "| Model | Field | Ours | Theirs | Cause | Why | Our evidence |", "|---|---|---|---|---|---|---|"]
        for r in bad:
            ours_v = f"{r['ours']:,}" if isinstance(r["ours"], int) else r["ours"]
            cause = CAUSES.get(r.get("cause"), r.get("cause", "unclassified"))
            lines.append(
                f"| {r['model']} | {LABEL.get(r['field'], r['field'])} | {ours_v} | {r['theirs']} | **{cause}** | "
                f"{r.get('cause_why', '')} | {r['our_evidence']} |"
            )
    lines += [
        "",
        "A disagreement is a prompt to re-check sources, not a correction: ArchWatch publishes what the model's own "
        "files state.",
        "",
        f"## Gallery models ArchWatch has not processed ({len(result['not_processed'])})",
        "",
        "| Date | Model | Company | Hub repo |",
        "|---|---|---|---|",
    ]
    for m in result["not_processed"]:
        lines.append(f"| {m['date']} | {m['gallery_name']} | {m['company']} | {m['repo'] or '(none listed)'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="reference repository, owner/name (or REFERENCE_REPO)")
    parser.add_argument("--path", help="YAML file in that repository (or REFERENCE_PATH)")
    parser.add_argument("--commit", help="commit to read (default: latest commit touching the file)")
    args = parser.parse_args(argv)
    config = reference_config(args.repo, args.path, args.commit)
    gallery = fetch(config, ROOT / "cache")
    records = {
        json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in (ROOT / "data/models").glob("*.json")
    }
    causes_path = REPORTS / "audit-causes.yaml"  # optional, hand-checked, kept out of git
    causes = (yaml.safe_load(causes_path.read_text()) or {}) if causes_path.exists() else {}
    result = attach_causes(compare(records, gallery), records, gallery, causes)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "agreement.json").write_text(
        json.dumps(dict(reference=dict(repo=config["repo"], commit=config["commit"]), **result), indent=2) + "\n"
    )
    (REPORTS / "agreement.md").write_text(report(result, config))
    agree = sum(r["agree"] for r in result["rows"])
    print(f"{agree}/{len(result['rows'])} values agree; report in cache/reference/agreement.md")


if __name__ == "__main__":
    main()
