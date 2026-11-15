"""Audit the deterministic pipeline against rasbt/llm-architecture-gallery's models.yml as an answer sheet.

    uv run --frozen python tools/reference_audit.py --analyze     # map + run our pipeline (no LLM calls)
    uv run --frozen python tools/reference_audit.py               # compare cached analyses, write the report

His values are NEVER a source: nothing here writes data/models, gold labels, diagrams or the site. Our side is
always our own published (verifier-approved) fields. Full analyses are cached under the ignored cache/ folder;
results/reference-audit.{md,json} hold the comparison only and are not part of the site.
"""

import argparse
import concurrent.futures as futures
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("reference_agreement", ROOT / "tools/reference_agreement.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

CACHE = ROOT / "cache" / "reference-audit"

# ArchWatch errors this audit exposed. Each was fixed in code with a regression test (tests/test_reference_audit.py),
# so it no longer appears as a difference; listed so the report stays honest about what was wrong.
FIXED = [
    (
        "MXFP4 expert blocks counted at one value per byte (gpt-oss)",
        "test_mxfp4_blocks_hold_two_values_per_byte_and_scales_are_metadata",
    ),
    ("FP4 expert flag on config.json ignored (DeepSeek-V4)", "test_native_fp4_experts_flagged_on_the_model_config"),
    (
        "Mistral consolidated.safetensors counted on top of the HF shards (Mistral Small 3.1/4)",
        "test_consolidated_export_is_not_counted_twice_but_extra_components_are_kept",
    ),
    (
        "Tensor-parallel shards merged by name, keeping one rank (grok-2)",
        "test_tensor_parallel_slices_count_every_rank",
    ),
    (
        "MTP layers stored after the decoder stack counted as main/active weights (DeepSeek-V3/R1, GLM, Qwen3.8-2.4T)",
        "test_mtp_layers_after_the_decoder_stack_are_auxiliary_not_active",
    ),
    (
        "Multimodal projectors counted as text weights (Mistral Small 4, Llama 4, Mistral Large 4)",
        "test_multimodal_projectors_are_not_text_weights",
    ),
    (
        "Native output heads (head.weight, output.weight) not recognized, so tied embeddings were misread",
        "test_native_output_heads_are_recognized",
    ),
    (
        "moe_num_experts / moe_top_k not read, so MoE models looked dense (Step-3.5, Step-5)",
        "test_moe_keys_with_moe_prefix_are_read",
    ),
    (
        "Comparison parser: 'Mamba-2' read as 2 layers, MTP counted as a layer, 128,000 vs 128K not normalised",
        "test_layer_mix_counts_leading_terms_and_skips_mtp",
    ),
]
GROUPS = ("ours_wrong", "his_wrong", "definition", "unresolved")


def latest_commit(config: dict) -> str:
    url = f"https://api.github.com/repos/{config['repo']}/commits?path={config['path']}&per_page=1"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())[0]["sha"]


def mapping(gallery: dict) -> list[dict]:
    """Map each entry to a Hub repo from its own config link only; anything else is listed, never guessed."""
    rows = []
    for name, entry in gallery.items():
        repo = (entry.get("config") or {}).get("repo")
        if repo:
            rows.append(dict(name=name, repo=repo, status="mapped", reason="config.repo link"))
            continue
        links = [entry.get(k) for k in ("license_url", "aa_url") if entry.get(k)]
        rows.append(
            dict(
                name=name,
                repo=None,
                status="unmapped",
                reason="no config.repo; other links disagree on the variant (" + "; ".join(links) + ")"
                if len(links) > 1
                else "no Hugging Face repo listed",
            )
        )
    return rows


REFRESH = True  # --reanalyze sets False: re-run the current code on the cached, commit-pinned sources


def analyze_one(repo: str, attempts: int = 4) -> dict:
    import time

    for n in range(attempts):
        result = _analyze_once(repo)
        if "429" not in result.get("error", "") or n == attempts - 1:
            return result
        time.sleep(60 * (n + 1))  # Hub rate limit: back off, then retry
    return result


def _analyze_once(repo: str) -> dict:
    from archwatch import hub, pipeline

    out = CACHE / "records" / (repo.replace("/", "__") + ".json")
    try:
        info = hub.repo_info(repo, refresh=REFRESH)
        if out.exists() and REFRESH:
            cached = json.loads(out.read_text())
            if cached.get("source_commit") == info["sha"] and "error" not in cached:
                return dict(repo=repo, status="analyzed", commit=info["sha"])
        record = pipeline.analyze(repo, info=info)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1, default=str))
        return dict(repo=repo, status="analyzed", commit=info["sha"])
    except Exception as error:  # noqa: BLE001 - every failure is reported, never hidden
        text = f"{type(error).__name__}: {error}"[:300]
        gated = any(s in text for s in ("401", "403", "Gated", "gated", "restricted"))
        result = dict(repo=repo, status="access_needed" if gated else "pipeline_error", error=text)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result))
        return result


def analyze(rows: list[dict], workers: int = 2) -> list[dict]:
    repos = [r["repo"] for r in rows if r["status"] == "mapped"]
    with futures.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(analyze_one, repos))
    (CACHE / "runs.json").write_text(json.dumps(results, indent=1))
    return results


def load_records() -> dict:
    records = {}
    for path in sorted((CACHE / "records").glob("*.json")):
        data = json.loads(path.read_text())
        if "error" not in data and "published" in data:
            records[data["id"]] = data
    return records


SLIDING = ("sliding_attention",)


def _near(value, his, tol):
    return value is not None and abs(value - his) <= tol + 1e-9 * max(1.0, his)


def _sig_tolerance(raw_number: str, unit_scale: float) -> float:
    """Rounding normalisation: '230' has 2 significant figures (trailing zeros of an integer are not significant)."""
    digits = raw_number.replace(",", "")
    if "." in digits:
        return 0.5 * 10 ** -len(digits.split(".")[1]) * unit_scale
    stripped = digits.rstrip("0")
    return 0.5 * 10 ** (len(digits) - len(stripped)) * unit_scale


def card_quote(record: dict, raw: str) -> str | None:
    """The sentence of the model's own README (at its pinned commit) that states his figure, if any."""
    import re

    path = ROOT / "cache" / record["id"].replace("/", "__") / record["source_commit"] / "README.md"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if text.startswith("---"):
        text = text.split("---", 2)[-1]  # YAML front matter is metadata (license links, base_model), not the card
    text = re.sub(r"https?://\S+|\(\S*\)", " ", text)  # URLs and link targets name repos, they state nothing
    m = re.match(r"\s*([\d,.]+)\s*([KMBT])?", str(raw))
    if not m:
        return None
    number, unit = m.group(1), m.group(2) or ""
    words = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}.get(unit, "")
    # Not part of a hyphenated name such as 'Qwen3-32B' or '30B-A3B'.
    patterns = [
        rf"(?<![\w.\-/]){re.escape(number)}\s*{unit}\b(?!-(?!A\d))"
        if unit
        else rf"(?<![\w.,\-]){re.escape(number)}(?![\d,])"
    ]
    if words:
        patterns.append(rf"(?<![\d.]){re.escape(number)}\s*{words}")
    if not unit:  # context written as 128,000 / 1,000,000 is stated in cards as 128K / 1M
        value = float(number.replace(",", ""))
        if value >= 1e6 and value % 1e6 == 0:
            patterns.append(rf"(?<![\w.\-/]){value / 1e6:g}\s*M\b")
        elif value >= 1000 and value % 1000 == 0:
            patterns.append(rf"(?<![\w.\-/]){value / 1000:g}\s*K\b")
        elif value >= 1024 and value % 1024 == 0:
            patterns.append(rf"(?<![\w.\-/]){value / 1024:g}\s*K\b")
    for pattern in patterns:
        hit = re.search(pattern, text, re.I)
        if hit:
            start, end = max(0, hit.start() - 90), min(len(text), hit.end() + 90)
            return " ".join(text[start:end].split()).replace("|", "/")
    return None


def definition_proof(record: dict, field: str, his) -> str | None:
    """Recompute the reference's definition from OUR published values; if it reproduces his number,
    the difference is a definition difference. Returns the evidence, or None if it does not."""
    if his is None:
        return None
    value, tol = his
    pub = record["published"]
    raw = his_raw.get((record["id"], field), "")
    import re as _re

    m = _re.match(r"\s*([\d,.]+)\s*([KMBT])?", raw)
    if m and pub.get(field) is not None and field in ("params_total", "params_active"):
        scale = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(2) or "", 1)
        if field == "params_active":
            am = _re.search(r"([\d.]+)\s*([KMBT])\s*active", raw)
            m = am or m
            scale = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(2) or "", 1)
        if abs(pub[field] - value) <= min(_sig_tolerance(m.group(1), scale), 0.025 * value):
            return (
                f"Rounding: his '{m.group(1)}{m.group(2) or ''}' has {len(m.group(1).replace(',', '').rstrip('0').replace('.', ''))} "
                f"significant figure(s); ours {pub[field]:,} rounds to it. Same quantity, different precision."
            )
    if field == "kv_cache_per_token_bf16":
        rows = record.get("cache", {}).get("breakdown", [])
        full = sum(r["grows_per_token_bytes"] or 0 for r in rows)
        window = pub.get("sliding_window")
        for r in rows:
            if r["layer_type"] in SLIDING and window and r.get("constant_bytes"):
                full += r["constant_bytes"] // window
        if _near(full, value, tol):
            return (
                f"Ours: growing cache per token = {pub.get('kv_cache_per_token_bytes') or sum(r['grows_per_token_bytes'] or 0 for r in rows):,} B "
                f"(sliding-window layers excluded: their cache is capped at the {window}-token window). His: full-retention "
                f"estimate counting every attention layer per token. Recomputed from our own published layer dims with "
                f"sliding layers included: {full:,} B, which matches his value."
            )
    if field in ("params_total", "params_active", "context_length"):
        token = raw
        if field == "params_active":
            am = _re.search(r"([\d.]+\s*[KMBT])\s*active", raw)
            token = am.group(1) if am else raw
        quote = card_quote(record, token.replace(" ", ""))
        if quote:
            ours = pub.get(field)
            what = {
                "params_total": "exact count of every stored tensor in the safetensors headers",
                "params_active": "per-token count from tensor headers (all non-expert weights + selected experts)",
                "context_length": "config.json max_position_embeddings (positions the released config supports)",
            }[field]
            return (
                f"His figure is the one the model card states (README.md at {record['source_commit'][:10]}: "
                f'"{quote}"). Ours ({ours:,}) is the {what}.'
            )
    if field in ("params_total", "params_active") and raw:
        token = raw
        if field == "params_active":
            am = _re.search(r"([\d.]+\s*[KMBT])\s*active", raw)
            token = am.group(1) if am else ""
        else:
            tm = _re.match(r"\s*([\d.]+\s*[KMBT])", raw)
            token = tm.group(1) if tm else ""
        token = token.replace(" ", "")
        name = record["id"].split("/", 1)[1]
        if (
            token
            and _re.search(rf"(?<![\d.])(A)?{_re.escape(token)}(?![\d.])", name, _re.I)
            and (field == "params_total" or f"A{token}".lower() in name.lower())
        ):
            return (
                f"His figure is the size in the model's name ('{name}'), a rounded label. Ours ({pub[field]:,}) is "
                + (
                    "the exact count of every stored tensor."
                    if field == "params_total"
                    else "the per-token count from tensor headers."
                )
            )
    if field in ("params_total", "params_active"):
        w = record["params"]["counted_from_weights"]
        comp = w["by_component"]
        base = pub.get(field)
        if base is None:
            return None
        scopes = {
            "without the vision tower": comp.get("vision", 0) + comp.get("vision_projector", 0),
            "without MTP/auxiliary heads": comp.get("auxiliary", 0),
            "without input embedding and output head": comp.get("embedding", 0) + comp.get("lm_head", 0),
            "without the input embedding": comp.get("embedding", 0),
        }
        names = list(scopes)
        for mask in range(1, 2 ** len(names)):
            chosen = [n for i, n in enumerate(names) if mask >> i & 1 and scopes[n]]
            if not chosen or len(chosen) != bin(mask).count("1"):
                continue
            if "without the input embedding" in chosen and "without input embedding and output head" in chosen:
                continue
            alt = base - sum(scopes[n] for n in chosen)
            if _near(alt, value, tol):
                return (
                    f"Ours: {base:,} ({'all stored tensors' if field == 'params_total' else 'per-token, all non-expert weights'}). "
                    f"His figure counts a narrower scope: ours {', '.join(chosen)} is {alt:,}, which matches his "
                    "value at his rounding. Component sizes from our tensor-header counts."
                )
    return None


his_raw: dict = {}


def classify(rows: list[dict], causes: dict, records: dict, gallery_theirs: dict) -> list[dict]:
    for r in rows:
        his_raw[(r["model"], r["field"])] = str(r["theirs"])
    for r in rows:
        if r["agree"]:
            continue
        c = (causes.get(r["model"]) or {}).get(r["field"])
        if c and c.get("group") in GROUPS:
            r["group"], r["evidence"] = c["group"], " ".join(c["evidence"].split())
            continue
        proof = definition_proof(records[r["model"]], r["field"], gallery_theirs[r["model"]].get(r["field"]))
        r["group"], r["evidence"] = ("definition", proof) if proof else ("unresolved", "")
    return rows


def report(meta: dict, mapping_rows: list, runs: list, rows: list) -> str:
    diffs = [r for r in rows if not r["agree"]]
    counts = {g: sum(r["group"] == g for r in diffs) for g in GROUPS}
    analyzed = [r for r in runs if r["status"] == "analyzed"]
    lines = [
        "# Reference audit (internal; not part of the site)",
        "",
        f"Answer sheet: `{meta['repo']}` `{meta['path']}`, last changed at commit `{meta['file_commit']}` "
        f"(repository head `{meta['head']}`). Its values are never a source for ArchWatch records.",
        "",
        f"Entries: {len(mapping_rows)}; mapped to a Hub repo: {sum(r['status'] == 'mapped' for r in mapping_rows)}; "
        f"analyzed by our deterministic pipeline: {len(analyzed)}; compared values: {len(rows)}; "
        f"agreeing: {sum(r['agree'] for r in rows)}; differences: {len(diffs)}.",
        "",
        "| Group | Differences |",
        "|---|---:|",
    ]
    lines += [f"| {g} | {counts[g]} |" for g in GROUPS]
    lines += [
        "",
        "Groups: **ours_wrong** = ArchWatch error still open; **his_wrong** = the model's own files contradict his value; "
        "**definition** = both are right under different definitions (both written down); **unresolved** = review queue.",
        "",
        "## ArchWatch errors found and fixed during this audit",
        "",
        "| Error | Regression test |",
        "|---|---|",
    ]
    lines += [f"| {what} | `{test}` |" for what, test in FIXED]
    lines += [
        "",
        "## Differences",
        "",
        "| Model | Field | Ours | His | Group | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(diffs, key=lambda r: (GROUPS.index(r["group"]), r["model"], r["field"])):
        ours = f"{r['ours']:,}" if isinstance(r["ours"], int) else r["ours"]
        lines.append(f"| {r['model']} | {r['field']} | {ours} | {r['theirs']} | {r['group']} | {r['evidence']} |")
    lines += [
        "",
        "## Not analyzed",
        "",
        "Not counted in the groups above: there is no ArchWatch value to compare. `access_needed` repos need the owner "
        "to request access; `pipeline_error` repos use formats the extractor does not support yet (open ArchWatch work).",
        "",
        "| Entry | Repo | Status | Reason |",
        "|---|---|---|---|",
    ]
    for m in mapping_rows:
        if m["status"] != "mapped":
            lines.append(f"| {m['name']} | - | unmapped | {m['reason']} |")
    for r in runs:
        if r["status"] != "analyzed":
            lines.append(f"| - | {r['repo']} | {r['status']} | {r.get('error', '')} |")
    lines += ["", "## Mapping", "", "| Entry | Hub repo | How |", "|---|---|---|"]
    lines += [f"| {m['name']} | {m['repo'] or '-'} | {m['reason']} |" for m in mapping_rows]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="run our pipeline on every mapped repo (no LLM)")
    parser.add_argument("--reanalyze", action="store_true", help="re-run current code on cached sources")
    parser.add_argument("--commit", help="answer-sheet commit (default: latest commit touching models.yml)")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/reference-gallery.yaml").read_text())
    file_commit = args.commit or latest_commit(config)
    gallery = ref.fetch(dict(config, commit=file_commit), ROOT / "cache")
    rows_map = mapping(gallery)
    CACHE.mkdir(parents=True, exist_ok=True)
    global REFRESH
    REFRESH = not args.reanalyze
    runs = analyze(rows_map) if (args.analyze or args.reanalyze) else json.loads((CACHE / "runs.json").read_text())
    causes = yaml.safe_load((ROOT / "config/reference-audit-causes.yaml").read_text()) or {}
    records = load_records()
    by_repo = {(e.get("config") or {}).get("repo", ""): e for e in gallery.values()}
    theirs = {repo: ref.theirs(by_repo[repo]) for repo in records if repo in by_repo}
    compared = classify(ref.compare(records, gallery)["rows"], causes, records, theirs)
    meta = dict(repo=config["repo"], path=config["path"], file_commit=file_commit, head=config["commit"])
    out = ROOT / "results"
    (out / "reference-audit.json").write_text(
        json.dumps(dict(reference=meta, mapping=rows_map, runs=runs, rows=compared), indent=1, default=str) + "\n"
    )
    (out / "reference-audit.md").write_text(report(meta, rows_map, runs, compared))
    diffs = [r for r in compared if not r["agree"]]
    print(
        {g: sum(r["group"] == g for r in diffs) for g in GROUPS}, f"{len(compared) - len(diffs)}/{len(compared)} agree"
    )


if __name__ == "__main__":
    main()
