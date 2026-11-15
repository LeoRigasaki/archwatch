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
import re
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
UNIT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _half_unit(number: str, scale: float) -> float:
    """The card's own rounding: '3B' covers 2.5-3.5B, '2.78T' covers 2.775-2.785T."""
    digits = number.replace(",", "")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    return 0.5 * 10**-decimals * scale


def his_range(field: str, raw: str, parsed) -> tuple[float, float, str] | None:
    """His value as an accepted interval: max(2%, his own rounding). A stated range stays a range."""
    raw = str(raw)
    if field == "params_active":
        pre = re.search(r"([\d.]+)\s*([KMBT])\s*active prefill.*?([\d.]+)\s*([KMBT])\s*active decode", raw)
        if pre:
            lo = float(pre.group(1)) * UNIT[pre.group(2)]
            hi = float(pre.group(3)) * UNIT[pre.group(4)]
            return (
                lo * 0.98,
                hi * 1.02,
                f"{pre.group(1)}{pre.group(2)} (prefill) to {pre.group(3)}{pre.group(4)} (decode)",
            )
        m = re.search(r"([\d.]+)\s*([KMBT])\s*active", raw)
        if not m:
            return None
        v = float(m.group(1)) * UNIT[m.group(2)]
        tol = max(0.02 * v, _half_unit(m.group(1), UNIT[m.group(2)]))
        return v - tol, v + tol, m.group(1) + m.group(2)
    if field == "params_total":
        m = re.match(r"\s*([\d.,]+)\s*([KMBT])", raw)
        if parsed is None or not m:
            return None
        v = parsed[0]
        tol = max(0.02 * v, parsed[1], _half_unit(m.group(1), UNIT[m.group(2)]))
        return v - tol, v + tol, raw.split(",")[0].split(";")[0].strip()
    if parsed is None:
        return None
    v, own = parsed
    tol = max(0.02 * v, own)
    return v - tol, v + tol, raw


def _subsets(items):
    for mask in range(1 << len(items)):
        yield [x for i, x in enumerate(items) if mask >> i & 1]


def candidates(record: dict, field: str, his_raw: str) -> list[tuple[str, float]]:
    """Our number recomputed under each named definition the reference might use."""
    pub = record["published"]
    out = []
    if field in ("params_total", "params_active"):
        comp = record["params"]["counted_from_weights"]["by_component"]
        active = record["params"].get("active") or {}
        base = pub.get(field)
        if base is None:
            return out
        if field == "params_total":
            removable = [
                ("vision tower and projector", comp.get("vision", 0) + comp.get("vision_projector", 0)),
                ("MTP / auxiliary heads", comp.get("auxiliary", 0)),
                ("lookup tables (Engram, n-gram, per-layer embeddings)", comp.get("lookup_embedding", 0)),
                ("input embedding", comp.get("embedding", 0)),
                ("output head", comp.get("lm_head", 0)),
            ]
            for chosen in _subsets([r for r in removable if r[1]]):
                rule = "all stored tensors" + (" excluding " + ", ".join(n for n, _ in chosen) if chosen else "")
                out.append((rule, base - sum(v for _, v in chosen)))
        else:
            removable = [("input embedding", -comp.get("embedding", 0)), ("output head", -comp.get("lm_head", 0))]
            addable = [("vision tower", comp.get("vision", 0) + comp.get("vision_projector", 0))]
            for chosen in _subsets([r for r in removable + addable if r[1]]):
                parts = [("excluding " if v < 0 else "including ") + n for n, v in chosen]
                rule = "per-token weights (unselected experts, MTP heads and lookup tables excluded)" + (
                    ", " + ", ".join(parts) if parts else ""
                )
                out.append((rule, base + sum(v for _, v in chosen)))
            if active.get("text_active") is not None and active["text_active"] != base:
                out.append(("per-token weights, all non-expert weights (current calculator)", active["text_active"]))
    elif field == "kv_cache_per_token_bf16":
        rows = record.get("cache", {}).get("breakdown", [])
        growing = sum(r["grows_per_token_bytes"] or 0 for r in rows)
        out.append(("growing cache per token (sliding-window layers capped)", growing))
        window = pub.get("sliding_window")
        full = growing + sum(
            r["constant_bytes"] // window
            for r in rows
            if r["layer_type"] in SLIDING and window and r.get("constant_bytes")
        )
        out.append(("full retention: every attention layer appends its KV per token (sliding layers not capped)", full))
    elif field == "context_length":
        out.append(("config.json max_position_embeddings", pub.get("context_length")))
        quote = card_quote(record, _context_token(his_raw))
        for label, stated in _stated_contexts(quote or ""):
            out.append((f'advertised context the model card states ("{label}"): "{quote}"', stated))
    return [(r, v) for r, v in out if v is not None]


def _context_token(raw: str) -> str:
    m = re.match(r"\s*([\d,.]+)\s*([KM])?", str(raw))
    return (m.group(1) + (m.group(2) or "")) if m else ""


def _stated_contexts(quote: str) -> list[tuple[str, float]]:
    """Every context figure a card sentence states, in tokens. Cards write K for both 1,000 and 1,024 ('128K'
    for 128,000 or 131,072), so both readings are offered; plain digit figures are taken as written."""
    out = []
    for m in re.finditer(r"(\d[\d,.]*)\s*([KkM])\b(?![A-Za-z0-9])|(\d{1,3}(?:,\d{3})+)\s*tokens", quote):
        if m.group(3):
            out.append((m.group(3), float(m.group(3).replace(",", ""))))
            continue
        n, unit = float(m.group(1).replace(",", "")), m.group(2).lower()
        label = m.group(1) + m.group(2)
        out += [(label, n * 1e3), (label, n * 1024)] if unit == "k" else [(label, n * 1e6), (label, n * 1024**2)]
    return out


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
        rf"(?<![\w.\-/]){re.escape(number)}\s*{unit}\b(?!-(?!A\d|token|context))"
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


def _gap_after(value: float, rng, his_raw: str) -> float:
    """Gap between the recomputed number and his figure: 0 inside a stated range, else relative to his value."""
    if "prefill" in str(rng[2]):
        return 0.0 if rng[0] <= value <= rng[1] else min(abs(value - rng[0]), abs(value - rng[1])) / rng[1]
    nominal = (rng[0] + rng[1]) / 2
    return abs(value - nominal) / nominal


def close(record: dict, field: str, his_raw: str, parsed, manual: dict | None, ours=None) -> dict:
    """Classify one difference. Definition only if a recomputation of OUR number lands in his accepted range."""
    rng = his_range(field, his_raw, parsed)
    ours = ours if ours is not None else record["published"].get(field)
    gap = abs(ours - parsed[0]) / parsed[0] if rng and isinstance(ours, (int, float)) and parsed and parsed[0] else None
    base = dict(gap=gap, recomputed=None, rule=None)
    if manual and manual.get("group") == "his_wrong":
        return dict(
            base, group="his_wrong", evidence=" ".join(manual["evidence"].split()), rule="primary source contradicts"
        )
    if manual and manual.get("recomputed") is not None and rng:
        rec = float(manual["recomputed"])
        if rng[0] <= rec <= rng[1]:
            return dict(
                base,
                group="definition",
                recomputed=rec,
                rule=manual["rule"],
                evidence=f"Rule: {manual['rule']}. Ours recomputed: {rec:,.0f}. His: {rng[2]} (accepted "
                f"{rng[0]:,.0f}-{rng[1]:,.0f}). {' '.join(manual['evidence'].split())}",
            )
    best = None
    for rule, value in candidates(record, field, his_raw) if rng else []:
        inside = rng[0] <= value <= rng[1]
        mid = (rng[0] + rng[1]) / 2
        score = (not inside, abs(value - mid))
        if best is None or score < best[0]:
            best = (score, rule, value, inside)
    if best and best[3]:
        _, rule, value, _ = best
        mid = (rng[0] + rng[1]) / 2
        return dict(
            base,
            group="definition",
            recomputed=value,
            rule=rule,
            evidence=f"Rule: {rule}. Ours recomputed: {value:,.0f}. His: {rng[2]} (accepted {rng[0]:,.0f}-{rng[1]:,.0f}); "
            f"gap after the rule {_gap_after(value, rng, his_raw):.2%}.",
        )
    if best:
        _, rule, value, _ = best
        why = f"Closest recomputation: {rule} = {value:,.0f}; his accepted range {rng[0]:,.0f}-{rng[1]:,.0f}."
    else:
        why = "No recomputation available for this field."
    return dict(
        base, group="ours_wrong", recomputed=best[2] if best else None, rule=best[1] if best else None, evidence=why
    )


def classify(rows: list[dict], causes: dict, records: dict, gallery_theirs: dict) -> list[dict]:
    for r in rows:
        if r["agree"]:
            continue
        manual = (causes.get(r["model"]) or {}).get(r["field"])
        ours = r["ours"] if isinstance(r["ours"], (int, float)) and not isinstance(r["ours"], bool) else None
        theirs_parsed = gallery_theirs[r["model"]].get(r["field"])
        r.update(close(records[r["model"]], r["field"], str(r["theirs"]), theirs_parsed, manual, ours=ours))
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
        "| Model | Field | Ours (published) | His | Gap | Ours recomputed | Rule | Group | Evidence |",
        "|---|---|---:|---|---:|---:|---|---|---|",
    ]
    for r in sorted(diffs, key=lambda r: (GROUPS.index(r["group"]), r["field"], r["model"])):
        ours = f"{r['ours']:,}" if isinstance(r["ours"], int) else r["ours"]
        gap = f"{r['gap']:.1%}" if r.get("gap") is not None else "-"
        rec = f"{r['recomputed']:,.0f}" if isinstance(r.get("recomputed"), (int, float)) else "-"
        rule = (r.get("rule") or "-").split(': "')[0]
        lines.append(
            f"| {r['model']} | {r['field']} | {ours} | {r['theirs']} | {gap} | {rec} | {rule} | {r['group']} | {r['evidence']} |"
        )
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
