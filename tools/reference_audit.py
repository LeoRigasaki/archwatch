"""Audit the deterministic pipeline against an external reference sheet used as an answer sheet.

    REFERENCE_REPO=owner/repo REFERENCE_PATH=file.yml uv run --frozen python tools/reference_audit.py --analyze
    REFERENCE_REPO=owner/repo REFERENCE_PATH=file.yml uv run --frozen python tools/reference_audit.py

The reference's values are NEVER a source: nothing here writes data/models, gold labels, diagrams or the site.
Our side is always our own published (verifier-approved) fields. The reference is named at run time and
downloaded into the ignored cache/ folder; the report goes to cache/reference/audit.{md,json} and is never
committed. Optional hand-checked causes are read from cache/reference/audit-causes.yaml.
"""

import argparse
import concurrent.futures as futures
import importlib.util
import json
import re
import sys
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
        "Inkling's native embedding names (model.llm.embed / unembed) not recognized as embedding and output head",
        "test_inkling_embed_and_unembed_are_embedding_and_head",
    ),
    (
        "Step-3.5 / Step-5 routed experts stacked in 3-D moe.*_proj tensors (and share_expert) not recognized, "
        "so every expert counted as active",
        "test_step_stacked_experts_and_share_expert_are_recognized",
    ),
    (
        "compressed-tensors int4 packed into int32 counted as one value per int32 (Kimi-K2.5/2.6/2.7)",
        "test_int32_pack_quantized_experts_hold_eight_int4_values",
    ),
    (
        "Comparison parser: 'Mamba-2' read as 2 layers, MTP counted as a layer, 128,000 vs 128K not normalised",
        "test_layer_mix_counts_leading_terms_and_skips_mtp",
    ),
]
GROUPS = ("ours_wrong", "his_wrong", "card_unreproducible", "name_rounding", "definition", "unresolved")


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


# One fixed rule per field. An exclusion is applied only when the model's own card states it for this figure.
DEFAULT_RULE = {
    "params_total": "default: all stored tensors minus MTP modules (reported separately) and minus vision tower "
    "and projector (text-only figure)",
    "params_active": "default: per-token count; unselected experts, MTP, lookup tables and the input embedding "
    "(row lookups) not active; output head counted (tied: shared matrix once, as the head)",
    "kv_cache_per_token_bf16": "default: full retention, every attention layer appends its KV per token "
    "(sliding-window layers not capped)",
}
APPROX = re.compile(r"(~|\\~|≈|approximately|approx\.)", re.I)
STATED = {
    # name: (pattern in the sentence that states his figure, what it means)
    "embedding": (
        r"non[- ]?embedding|exclud\w*[^.|]{0,40}embedding|without[^.|]{0,20}embedding",
        "input embedding excluded; output head too unless the card says it is included",
    ),
    "head_included": (r"includ\w*[^.|]{0,20}(output|lm)[ _-]?head", "output head included"),
    "lookup": (
        r"\d[\d.]*\s*[BM]\b[^.|]{0,25}(engram|n-gram|ngram)|exclud\w*[^.|]{0,40}(engram|n-gram|per-layer embed)",
        "lookup tables stated separately",
    ),
    "vision_included": (r"includ\w*[^.|]{0,25}(vision|perception|image) encoder", "vision encoder included"),
}


def _card_text(record: dict) -> str | None:
    path = ROOT / "cache" / record["id"].replace("/", "__") / record["source_commit"] / "README.md"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if text.startswith("---"):
        text = text.split("---", 2)[-1]  # YAML front matter is metadata, not the card
    text = re.sub(r"\]\([^)\s]*\)", "]", text)  # markdown link targets name repos, they state nothing
    return re.sub(r"https?://\S+", " ", text)


def card_statements(record: dict, number: str, unit: str) -> list[dict]:
    """Every sentence of the model's own card that states this figure, with any footnote it points to and the
    text just before the number (to see a '~' / 'approximately')."""
    text = _card_text(record)
    if not text or not number:
        return []
    pattern = rf"(?<![\w.\-/]){re.escape(number)}\s*{unit}\b(?!-(?!A\d|token|context))"
    out = []
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z*])", line):
            for hit in re.finditer(pattern, sentence):
                quote = " ".join(sentence.split()).replace("|", "/")
                for mark in set(re.findall(r"[¹²³⁴⁵]", sentence)):
                    note = next((ln for ln in text.splitlines() if ln.strip().startswith(mark)), None)
                    if note:
                        quote += " [footnote] " + " ".join(note.split())
                out.append(dict(quote=quote, before=sentence[max(0, hit.start() - 40) : hit.start()]))
    return out


def _his_number(field: str, raw: str) -> tuple[str, str]:
    raw = str(raw)
    m = re.search(r"([\d.]+)\s*([KMBT])\s*active", raw) if field == "params_active" else None
    m = m or (re.match(r"\s*([\d.,]+)\s*([KMBT])", raw) if field == "params_total" else None)
    return (m.group(1), m.group(2)) if m else ("", "")


def name_sizes(repo: str, field: str) -> list[float]:
    """Sizes written in the model name: '12B-A2.5B' gives total 12B and active 2.5B."""
    name = repo.split("/")[-1]
    out = []
    for m in re.finditer(r"(?<![A-Za-z0-9.])(A?)(\d+(?:\.\d+)?)([BMT])(?![A-Za-z0-9])", name, re.I):
        active = m.group(1).upper() == "A"
        if active == (field == "params_active"):
            out.append(float(m.group(2)) * UNIT[m.group(3).upper()])
    return out


def fixed_rule(
    record: dict, field: str, his_raw: str, his_value: float | None = None
) -> tuple[str, float, list[str]] | None:
    """Our number under the one fixed rule, adjusted only by exclusions the card states for his figure."""
    pub = record["published"]
    if field == "kv_cache_per_token_bf16":
        rows = record.get("cache", {}).get("breakdown", [])
        window = pub.get("sliding_window")
        full = sum(r["grows_per_token_bytes"] or 0 for r in rows) + sum(
            r["constant_bytes"] // window
            for r in rows
            if r["layer_type"] in SLIDING and window and r.get("constant_bytes")
        )
        return DEFAULT_RULE[field], full, []
    if field not in ("params_total", "params_active") or pub.get(field) is None:
        return None
    comp = record["params"]["counted_from_weights"]["by_component"]
    stated = {}
    number, unit = _his_number(field, his_raw)
    statements = card_statements(record, number, unit)
    if (
        his_value is not None
        and number
        and abs(float(number.replace(",", "")) * UNIT[unit] - his_value) > 1e-6 * his_value
    ):
        statements = []  # his figure is a sum ('100B core ... plus 40B'): the card sentence states a part, not it
    for st in statements:
        for key, (pattern, _) in STATED.items():
            if key not in stated and re.search(pattern, st["quote"], re.I):
                stated[key] = st["quote"]
    value, parts, quotes = pub[field], [DEFAULT_RULE[field]], []
    vision = comp.get("vision", 0) + comp.get("vision_projector", 0)
    if field == "params_total":
        if "vision_included" in stated:
            parts, quotes = ["card: all stored tensors including the vision encoder"], [stated["vision_included"]]
        else:
            value -= vision
    removals = []
    tied = not comp.get("lm_head", 0)
    if "embedding" in stated and field == "params_total":
        removals.append(("input embedding", comp.get("embedding", 0), stated["embedding"]))
        if "head_included" not in stated and not tied:
            removals.append(("output head", comp.get("lm_head", 0), stated["embedding"]))
    if "embedding" in stated and field == "params_active" and "head_included" not in stated:
        # the default already leaves out the input embedding; "non-embedding" also drops the head
        # (tied: the shared matrix, which the default counts once as the head)
        removals.append(("output head", comp.get("embedding" if tied else "lm_head", 0), stated["embedding"]))
    if field == "params_total" and "lookup" in stated:
        removals.append(("lookup tables", comp.get("lookup_embedding", 0), stated["lookup"]))
    removals = [r for r in removals if r[1]]
    if removals:
        parts.append("card states: excluding " + ", ".join(n for n, _, _ in removals))
        value -= sum(v for _, v, _ in removals)
        quotes = list(dict.fromkeys(quotes + [q for _, _, q in removals]))
    return "; ".join(parts), value, quotes


def approx_quote(record: dict, field: str, his_raw: str) -> str | None:
    """The card sentence that gives his figure as approximate ('~23B', 'Approximately 428B')."""
    for st in card_statements(record, *_his_number(field, his_raw)):
        if APPROX.search(st["before"]):
            return st["quote"]
    return None


def candidates(record: dict, field: str, his_raw: str) -> list[tuple[str, float]]:
    """Context only: the config value, and every figure the card's own sentence states (K read as 1000 or 1024)."""
    pub = record["published"]
    out = []
    if field == "context_length":
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


def _nominal(record: dict, field: str, his_raw: str, value: float) -> dict | None:
    """name_rounding: his figure is the size in the model name or a card '~' figure, ours is outside 2% of it but
    within half a unit of its last stated digit ('3B' = 2.5-3.5B, '2.5B' = 2.45-2.55B, '~23B' = 22.5-23.5B)."""
    if field not in ("params_total", "params_active"):
        return None
    number, unit = _his_number(field, his_raw)
    if not number:
        return None
    his_value = float(number.replace(",", "")) * UNIT[unit]
    half = _half_unit(number, UNIT[unit])
    if abs(value - his_value) <= 0.02 * his_value or abs(value - his_value) > half:
        return None
    in_name = any(abs(s - his_value) < 1e-6 * his_value for s in name_sizes(record["id"], field))
    approx = None if in_name else approx_quote(record, field, his_raw)
    if not (in_name or approx):
        return None
    where = (
        f"his {number}{unit} is the size in the model name ({record['id'].split('/')[-1]})"
        if in_name
        else f'the card gives it as approximate: "{approx}"'
    )
    rule = DEFAULT_RULE[field]
    return dict(
        group="name_rounding",
        recomputed=value,
        rule="nominal size: " + ("model name" if in_name else "card '~' figure"),
        card_quote=[approx] if approx else None,
        evidence=f"Nominal figure: {where}. Ours is the exact count from the weights ({rule} = {value:,.0f}), "
        f"{abs(value - his_value) / his_value:.2%} from it and within half a unit of its last digit "
        f"({his_value - half:,.0f}-{his_value + half:,.0f}).",
    )


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
    fixed = (
        fixed_rule(record, field, his_raw, parsed[0] if field == "params_total" and parsed else None) if rng else None
    )
    if fixed:
        rule, value, quotes = fixed
        nominal = _nominal(record, field, his_raw, value) if not quotes else None
        if nominal:
            return dict(base, **nominal)
        if rng[0] <= value <= rng[1]:
            card = "".join(f' Card: "{q}".' for q in quotes)
            return dict(
                base,
                group="definition",
                recomputed=value,
                rule=rule,
                card_quote=quotes or None,
                evidence=f"Rule: {rule}. Ours recomputed: {value:,.0f}. His: {rng[2]} (accepted "
                f"{rng[0]:,.0f}-{rng[1]:,.0f}); gap after the rule {_gap_after(value, rng, his_raw):.2%}.{card}",
            )
    else:
        hits = [(r, v) for r, v in (candidates(record, field, his_raw) if rng else []) if rng[0] <= v <= rng[1]]
        if hits:
            rule, value = hits[0]
            return dict(
                base,
                group="definition",
                recomputed=value,
                rule=rule,
                card_quote=[rule.split(': "', 1)[1].rstrip('"')] if ': "' in rule else None,
                evidence=f"Rule: {rule}. Ours recomputed: {value:,.0f}. His: {rng[2]} (accepted "
                f"{rng[0]:,.0f}-{rng[1]:,.0f}); gap after the rule {_gap_after(value, rng, his_raw):.2%}.",
            )
    if manual and manual.get("group") == "ours_wrong":
        return dict(base, group="ours_wrong", evidence="Bug: " + " ".join(manual["evidence"].split()))
    read = tensors_read(record)
    checked = (
        read["text"]
        if field in ("params_total", "params_active")
        else "config.json and the model card at the pinned commit; neither states his figure."
    )
    if fixed:
        checked += f" Fixed rule: {fixed[0]} = {fixed[1]:,.0f}; his accepted range {rng[0]:,.0f}-{rng[1]:,.0f}."
    if field in ("params_total", "params_active") and not read["complete"]:
        return dict(base, group="ours_wrong", recomputed=fixed and fixed[1], rule=fixed and fixed[0], evidence=checked)
    return dict(
        base,
        group="card_unreproducible",
        recomputed=fixed[1] if fixed else None,
        rule=fixed[0] if fixed else None,
        evidence=("Checked: " + checked) if fixed else f"Checked: {checked} No rule reproduces his figure.",
    )


def tensors_read(record: dict) -> dict:
    """Did we read every tensor the repo lists? Shard index (if any) against the headers actually read."""
    counted = (record.get("params") or {}).get("counted_from_weights") or {}
    n = counted.get("tensor_count")
    folder = ROOT / "cache" / record["id"].replace("/", "__") / record["source_commit"]
    index = folder / "model.safetensors.index.json"
    skipped = counted.get("skipped_scale_tensors", 0) + counted.get("skipped_buffer_tensors", 0)
    if index.exists():
        listed = json.loads(index.read_text()).get("weight_map", {})
        shards = len(set(listed.values()))
        headers = folder / "_tensors.json"
        read = set(json.loads(headers.read_text())) if headers.exists() else set()
        missing = len(set(listed) - read) if read else None
        complete = missing == 0
        text = (
            f"read {len(read) if read else n:,} tensor headers; the shard index lists {len(listed):,} in {shards} "
            f"shards ({'none missing' if complete else f'{missing} missing' if missing is not None else 'headers not cached'}); "
            f"{skipped} scale/buffer tensors counted as metadata."
        )
        return dict(complete=complete, text=text)
    return dict(
        complete=n is not None,
        text=f"single safetensors file, all {n or 0:,} tensor headers read; {skipped} scale/buffer tensors counted as metadata.",
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


def _is_default(row: dict) -> bool:
    rule = row.get("rule") or ""
    return rule.startswith("default:") and "card states" not in rule


def group_of(row: dict) -> str:
    return "agree" if row["agree"] else row["group"]


def moved_rows(previous: list[dict], rows: list[dict]) -> list[tuple]:
    before = {(r["model"], r["field"]): group_of(r) for r in previous}
    return [
        (r["model"], r["field"], before.get((r["model"], r["field"]), "not compared"), group_of(r))
        for r in rows
        if before.get((r["model"], r["field"]), "not compared") != group_of(r)
    ]


def report(meta: dict, mapping_rows: list, runs: list, rows: list, moved: list | None = None) -> str:
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
        "Groups: **ours_wrong** = a confirmed bug in our code (or weights we did not read); **his_wrong** = the "
        "model's own files contradict his value; **card_unreproducible** = we read every tensor and no fixed or "
        "card-stated rule reproduces the card's figure (what we checked is listed); **name_rounding** = his value is "
        "the size in the model name or a card '~' figure, and our exact count is within half a unit of its last "
        "digit; "
        "**definition** = our number under the fixed rule (or a rule the card states) lands in his accepted range; "
        "**unresolved** = review queue.",
        "",
        "Fixed rules (the same for every model without a card statement): "
        + "; ".join(f"`{f}` = {r.split(': ', 1)[1]}" for f, r in DEFAULT_RULE.items())
        + ". An exclusion (embedding, output head, MTP, lookup tables) or a vision inclusion is applied only when "
        "the card sentence that states his figure says so; that sentence is quoted below. Accepted range: 2% or "
        "the card's own rounding.",
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
    special = [r for r in diffs if r["group"] in ("definition", "name_rounding") and not _is_default(r)]
    lines += [
        "",
        "## Rows not on the fixed default rule",
        "",
        "| Model | Field | Group | Rule | Source quote |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(special, key=lambda r: (r["group"], r["field"], r["model"])):
        quote = " / ".join(r.get("card_quote") or [])
        if not quote and r["group"] == "name_rounding":
            quote = "model name: " + r["model"].split("/")[-1]
        elif not quote:
            quote = "(file evidence) " + r["evidence"].split(". Ours: ", 1)[-1]
        rule = (r["rule"] or "").split(': "')[0]
        lines.append(f"| {r['model']} | {r['field']} | {r['group']} | {rule} | {quote} |")
    if moved:
        lines += [
            "",
            "## Rows that changed group since the previous run",
            "",
            "| Model | Field | Before | Now |",
            "|---|---|---|---|",
        ]
        lines += [f"| {m} | {f} | {a} | {b} |" for m, f, a, b in moved]
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
    parser.add_argument("--repo", help="reference repository, owner/name (or REFERENCE_REPO)")
    parser.add_argument("--path", help="YAML file in that repository (or REFERENCE_PATH)")
    parser.add_argument("--commit", help="answer-sheet commit (default: latest commit touching the file)")
    args = parser.parse_args()
    config = ref.reference_config(args.repo, args.path, args.commit)
    file_commit = config["commit"]
    gallery = ref.fetch(config, ROOT / "cache")
    rows_map = mapping(gallery)
    CACHE.mkdir(parents=True, exist_ok=True)
    global REFRESH
    REFRESH = not args.reanalyze
    runs = analyze(rows_map) if (args.analyze or args.reanalyze) else json.loads((CACHE / "runs.json").read_text())
    causes_path = ref.REPORTS / "audit-causes.yaml"  # optional, hand-checked, kept out of git
    causes = (yaml.safe_load(causes_path.read_text()) or {}) if causes_path.exists() else {}
    records = load_records()
    by_repo = {(e.get("config") or {}).get("repo", ""): e for e in gallery.values()}
    theirs = {repo: ref.theirs(by_repo[repo]) for repo in records if repo in by_repo}
    compared = classify(ref.compare(records, gallery)["rows"], causes, records, theirs)
    meta = dict(repo=config["repo"], path=config["path"], file_commit=file_commit, head=file_commit)
    out = ref.REPORTS
    out.mkdir(parents=True, exist_ok=True)
    previous_path = out / "audit.json"
    previous = json.loads(previous_path.read_text())["rows"] if previous_path.exists() else []
    moved = moved_rows(previous, compared)
    (out / "audit.json").write_text(
        json.dumps(dict(reference=meta, mapping=rows_map, runs=runs, rows=compared), indent=1, default=str) + "\n"
    )
    (out / "audit.md").write_text(report(meta, rows_map, runs, compared, moved))
    diffs = [r for r in compared if not r["agree"]]
    print(
        {g: sum(r["group"] == g for r in diffs) for g in GROUPS}, f"{len(compared) - len(diffs)}/{len(compared)} agree"
    )


if __name__ == "__main__":
    main()
