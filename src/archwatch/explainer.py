"""'What changed vs predecessor' notes written from the diff JSON only, with code checks (no LLM grading).

The writer sees a deterministic digest of the record's diff: plain-word field labels, numbers already
formatted, layer mixes as text, each numeric change's direction computed by code, withheld values marked
"not published for <model>", and the tensor-group summary. Checks: every number in the note occurs in that
digest, no claim words, no raw field keys or tensor paths, every direction word agrees with the direction of
each change whose numbers share its sentence, and a word limit.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .labels import FIELD_NAMES, KIND_LABELS

ROOT = Path(__file__).resolve().parents[2]

PLAIN = {
    "layer_type_counts": "layer mix",
    "attention_type": "attention type",
    "num_layers": "number of layers",
    "hidden_size": "hidden size",
    "num_attention_heads": "attention heads",
    "num_kv_heads": "key-value heads",
    "head_dim": "head size",
    "v_head_dim": "value head size",
    "ffn_intermediate": "feed-forward width",
    "ffn_gated": "feed-forward type",
    "short_conv_kernel": "convolution kernel size",
    "sliding_window": "sliding window",
    "context_length": "context length",
    "vocab_size": "vocabulary",
    "params_total": "total parameters",
    "params_active": "active parameters",
    "kv_cache_per_token_bytes": "growing cache per token",
    "moe_num_experts": "routed experts",
    "moe_active_experts": "experts used per token",
    "moe_shared_experts": "shared experts",
    "moe_expert_intermediate": "expert width",
    "moe_latent_dim": "MoE latent width",
    "dsa_index_topk": "sparse-attention token budget",
    "generation_type": "generation method",
    "canvas_length": "diffusion canvas length",
    "activation": "activation function",
}
# Most important architectural change first, then size, then the rest. model_type is never described.
ORDER = [
    "generation_type",
    "layer_type_counts",
    "attention_type",
    "moe_num_experts",
    "moe_active_experts",
    "moe_shared_experts",
    "moe_expert_intermediate",
    "moe_latent_dim",
    "dsa_index_topk",
    "canvas_length",
    "params_total",
    "params_active",
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
    "ffn_intermediate",
    "context_length",
    "vocab_size",
    "kv_cache_per_token_bytes",
]
SKIP = {"model_type"}
BOOL = {
    "ffn_gated": ("gated", "not gated"),
    "qk_norm": ("used", "not used"),
    "attention_output_gate": ("present", "absent"),
    "tied_embeddings": ("tied", "separate"),
}
VALUES = {"silu": "SiLU", "gelu": "GELU", "relu2": "squared ReLU", "block_diffusion": "block diffusion"}


def short(repo: str) -> str:
    return repo.split("/", 1)[-1]


NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred|thousand|million|billion)\b",
    re.I,
)
AWKWARD = re.compile(r"\b(having been|head dim|\(q/k\)|ffn|kv heads|model type)\b", re.I)
BANNED = re.compile(r"\b(benchmark|faster|slower|better|worse|accuracy|score|trained|training)\b", re.I)
UP = re.compile(
    r"\b(up|increas\w*|gr[eo]w\w*|rose|rise\w*|ris(?:ing|es)|doubl\w*|more|larger|higher|expand\w*)\b", re.I
)
DOWN = re.compile(
    r"\b(down|decreas\w*|shr[ai]nk\w*|fell|fall\w*|drop\w*|reduc\w*|halv\w*|fewer|less|smaller|lower)\b", re.I
)


def _label(field: str) -> str:
    return PLAIN.get(field, FIELD_NAMES.get(field, field.replace("_", " ")).lower())


def _human(field: str, value, digits: int | None = None):
    """Readable form; the check verifies it against the exact diff value within the stated rounding."""
    if isinstance(value, bool):
        return BOOL.get(field, ("yes", "no"))[0 if value else 1]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str):
        return VALUES.get(value, value.replace("_", " "))
    if isinstance(value, dict):
        return " + ".join(f"{n} {KIND_LABELS.get(k, k.replace('_', ' '))}" for k, n in value.items())
    if isinstance(value, (int, float)) and field.startswith("params"):
        for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
            if value >= div:
                x = value / div
                if digits is not None:
                    return f"{x:.{digits}f}{unit}"
                return f"{x:.2f}{unit}" if x < 10 else f"{x:.1f}{unit}" if x < 100 else f"{x:.0f}{unit}"
    if isinstance(value, (int, float)) and field.endswith("_bytes"):
        if value < 1024:
            return f"{value} bytes"
        return f"{value / 1024:.{digits or 0}f} KB"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _show(field: str, value, model: str, record: dict | None) -> str:
    if value is not None:
        return _human(field, value)
    check = next((c for c in (record or {}).get("checks", []) if c["field"] == field), None)
    if check and check["status"] == "conflict":
        claims = ", ".join(
            f"{c['evidence'].split(':', 1)[0]} says {_human(field, c['value'])}" for c in check["candidates"]
        )
        return f"in review for {short(model)}: {claims}"
    return f"not stated for {short(model)}"


def _namespace_change(tensors: dict):
    """Detect a weight-namespace rename: listed added/removed groups that match after the first path segment."""
    added = {a["pattern"].split(".", 1)[-1]: a for a in tensors.get("added", [])}
    removed = tensors.get("removed", [])
    pairs = [
        (r["pattern"].split(".", 1)[0], added[r["pattern"].split(".", 1)[-1]]["pattern"].split(".", 1)[0])
        for r in removed
        if r["pattern"].split(".", 1)[-1] in added and added[r["pattern"].split(".", 1)[-1]]["params"] == r["params"]
    ]
    if removed and len(pairs) * 2 >= len(removed):
        return sorted(set(pairs))
    return None


def diff_input(record: dict) -> dict:
    pred = record.get("predecessor") or {}
    if not pred or not record.get("diff_vs_predecessor"):
        raise ValueError("Record has no predecessor diff")
    old, new = pred["id"], record["id"]
    pred_path = ROOT / "data/models" / (old.replace("/", "__") + ".json")
    pred_record = json.loads(pred_path.read_text()) if pred_path.exists() else None
    rows = [r for r in record["diff_vs_predecessor"] if r["field"] not in SKIP]
    rows.sort(key=lambda r: ORDER.index(r["field"]) if r["field"] in ORDER else len(ORDER))
    changes = []
    for row in rows:
        before, after = row.get("before"), row.get("after")
        direction = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool):
            direction = "up" if after > before else "down" if after < before else None
        b_text, a_text = _show(row["field"], before, old, pred_record), _show(row["field"], after, new, record)
        if direction and b_text == a_text:  # rounding hid a real change: add digits until it shows
            for digits in range(1, 7):
                b_text, a_text = _human(row["field"], before, digits), _human(row["field"], after, digits)
                if b_text != a_text:
                    break
        changes.append(
            dict(
                what=_label(row["field"]),
                before=b_text,
                after=a_text,
                direction=direction,
                exact=[v for v in (before, after) if isinstance(v, (int, float)) and not isinstance(v, bool)],
            )
        )
    tensors = record.get("tensor_diff_vs_predecessor") or {}
    renamed = _namespace_change(tensors)
    if renamed:
        weights = (
            "The weight namespace changed ({}), so added and removed tensor-group totals are not comparable.".format(
                ", ".join(f"{a} to {b}" for a, b in renamed)
            )
        )
    else:
        add_n, rem_n = tensors.get("added_count", 0), tensors.get("removed_count", 0)
        add = (
            f"{add_n} weight groups added ({_human('params', tensors.get('added_total') or 0)} parameters)"
            if add_n
            else "no weight groups added"
        )
        rem = (
            f"{rem_n} removed ({_human('params', tensors.get('removed_total') or 0)} parameters)"
            if rem_n
            else "none removed"
        )
        weights = f"{add}; {rem}."
    exact = [
        tensors.get("added_total"),
        tensors.get("removed_total"),
        tensors.get("added_count"),
        tensors.get("removed_count"),
    ]
    return dict(
        predecessor=short(old),
        successor=short(new),
        changes=changes,
        weights=weights,
        exact_values=sorted({v for c in changes for v in c["exact"]} | {v for v in exact if v}),
    )


def input_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _numbers(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return re.findall(r"\d+(?:\.\d+)?", text)


def _canonical(n: str) -> str:
    return str(int(float(n))) if float(n).is_integer() else str(float(n))


def allowed_numbers(data: dict) -> set[str]:
    shown = {k: v for k, v in data.items() if k != "exact_values"}
    return {_canonical(n) for n in _numbers(json.dumps(shown, ensure_ascii=False))}


UNIT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "KB": 1024, "MB": 1024**2}


def _rounds_from(token: str, unit: str, exact: list) -> bool:
    """'2.78' + 'T' is allowed if some exact diff value rounds to it at the shown precision."""
    value = float(token) * UNIT[unit]
    step = 10 ** -(len(token.split(".")[1]) if "." in token else 0) * UNIT[unit]
    return any(abs(e - value) <= step / 2 + 1e-9 * max(1, e) for e in exact)


def unverified_numbers(text: str, data: dict) -> list[str]:
    allowed, exact, bad = allowed_numbers(data), data.get("exact_values", []), []
    clean = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s?(KB|MB|K|M|B|T)?\b", clean):
        token, unit = m.group(1), m.group(2)
        if unit and _rounds_from(token, unit, exact):
            continue
        if not unit and (_canonical(token) in allowed or _canonical(token) in {_canonical(str(e)) for e in exact}):
            continue
        if unit and _canonical(token) in allowed and f"{token}{unit}" in clean.replace(" ", ""):
            if (token + unit) in json.dumps(data).replace(" ", "") or (token + " " + unit) in json.dumps(data):
                continue
        bad.append(token + (unit or ""))
    return sorted(set(bad))


def word_limit(data: dict, settings: dict) -> int:
    return min(
        settings["max_words_cap"], settings["max_words_base"] + settings["max_words_per_change"] * len(data["changes"])
    )


def format_numbers(text: str) -> str:
    """Formatting only: add thousands separators to bare integers of five or more digits."""
    return re.sub(r"(?<![\d,.])\d{5,}(?![\d,]|\.\d)", lambda m: f"{int(m.group(0)):,}", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def _clause_numbers_ok(clause_nums: set[str], change: dict) -> bool:
    """Every number in a clause about one field must be that field's own before or after value."""
    own = {_canonical(n) for n in _numbers(f"{change['before']} {change['after']}")}
    exact = [e for e in change.get("exact", []) if isinstance(e, (int, float))]
    if len(exact) == 2:  # the change itself ('rose by 178,526,208') is a claim about this field too
        delta = abs(exact[1] - exact[0])
        own.add(_canonical(str(delta)))
        exact = exact + [delta]
    for n in clause_nums:
        if n in own:
            continue
        if any(_rounds_from(n, unit, exact) for unit in UNIT):
            continue
        return False
    return True


CLAIM_NUMBER = r"\d+(?:\.\d+)?\s*(?:KB|MB|K|M|B|T)?\b"
CLAIM_ALIASES = {
    "number of layers": ("number of layers", "layers"),
    "attention heads": ("attention heads", "query heads"),
    "key-value heads": ("key-value heads", "kv heads"),
    "growing cache per token": ("growing cache per token", "cache per token"),
}


def _matches_exact_claim(token: str, expected: float) -> bool:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(KB|MB|K|M|B|T)?", token.strip())
    if not match:
        return False
    number, unit = match.groups()
    return _rounds_from(number, unit, [expected]) if unit else float(number) == expected


def _claim_role_problems(clause: str, change: dict) -> list[str]:
    """Bind an ordered transition and its delta to one scalar field, including the stated unit."""
    exact = change.get("exact", [])
    if len(exact) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in exact):
        return []
    before, after = exact
    problems = []
    transition = re.search(rf"(?:\bfrom\s+)?({CLAIM_NUMBER})\s*(?:\bto\b|→|->)\s*({CLAIM_NUMBER})", clause)
    if transition and not (_matches_exact_claim(transition[1], before) and _matches_exact_claim(transition[2], after)):
        problems.append(f"{change['what']}: before/after values are reversed or do not match the field")
    delta = re.search(rf"\bby\s+({CLAIM_NUMBER})", clause)
    if delta and not _matches_exact_claim(delta[1], abs(after - before)):
        problems.append(f"{change['what']}: stated delta does not match this field's change")
    if transition or delta:
        wording = clause
        for alias in CLAIM_ALIASES.get(change["what"], (change["what"],)):
            wording = re.sub(re.escape(alias), "", wording, flags=re.I)
        up, down = bool(UP.search(wording)), bool(DOWN.search(wording))
        if (up and not down and after < before) or (down and not up and after > before):
            problems.append(f"{change['what']}: direction contradicts this field's change")
    return problems


def field_claim_problems(text: str, data: dict) -> list[str]:
    """Check each claim as field + before + after together: a clause naming a changed field may only carry that
    field's own values ('attention heads 32 to 24' is rejected even if 24 is another field's value)."""
    problems = []
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    for sentence in _sentences(text):
        for clause in re.split(r",(?!\d)|;|\band\b|\bwhile\b|\bwhereas\b", sentence):
            low = clause.lower()
            matches = [
                (len(alias), index, c)
                for index, c in enumerate(data["changes"])
                for alias in CLAIM_ALIASES.get(c["what"], (c["what"].lower(),))
                if re.search(rf"\b{re.escape(alias)}\b", low)
            ]
            if not matches:
                continue
            change = max(matches, key=lambda item: (item[0], -item[1]))[2]
            nums = {_canonical(n) for n in _numbers(clause)}
            if nums and not _clause_numbers_ok(nums, change):
                problems.append(
                    f'{change["what"]}: "{clause.strip()}" does not match {change["before"]} to {change["after"]}'
                )
            problems += _claim_role_problems(clause, change)
    return problems


def check(text: str, data: dict, *, max_words: int) -> dict:
    problems = [f"number {n} does not match any value in the diff" for n in unverified_numbers(text, data)]
    problems += field_claim_problems(text, data)
    problems += [
        f'claim word "{w}" is outside the diff' for w in sorted({m.group(0).lower() for m in BANNED.finditer(text)})
    ]
    raw = sorted(set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b", text)) - set(_numbers(text)))
    values = json.dumps(data, ensure_ascii=False)
    raw = [t for t in raw if t not in values or f'"{t}"' in json.dumps(list(FIELD_NAMES))]
    raw += sorted(set(re.findall(r"\b\w+(?:\.\w+)*\.\*\.[\w.]+", text)))
    problems += [f'raw field or tensor name "{t}"; use plain words' for t in raw]
    for sentence in _sentences(text):
        inherited = None  # a direction word carries over to later clauses that have none of their own
        for clause in re.split(r",|;|\band\b|\bwhile\b", sentence):
            up, down = UP.search(clause), DOWN.search(clause)
            word = (up or down) if not (up and down) else None
            if word:
                inherited = ("up" if up else "down", word.group(0))
            found = {_canonical(n) for n in _numbers(clause)}
            for c in data["changes"]:
                nums = {_canonical(n) for n in _numbers(c["before"] + " " + c["after"])}
                if not c["direction"] or not nums or not nums <= found or not inherited:
                    continue
                if inherited[0] != c["direction"]:
                    problems.append(
                        f'"{inherited[1]}" contradicts {c["what"]} ({c["before"]} to {c["after"]}, {c["direction"]})'
                    )
    problems += [
        f'awkward or raw wording "{w}"; use plain English'
        for w in sorted({m.group(0).lower() for m in AWKWARD.finditer(text)})
    ]
    problems += [
        f'number word "{w}"; write numbers as digits'
        for w in sorted({m.group(0).lower() for m in NUMBER_WORDS.finditer(text)})
    ]
    words = len(text.split())
    if words > max_words:
        problems.append(f"{words} words; the limit is {max_words}")
    if "#" in text or re.search(r"^\s*[-*]\s", text, re.M) or "{" in text:
        problems.append("plain prose only: no headings, lists or braces")
    return dict(passed=not problems, problems=problems, numbers=_numbers(text), words=words)
