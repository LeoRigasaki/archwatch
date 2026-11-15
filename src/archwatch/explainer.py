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

from .labels import FIELD_NAMES, KIND_LABELS

NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred|thousand|million|billion)\b",
    re.I,
)
BANNED = re.compile(r"\b(benchmark|faster|slower|better|worse|accuracy|score|trained|training)\b", re.I)
UP = re.compile(r"\b(increas\w*|gr[eo]w\w*|rose|rise\w*|ris(?:ing|es)|doubl\w*|more|larger|higher|expand\w*)\b", re.I)
DOWN = re.compile(r"\b(decreas\w*|shr[ai]nk\w*|fell|fall\w*|drop\w*|reduc\w*|halv\w*|fewer|less|smaller|lower)\b", re.I)


def _label(field: str) -> str:
    return FIELD_NAMES.get(field, field.replace("_", " ").capitalize())


def _show(value, model: str) -> str:
    if value is None:
        return f"not published for {model}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        return " + ".join(f"{n} {KIND_LABELS.get(k, k.replace('_', ' '))}" for k, n in value.items())
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


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
    changes = []
    for row in record["diff_vs_predecessor"]:
        before, after = row.get("before"), row.get("after")
        direction = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool):
            direction = "up" if after > before else "down" if after < before else None
        changes.append(
            dict(what=_label(row["field"]), before=_show(before, old), after=_show(after, new), direction=direction)
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
            f"{add_n} tensor groups added ({tensors.get('added_total') or 0:,} parameters)"
            if add_n
            else "no tensor groups added"
        )
        rem = f"{rem_n} removed ({tensors.get('removed_total') or 0:,} parameters)" if rem_n else "none removed"
        weights = f"{add}; {rem}."
    return dict(predecessor=old, successor=new, changes=changes, weights=weights)


def input_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _numbers(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return re.findall(r"\d+(?:\.\d+)?", text)


def _canonical(n: str) -> str:
    return str(int(float(n))) if float(n).is_integer() else str(float(n))


def allowed_numbers(data: dict) -> set[str]:
    return {_canonical(n) for n in _numbers(json.dumps(data, ensure_ascii=False))}


def word_limit(data: dict, settings: dict) -> int:
    return min(
        settings["max_words_cap"], settings["max_words_base"] + settings["max_words_per_change"] * len(data["changes"])
    )


def format_numbers(text: str) -> str:
    """Formatting only: add thousands separators to bare integers of five or more digits."""
    return re.sub(r"(?<![\d,.])\d{5,}(?![\d,]|\.\d)", lambda m: f"{int(m.group(0)):,}", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def check(text: str, data: dict, *, max_words: int) -> dict:
    allowed = allowed_numbers(data)
    problems = [
        f"number {n} is not in the diff" for n in sorted({n for n in _numbers(text) if _canonical(n) not in allowed})
    ]
    problems += [
        f'claim word "{w}" is outside the diff' for w in sorted({m.group(0).lower() for m in BANNED.finditer(text)})
    ]
    raw = sorted(set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b", text)) - set(_numbers(text)))
    values = json.dumps(data, ensure_ascii=False)
    raw = [t for t in raw if t not in values or f'"{t}"' in json.dumps(list(FIELD_NAMES))]
    raw += sorted(set(re.findall(r"\b\w+(?:\.\w+)*\.\*\.[\w.]+", text)))
    problems += [f'raw field or tensor name "{t}"; use plain words' for t in raw]
    for sentence in _sentences(text):
        found = {_canonical(n) for n in _numbers(sentence)}
        for c in data["changes"]:
            if not c["direction"]:
                continue
            nums = {_canonical(n) for n in _numbers(c["before"] + " " + c["after"])}
            if not nums <= found:
                continue
            wrong = DOWN if c["direction"] == "up" else UP
            hit = wrong.search(sentence)
            if hit:
                problems.append(
                    f'"{hit.group(0)}" contradicts {c["what"]} ({c["before"]} to {c["after"]}, {c["direction"]}) '
                    "in the same sentence"
                )
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
