"""Parse the model card's summary table (the claims a lab makes about its own model).

Cards are written for people, so values are rounded ("2.8T", "160K"). Each parsed value keeps
its raw text and a tolerance implied by that rounding, so the verifier compares fairly.
"""

from __future__ import annotations

import html
import re

LABELS = {
    "total parameters": "params_total",
    "parameters": "params_total",
    "activated parameters": "params_active",
    "active parameters": "params_active",
    "number of layers": "num_layers",
    "layers": "num_layers",
    "num hidden layers": "num_layers",
    "hidden size": "hidden_size",
    "hidden dimension": "hidden_size",
    "num attention heads": "num_attention_heads",
    "num key value heads": "num_kv_heads",
    "intermediate size": "ffn_intermediate",
    "number of dense layers": "dense_layers",
    "attention-layer composition": "layer_composition",
    "architecture": "architecture",
    "attention hidden dimension": "hidden_size",
    "number of attention heads": "num_attention_heads",
    "dsa index head dimension": "dsa_index_head_dim",
    "dsa index number of heads": "dsa_index_num_heads",
    "dsa index top-k": "dsa_index_topk",
    "latent moe dimension": "moe_latent_dim",
    "moe hidden dimension (per expert)": "moe_expert_intermediate",
    "number of experts": "moe_num_experts",
    "selected experts per token": "moe_active_experts",
    "number of shared experts": "moe_shared_experts",
    "vocabulary size": "vocab_size",
    "vocabulary": "vocab_size",
    "context length": "context_length",
    "attention mechanism": "attention_mechanism",
    "activation function": "activation",
    "vision encoder": "vision_encoder",
    "parameters of vision encoder": "vision_params",
    "quantization": "quantization",
    "modality": "modality",
    "license": "license",
}
SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _clean(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " / ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip(" *`\t\n")


def parse_number(raw: str) -> tuple[float, float] | None:
    """'2.8T' -> (2.8e12, 0.05e12); '1048576' -> (1048576, 0); '160K' -> (160e3, ...) with K ambiguity handled by caller."""
    m = re.fullmatch(r"~?\s*([\d.,]+)\s*([KMBT])?\b.*", raw.strip(), re.I)
    if not m:
        return None
    num, suf = m.group(1).replace(",", ""), m.group(2)
    try:
        v = float(num)
    except ValueError:
        return None
    decimals = len(num.split(".")[1]) if "." in num else 0
    scale = SUFFIX.get((suf or "").upper(), 1)
    tol = (0.5 * 10 ** (-decimals)) * scale if suf else 0
    return v * scale, tol


def _pairs(readme):
    readme = re.sub(r"(?ms)^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", "", readme)
    pairs = []
    for match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", readme, re.S | re.I):
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", match[1], re.S | re.I)
        if len(cells) == 2:
            pairs.append((match.start(), cells[0], cells[1]))
    # Also accept two cells without an enclosing <tr> (legacy card fragments).
    for match in re.finditer(
        r"<td[^>]*>\s*<strong>(.*?)</strong>(.*?)</td>\s*<td[^>]*>(.*?)</td>", readme, re.S | re.I
    ):
        pairs.append((match.start(), match[1] + match[2], match[3]))
    offset = 0
    for line in readme.splitlines(keepends=True):
        row = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        if len(row) == 2:
            pairs.append((offset, row[0], row[1].replace(r"\|", "|")))
        bullet = re.match(r"\s*[-*+]\s+(.+?)\s*:\s*(.*?)\s*$", line)
        if bullet:
            pairs.append((offset, bullet[1], bullet[2]))
        offset += len(line)
    return [(label, raw) for _, label, raw in sorted(pairs)]


def parse_card(readme: str) -> dict:
    out: dict = {}
    for label, raw in _pairs(readme):
        label = _clean(label).lower().replace("_", " ")
        key = LABELS.get(label)
        if not key or key in out:
            continue
        raw = _clean(raw)
        entry = {"raw": raw}
        num = parse_number(raw)
        if num and key not in (
            "layer_composition",
            "architecture",
            "attention_mechanism",
            "activation",
            "vision_encoder",
            "quantization",
            "modality",
            "license",
        ):
            entry["value"], entry["tolerance"] = num
            entry["k_suffix"] = bool(re.search(r"\dK\b", raw, re.I))
        out[key] = entry
    # some cards put the layer mix under "Architecture" ("24 layers: 13 gated short convolutions, ...")
    if "layer_composition" not in out and re.search(r"\d+\s+\w", (out.get("architecture") or {}).get("raw", "")):
        out["layer_composition"] = dict(out["architecture"])
    if "layer_composition" in out:
        raw = re.sub(r"^\d+\s+layers:\s*", "", out["layer_composition"]["raw"])
        out["layer_composition"]["parts"] = {
            name.strip(): int(n)
            for n, name in re.findall(r"(\d+)\s+([A-Za-z][A-Za-z\- ]+?)(?=\s*(?:,|\+|\*|$| and))", raw)
        }
    return out
