"""Build the static site from published records only.

    uv run --frozen python build_site.py

Pages: site/index.html (explore), site/models/<stem>.html (one per published model), site/compare/<stem>.html
(a model next to its predecessor, only when both are published) and site/compare/index.html. Every number comes
from published, verifier-approved record fields or from the cache calculator run on them; every sentence is
written by code from those fields, except an explainer note that passed every check for the current diff. Held
records appear by name and a plain reason only; fine-tunes are out of scope and not listed. Plain HTML/CSS with a
small vanilla script for the filters, the memory planner and the theme switch.
"""

import datetime as dt
import hashlib
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

from archwatch import concept_diagrams, concept_examples, diagram, generated, pipeline
from archwatch import concepts as concept_data
from archwatch.artifacts import write_if_changed
from archwatch.explainer import diff_input, input_hash
from archwatch.labels import KIND_LABELS
from archwatch.palette import colors
from archwatch.publication import PublicationError, plain_hold, render_cache, render_values
from archwatch.sources import source_url
from archwatch.watcher import derivative_relation

ROOT = Path(__file__).parent
CODE = Path(__file__).parent  # site settings live with the code; ROOT is where records are read and pages written
LABEL = KIND_LABELS
SHORT = {
    "kda_linear": "KDA linear",
    "gated_dsa_mla": "Gated DSA",
    "full_attention_mla": "MLA (full)",
    "short_conv": "Short conv",
    "sliding_attention": "Sliding window",
    "full_attention": "Global attention",
    "mamba2": "Mamba-2",
    "mlp": "MLP",
}
COLOR = colors()  # one colour per layer type, shared with the code-drawn diagrams
LINEAR, SPARSE, NON_ATTENTION = {"kda_linear"}, {"gated_dsa_mla"}, {"kda_linear", "short_conv", "mamba2"}
MLA_KINDS = {"gated_dsa_mla", "full_attention_mla"}
LAYER_FILTERS = (
    ("linear", "Linear attention", {"kda_linear"}),
    ("sparse", "Sparse attention", {"gated_dsa_mla"}),
    ("sliding", "Sliding window", {"sliding_attention"}),
    ("conv", "Short convolution", {"short_conv"}),
    ("mamba", "Mamba-2", {"mamba2"}),
)
SIZE_FILTERS = (("s", "Under 3B", 0, 3e9), ("m", "3B to 30B", 3e9, 30e9), ("l", "Over 30B", 30e9, float("inf")))
# Plain labels for every published field. The config key is shown under the label, never instead of it.
PLAIN = {
    "params_total": "Total parameters",
    "params_active": "Active parameters per token",
    "params_active_non_embedding": "Active parameters, excluding embeddings",
    "params_mtp": "Multi-token-prediction module (not in the total)",
    "num_layers": "Layers",
    "layer_type_counts": "Layer mix",
    "context_length": "Context length",
    "hidden_size": "Hidden size (width)",
    "attention_type": "Attention type",
    "num_attention_heads": "Attention heads",
    "num_kv_heads": "Key-value heads",
    "head_dim": "Head size",
    "v_head_dim": "Value head size",
    "vocab_size": "Vocabulary",
    "moe_num_experts": "Routed experts",
    "moe_active_experts": "Experts used per token",
    "moe_shared_experts": "Shared experts",
    "moe_expert_intermediate": "Expert hidden size",
    "moe_shared_expert_intermediate": "Shared-expert hidden size",
    "moe_latent_dim": "Expert latent width",
    "moe_router": "Expert router",
    "dense_layers": "Layers with a dense FFN",
    "dense_ffn_intermediate": "Dense FFN hidden size",
    "ffn_intermediate": "FFN hidden size",
    "ffn_gated": "Gated FFN",
    "activation": "Activation function",
    "activation_card_name": "Activation (name on the card)",
    "attention_output_gate": "Gate on the attention output",
    "attn_res_block_size": "Residual-mixing block size",
    "block_layout": "Block layout",
    "canvas_length": "Diffusion canvas length",
    "decoder_type": "Design",
    "dsa_index_head_dim": "Sparse-attention indexer head size",
    "dsa_index_key_width": "Sparse-attention indexer key width",
    "dsa_index_num_heads": "Sparse-attention indexer heads",
    "dsa_index_topk": "Tokens each sparse layer attends to",
    "generation_type": "Generation method",
    "global_head_dim": "Global-layer head size",
    "hyper_connection_multiplier": "Residual streams",
    "input_layers": "Input layers",
    "output_layers": "Output layers",
    "recurrent_layers": "Recurrent layers",
    "recurrent_loops": "Recurrent passes",
    "recurrent_kv_cache_mode": "Cache in recurrent passes",
    "linear_attn_conv_kernel": "Linear-attention convolution width",
    "linear_attn_head_dim": "Linear-attention head size",
    "linear_attn_num_heads": "Linear-attention heads",
    "linear_conv_kernel": "Linear-attention convolution width",
    "linear_key_head_dim": "Linear-attention key head size",
    "linear_num_key_heads": "Linear-attention key heads",
    "linear_num_value_heads": "Linear-attention value heads",
    "linear_value_head_dim": "Linear-attention value head size",
    "mamba_activation": "Mamba activation",
    "mamba_conv_channels": "Mamba convolution channels",
    "mamba_conv_kernel": "Mamba convolution width",
    "mamba_groups": "Mamba groups",
    "mamba_head_dim": "Mamba head size",
    "mamba_num_heads": "Mamba heads",
    "mamba_state_size": "Mamba state size",
    "mla_cached_width_per_token": "Values cached per token in MLA layers",
    "mla_kv_lora_rank": "MLA key-value compression width",
    "mla_q_lora_rank": "MLA query compression width",
    "mla_qk_rope_head_dim": "MLA position-encoded key size",
    "mla_use_nope": "Sparse layers without position encoding",
    "model_type": "Architecture name in the config",
    "num_global_kv_heads": "Global-layer key-value heads",
    "num_mtp_layers": "Multi-token-prediction layers",
    "qk_norm": "Normalised queries and keys",
    "released_weights_format": "Released weight format",
    "rope": "Position encoding",
    "sandwich_norm": "Norm before and after each block",
    "shared_kv_layers": "Layers sharing a key-value cache",
    "short_conv_kernel": "Convolution width",
    "sliding_window": "Sliding window",
    "tied_embeddings": "Input and output embeddings shared",
    "vision_encoder_layers": "Vision encoder layers",
    "vision_hidden_size": "Vision encoder width",
    "vision_params": "Vision encoder parameters",
    "ngram_layer_ids": "Layers with n-gram memory",
    "release_date": "Release date (model card)",
}
KEY_FACTS = (
    "params_total",
    "params_active",
    "params_active_non_embedding",
    "num_layers",
    "layer_type_counts",
    "context_length",
    "hidden_size",
    "attention",
    "experts",
)
NOT_SHOWN = {"block_shapes", "parameter_components", "weight_header_summary", "layer_types", "layer_pattern_per_layer"}
NOT_SHOWN |= {"layer_index_base_in_config"}
NOT_A_CHANGE = {"model_type"}  # a renamed config class is not an architecture change
SOURCE_WORDS = {
    "config": "config.json",
    "weights": "weight file headers",
    "code": "modeling code",
    "card": "model card",
    "derived": "computed from the facts above",
}
STATUS_WORDS = {"verified": "verified", "single source": "one source", "differs from card": "differs from card"}
PRECISION = (("16", "16-bit (full)", 2.0), ("8", "8-bit", 1.0), ("4", "4-bit", 0.5))
DEVICES = (("8 GB laptop", 8e9), ("16 GB laptop", 16e9), ("24 GB GPU", 24e9))


def e(x):
    return html.escape(str(x))


def settings():
    return yaml.safe_load((CODE / "config/site.yaml").read_text())


def concepts():
    return concept_data.registry()


def concept_keys(pub):
    return [match["key"] for match in concept_data.from_values(pub)]


def concept_links(pub, up="", limit=None, rec=None, stem=None):
    guide = concepts()
    matches = concept_data.for_record(rec) if rec is not None else concept_data.from_values(pub)
    links = []
    for match in matches[:limit]:
        key = match["key"]
        status = STATUS_WORDS.get(match.get("status"), "")
        links.append(
            f'<a href="{up}concepts/{key}.html" title="{e(match["scope"] + (" Evidence: " + status if status else ""))}">{e(guide[key]["short"])}</a>'
        )
    if limit is not None and len(matches) > limit and rec:
        stem = stem or rec["id"].replace("/", "__")
        links.append(f'<a href="{up}models/{e(stem)}.html#concept-evidence">+{len(matches) - limit} with evidence</a>')
    return '<div class="concept-chips" aria-label="Related concepts">' + "".join(links) + "</div>"


def concept_evidence(rec, stem):
    guide = concepts()
    entries = []
    for match in concept_data.for_record(rec):
        key = match["key"]
        fields = []
        for field in match["fields"]:
            value = json.dumps(field["value"], ensure_ascii=False, sort_keys=True)
            sources = ", ".join(
                f'<a href="{e(source["url"])}">{e(source["claim"])}</a>' if source["url"] else e(source["claim"])
                for source in field["sources"]
            )
            fields.append(
                f'<tr><th scope="row">{e(PLAIN.get(field["field"], field["field"].replace("_", " ")))}</th><td><code>{e(value)}</code></td><td>{e(STATUS_WORDS.get(field["status"], field["status"]))}<br>{sources}</td></tr>'
            )
        entries.append(
            f"""<details class="concept-proof" id="concept-{key}"><summary><a href="../concepts/{key}.html">{e(guide[key]["short"])}</a><span class="st st-{e(match["status"].replace(" ", "-"))}">{e(STATUS_WORDS.get(match["status"], match["status"]))}</span><span class="proof-hint">Inspect evidence ↓</span></summary><p>{e(match["scope"])}</p><div class="tablebox"><table><thead><tr><th scope="col">Supporting field</th><th scope="col">Published value</th><th scope="col">Status and pinned sources</th></tr></thead><tbody>{"".join(fields)}</tbody></table></div></details>"""
        )
    return f"""<section id="concept-evidence"><div class="sechead"><h2>Related concepts, with evidence</h2><a class="btn" download href="../data/concepts/{e(stem)}.json">Download concept evidence ↓</a></div><p class="sub">Each association follows a deterministic rule using published fields. The badge reflects the weakest supporting field. A calculation does not count as another independent source. Unlisted concepts are unknown or uncovered, not confirmed absent.</p>{"".join(entries)}<p class="hint">This versioned evidence accompanies the diagram. It adds no claims or wiring to the SVG above.</p></section>"""


# ---------------------------------------------------------------- formatting


def sig(x, digits=3):
    """x to `digits` significant figures, trailing zeros dropped (27.0 -> 27, 1.734 -> 1.73)."""
    if x == 0:
        return "0"
    text = f"{x:.{digits}g}"
    if "e" in text:
        text = f"{x:.0f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def human(n):
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= d:
            if unit in ("T", "B"):
                return f"{n / d:.2f}{unit}"
            if round(n / d) >= 1000:
                return f"{n / 1e9:.2f}B" if unit == "M" else f"{n / 1e6:.2f}M"
            return f"{n / d:.0f}{unit}"
    return f"{n:,}"


def tokens(n):
    n = int(n)
    for div, unit in ((2**20, "M"), (10**6, "M"), (1024, "K"), (1000, "K")):
        if n >= div and n % div == 0:
            return f"{n // div}{unit}"
    return f"{n:,}"


def size(b):
    for unit, d in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if b >= d:
            return f"{b / d:.2f} {unit}" if unit == "GiB" else f"{sig(b / d)} {unit}"
    return f"{b:,} B"


def date(iso, year=True):
    d = dt.date.fromisoformat(iso[:10])
    return f"{d.day} {d:%b %Y}" if year else f"{d.day} {d:%b}"


def show(field, v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if field == "layer_type_counts":
        return ", ".join(f"{n} {SHORT.get(k, k)}" for k, n in v.items())
    if isinstance(v, dict):
        return "; ".join(f"{k.replace('_', ' ')}: {show(k, n)}" for k, n in v.items())
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int) and (field.startswith("params") or field == "vision_params"):
        return f"{v:,}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def when(rec, pub):
    """(label, ISO date) for display. A release date only when the model's own card states one; otherwise the
    Hub creation time, labelled as such. Creation time is never called a release."""
    if pub.get("release_date"):
        return "released", pub["release_date"]
    return "added to Hub", rec["hub_created_at"][:10]


def card_status(c):
    """Shown status of a check. Our active count is published by one fixed rule; when the card's own figure is
    outside its rounding of ours, the row says so and shows the card's figure. It is never shown as verified."""
    ref = c.get("card_reference")
    if ref and not ref["agrees"]:
        return "differs from card", f'card says "{ref["raw"]}"'
    return c["status"], None


# ---------------------------------------------------------------- record views


def kinds_in_order(pub):
    return list(dict.fromkeys(pub["layer_types"]))


def design(pub):
    kinds = set(pub["layer_types"]) - {"mlp"}
    if "moe_num_experts" in pub:
        return "MoE"
    if kinds & NON_ATTENTION and kinds - NON_ATTENTION:
        return "Hybrid"
    return "Dense"


def hybrid(pub):
    kinds = set(pub["layer_types"]) - {"mlp"}
    return bool(kinds & NON_ATTENTION and kinds - NON_ATTENTION)


def mix_text(pub, sep=" · "):
    return sep.join(f"{SHORT.get(k, k)} {pub['layer_type_counts'][k]}" for k in kinds_in_order(pub))


def cells(pub, label):
    """Every layer as one cell, layer 1 on the left."""
    spans = "".join(f'<i style="background:{COLOR.get(k, "#999")}"></i>' for k in pub["layer_types"])
    return f'<div class="cells" role="img" aria-label="{e(label)}">{spans}</div>'


def attention_summary(pub):
    kind = f"{pub['attention_type']} · " if pub.get("attention_type") else ""
    if set(pub["layer_types"]) & MLA_KINDS:
        latent = f"; KV latent width {pub['mla_kv_lora_rank']}" if "mla_kv_lora_rank" in pub else ""
        return f"{kind}{pub['num_attention_heads']} query heads{latent}"
    kv = f" ({pub['num_kv_heads']} key-value)" if "num_kv_heads" in pub else ""
    return f"{kind}{pub['num_attention_heads']} heads{kv}"


def describe(pub):
    """One technical sentence written by code from published fields."""
    counts, n = pub["layer_type_counts"], pub["num_layers"]
    extra = {
        "short_conv": f" (kernel {pub['short_conv_kernel']})" if pub.get("short_conv_kernel") else "",
        "sliding_attention": f" (window {pub['sliding_window']:,})" if pub.get("sliding_window") else "",
        "gated_dsa_mla": f" (top-k {pub['dsa_index_topk']:,})" if pub.get("dsa_index_topk") else "",
        "full_attention": f" ({pub['attention_type']})" if pub.get("attention_type") else "",
    }
    if len(counts) == 1:
        kind = next(iter(counts))
        head = (
            f" ({pub['num_attention_heads']} heads, {pub['num_kv_heads']} KV heads)"
            if pub.get("num_attention_heads") and pub.get("num_kv_heads")
            else ""
        )
        if kind in MLA_KINDS:
            head = f" ({pub['num_attention_heads']} query heads, KV latent width {pub['mla_kv_lora_rank']})"
        what = pub.get("attention_type") if kind == "full_attention" and pub.get("attention_type") else LABEL[kind]
        text = f"{'MoE' if 'moe_num_experts' in pub else 'Dense'} decoder, {what} in all {n} layers{head}"
    else:
        text = ", ".join(f"{counts[k]} {LABEL.get(k, k)} layers{extra.get(k, '')}" for k in kinds_in_order(pub))
    if "moe_num_experts" in pub:
        shared = f" + {pub['moe_shared_experts']} shared" if pub.get("moe_shared_experts") else ""
        text += f"; {pub['moe_num_experts']} routed experts, {pub['moe_active_experts']} per token{shared}"
    return text + "."


def explainer_note(rec):
    """A saved explainer note, only if it passed every code check and was written for the current diff."""
    path = ROOT / "results/explainer" / (rec["id"].replace("/", "__") + ".json")
    if not path.exists():
        return None
    note = json.loads(path.read_text())
    pred = (rec.get("predecessor") or {}).get("id")
    pred_path = ROOT / "data/models" / (pred.replace("/", "__") + ".json") if pred else None
    if not pred_path or not pred_path.exists():
        return None
    # the diff is recomputed from published values, never taken from the record's cached copy
    fresh = dict(rec, diff_vs_predecessor=pipeline.diff(rec, json.loads(pred_path.read_text())))
    try:
        current = input_hash(diff_input(fresh))
    except (ValueError, KeyError):
        return None
    if not note.get("passed") or note.get("input_sha256") != current:
        return None
    return note.get("text")


def significant(field, before, after):
    """Changes worth stating: the architecture and the cache always; numbers only when they move by at least 1%
    (and parameter counts by at least 0.1B). Weight-group bookkeeping is never listed."""
    if field in ("layer_type_counts", "kv_cache_per_token_bytes"):
        return True
    if before is None or after is None:
        return True
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool):
        if field.startswith("params") and abs(after - before) < 1e8:
            return False  # under 0.1B parameters
        return before == 0 or abs(after - before) / abs(before) >= 0.01
    return True


IMPLIED = (
    # (field prefix, layer kinds whose absence makes the value implied, what the architecture implies)
    ("dsa_index", {"gated_dsa_mla"}, "all earlier tokens"),
    ("sliding_window", {"sliding_attention"}, "no sliding window"),
    ("short_conv_kernel", {"short_conv"}, "no convolution layers"),
    ("mla_", {"gated_dsa_mla", "full_attention_mla"}, "no MLA layers"),
    ("linear_", {"kda_linear"}, "no linear-attention layers"),
    ("mamba_", {"mamba2"}, "no Mamba layers"),
)


def implied(field, pub):
    """What the architecture implies for a field the files do not state; None when nothing is implied."""
    if not pub:
        return None
    if field == "params_active" and "moe_num_experts" not in pub:
        return "all (dense)"
    if field.startswith("moe_") and "moe_num_experts" not in pub:
        return "none (dense)"
    kinds = set(pub.get("layer_types", []))
    for prefix, members, text in IMPLIED:
        if field.startswith(prefix) and not kinds & members:
            return text if prefix != "dsa_index" or field == "dsa_index_topk" else "no sparse-attention indexer"
    return None


def change_value(field, v, pub=None):
    """Parameter counts and byte sizes to 3 significant figures; configuration integers stay exact. A missing
    value the architecture implies is derived ('all earlier tokens'); 'not stated' only when nothing is implied."""
    if v is None:
        return implied(field, pub) or "not stated"
    if field in ("kv_cache_per_token_bytes", "kv"):
        return size(v)
    if field == "layer_type_counts":
        return " + ".join(f"{n} {SHORT.get(k, k)}" for k, n in v.items())
    if field.startswith("params") and isinstance(v, (int, float)):
        for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
            if v >= d:
                return f"{sig(v / d)}{unit}"
    return show(field, v)


def changes(rec, before):
    """(label, before, after) for each significant change since the predecessor, rounded to 3 figures."""
    out = []
    for row in pipeline.diff(rec, before):
        field = row["field"]
        if field in NOT_SHOWN | NOT_A_CHANGE or not significant(field, row["before"], row["after"]):
            continue
        label = "Cache per token" if field == "kv_cache_per_token_bytes" else PLAIN.get(field, field)
        out.append(
            (
                label,
                change_value(field, row["before"], before["published"]),
                change_value(field, row["after"], rec["published"]),
            )
        )
    return out


def change_chip(field, before, after, old_pub=None, new_pub=None):
    if before is None and after is not None and implied(field, old_pub):
        return ("new limit", "new") if field == "dsa_index_topk" else ("changed", "up")
    if after is None and before is not None and implied(field, new_pub):
        return "removed", "warn"
    if before == after:
        return "same", "same"
    if before is None:
        return "now stated", "new"
    if after is None:
        return "not stated", "warn"
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool):
        if after == 2 * before:
            return "doubled", "up"
        if 2 * after == before:
            return "halved", "warn"
        pct = 100 * (after - before) / before
        if abs(pct) < 0.1:
            return ("<+0.1%" if pct > 0 else "<-0.1%"), "same"
        return f"{pct:+.{0 if abs(pct) >= 1 else 1}f}%", "up" if after > before else "warn"
    return "changed", "up"


# ---------------------------------------------------------------- page frame


def frame(title, body, *, depth=0, active="", description=""):
    up = "../" * depth
    repo = settings()["repo_url"]
    nav = "".join(
        f'<a href="{href}"{" aria-current=page" if key == active else ""}>{text}</a>'
        for key, href, text in (
            ("explore", f"{up}index.html", "Explore"),
            ("compare", f"{up}compare/index.html", "Compare"),
            ("concepts", f"{up}concepts/index.html", "Concept guide"),
            ("review", f"{up}review/index.html", "Evidence"),
            ("github", repo, "GitHub ↗"),
        )
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><meta name="description" content="{e(description or title)}">
<script>try{{var t=localStorage.getItem('aw-theme');if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<link rel="icon" href="{up}favicon.svg" type="image/svg+xml">
<link rel="alternate" type="application/rss+xml" title="ArchWatch analyses" href="{up}feed.xml">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap">
<link rel="stylesheet" href="{up}style.css?v={hashlib.sha256(CSS.encode()).hexdigest()[:12]}"></head>
<body><a class="skip" href="#main">Skip to content</a>
<header class="top"><div class="wrap topbar"><a class="brand" href="{up}index.html"><span class="brandmark" aria-hidden="true"><i></i><i></i><i></i></span>ArchWatch<span class="brand-sub">Architecture field notes</span></a>
<button class="menubtn btn" type="button" aria-expanded="false" aria-controls="main-nav">Menu</button>
<nav id="main-nav" aria-label="Main">{nav}</nav>
<button type="button" class="themebtn" aria-pressed="false" aria-label="Dark theme"><span class="knob" aria-hidden="true"></span>Dark</button></div></header>
<main id="main" class="wrap">{body}</main>
<footer id="method" class="foot"><div class="wrap footergrid"><div><a class="brand" href="{up}index.html">ArchWatch</a><p class="foot-tag">Architecture you can trace.</p><a href="{up}feed.xml">Analysis feed ↗</a> · <a href="{e(repo)}">Source code ↗</a></div><div><b>How to read this site</b><p>Model values come from primary files at a pinned commit: config, weight headers, modeling code and model cards. Verified means two sources agree; one-source and computed values are labelled. A model with unresolved required facts stays in review.</p><p>Memory is a mathematical estimate, not a device benchmark. Approved generated diagrams and code-drawn fallbacks are identified on each page. <a href="{up}review/index.html">See coverage and limits →</a></p><span class="byline">{e(last_run())}</span></div></div></footer>
<script src="{up}site.js?v={hashlib.sha256(JS.encode()).hexdigest()[:12]}" defer></script></body></html>"""


_LAST = {}


def last_run():
    if ROOT not in _LAST:
        stamps = [json.loads(p.read_text()).get("analyzed_at", "") for p in (ROOT / "data/models").glob("*.json")]
        latest = max((s for s in stamps if s), default="")
        _LAST[ROOT] = f"Last analysis {date(latest)}" if latest else ""
    return _LAST[ROOT]


# ---------------------------------------------------------------- inventory


def inventory():
    """Published records, and held repos with plain reasons (no numbers). Fine-tunes are out of scope: not listed."""
    seen_path = ROOT / "data/seen.json"
    seen = json.loads(seen_path.read_text()).get("models", {}) if seen_path.exists() else {}
    published, held, records = {}, {}, {}
    for path in sorted((ROOT / "data/models").glob("*.json")):
        rec = json.loads(path.read_text())
        records[rec["id"]] = (path.stem, rec)
    for repo, (stem, rec) in sorted(records.items()):
        if derivative_relation(rec.get("hub_metadata") or {}):
            continue
        state = seen.get(repo, {})
        try:
            if (
                state.get("status") in ("review", "error", "rejected")
                and state.get("last_observed_commit") == rec["source_commit"]
            ):
                raise PublicationError(state.get("reason", "Held by watcher"))
            pub = render_values(rec)
            render_cache(pub)
            published[repo] = (stem, rec, pub)
        except (PublicationError, KeyError, TypeError) as error:
            held[repo] = plain_reason(str(error), rec.get("published", {}))
    for repo, state in sorted(seen.items()):
        if repo in records or "base_model relation" in state.get("reason", ""):
            continue
        held[repo] = plain_reason(f"{state.get('status', 'unprocessed')}: {state.get('reason', '')}", {})
    return published, held, seen


def plain_reason(reason, values):
    """A short, number-free reason for the public held list."""
    configured = plain_hold(values)
    if configured:
        return configured
    text = reason.lower()
    if "403" in text or "401" in text or "gated" in text:
        return "Gated repository: waiting for access to the weights."
    if "config.json" in text or "format" in text:
        return "Released in a file format ArchWatch cannot read yet."
    if "layer pattern" in text:
        return "Layer pattern not yet confirmed by two sources."
    if "required published inputs" in text:
        return "A fact the diagram and cache need is still in review."
    if "pending" in text or "unprocessed" in text:
        return "Found; not analysed yet."
    return "Held for review."


def published_predecessor(rec, published):
    pred = (rec.get("predecessor") or {}).get("id")
    return published.get(pred) if pred else None


def capped(pub, cache):
    """(window, bytes when the window is full) when every growing layer is capped by a sliding window."""
    sliding = [r for r in cache["breakdown"] if r["layer_type"] == "sliding_attention"]
    if cache["grows_per_token_bytes"] or not sliding or not pub.get("sliding_window"):
        return None
    return int(pub["sliding_window"]), sum(r["constant_bytes"] for r in sliding)


def cache_value(pub, cache=None):
    """(value, label) for cache per token; a capped cache shows its full-window size, never '0 B'."""
    cache = cache or render_cache(pub)
    cap = capped(pub, cache)
    if cap:
        return size(cap[1]), f"cache, capped at {cap[0]:,} tokens"
    return size(cache["grows_per_token_bytes"]), "cache / token"


def cache_at(pub, cache, n):
    """Growing cache at n tokens; capped layers count up to their window."""
    cap = capped(pub, cache)
    if cap:
        return cap[1] * min(n, cap[0]) // cap[0]
    return cache["grows_per_token_bytes"] * n


# ---------------------------------------------------------------- explore (home)


def model_card(stem, rec, pub, published, pubs):
    org, name = rec["id"].split("/")
    kinds = set(pub["layer_types"])
    flags = [key for key, _, members in LAYER_FILTERS if kinds & members]
    data = dict(
        publisher=org,
        design=design(pub).lower(),
        hybrid="1" if hybrid(pub) else "0",
        layers=" ".join(flags),
        params=int(pub["params_total"]),
        date=rec["hub_created_at"][:10],
        name=name.lower(),
        search=" ".join(
            [
                rec["id"],
                pubs.get(org, org),
                design(pub),
                mix_text(pub),
                describe(pub),
                *(concepts()[key]["title"] for key in concept_keys(pub)),
            ]
        ).lower(),
    )
    attrs = " ".join(f'data-{k}="{e(v)}"' for k, v in data.items())
    return f"""<li class="card" {attrs}><article aria-labelledby="h-{e(stem)}">
<div class="cardhead"><span>{e(pubs.get(org, org))}</span><span>{e(when(rec, pub)[0])} <time datetime="{e(when(rec, pub)[1])}">{e(date(when(rec, pub)[1]))}</time></span></div>
<h3 id="h-{e(stem)}"><a href="models/{e(stem)}.html">{e(name)}</a> <span class="tag">{e(design(pub))}</span></h3>
<p class="line">{e(describe(pub))}</p>
{concept_links(pub, limit=4, rec=rec, stem=stem)}
{cells(pub, "Layer order, layer 1 on the left: " + mix_text(pub))}<p class="mixcap">{e(mix_text(pub))}</p>
<dl class="mini"><div><dt>parameters</dt><dd>{e(human(pub["params_total"]))}</dd></div>
<div><dt>layers</dt><dd>{pub["num_layers"]}</dd></div><div><dt>context</dt><dd>{e(tokens(pub["context_length"]))}</dd></div>
<div><dt>{e(cache_value(pub)[1])}</dt><dd>{e(cache_value(pub)[0])}</dd></div></dl>
<div class="cardbtns"><a class="btn primary" href="models/{e(stem)}.html">Inspect model ↗</a><a class="btn" href="models/{e(stem)}.html#fit">Will it fit?</a><button type="button" class="btn compare-pick" data-stem="{e(stem)}" data-name="{e(name)}" aria-pressed="false" aria-label="Add {e(name)} to comparison">+ Compare</button></div>
</article></li>"""


def table_row(stem, rec, pub, pubs):
    org, name = rec["id"].split("/")
    return (
        f'<tr><th scope=row><a href="models/{e(stem)}.html">{e(name)}</a><span class="detail">{e(pubs.get(org, org))} · '
        f"{e(when(rec, pub)[0])} {e(date(when(rec, pub)[1]))}</span></th><td>{e(design(pub))}</td><td class=mono>{e(human(pub['params_total']))}</td>"
        f"<td class=mono>{pub['num_layers']}</td><td class=mono>{e(tokens(pub['context_length']))}</td>"
        f"<td class=mono>{e(cache_value(pub)[0])}{'' if cache_value(pub)[1] == 'cache / token' else '<span class=detail>' + e(cache_value(pub)[1].replace('cache, ', '')) + '</span>'}</td><td>{cells(pub, mix_text(pub))}</td></tr>"
    )


def home(published, held, seen):
    pubs = settings()["publishers"]
    order = sorted(published.values(), key=lambda x: (x[1]["hub_created_at"] or "", x[1]["id"]), reverse=True)
    cards = "".join(model_card(s, r, p, published, pubs) for s, r, p in order)
    rows = "".join(table_row(s, r, p, pubs) for s, r, p in order)
    newest = "".join(
        f'<li><span class="mono">{e(date(r["hub_created_at"], year=False))}</span> <a href="models/{e(s)}.html">'
        f"{e(r['id'].split('/')[-1])}</a></li>"
        for s, r, _ in order[:3]
    )
    pubs_list = [p for _, _, p in published.values()]

    def box(group, value, label, count, checked=False):
        return (
            f'<li><label><input type="checkbox" name="{group}" value="{e(value)}"{" checked" if checked else ""}>'
            f'{e(label)}</label><span class="count">{count}</span></li>'
        )

    designs = "".join(
        box("design", key, label, sum(1 for p in pubs_list if test(p)), True)
        for key, label, test in (
            ("dense", "Dense", lambda p: design(p) != "MoE"),
            ("moe", "Mixture of experts", lambda p: design(p) == "MoE"),
            ("hybrid", "Hybrid (not only attention)", hybrid),
        )
    )
    layers = "".join(
        box("layers", key, label, sum(1 for p in pubs_list if set(p["layer_types"]) & members))
        for key, label, members in LAYER_FILTERS
        if any(set(p["layer_types"]) & members for p in pubs_list)
    )
    sizes = "".join(
        box("size", key, label, sum(1 for p in pubs_list if lo <= p["params_total"] < hi))
        for key, label, lo, hi in SIZE_FILTERS
    )
    publisher_options = "".join(
        f'<option value="{e(org)}">{e(pubs.get(org, org))}</option>'
        for org in sorted({repo.split("/")[0] for repo in published}, key=lambda x: pubs.get(x, x).lower())
    )
    feature = ""
    if order:
        stem, record, values = order[0]
        feature = f"""<a class="featured" href="models/{e(stem)}.html"><span class="eyebrow">Latest in the collection <span>↗</span></span>
<h2>{e(record["id"].split("/")[-1])}</h2><p>{e(describe(values))}</p>{cells(values, "Published layer order: " + mix_text(values))}
<div class="feature-bottom"><span>{e(human(values["params_total"]))} parameters</span><span>{values["num_layers"]} layers</span><span>{e(tokens(values["context_length"]))} context</span></div></a>"""
    held_rows = "".join(f"<li><b>{e(repo.split('/')[-1])}</b><span>{e(r)}</span></li>" for repo, r in held.items())
    body = f"""<section class="hero homehero"><div><p class="eyebrow"><span class="status-dot"></span> Open model architecture observatory</p><h1>Understand the model.<br><em>Inspect the evidence.</em></h1>
<p class="lead">Explore how open-weight models are built, compare their moving parts, and plan memory from facts you can trace to the source.</p>
<div class="hero-links"><a class="btn primary" href="#collection">Explore the collection ↓</a><a href="concepts/index.html">Start with a concept →</a></div>
<dl class="collection-stats"><div><dt>explorable models</dt><dd>{len(published):02d}</dd></div><div><dt>in review</dt><dd>{len(held):02d}</dd></div><div><dt>publishers tracked</dt><dd>{len({repo.split("/")[0] for repo in set(published) | set(held)}):02d}</dd></div></dl></div>{feature}</section>
<div id="collection" class="collectionhead"><div><p class="eyebrow">The collection</p><h2>Find an architecture worth understanding.</h2></div><p class="hint">Published facts only. No weight downloads required.</p></div>
<div class="searchrow"><label class="sr" for="q">Search models</label><span aria-hidden="true">⌕</span>
<input id="q" type="search" placeholder="Search models, publishers, or a layer type…" autocomplete="off"><span class="search-hint">Search the published collection</span></div>
<section class="newbar" aria-label="Newest on the Hub"><b>Newest on the Hub</b><ul>{newest}</ul></section>
<div class="explore">
<aside class="filters" aria-label="Filters"><details id="filter-panel" open><summary>Filter collection <span aria-hidden="true">＋</span></summary><form id="filters">
<fieldset><legend><label for="publisher">Publisher</label></legend><select id="publisher" name="publisher"><option value="">All publishers</option>{publisher_options}</select></fieldset>
<fieldset><legend>Design</legend><ul>{designs}</ul></fieldset>
<fieldset><legend>Layer types inside</legend><ul>{layers}</ul></fieldset>
<fieldset><legend>Size</legend><ul>{sizes}</ul></fieldset>
<fieldset><legend>Added to Hub</legend><ul>
<li><label><input type="radio" name="released" value="any" checked>Any time</label></li>
<li><label><input type="radio" name="released" value="6m">Last 6 months</label></li>
<li><label><input type="radio" name="released" value="year">This year</label></li></ul></fieldset>
<button type="reset" class="linkbtn">Reset filters</button></form></details></aside>
<section id="models" aria-labelledby="count">
<div class="listhead"><h2 id="count"><span class="shown">{len(published)}</span> models</h2>
<div class="tools"><label for="sort">Sort</label><select id="sort"><option value="new">Newest first</option>
<option value="big">Largest first</option><option value="small">Smallest first</option><option value="name">Name</option></select>
<div class="seg" role="group" aria-label="View"><button type="button" data-view="cards" aria-pressed="true">Cards</button>
<button type="button" data-view="table" aria-pressed="false">Table</button></div></div></div>
<ul class="cards">{cards}</ul>
<div class="tablebox tableview" hidden><table class="facts"><thead><tr><th scope=col>Model</th><th scope=col>Design</th>
<th scope=col>Parameters</th><th scope=col>Layers</th><th scope=col>Context</th><th scope=col>Cache / token</th>
<th scope=col>Layer order</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="empty" hidden>No model matches these filters. Try a broader search or reset the filters.</p>
<details class="heldbox"><summary>{len(held)} models found but not yet published</summary>
<p class="sub">Analysed, but not published until every rule for them is checked. Fine-tunes are out of scope and not listed.</p>
<ul>{held_rows}</ul></details>
</section></div>
<div class="comparetray" id="compare-tray" hidden><div><b>Compare your selection</b><p id="compare-selection" role="status" aria-live="polite"></p></div><div class="actions"><button type="button" class="btn" id="clear-selection">Clear</button><a class="btn primary" id="compare-selection-link" href="compare/index.html" aria-disabled="true">Select at least 2</a></div></div>"""
    return frame(
        "ArchWatch: find, compare and size open LLM architectures",
        body,
        active="explore",
        description="Architectures of open-weight models, read from their own files and checked.",
    )


# ---------------------------------------------------------------- model page


def config_key(c):
    """The config key behind a fact (small text under its label), or where else it was read."""
    for x in c["candidates"]:
        src, _, detail = x["evidence"].partition(":")
        if src == "config":
            return detail.split(" ")[0]
    first = c["candidates"][0]["evidence"] if c["candidates"] else ""
    src, _, detail = first.partition(":")
    return f"{SOURCE_WORDS.get(src, src)}: {detail}"[:80]


def source_links(rec, c):
    """Each source of a fact, linked to its file at the pinned commit (config key or code line where known)."""
    links = {}
    for x in c["candidates"]:
        src, _, detail = x["evidence"].partition(":")
        word = SOURCE_WORDS.get(src, src)
        url = source_url(rec, x["evidence"])
        if src == "code":
            m = re.match(r"([\w./-]+\.py)(?:#L(\d+))?", detail)
            word = f"{m.group(1)} line {m.group(2)}" if m and m.group(2) else (m.group(1) if m else word)
        if word not in links:
            links[word] = url
    return ", ".join(f'<a href="{e(u)}">{e(w)}</a>' if u else e(w) for w, u in links.items())


def fact_row(rec, label, key, value, c):
    status, note = card_status(c) if c else ("computed", None)
    word = STATUS_WORDS.get(status, status)
    sources = source_links(rec, c) if c else "computed from the facts above"
    extra = f'<span class="note">{e(note)}; recorded, not a gate</span>' if note else ""
    return (
        f'<tr><th scope=row>{e(label)}<span class="key">{e(key)}</span></th><td class="mono">{e(value)}</td>'
        f'<td>{sources}{extra}</td><td><span class="st st-{word.replace(" ", "-")}">{e(word)}</span></td></tr>'
    )


def facts(rec, pub, cache):
    checks = {c["field"]: c for c in rec["checks"]}
    shown = set()
    key_rows = []

    def one(field, label=None, value=None):
        if field not in pub or field not in checks:
            return
        shown.add(field)
        key_rows.append(
            fact_row(
                rec,
                label or PLAIN.get(field, field),
                config_key(checks[field]),
                value or show(field, pub[field]),
                checks[field],
            )
        )

    for field in KEY_FACTS:
        if field == "attention":
            if "num_attention_heads" in pub:
                value = attention_summary(pub)
                one("num_attention_heads", "Attention type and heads", value)
                shown.update({"attention_type", "num_kv_heads"})
        elif field == "experts":
            if "moe_num_experts" in pub:
                value = f"{pub['moe_num_experts']} routed / {pub['moe_active_experts']} per token / {pub.get('moe_shared_experts', 0)} shared"
                one("moe_num_experts", "Experts (routed / per token / shared)", value)
                shown.update({"moe_active_experts", "moe_shared_experts"})
        else:
            one(field)
    key_rows.append(
        f'<tr><th scope=row>Cache per token<span class="key">bf16; sliding-window and constant states excluded</span></th>'
        f"<td class=mono>{e(cache_fact(pub, cache))}</td><td>ArchWatch calculator</td>"
        f'<td><span class="st st-computed">computed</span></td></tr>'
    )
    rest = []
    for field in sorted(pub, key=lambda f: PLAIN.get(f, f).lower()):
        if field in shown or field in NOT_SHOWN or field not in checks:
            continue
        rest.append(
            fact_row(
                rec,
                PLAIN.get(field, field.replace("_", " ").capitalize()),
                config_key(checks[field]),
                show(field, pub[field]),
                checks[field],
            )
        )
    for field, c in sorted(checks.items()):
        if field not in pub and field not in NOT_SHOWN:  # held for review: no value shown
            rest.append(
                f'<tr><th scope=row>{e(PLAIN.get(field, field.replace("_", " ").capitalize()))}<span class="key">{e(config_key(c))}</span></th>'
                f'<td class=mono>held for review</td><td>sources disagree</td><td><span class="st st-held-for-review">held for review</span></td></tr>'
            )
    head = "<thead><tr><th scope=col>Fact</th><th scope=col>Value</th><th scope=col>Read from</th><th scope=col>Status</th></tr></thead>"
    return (
        f'<div class="tablebox"><table class="facts">{head}<tbody>{"".join(key_rows)}</tbody></table>'
        f'<details class="allfields"><summary>All {len(rest) + len(key_rows) - 1} fields</summary>'
        f'<table class="facts">{head}<tbody>{"".join(rest)}</tbody></table></details></div>'
    )


def cache_fact(pub, cache):
    cap = capped(pub, cache)
    if cap:
        return f"capped at {cap[0]:,} tokens: {cap[1]:,} B ({size(cap[1])}) when the window is full"
    return f"{cache['grows_per_token_bytes']:,} B"


def fit_planner(pub, cache):
    ctx = int(pub["context_length"])
    contexts = [n for n in settings()["memory_contexts"] if n < ctx] + [ctx]
    default_ctx = 32768 if 32768 in contexts else contexts[-1]
    combos = {}
    for n in contexts:
        kv = cache_bytes(cache_parts(pub, cache), n)
        for key, _, per in PRECISION:
            weights = pub["params_total"] * per
            total = weights + kv
            combos[f"{n}-{key}"] = dict(
                total=size(total),
                weights=size(weights),
                cache=size(kv),
                weights_bytes=weights,
                cache_bytes=kv,
                devices=[
                    [
                        min(100, round(100 * total / cap, 1)),
                        "fits" if total <= 0.75 * cap else "tight" if total <= cap else "too big",
                    ]
                    for _, cap in DEVICES
                ],
            )
    pick = combos[f"{default_ctx}-16"]
    ctx_btns = "".join(
        f'<button type="button" data-ctx="{n}" aria-pressed="{"true" if n == default_ctx else "false"}">{e(tokens(n))}{" (max)" if n == ctx else ""}</button>'
        for n in contexts
    )
    prec_btns = "".join(
        f'<button type="button" data-prec="{k}" aria-pressed="{"true" if k == "16" else "false"}">{e(label)}</button>'
        for k, label, _ in PRECISION
    )
    bars = "".join(
        f'<li><span>{e(name)}</span><span class="track"><span class="fill" style="width:{pct}%"></span></span>'
        f'<span class="verdict v-{word.replace(" ", "-")}">{e(word)}</span></li>'
        for (name, _), (pct, word) in zip(DEVICES, pick["devices"])
    )
    return f"""<section id="fit" class="panel planner" aria-labelledby="fit-h" data-combos='{e(json.dumps(combos))}'>
<h2 id="fit-h">Will it fit on my machine?</h2>
<p class="sub">Theoretical weight storage plus persistent decoding state. Start with one sequence, then plan your own memory budget.</p>
<div class="choice"><span>Context length</span><div class="seg" role="group" aria-label="Context length">{ctx_btns}</div></div>
<div class="choice"><span>Weight precision</span><div class="seg" role="group" aria-label="Weight precision">{prec_btns}</div></div>
<div class="planner-inputs"><label>Device memory (GB)<input id="device-memory" type="number" min="1" max="100000" value="16" step="1"></label><label>Concurrent sequences<input id="sequences" type="number" min="1" max="1024" value="1" step="1"></label><label>Planning headroom (%)<input id="headroom" type="number" min="0" max="90" value="20" step="1"></label></div>
<div class="fitsum" aria-live="polite"><p class="big mono"><span data-out="total">{e(pick["total"])}</span></p>
<p class="mono split2">weights <span data-out="weights">{e(pick["weights"])}</span><br>cache <span data-out="cache">{e(pick["cache"])}</span></p></div>
<ul class="devices">{bars}</ul>
<p class="budget-result" id="budget-result" role="status" aria-live="polite"></p>
<p class="hint">Weights are shared; each sequence needs its own persistent state. Device GB uses decimal units; displayed storage uses binary units. Headroom is your planning allowance, not measured runtime overhead. The reference bars use 75% as their “fits” threshold.</p>
<details class="planner-assumptions"><summary>Formula and assumptions</summary><p>Weights = {pub["params_total"]:,} parameters × bytes per parameter. {e(cache_rule(pub, cache))}. Weight precision does not change cache precision. Total = weights + state per sequence × sequences.</p><p>{e(" ".join(cache.get("assumptions", [])))}</p><p>Runtime allocations, quantization metadata, temporary activations, offloading and device behavior are not included. A mathematical fit is not a runtime guarantee.</p></details></section>"""


def cache_rule(pub, cache):
    cap = capped(pub, cache)
    if cap:
        return (
            f"Cache is capped at {cap[0]:,} tokens: {size(cap[1])} once the window is full, then it stops growing"
            + (
                f" (plus {size(cache['constant_bytes'] - cap[1])} of fixed-size state)"
                if cache["constant_bytes"] > cap[1]
                else ""
            )
        )
    fixed = f" + {size(cache['constant_bytes'])} of fixed-size state" if cache["constant_bytes"] else ""
    return f"Cache = {cache['grows_per_token_bytes']:,} B per token × context{fixed}"


def layer_section(pub):
    legend = "".join(
        f'<li><span class="sw" style="background:{COLOR.get(k, "#999")}" aria-hidden="true"></span><b>{pub["layer_type_counts"][k]}</b> {e(LABEL.get(k, k))}</li>'
        for k in kinds_in_order(pub)
    )
    return f"""<section id="layers" aria-labelledby="layers-h"><h2 id="layers-h">Layer by layer</h2>
<p class="sub">Layer 1 is on the left.</p>{cells(pub, "Layer order, layer 1 on the left: " + mix_text(pub))}
<ul class="legendboxes">{legend}</ul></section>"""


def model_page(stem, rec, pub, published):
    cache = render_cache(pub)
    org, name = rec["id"].split("/")
    pubs = settings()["publishers"]
    before = published_predecessor(rec, published)
    checks = {c["field"]: c for c in rec["checks"]}
    key = [f for f in ("params_total", "num_layers", "layer_type_counts", "context_length") if f in pub]
    verified = [f for f in key if card_status(checks[f])[0] == "verified"]
    badge = "Key facts verified" if len(verified) == len(key) else f"{len(verified)} of {len(key)} key facts verified"
    tiles = [
        (human(pub["params_total"]), "parameters"),
        (human(pub["params_active"]), "active per token")
        if "params_active" in pub
        else ("all", "active per token (dense)"),
        (str(pub["num_layers"]), "layers"),
        (f"{int(pub['context_length']):,}", "context tokens"),
        cache_value(pub, cache),
    ]
    t_html = "".join(f"<div><dt>{e(b)}</dt><dd class=mono>{e(a)}</dd></div>" for a, b in tiles)
    compare = ""
    changed_html = ""
    if before:
        pred_name = before[1]["id"].split("/")[-1]
        compare = f'<a class="btn" href="../compare/{e(stem)}.html">Compare with {e(pred_name)}</a>'
        items = "".join(
            f"<li><b>{e(label)}</b>: <span class=mono>{e(a)}</span> &rarr; <span class=mono>{e(b)}</span></li>"
            for label, a, b in changes(rec, before[1])
        )
        changed_html = f"""<section aria-labelledby="chg"><h2 id="chg">What changed since {e(pred_name)}</h2>
<ul class="changes">{items or "<li>No architectural change above 1%.</li>"}</ul>
<p class="byline">Listed by code from the verified diff: changes under 1% (or under 0.1B parameters) and weight-group bookkeeping are left out; numbers to 3 significant figures.</p></section>"""
    chosen = generated.select(rec)
    if chosen:
        svg_path = ROOT / "site/diagrams" / f"{stem}.svg"
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(svg_path, chosen[0])
        n = len(chosen[1]["checks_passed"])
        art = f'<img src="../diagrams/{e(stem)}.svg" alt="Architecture diagram of {e(name)}: {e(mix_text(pub))}">'
        art_label = f"Drawn by Gemini from verified fields, {n} of {n} checks passed"
        art_link = f' · <a href="../diagrams/{e(stem)}.svg">Open full size</a>'
    else:
        art = diagram.render(rec)
        art_label = "Drawn by code from verified fields; no generated diagram has passed every check yet"
        art_link = ""
    commit = rec["source_commit"]
    tabs = "".join(
        f'<a href="#{a}">{t}</a>'
        for a, t in (
            ("short", "Overview"),
            ("layers", "Layers"),
            ("diagram", "Diagram"),
            ("fit", "Memory planner"),
            ("concept-evidence", "Concept evidence"),
            ("facts", "Facts and sources"),
        )
    )
    body = f"""<nav class="crumbs" aria-label="Breadcrumb"><a href="../index.html">Explore</a> / {e(pubs.get(org, org))}</nav>
<div class="titlerow"><div><h1>{e(name)}</h1>
<p class="sub"><a href="https://huggingface.co/{e(rec["id"])}">{e(rec["id"])}</a> · {e(when(rec, pub)[0])} {e(date(when(rec, pub)[1]))}{" (model card)" if pub.get("release_date") else ""} ·
read at commit <a class="mono" href="https://huggingface.co/{e(rec["id"])}/tree/{e(commit)}">{e(commit[:10])}</a></p></div>
<div class="actions"><span class="badge">{e(badge)}</span>{compare}</div></div>
<nav class="tabs" aria-label="On this page">{tabs}</nav>
<dl class="stats tiles">{t_html}</dl>
<section id="short" aria-labelledby="short-h"><h2 id="short-h">In short</h2><p class="prose">{e(describe(pub))}</p>
<p class="byline">Written by code from the published facts.</p><h3 class="concept-heading">Understand the mechanisms</h3>{concept_links(pub, "../", rec=rec)}</section>
{changed_html}
{layer_section(pub)}
<section id="diagram" aria-labelledby="arch"><div class="sechead"><h2 id="arch">Diagram</h2><p class="hint">{e(art_label)}{art_link}</p></div>
<figure class="panel diagram">{art}</figure></section>
<div class="diagram-tools"><button class="btn" type="button" id="expand-diagram">Expand diagram ⤢</button><button class="btn" type="button" id="download-diagram" data-filename="{e(stem)}.svg">Download SVG ↓</button></div>
<dialog id="diagram-dialog" aria-labelledby="diagram-dialog-title"><div class="dialogbar"><h2 id="diagram-dialog-title">{e(name)} architecture</h2><button class="btn" type="button" id="close-diagram">Close ×</button></div><div class="dialog-art"></div></dialog>
{fit_planner(pub, cache)}
{concept_evidence(rec, stem)}
<section id="facts" aria-labelledby="facts-h"><h2 id="facts-h">Facts and where they come from</h2>
<p class="sub">Verified means two independent files agree. <a download href="../data/models/{e(stem)}.json">Download published data + source links (JSON) ↓</a></p>{facts(rec, pub, cache)}</section>"""
    return frame(f"{name} architecture · ArchWatch", body, depth=1, active="explore", description=describe(pub))


def model_section(name, anchor):
    """One model page from data/models/<name>.json (raises PublicationError when it is not publishable)."""
    rec = json.loads((ROOT / "data/models" / f"{name}.json").read_text())
    pub = render_values(rec)
    return model_page(name, rec, pub, {})


# ---------------------------------------------------------------- compare tool data (site/data/models.json)

TABLE_FIRST = (
    "layer_type_counts",
    "params_total",
    "params_active",
    "params_active_non_embedding",
    "params_mtp",
    "num_layers",
    "context_length",
    "cache",
    "hidden_size",
    "attention_type",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
    "moe_num_experts",
    "moe_active_experts",
    "moe_shared_experts",
    "vocab_size",
)
CHART_MIN = 1024
CONTEXTS = (8192, 32768, 131072, 1048576)
SLOT_COLORS = ("#0B6E66", "#C2410C", "#6D28D9")  # line colours for compare slots A, B, C (not layer types)


def cache_parts(pub, cache):
    """Cache split into what grows per token, what is capped by a sliding window, and fixed-size state."""
    sliding = [r for r in cache["breakdown"] if r["layer_type"] == "sliding_attention" and r["constant_bytes"]]
    window = int(pub.get("sliding_window") or 0)
    full = sum(r["constant_bytes"] for r in sliding) if window else 0
    return dict(
        grow=cache["grows_per_token_bytes"],
        window=window if full else 0,
        window_full=full,
        fixed=cache["constant_bytes"] - full,
    )


def cache_bytes(parts, n):
    """Cache for one sequence at n tokens: growing layers scale with n, windowed layers up to their window, plus
    fixed-size state. At or past the window this equals the calculator's grows_per_token x n + constant."""
    windowed = parts["window_full"] * min(n, parts["window"]) // parts["window"] if parts["window"] else 0
    return parts["grow"] * n + windowed + parts["fixed"]


def chart_grid(max_context):
    points, n = [], CHART_MIN
    while n < max_context:
        points.append(n)
        n *= 2
    return points + [max_context]


def display(field, value):
    if field == "layer_type_counts":
        return " + ".join(f"{n} {SHORT.get(k, k)}" for k, n in value.items())
    if field.startswith("params") and isinstance(value, (int, float)):
        for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
            if value >= d:
                return f"{sig(value / d)}{unit}"
    return show(field, value)


def note_follows_filter(rec, before, note):
    """A saved note may be shown only if it states no change the site leaves out (under 1%, bookkeeping)."""
    if "weight group" in note.lower():
        return False
    hidden = set()
    for row in pipeline.diff(rec, before):
        if not significant(row["field"], row["before"], row["after"]):
            for v in (row["before"], row["after"]):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    hidden.update({f"{v:,}", str(v), change_value(row["field"], v)})
    numbers = {x.replace(",", "") for x in re.findall(r"\d[\d,.]*[KMBT]?", note)}
    return not any(h.replace(",", "") in numbers for h in hidden)


def model_data(stem, rec, pub, published, max_context):
    cache = render_cache(pub)
    parts = cache_parts(pub, cache)
    org, name = rec["id"].split("/")
    fields = {}
    for field, value in pub.items():
        if (
            field in NOT_SHOWN | NOT_A_CHANGE
            or field not in PLAIN
            or isinstance(value, (list, dict))
            and field != "layer_type_counts"
        ):
            continue
        raw = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        fields[field] = [display(field, value), raw]
    value, label = cache_value(pub, cache)
    fields["cache"] = [
        value if label == "cache / token" else f"{value} ({label.replace('cache, ', '')})",
        parts["grow"],
    ]
    derived = derived_fields(pub)
    before = published_predecessor(rec, published)
    note = explainer_note(rec) if before else None
    if note and not note_follows_filter(rec, before[1], note):
        note = None
    grid = [n for n in chart_grid(max_context) if n <= int(pub["context_length"])] + [int(pub["context_length"])]
    return dict(
        stem=stem,
        id=rec["id"],
        source_commit=rec["source_commit"],
        source_url=f"https://huggingface.co/{rec['id']}/tree/{rec['source_commit']}",
        name=name,
        publisher=settings()["publishers"].get(org, org),
        date=f"{when(rec, pub)[0]} {date(when(rec, pub)[1])}",
        design=design(pub),
        hybrid=hybrid(pub),
        layers=pub["layer_types"],
        counts=pub["layer_type_counts"],
        context=int(pub["context_length"]),
        params=dict(
            total=pub["params_total"],
            active=pub.get("params_active"),
            active_non_embedding=pub.get("params_active_non_embedding"),
            mtp=pub.get("params_mtp"),
        ),
        cache=parts,
        cache_label=fields["cache"][0],
        series=[[n, cache_bytes(parts, n)] for n in sorted(set(grid))],
        fields=fields,
        derived=derived,
        predecessor=before[0] if before else None,
        note=note,
    )


KEY_ROWS = (
    "design",
    "params_total",
    "params_active",
    "params_active_non_embedding",
    "num_layers",
    "layer_type_counts",
    "context_length",
    "cache",
    "hidden_size",
    "attention",
    "experts",
    "vocab_size",
)
MERGED = {  # shown inside a combined row instead of on their own
    "attention_type",
    "num_attention_heads",
    "num_kv_heads",
    "moe_num_experts",
    "moe_active_experts",
    "moe_shared_experts",
    "ffn_intermediate",
    "dense_ffn_intermediate",
}
DERIVED_LABELS = {
    "design": "Design",
    "attention": "Attention type and heads",
    "experts": "Experts (routed / per token / shared)",
    "ffn": "FFN hidden size",
}


def derived_fields(pub):
    """Rows computed from published values, or implied by the architecture (never read from a held field)."""
    out = {"design": [design(pub) + (" (hybrid)" if hybrid(pub) and design(pub) == "MoE" else ""), None]}
    if "num_attention_heads" in pub:
        out["attention"] = [attention_summary(pub), None]
    if "moe_num_experts" in pub:
        out["experts"] = [
            f"{pub['moe_num_experts']} / {pub['moe_active_experts']} / {pub.get('moe_shared_experts', 0)}",
            pub["moe_num_experts"],
        ]
    else:
        out["experts"] = ["none (dense)", None]
    comp = pub.get("parameter_components") or {}
    if "moe_num_experts" not in pub:
        out["params_active"] = [f"all, {display('params_total', pub['params_total'])} (dense)", pub["params_total"]]
        if "params_active_non_embedding" not in pub and comp:
            # dense: every parameter is active; 'excluding embeddings' drops the input embedding and output head
            value = pub["params_total"] - comp.get("embedding", 0) - comp.get("lm_head", 0)
            out["params_active_non_embedding"] = [f"{display('params_total', value)} (dense)", value]
    if "v_head_dim" not in pub and "head_dim" in pub and not set(pub["layer_types"]) & MLA_KINDS:
        out["v_head_dim"] = [f"{pub['head_dim']:,} (same as head size)", pub["head_dim"]]
    if "ffn_intermediate" in pub:
        out["ffn"] = [f"{pub['ffn_intermediate']:,}", pub["ffn_intermediate"]]
    elif "dense_ffn_intermediate" in pub:
        out["ffn"] = [f"{pub['dense_ffn_intermediate']:,} (dense layers)", pub["dense_ffn_intermediate"]]
    return out


def suggestions(models):
    """Predecessor pairs, then code-chosen pairs: closest total parameters across different designs, and a dense
    model next to a hybrid one of similar size."""
    out, seen = [], set()

    def add(a, b, why):
        key = tuple(sorted((a["stem"], b["stem"])))
        if key not in seen:
            seen.add(key)
            out.append(dict(a=a["stem"], b=b["stem"], label=f"{a['name']} vs {b['name']}", why=why))

    by_stem = {m["stem"]: m for m in models}
    for m in models:
        if m["predecessor"] in by_stem:
            add(by_stem[m["predecessor"]], m, "predecessor")

    def gap(a, b):
        return abs(math.log(a["params"]["total"] / b["params"]["total"]))

    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
    combos = sorted({tuple(sorted((a["design"], b["design"]))) for a, b in pairs if a["design"] != b["design"]})
    for combo in combos:  # closest total parameters for each combination of different designs
        best = min(
            (p for p in pairs if tuple(sorted((p[0]["design"], p[1]["design"]))) == combo), key=lambda p: gap(*p)
        )
        add(*best, f"closest total parameters, {combo[0]} vs {combo[1]}")
    return out


def compare_data(published):
    max_context = max((int(p["context_length"]) for _, _, p in published.values()), default=1)
    models = [model_data(s, r, p, published, max_context) for s, r, p in published.values()]
    models.sort(key=lambda m: m["name"].lower())
    labels = {f: PLAIN.get(f, f) for m in models for f in list(m["fields"]) + list(m["derived"]) if f not in MERGED}
    labels.update({f: DERIVED_LABELS[f] for f in DERIVED_LABELS if f in labels})
    labels["cache"] = "Cache per token"
    implied_values = {
        m["stem"]: {
            f: implied(f, published[m["id"]][2])
            for f in labels
            if f not in m["fields"] and implied(f, published[m["id"]][2])
        }
        for m in models
    }
    order = [f for f in KEY_ROWS if f in labels]
    rest = sorted((f for f in labels if f not in KEY_ROWS), key=lambda f: labels[f].lower())
    return dict(
        colors=COLOR,
        kinds={k: SHORT.get(k, k) for k in COLOR},
        labels=labels,
        order=order,
        rest=rest,
        implied=implied_values,
        contexts=[n for n in CONTEXTS if n <= max_context],
        precision=[dict(key=k, label=lb, bytes=b) for k, lb, b in PRECISION],
        devices=[dict(label=lb, bytes=b) for lb, b in DEVICES],
        slots=SLOT_COLORS,
        models=models,
        suggested=suggestions(models),
    )


def compare_tool(data, default=None):
    """The compare tool page. `default` = (a, b) pre-selects a pair (the /compare/<pair> pages)."""
    pairs = [s for s in data["suggested"] if s["why"] == "predecessor"]
    nojs = (
        "".join(f'<li><a href="../compare/{e(s["b"])}.html">{e(s["label"])}</a></li>' for s in pairs)
        or "<li>No published model has a published predecessor yet.</li>"
    )
    chips = "".join(
        f'<li><a class="chipbtn" href="../compare/index.html?a={e(s["a"])}&amp;b={e(s["b"])}" data-a="{e(s["a"])}" data-b="{e(s["b"])}">'
        f"{e(s['label'])}<span>{e(s['why'])}</span></a></li>"
        for s in data["suggested"]
    )
    options = "".join(f'<option value="{e(m["name"])}">{e(m["publisher"])}</option>' for m in data["models"])
    pick = lambda slot, label: (  # noqa: E731
        f'<div class="slot compare-model" data-slot="{slot}"{" hidden" if slot == "c" else ""}>'
        f'<div class="slot-top"><label for="pick-{slot}">{label}</label><button type="button" class="linkbtn change-model" data-edit="{slot}">Change</button></div>'
        f'<input id="pick-{slot}" list="models-list" autocomplete="off" placeholder="Choose a published model" aria-describedby="slot-meta-{slot}">'
        f'<span class="slotmeta" id="slot-meta-{slot}"></span><div class="slot-preview"></div>'
        + ('<button type="button" class="linkbtn" id="clear-c">Remove C</button>' if slot == "c" else "")
        + "</div>"
    )
    preset = f' data-a="{e(default[0])}" data-b="{e(default[1])}"' if default else ""
    body = f"""<div class="compare-title"><div><p class="eyebrow">Side by side</p><h1>Compare</h1><p class="sub">See what changed, inspect the evidence, and compare memory under the same assumptions.</p></div><div class="seg reading-mode" role="group" aria-label="Reading mode"><button type="button" data-reading="plain" aria-pressed="true">Plain</button><button type="button" data-reading="technical" aria-pressed="false">Technical</button></div></div>
<noscript><p class="lead">The compare tool needs JavaScript. Each model next to its predecessor:</p><ul class="plainlist">{nojs}</ul></noscript>
<p id="compare-error" class="panel" role="alert" hidden>Could not load the published model data. Reload the page to try again.</p>
<div id="tool" class="tool" data-reading="plain" data-version="{hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]}" hidden{preset}>
<datalist id="models-list">{options}</datalist>
<form class="comparison-models" id="pickers" aria-label="Models to compare">{pick("a", "Model A")}{pick("b", "Model B")}{pick("c", "Model C (optional)")}
<button type="button" class="add-third" id="add-third"><span aria-hidden="true">＋</span>Add a third model</button>
<div class="compare-actions"><button type="button" class="btn" id="swap" aria-label="Swap model A and model B">⇄ Swap A and B</button><details class="compare-suggestions"><summary>Suggested comparisons</summary><ul class="suggest">{chips}</ul></details><span class="hint">Search a card to change its model.</span></div></form><p id="selection-error" class="hint" role="alert"></p>
<section class="panel compare-summary" aria-labelledby="kd"><div class="sechead"><h2 id="kd">What changed</h2><span class="eyebrow">Computed from published fields</span></div><div id="keydiff"></div></section>
<section aria-labelledby="fbf"><div class="sechead"><h2 id="fbf">Field by field</h2><div class="compare-table-actions"><label class="toggle"><input type="checkbox" id="onlydiff"> Only show differences</label><button type="button" class="btn" id="copylink">Copy link</button><button type="button" class="btn" id="export-csv">Export CSV ↓</button></div></div><p class="hint" id="copied" aria-live="polite"></p><div class="tablebox compare-fields" id="fields"></div></section>
<section class="panel" aria-labelledby="cc"><h2 id="cc">Cache memory as context grows</h2><div id="cache-summarybars"></div><p class="hint">Persistent state for one sequence, including fixed state and window caps. Attention uses bf16; recurrent state can use fp32 under the model's cache rules. Contexts stay within both models' limits.</p><details class="compare-chart-details"><summary>Explore the full cache curves</summary><div id="chart"></div><details><summary>Show the numbers</summary><div id="chartdata" class="tablebox"></div></details></details></section>
<details class="compare-layer-details"><summary>Inspect layer strips in detail</summary><section aria-labelledby="ls"><h2 id="ls">Layer by layer</h2><p class="sub">Every strip has the same width; layer 1 is on the left.</p><div id="strips"></div></section></details>
<section class="panel" aria-labelledby="fit-h"><h2 id="fit-h">Will they fit?</h2><p class="sub">Weight storage plus persistent state for one conversation. Runtime overhead is excluded. “Tight” means over 75% of device memory.</p><div class="choice"><span>Context length</span><div class="seg" role="group" aria-label="Context length" id="ctxseg"></div></div><div class="choice"><span>Weight precision</span><div class="seg" role="group" aria-label="Weight precision" id="precseg"></div></div><div id="fit-compare" aria-live="polite"></div></section></div>"""
    page = frame("Compare · ArchWatch", body, depth=1, active="compare")
    version = hashlib.sha256(COMPARE_JS.encode()).hexdigest()[:12]
    return page.replace("</body>", f'<script src="../compare.js?v={version}" defer></script></body>')


# ---------------------------------------------------------------- assets


def lessons():
    return yaml.safe_load((CODE / "config/concept-lessons.yaml").read_text())


def input_value(value):
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, bool)) else str(value)
    return text if len(text) <= 80 else "see the model page"


def worked_example(lesson, published, title):
    """A model-backed example is computed from the record at build time; otherwise the lesson's toy example."""
    example = lesson["example"]
    if not example.get("model"):
        note = f"No published model uses {title.lower() if title[1:2].islower() else title} yet, so this example uses toy numbers."
        return (
            f'<p class="toy-note">{e(note)}</p>',
            example["setup"],
            example["steps"],
            example["result"],
            lesson.get("code"),
        )
    if example["model"] not in published:  # held or removed: never show numbers from an unpublished record
        name = example["model"].split("/")[-1]
        return (
            f'<p class="toy-note">This example is computed from {e(name)}, which is not published at the moment.</p>',
            "",
            [],
            "",
            None,
        )
    w = concept_examples.worked(example, published)
    inputs = ", ".join(
        f"<code>{e(name)}</code> = {e(input_value(value))} ({e(STATUS_WORDS[status])})"
        for name, value, status in w["fields"]
    )
    note = (
        f'<p class="toy-note">Computed when the site is built, from the published record of <a href="../models/{e(w["stem"])}.html">{e(w["values"]["model"])}</a> '
        f'at commit <span class="mono">{e(w["commit"][:10])}</span>. Inputs: {inputs}. The model page shows the same values.</p>'
    )
    return note, w["setup"], w["steps"], w["result"], w["code"]


ATTENTION_LABS = {"mha", "gqa", "kv-cache", "sliding-window"}
TOY_LAB = dict(model="toy decoder", layers=2, hq=8, kv=2, hd=64, context=4096, window=512)


def lab_bytes(preset, windowed):
    retained = min(preset["context"], preset["window"]) if windowed and preset.get("window") else preset["context"]
    return retained, 2 * preset["layers"] * retained * preset["kv"] * preset["hd"] * 2


def concept_lab(key, lesson, published):
    """Calculators start from the worked example's model; toy values are a labelled reset option."""
    example = lesson["example"]
    model = published.get(example.get("model"))
    if key in ATTENTION_LABS:
        toy = dict(TOY_LAB, kv=8 if key == "mha" else 2)
        start = toy
        if model:
            start = {k: v for k, v in concept_examples.lab_preset(model[1]).items() if k != "bytes"}
        windowed = key == "sliding-window" and bool(start.get("window"))
        retained, nbytes = lab_bytes(start, windowed)
        baseline = 2 * start["layers"] * start["context"] * start["hq"] * start["hd"] * 2
        kind = "MHA" if start["kv"] == start["hq"] else "MQA" if start["kv"] == 1 else "GQA"
        source = (
            f"Starts from {e(start['model'])}'s published values: {start['hq']} query heads, head width {start['hd']}, bf16 cache."
            if model
            else "No published model uses this yet, so the calculator starts from toy values."
        )
        presets = json.dumps({"model": start, "toy": toy, "windowed": windowed}, sort_keys=True)
        buttons = (
            f'<div class="lab-presets"><button type="button" class="btn" data-preset="model">Use {e(start["model"])} values</button>'
            f'<button type="button" class="btn" data-preset="toy">Reset to toy values</button></div>'
            if model
            else ""
        )
        return (
            f"""<section id="try-it" class="concept-lab attention-lab" data-presets='{e(presets)}'><p class="eyebrow">Try the arithmetic</p><h2>Explore the cache trade-off</h2>"""
            f"""<p data-lab="source">{source}</p>{buttons}<div class="lab-controls">"""
            f"""<label>Context tokens<input name="toy-context" type="number" min="1" max="1048576" step="1" value="{start["context"]}"></label>"""
            f"""<label>KV heads<select name="toy-kv"><option value="{start["kv"]}" selected>{start["kv"]}</option></select></label>"""
            f"""<label>Attention layers<input name="toy-layers" type="number" min="1" max="256" step="1" value="{start["layers"]}"></label>"""
            f"""<label>Retention<select name="toy-window"><option value="full"{"" if windowed else " selected"}>Full history</option>"""
            + (
                f"""<option value="{start["window"]}"{" selected" if windowed else ""}>Window {start["window"]:,}</option>"""
                if start.get("window")
                else ""
            )
            + f"""</select></label></div>"""
            f"""<div class="lab-result" aria-live="polite"><b data-toy="type">{kind}</b><strong data-toy="bytes">{e(lab_size(nbytes))}</strong>"""
            f"""<span><span data-toy="retained">{retained:,}</span> retained positions; full-history MHA baseline <span data-toy="baseline">{e(lab_size(baseline))}</span>.</span>"""
            f"""<code data-toy="formula">2 × {start["layers"]} × {retained} × {start["kv"]} × {start["hd"]} × 2 = {nbytes:,} bytes</code></div>"""
            f"""<p class="hint" data-lab="toy-note"{"" if not model else " hidden"}>Toy values: 8 query heads of width 64. They are not any model's facts.</p>"""
            f"""<p class="hint">Attention cache only: no weights, recurrent state or runtime overhead.</p></section>"""
        )
    if key == "rmsnorm":
        width = ""
        if model:
            width = f" {e(model[1]['id'].split('/')[-1])} normalizes {render_values(model[1])['hidden_size']:,} values per token in every norm; its activations are not published, so try a short vector of your own."
        return f'<section id="try-it" class="concept-lab norm-lab"><p class="eyebrow">Try the arithmetic</p><h2>Normalize your own vector</h2><p>{width.strip() or "Try a short vector of your own."}</p><label>Vector components<input type="text" value="3, 4" aria-label="Vector components"></label><output aria-live="polite"></output><p class="hint">Toy vector, learned scales set to 1.</p></section>'
    return ""


def lab_size(n):
    for unit, power in (("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if n >= 2**power:
            return f"{n / 2**power:.2f} {unit}"
    return f"{n} B"


def teaching_sections(key, item, lesson, published):
    note, setup, example_steps, result, code_text = worked_example(lesson, published, item["title"])
    symbols = "".join(
        f"<div><dt>{e(symbol)}</dt><dd>{e(meaning)}</dd></div>" for symbol, meaning in lesson["symbols"].items()
    )
    steps = "".join(f"<li>{e(step)}</li>" for step in example_steps)
    pitfalls = "".join(f"<li>{e(pitfall)}</li>" for pitfall in lesson["pitfalls"])
    related = "".join(f'<a href="{e(slug)}.html">{e(concepts()[slug]["short"])} →</a>' for slug in lesson["related"])
    lab = concept_lab(key, lesson, published)
    code = (
        f'<details class="lesson-code"><summary>Minimal Python illustration</summary><p>This small example demonstrates the arithmetic, not a full model implementation.</p><pre><code>{e(code_text)}</code></pre></details>'
        if code_text
        else ""
    )
    return f"""<section id="why"><h2>Why this mechanism exists</h2><p>{e(lesson["why"])}</p></section><section id="mechanism"><h2>Follow the operation</h2><p>{e(item["mechanism"])}</p><figure class="lesson-figure">{concept_diagrams.render(key, lesson)}<figcaption>Original ArchWatch teaching schematic. <a download href="figures/{key}.svg">Download SVG ↓</a></figcaption></figure></section><section id="worked-example" class="worked-example"><p class="eyebrow">A worked example</p><h2>Make it concrete</h2>{note}<p>{e(setup)}</p><ol>{steps}</ol><p class="example-result">{e(result)}</p></section><section id="equation"><h2>Equation and notation</h2><pre class="equation">{e(lesson["equation"])}</pre><dl class="notation">{symbols}</dl>{code}</section>{lab}<section id="memory"><h2>What it means for memory</h2><p>{e(item["memory"])}</p></section><section id="pitfalls"><h2>Where comparisons go wrong</h2><ul>{pitfalls}</ul><p class="lesson-takeaway">{e(lesson["takeaway"])}</p></section><section id="verify"><h2>How to verify it in a model</h2><p>{e(item["verify"])}</p></section><div class="related-lessons"><span class="eyebrow">Follow the connection</span>{related}</div>"""


DRAFT = '<span class="draft" title="Not yet reviewed by the maintainer">draft</span>'


def guide_pages(published):
    guide = concepts()
    lesson_data = lessons()
    model_matches = {
        stem: {hit["key"]: hit for hit in concept_data.for_record(rec)} for stem, rec, _ in published.values()
    }
    cards, pages = [], {}
    for key, item in guide.items():
        draft = "" if item["reviewed"] else DRAFT
        matches = [(stem, rec, pub) for stem, rec, pub in published.values() if key in model_matches[stem]]
        state = (
            f'<span class="mono">{len(matches)} published example{"s" if len(matches) != 1 else ""}</span>'
            if matches
            else '<span class="nopub-tag">no published model yet</span>'
        )
        cards.append(
            (
                not matches,
                len(cards),
                f'''<li data-published="{"1" if matches else "0"}" data-concept="{e(" ".join([item["title"], item["short"], *item.get("aliases", [])]).lower())}" data-category="{e(item["category"])}"><a class="guidecard" href="{key}.html"><span class="eyebrow">{e(item["category"])}{draft}</span><h2>{e(item["title"])} <span>↗</span></h2><p>{e(item["summary"])}</p>{state}</a></li>''',
            )
        )
        models = "".join(
            f'<li><a href="../models/{e(stem)}.html#concept-{key}"><b>{e(rec["id"].split("/")[-1])}</b><span>{e(model_matches[stem][key]["scope"])}</span><span>{e(STATUS_WORDS[model_matches[stem][key]["status"]])} · inspect supporting fields →</span></a></li>'
            for stem, rec, pub in matches
        )
        criteria = json.dumps(item["rule"], sort_keys=True, indent=2)
        body = f'''<nav class="crumbs" aria-label="Breadcrumb"><a href="../concepts/index.html">Concept guide</a> / {e(item["short"])}</nav>
<div class="guidehero"><p class="eyebrow">{e(item["category"])} / Field guide</p><h1>{e(item["title"])}</h1>{draft and f'<p class="draft-note">{DRAFT} Not yet reviewed by the maintainer.</p>'}<p class="lead">{e(item["summary"])}</p><p class="byline">An original ArchWatch explanation. The rule below associates models from approved source fields; this guide never supplies model facts.</p></div>
<nav class="lesson-nav" aria-label="Guide sections"><a href="#why">Motivation</a><a href="#worked-example">Worked example</a><a href="#equation">Equation</a><a href="#pitfalls">Pitfalls</a><a href="#verify">Verify</a><a href="#examples">Model evidence</a></nav>
<div class="guide-columns lesson-columns"><article class="prose lesson-prose">{teaching_sections(key, item, lesson_data[key], published)}<p class="primary-source">Read the primary source ↗<br><a href="{e(item["source"]["url"])}">{e(item["source"]["title"])}</a></p><details class="concept-rule"><summary>Inspect the matching rule</summary><p>Scope: {e(item["scope"])}</p><pre>{e(criteria)}</pre><p>Additional semantic guards enforce scope, head relationships and nested operations. Every supporting field needs a current passing verifier decision. Missing fields never establish absence.</p></details></article>
<aside class="panel lesson-examples" id="examples"><p class="eyebrow">Explore the idea in practice</p><h2>Published examples</h2><ul class="guide-models">{models or "<li>No qualified published example yet. This is a guide, not a claim that the supported models lack this mechanism.</li>"}</ul><p class="hint">A one-source association remains labelled. Held architectures are excluded. <a href="../review/index.html">Inspect the review queue →</a></p></aside></div>'''
        pages[key + ".html"] = frame(
            f"{item['title']} · ArchWatch guide", body, depth=1, active="concepts", description=item["summary"]
        )
    categories = "".join(
        f'<option value="{e(category)}">{e(category)}</option>'
        for category in sorted({item["category"] for item in guide.values()})
    )
    body = f"""<p class="eyebrow">The concept guide</p><h1>Follow the moving parts.</h1><p class="lead">{len(guide)} concepts and companion topics. Follow a mechanism from its explanation to a matching rule, then inspect the primary evidence in a published model.</p><p class="hint">Guides cover more mechanisms than the current model collection. An empty example list means not yet qualified, not absent.</p><div class="concept-filters"><label for="concept-search">Find a concept<input type="search" id="concept-search" placeholder="Try RoPE, MTP, normalization…"></label><label for="concept-category">Category<select id="concept-category"><option value="">All categories</option>{categories}</select></label></div><p class="hint" id="concept-count" role="status" aria-live="polite">{len(guide)} guides</p><ul class="guide-grid">{"".join(card for _, _, card in sorted(cards))}</ul><p id="concept-empty" hidden>No guide matches this search. Try a different name or category.</p><p><a download href="../data/concepts.json">Download the concept registry and rules ↓</a></p>"""
    pages["index.html"] = frame("LLM architecture concept guide · ArchWatch", body, depth=1, active="concepts")
    return pages


def review_page(published, held, seen):
    counts = Counter()
    layers = Counter()
    for _, rec, pub in published.values():
        for check in rec.get("checks", []):
            if check["field"] in pub:
                counts[card_status(check)[0]] += 1
        layers.update(pub["layer_type_counts"])
    total = sum(counts.values())
    evidence = "".join(
        f'<li><span>{e(status)}</span><b class="mono">{n}</b><span class="track"><span class="fill" style="width:{100 * n / max(1, total):.1f}%"></span></span></li>'
        for status, n in sorted(counts.items())
    )
    mix = "".join(
        f'<li><span><i class="sw" style="background:{COLOR.get(key, "#999")}"></i>{e(LABEL.get(key, key))}</span><b class="mono">{n}</b></li>'
        for key, n in sorted(layers.items(), key=lambda pair: (-pair[1], pair[0]))
    )
    pending = "".join(
        f'<tr><th scope="row"><a href="https://huggingface.co/{e(repo)}">{e(repo)} ↗</a></th><td>{e(reason)}</td><td>{e(seen.get(repo, {}).get("status", "review"))}</td></tr>'
        for repo, reason in sorted(held.items())
    )
    body = f'''<p class="eyebrow">Evidence & coverage</p><h1>Show the work.<br><em>Keep uncertainty visible.</em></h1><p class="lead">A collected model is not automatically a published architecture. Required facts and the layer pattern must pass the publication rules first.</p>
<dl class="collection-stats"><div><dt>published models</dt><dd>{len(published):02d}</dd></div><div><dt>held models</dt><dd>{len(held):02d}</dd></div><div><dt>published field checks</dt><dd>{total}</dd></div></dl>
<div class="pair"><section class="panel"><h2>Where the evidence stands</h2><p class="hint">Field checks in published models. Counts describe evidence coverage, not accuracy.</p><ul class="evidence-bars">{evidence}</ul></section><section class="panel"><h2>Inside the collection</h2><p class="hint">Published decoder layers, counted once per model record. Related releases can share a design.</p><ul class="coverage-list">{mix}</ul></section></div>
<section><h2>How a fact earns its label</h2><div class="method-grid"><article><span class="st st-verified">Verified</span><h3>Agreement across sources</h3><p>Two source types support the value. Follow their pinned links and inspect the evidence yourself.</p></article><article><span class="st st-one-source">One source</span><h3>Useful, with a limit</h3><p>A published value has one supporting source. It is labelled separately from cross-checked values.</p></article><article><span class="st st-computed">Computed</span><h3>Show the calculation</h3><p>Memory and derived values use published inputs. They are estimates under stated assumptions.</p></article></div></section>
<section><h2>In the review queue</h2><p class="sub">Names and hold reasons only. Unresolved architecture values are excluded from model pages, comparisons and downloads.</p><div class="tablebox"><table><thead><tr><th scope="col">Repository</th><th scope="col">Why it is held</th><th scope="col">Watcher state</th></tr></thead><tbody>{pending or '<tr><td colspan="3">Nothing is held.</td></tr>'}</tbody></table></div></section>
<section class="panel prose"><h2>Know the boundaries</h2><p>This collection covers supported primary model releases, not every model on the Hub. A newer release can remain in review while an older supported model is explorable. Fine-tunes and conversions are outside this gallery's architecture scope.</p><p>Header parameter counts measure stored tensors under the project's counting convention. Multi-token prediction modules may be reported separately. Memory estimates omit runtime allocations, quantization metadata and device-specific behavior.</p><p>Concept guides explain mechanisms; they do not supply facts to the extractor. Generated diagrams arrange published data and pass deterministic checks before selection. A checked diagram is still a representation, not independent verification of the source model.</p><a href="{e(settings()["repo_url"])}/tree/main/evals">Inspect gold labels and evaluation methodology ↗</a></section>'''
    return frame("Evidence, review queue and coverage · ArchWatch", body, depth=1, active="review")


def published_export(rec, pub):
    return {
        "id": rec["id"],
        "source_commit": rec["source_commit"],
        "hub_created_at": rec.get("hub_created_at"),
        "analyzed_at": rec.get("analyzed_at"),
        "published": pub,
        "evidence": {
            c["field"]: {
                "status": card_status(c)[0],
                "sources": sorted({u for x in c.get("candidates", []) if (u := source_url(rec, x["evidence"]))}),
            }
            for c in rec.get("checks", [])
            if c["field"] in pub
        },
        "memory_assumptions": render_cache(pub).get("assumptions", []),
        "concept_evidence": concept_data.evidence_spec(rec),
    }


def analysis_feed(published):
    base = settings().get("site_url", "").rstrip("/")
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    for key, value in [
        ("title", "ArchWatch analysis feed"),
        ("link", base),
        ("description", "Published architecture analyses. Dates are analysis times, not model release dates."),
    ]:
        ET.SubElement(channel, key).text = value
    for stem, rec, pub in sorted(
        published.values(), key=lambda row: (row[1].get("analyzed_at", ""), row[1]["id"]), reverse=True
    ):
        item = ET.SubElement(channel, "item")
        link = f"{base}/models/{stem}.html"
        for key, value in [
            ("title", rec["id"].split("/")[-1]),
            ("link", link),
            ("description", describe(pub)),
            ("guid", f"{link}#{rec['source_commit']}"),
        ]:
            ET.SubElement(item, key).text = value
        stamp = rec.get("analyzed_at")
        if stamp:
            moment = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            ET.SubElement(item, "pubDate").text = moment.strftime("%a, %d %b %Y %H:%M:%S %z")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(rss, encoding="unicode") + "\n"


CSS = (CODE / "web/style.css").read_text()

JS = (CODE / "web/site.js").read_text()

COMPARE_JS = (CODE / "web/compare.js").read_text()


def main():
    published, held, seen = inventory()
    site = ROOT / "site"
    write_if_changed(site / "style.css", CSS.lstrip())
    write_if_changed(site / "site.js", JS.lstrip())
    write_if_changed(site / "favicon.svg", (CODE / "web/favicon.svg").read_text())
    write_if_changed(site / "feed.xml", analysis_feed(published))
    write_if_changed(site / "index.html", home(published, held, seen))
    write_if_changed(
        site / "data/concepts.json",
        json.dumps({"schema_version": "archwatch-concept-registry/1", "concepts": concepts()}, indent=2, sort_keys=True)
        + "\n",
    )
    guides = guide_pages(published)
    for name, content in guides.items():
        write_if_changed(site / "concepts" / name, content)
    for key, lesson in lessons().items():
        write_if_changed(site / "concepts/figures" / f"{key}.svg", concept_diagrams.render(key, lesson))
    write_if_changed(site / "review/index.html", review_page(published, held, seen))
    data = compare_data(published)
    write_if_changed(site / "data" / "models.json", json.dumps(data, indent=1, sort_keys=True) + "\n")
    write_if_changed(site / "compare.js", COMPARE_JS.lstrip())
    pairs, expected = [], {"index.html"}
    for repo, (stem, rec, pub) in published.items():
        write_if_changed(site / "models" / f"{stem}.html", model_page(stem, rec, pub, published))
        write_if_changed(
            site / "data/models" / f"{stem}.json",
            json.dumps(published_export(rec, pub), indent=2, sort_keys=True) + "\n",
        )
        write_if_changed(
            site / "data/concepts" / f"{stem}.json",
            json.dumps(concept_data.evidence_spec(rec), indent=2, sort_keys=True) + "\n",
        )
        before = published_predecessor(rec, published)
        if before:
            write_if_changed(site / "compare" / f"{stem}.html", compare_tool(data, (before[0], stem)))
            pairs.append(stem)
            expected.add(f"{stem}.html")
    write_if_changed(site / "compare" / "index.html", compare_tool(data))
    # pages of records that are no longer published are removed, so held numbers never stay online
    stems = {stem for stem, _, _ in published.values()}
    for folder, keep in (("models", {f"{s}.html" for s in stems}), ("compare", expected), ("concepts", set(guides))):
        for path in (site / folder).glob("*.html"):
            if path.name not in keep:
                path.unlink()
    for folder in ("models", "concepts"):
        for path in (site / "data" / folder).glob("*.json"):
            if path.stem not in stems:
                path.unlink()
    approved = {f"{s}.svg" for s, rec, _ in published.values() if generated.select(rec)}
    for path in (site / "diagrams").glob("*.svg") if (site / "diagrams").exists() else []:
        if path.name not in approved:
            path.unlink()
    print(f"site: {len(published)} model pages, {len(pairs)} compare pages, {len(held)} held")


if __name__ == "__main__":
    main()
