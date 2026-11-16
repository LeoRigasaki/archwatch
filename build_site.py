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
import html
import json
import math
import re
from pathlib import Path

import yaml

from archwatch import diagram, generated, pipeline
from archwatch.artifacts import write_if_changed
from archwatch.explainer import diff_input, input_hash
from archwatch.labels import KIND_LABELS
from archwatch.palette import colors
from archwatch.publication import PublicationError, plain_hold, render_cache, render_values
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
            ("method", f"{up}index.html#method", "Method"),
            ("github", repo, "GitHub"),
        )
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><meta name="description" content="{e(description or title)}">
<script>try{{var t=localStorage.getItem('aw-theme');if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap">
<link rel="stylesheet" href="{up}style.css"></head>
<body><a class="skip" href="#main">Skip to content</a>
<header class="top"><div class="wrap topbar"><a class="brand" href="{up}index.html">ArchWatch</a><nav aria-label="Main">{nav}</nav>
<span class="lastrun">{e(last_run())}</span>
<button type="button" class="themebtn" aria-pressed="false" aria-label="Dark theme"><span class="knob" aria-hidden="true"></span>Dark</button></div></header>
<main id="main" class="wrap">{body}</main>
<footer id="method" class="foot"><div class="wrap"><p>Every number is read from the model's own files at a pinned
commit (config.json, weight file headers, modeling code, model card) and labelled verified, one source or
computed. Verified means two independent files agree; a disagreement is held for a person to resolve. Cache and
memory figures come from ArchWatch's calculator. Diagrams are drawn by Gemini from verified fields and checked by
code. Rebuilt daily by GitHub Actions on free tiers. <a href="{e(repo)}">View the code</a>.</p></div></footer>
<script src="{up}site.js" defer></script></body></html>"""


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
        design=design(pub).lower(),
        hybrid="1" if hybrid(pub) else "0",
        layers=" ".join(flags),
        params=int(pub["params_total"]),
        date=rec["hub_created"][:10],
        name=name.lower(),
        search=" ".join([rec["id"], pubs.get(org, org), design(pub), mix_text(pub), describe(pub)]).lower(),
    )
    attrs = " ".join(f'data-{k}="{e(v)}"' for k, v in data.items())
    before = published_predecessor(rec, published)
    compare = (
        f'<a class="btn" href="compare/{e(stem)}.html">Compare with {e(before[1]["id"].split("/")[-1])}</a>'
        if before
        else ""
    )
    return f"""<li class="card" {attrs}><article aria-labelledby="h-{e(stem)}">
<div class="cardhead"><span>{e(pubs.get(org, org))}</span><time datetime="{e(rec["hub_created"][:10])}">{e(date(rec["hub_created"]))}</time></div>
<h3 id="h-{e(stem)}"><a href="models/{e(stem)}.html">{e(name)}</a> <span class="tag">{e(design(pub))}</span></h3>
<p class="line">{e(describe(pub))}</p>
{cells(pub, "Layer order, layer 1 on the left: " + mix_text(pub))}<p class="mixcap">{e(mix_text(pub))}</p>
<dl class="mini"><div><dt>parameters</dt><dd>{e(human(pub["params_total"]))}</dd></div>
<div><dt>layers</dt><dd>{pub["num_layers"]}</dd></div><div><dt>context</dt><dd>{e(tokens(pub["context_length"]))}</dd></div>
<div><dt>{e(cache_value(pub)[1])}</dt><dd>{e(cache_value(pub)[0])}</dd></div></dl>
<div class="cardbtns">{compare}<a class="btn" href="models/{e(stem)}.html#fit">Will it fit?</a><a class="btn" href="models/{e(stem)}.html">Open</a></div>
</article></li>"""


def table_row(stem, rec, pub, pubs):
    org, name = rec["id"].split("/")
    return (
        f'<tr><th scope=row><a href="models/{e(stem)}.html">{e(name)}</a><span class="detail">{e(pubs.get(org, org))} · '
        f"{e(date(rec['hub_created']))}</span></th><td>{e(design(pub))}</td><td class=mono>{e(human(pub['params_total']))}</td>"
        f"<td class=mono>{pub['num_layers']}</td><td class=mono>{e(tokens(pub['context_length']))}</td>"
        f"<td class=mono>{e(cache_value(pub)[0])}{'' if cache_value(pub)[1] == 'cache / token' else '<span class=detail>' + e(cache_value(pub)[1].replace('cache, ', '')) + '</span>'}</td><td>{cells(pub, mix_text(pub))}</td></tr>"
    )


def home(published, held, seen):
    pubs = settings()["publishers"]
    order = sorted(published.values(), key=lambda x: (x[1]["hub_created"] or "", x[1]["id"]), reverse=True)
    cards = "".join(model_card(s, r, p, published, pubs) for s, r, p in order)
    rows = "".join(table_row(s, r, p, pubs) for s, r, p in order)
    newest = "".join(
        f'<li><span class="mono">{e(date(r["hub_created"], year=False))}</span> <a href="models/{e(s)}.html">'
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
    held_rows = "".join(f"<li><b>{e(repo.split('/')[-1])}</b><span>{e(r)}</span></li>" for repo, r in held.items())
    body = f"""<section class="hero"><h1>Find, compare and size any open LLM architecture.</h1>
<p class="lead">Read from each model's own files at a pinned commit. Every number is labelled with where it came from.</p>
<div class="searchrow"><label class="sr" for="q">Search models</label>
<input id="q" type="search" placeholder="Search a model, lab or idea: Kimi, Liquid, sliding window" autocomplete="off"></div></section>
<section class="newbar" aria-label="Newest releases"><b>Newest</b><ul>{newest}</ul></section>
<div class="explore">
<aside class="filters" aria-label="Filters"><form id="filters">
<fieldset><legend>Design</legend><ul>{designs}</ul></fieldset>
<fieldset><legend>Layer types inside</legend><ul>{layers}</ul></fieldset>
<fieldset><legend>Size</legend><ul>{sizes}</ul></fieldset>
<fieldset><legend>Released</legend><ul>
<li><label><input type="radio" name="released" value="any" checked>Any time</label></li>
<li><label><input type="radio" name="released" value="6m">Last 6 months</label></li>
<li><label><input type="radio" name="released" value="year">This year</label></li></ul></fieldset>
<button type="reset" class="linkbtn">Reset filters</button></form></aside>
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
<p class="empty" hidden>No model matches these filters.</p>
<details class="heldbox"><summary>{len(held)} models found but not yet published</summary>
<p class="sub">Analysed, but not published until every rule for them is checked. Fine-tunes are out of scope and not listed.</p>
<ul>{held_rows}</ul></details>
</section></div>"""
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


def fact_row(label, key, value, c):
    status, note = card_status(c) if c else ("computed", None)
    word = STATUS_WORDS.get(status, status)
    sources = (
        ", ".join(dict.fromkeys(SOURCE_WORDS.get(x["evidence"].split(":")[0], "") for x in c["candidates"]))
        if c
        else "computed from the facts above"
    )
    extra = f'<span class="note">{e(note)}; recorded, not a gate</span>' if note else ""
    return (
        f'<tr><th scope=row>{e(label)}<span class="key">{e(key)}</span></th><td class="mono">{e(value)}</td>'
        f'<td>{e(sources)}{extra}</td><td><span class="st st-{word.replace(" ", "-")}">{e(word)}</span></td></tr>'
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
                label or PLAIN.get(field, field),
                config_key(checks[field]),
                value or show(field, pub[field]),
                checks[field],
            )
        )

    for field in KEY_FACTS:
        if field == "attention":
            if "num_attention_heads" in pub:
                kv = f" ({pub['num_kv_heads']} key-value)" if "num_kv_heads" in pub else ""
                value = f"{pub.get('attention_type', '')} · {pub['num_attention_heads']} heads{kv}".lstrip(" ·")
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
        kv = cache["grows_per_token_bytes"] * n + cache["constant_bytes"]
        for key, _, per in PRECISION:
            weights = pub["params_total"] * per
            total = weights + kv
            combos[f"{n}-{key}"] = dict(
                total=size(total),
                weights=size(weights),
                cache=size(kv),
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
<p class="sub">Weights plus the cache that grows with context, for one conversation. Runtime overhead is not included.</p>
<div class="choice"><span>Context length</span><div class="seg" role="group" aria-label="Context length">{ctx_btns}</div></div>
<div class="choice"><span>Weight precision</span><div class="seg" role="group" aria-label="Weight precision">{prec_btns}</div></div>
<div class="fitsum" aria-live="polite"><p class="big mono"><span data-out="total">{e(pick["total"])}</span></p>
<p class="mono split2">weights <span data-out="weights">{e(pick["weights"])}</span><br>cache <span data-out="cache">{e(pick["cache"])}</span></p></div>
<ul class="devices">{bars}</ul>
<p class="sub">Weights = {pub["params_total"]:,} parameters × bytes per parameter. {e(cache_rule(pub, cache))}, bf16. "Tight" means over 75% of the device's memory.</p></section>"""


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
            ("facts", "Facts and sources"),
        )
    )
    body = f"""<nav class="crumbs" aria-label="Breadcrumb"><a href="../index.html">Explore</a> / {e(pubs.get(org, org))}</nav>
<div class="titlerow"><div><h1>{e(name)}</h1>
<p class="sub"><a href="https://huggingface.co/{e(rec["id"])}">{e(rec["id"])}</a> · released {e(date(rec["hub_created"]))} ·
read at commit <a class="mono" href="https://huggingface.co/{e(rec["id"])}/tree/{e(commit)}">{e(commit[:10])}</a></p></div>
<div class="actions"><span class="badge">{e(badge)}</span>{compare}</div></div>
<nav class="tabs" aria-label="On this page">{tabs}</nav>
<dl class="stats tiles">{t_html}</dl>
<section id="short" aria-labelledby="short-h"><h2 id="short-h">In short</h2><p class="prose">{e(describe(pub))}</p>
<p class="byline">Written by code from the published facts.</p></section>
{changed_html}
{layer_section(pub)}
<section id="diagram" aria-labelledby="arch"><div class="sechead"><h2 id="arch">Diagram</h2><p class="hint">{e(art_label)}{art_link}</p></div>
<figure class="panel diagram">{art}</figure></section>
{fit_planner(pub, cache)}
<section id="facts" aria-labelledby="facts-h"><h2 id="facts-h">Facts and where they come from</h2>
<p class="sub">Verified means two independent files agree.</p>{facts(rec, pub, cache)}</section>"""
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
        name=name,
        publisher=settings()["publishers"].get(org, org),
        date=date(rec["hub_created"]),
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
        kv = f" ({pub['num_kv_heads']} key-value)" if "num_kv_heads" in pub else ""
        kind = f"{pub['attention_type']} · " if pub.get("attention_type") else ""
        out["attention"] = [f"{kind}{pub['num_attention_heads']} heads{kv}", None]
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
    max_context = max(int(p["context_length"]) for _, _, p in published.values())
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
        f'<div class="slot" data-slot="{slot}"><label for="pick-{slot}">{label}</label>'
        f'<input id="pick-{slot}" list="models-list" autocomplete="off" placeholder="Search models">'
        f'<span class="slotmeta"></span></div>'
    )
    preset = f' data-a="{e(default[0])}" data-b="{e(default[1])}"' if default else ""
    body = f"""<h1>Compare</h1>
<noscript><p class="lead">The compare tool needs JavaScript. Each model next to its predecessor:</p><ul class="plainlist">{nojs}</ul></noscript>
<div id="tool" class="tool" hidden{preset}>
<section aria-labelledby="sugg"><h2 id="sugg" class="sr">Suggested comparisons</h2><ul class="suggest">{chips}</ul></section>
<datalist id="models-list">{options}</datalist>
<form class="panel pickers" id="pickers" aria-label="Models to compare">
{pick("a", "Model A")}<button type="button" class="btn swap" id="swap" aria-label="Swap model A and model B">Swap</button>{pick("b", "Model B")}
{pick("c", "Model C (optional)")}<div class="pickact"><button type="button" class="btn" id="clear-c">Remove C</button>
<button type="button" class="btn" id="copylink">Copy link</button><span class="sub" id="copied" aria-live="polite"></span></div></form>
<section class="panel" aria-labelledby="kd"><h2 id="kd">Key differences</h2><div id="keydiff"></div></section>
<section aria-labelledby="ls"><h2 id="ls">Layer by layer</h2><p class="sub">Every strip has the same width; layer 1 is on the left.</p><div id="strips"></div></section>
<section class="panel" aria-labelledby="cc"><h2 id="cc">Cache as context grows</h2>
<p class="sub">bf16, one sequence. Growing layers rise with context, sliding-window layers stop at their window, fixed-size state is a flat floor. Each line ends at the model's maximum context.</p>
<div id="chart"></div><details><summary>Show the numbers</summary><div id="chartdata" class="tablebox"></div></details></section>
<section class="panel" aria-labelledby="fit-h"><h2 id="fit-h">Will they fit?</h2>
<p class="sub">Weights plus cache for one conversation; runtime overhead not included. "Tight" means over 75% of the device's memory.</p>
<div class="choice"><span>Context length</span><div class="seg" role="group" aria-label="Context length" id="ctxseg"></div></div>
<div class="choice"><span>Weight precision</span><div class="seg" role="group" aria-label="Weight precision" id="precseg"></div></div>
<div id="fit-compare" aria-live="polite"></div></section>
<section aria-labelledby="fbf"><div class="sechead"><h2 id="fbf">Field by field</h2>
<label class="toggle"><input type="checkbox" id="onlydiff"> Only show differences</label></div>
<div class="tablebox" id="fields"></div></section>
</div>"""
    page = frame("Compare · ArchWatch", body, depth=1, active="compare")
    return page.replace("</body>", '<script src="../compare.js" defer></script></body>')


# ---------------------------------------------------------------- assets

CSS = """
:root{--bg:#F7F7F5;--fg:#111418;--mut:#5b6169;--line:#e3e3df;--card:#fff;--acc:#0B6E66;--soft:#efefeb;--accsoft:#0B6E6618;
--warn:#f3e6c4;--warnfg:#7a5a00;--bad:#f8dcd7;--badfg:#9b2c1f;--new:#dde8f7;--newfg:#244c86;--diff:#ece4fb;--difffg:#5b34b0;color-scheme:light}
:root[data-theme=dark]{--bg:#111418;--fg:#e9ebee;--mut:#9aa1a9;--line:#2a2f36;--card:#181c21;--acc:#4fd1c0;--soft:#1f242a;--accsoft:#4fd1c022;
--warn:#3a3020;--warnfg:#e8c46a;--bad:#3d2421;--badfg:#f2a196;--new:#1e2a3d;--newfg:#9cc0f2;--diff:#2b2340;--difffg:#c4b0f5;color-scheme:dark}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111418;--fg:#e9ebee;--mut:#9aa1a9;--line:#2a2f36;--card:#181c21;--acc:#4fd1c0;--soft:#1f242a;--accsoft:#4fd1c022;
--warn:#3a3020;--warnfg:#e8c46a;--bad:#3d2421;--badfg:#f2a196;--new:#1e2a3d;--newfg:#9cc0f2;--diff:#2b2340;--difffg:#c4b0f5;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 Inter,system-ui,sans-serif}
h1,h2,h3{font-family:"Space Grotesk",Inter,sans-serif;letter-spacing:-.01em;line-height:1.15;margin:0}
h1{font-size:clamp(32px,4.4vw,48px);font-weight:600}h2{font-size:24px;font-weight:600}h3{font-size:20px;font-weight:600}
a{color:var(--acc)}a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.mono,.stats dd,.mini dd,.count{font-family:"IBM Plex Mono",ui-monospace,monospace}
.wrap{max-width:1200px;margin:0 auto;padding:0 20px}
.skip{position:absolute;left:-999px}.skip:focus{left:12px;top:12px;background:var(--card);padding:8px;z-index:9}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.top{background:var(--card);border-bottom:1px solid var(--line)}
.topbar{display:flex;align-items:center;gap:28px;height:60px}
.brand{font:600 20px "Space Grotesk",sans-serif;color:var(--fg);text-decoration:none}
.top nav{display:flex;gap:22px}.top nav a{color:var(--mut);text-decoration:none;font-size:15px}
.top nav a[aria-current]{color:var(--fg);font-weight:500}
.lastrun{margin-left:auto;font-size:13px;color:var(--mut)}
.themebtn{font:inherit;font-size:14px;color:var(--mut);background:none;border:0;display:flex;align-items:center;gap:8px;cursor:pointer}
.knob{width:34px;height:20px;border-radius:99px;background:var(--line);position:relative}
.knob::after{content:"";position:absolute;top:3px;left:3px;width:14px;height:14px;border-radius:50%;background:var(--card);transition:left .15s}
.themebtn[aria-pressed=true] .knob{background:var(--acc)}.themebtn[aria-pressed=true] .knob::after{left:17px}
main.wrap{padding:40px 20px 72px}
.lead{font-size:18px;color:var(--mut);max-width:640px;margin:16px 0 24px}
.sub{color:var(--mut);margin:6px 0 14px}.hint{color:var(--mut);font-size:14px;margin:0}.byline{font-size:13px;color:var(--mut)}
.searchrow input{width:100%;max-width:520px;font:inherit;padding:12px 16px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.newbar{display:flex;align-items:center;gap:24px;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 20px;margin:28px 0}
.newbar ul{list-style:none;display:flex;gap:28px;flex-wrap:wrap;margin:0;padding:0;font-size:15px}.newbar .mono{color:var(--mut);font-size:13px}
.newbar a{color:var(--fg);text-decoration:none}.newbar a:hover{text-decoration:underline}
.explore{display:grid;grid-template-columns:200px 1fr;gap:28px;align-items:start}
.filters fieldset{border:0;margin:0 0 22px;padding:0}.filters legend{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin-bottom:8px;font-weight:600}
.filters ul{list-style:none;margin:0;padding:0}.filters li{display:flex;justify-content:space-between;font-size:14.5px;padding:3px 0}
.filters label{display:flex;gap:8px;align-items:center;cursor:pointer}.filters input{accent-color:var(--acc);width:16px;height:16px;margin:0}
.count{color:var(--mut);font-size:13px}.linkbtn{background:none;border:0;padding:0;font:inherit;font-size:14px;color:var(--acc);text-decoration:underline;cursor:pointer}
.listhead{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:16px}
.tools{display:flex;align-items:center;gap:10px;font-size:14px;color:var(--mut)}
.tools select{font:inherit;font-size:14px;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.seg{display:inline-flex;gap:6px;flex-wrap:wrap}.seg button,.btn{font:inherit;font-size:14px;padding:6px 14px;border-radius:99px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;text-decoration:none;display:inline-block}
.seg button[aria-pressed=true]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.cards{list-style:none;padding:0;margin:0 0 28px;display:grid;grid-template-columns:repeat(auto-fill,minmax(min(330px,100%),1fr));gap:18px}
.card article{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;height:100%;display:flex;flex-direction:column}
.cardhead{display:flex;justify-content:space-between;font-size:13px;color:var(--mut)}
.card h3{margin:10px 0 10px;font-size:20px}.card h3 a{color:var(--fg);text-decoration:none}.card h3 a:hover{color:var(--acc)}
.tag{font:500 12px Inter,sans-serif;background:var(--soft);color:var(--fg);padding:2px 9px;border-radius:99px;vertical-align:3px;margin-left:6px}
.line{margin:0 0 14px;font-size:15px}
.cells{display:flex;gap:2px;height:12px;overflow:hidden}.cells i{flex:1 1 0;border-radius:2px;min-width:0}
@media (max-width:860px){.cells{gap:1px}}
.mixcap{font-size:12.5px;color:var(--mut);margin:6px 0 12px}
.mini{display:grid;grid-template-columns:repeat(4,1fr);margin:auto 0 14px;padding-top:12px;border-top:1px solid var(--line)}
.mini div,.stats>div{display:flex;flex-direction:column-reverse}.mini dd{margin:0;font-size:15px}.mini dt{font-size:12px;color:var(--mut)}
.cardbtns{display:flex;gap:8px;flex-wrap:wrap}
.empty{color:var(--mut)}
.heldbox{margin-top:28px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 20px}
.heldbox summary{cursor:pointer;font-weight:500}.heldbox ul{list-style:none;padding:0;margin:0}
.heldbox li{display:grid;grid-template-columns:minmax(160px,30%) 1fr;gap:16px;padding:9px 0;border-top:1px solid var(--line);font-size:14.5px}
.heldbox li span{color:var(--mut)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin:24px 0 40px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.stats>div{padding:18px 22px;border-right:1px solid var(--line)}.stats>div:last-child{border-right:0}
.stats dd{margin:0;font-size:26px;font-weight:500}.stats dt{color:var(--mut);font-size:14px}
.sechead{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin:0 0 16px}
.tablebox{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow-x:auto;margin-bottom:48px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px 18px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child>*{border-bottom:0}thead th{background:var(--soft);font-size:14px;font-weight:600}tbody th{font-weight:400}
.facts td.mono{overflow-wrap:anywhere}.key,.detail,.note{display:block;font-size:12px;color:var(--mut);font-family:"IBM Plex Mono",monospace;overflow-wrap:anywhere}
.note{color:var(--difffg);font-family:Inter,sans-serif}.detail{font-family:Inter,sans-serif}
.tableview .cells{min-width:160px}
.allfields{border-top:1px solid var(--line)}.allfields summary{padding:12px 18px;cursor:pointer;font-weight:500}
.st,.chip{display:inline-block;font-size:13px;padding:1px 10px;border-radius:99px;white-space:nowrap}
.st-verified,.chip-up{background:var(--accsoft);color:var(--acc)}.st-one-source,.st-computed,.chip-warn{background:var(--warn);color:var(--warnfg)}
.st-differs-from-card{background:var(--diff);color:var(--difffg)}.st-held-for-review{background:var(--bad);color:var(--badfg)}
.chip-same{background:var(--soft);color:var(--mut)}.chip-new{background:var(--new);color:var(--newfg)}
.crumbs{font-size:14px;color:var(--mut);margin-bottom:8px}
.titlerow{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap}
.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.badge{background:var(--accsoft);color:var(--acc);padding:6px 14px;border-radius:99px;font-size:14px;font-weight:500}
.tabs{display:flex;gap:22px;flex-wrap:wrap;margin:22px 0 0;padding:10px 18px;background:var(--card);border:1px solid var(--line);border-radius:10px;font-size:14px}
.tabs a{text-decoration:none}
section{margin-bottom:44px}section h2{margin-bottom:10px}.prose{max-width:760px;font-size:16.5px}
.changes{padding-left:20px;max-width:820px}.changes li{margin:4px 0}
.legendboxes{list-style:none;display:flex;gap:10px;flex-wrap:wrap;padding:0;margin:14px 0 0}
.legendboxes li{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px;font-size:14.5px}
#layers .cells{height:28px;gap:3px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px 24px}
.diagram{margin:0;overflow-x:auto;background:#fff}.diagram img,.diagram svg{display:block;width:100%;height:auto;min-width:720px}
.choice{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:10px 0}.choice>span{width:130px;font-size:14px;color:var(--mut)}
.fitsum{display:flex;align-items:flex-end;gap:28px;flex-wrap:wrap;margin:14px 0}.big{font-size:44px;margin:0;line-height:1}
.split2{font-size:14px;color:var(--mut);margin:0}
.devices{list-style:none;padding:0;margin:8px 0}.devices li{display:grid;grid-template-columns:150px 1fr 90px;gap:14px;align-items:center;padding:4px 0;font-size:14px}
.track{height:8px;background:var(--soft);border-radius:99px;overflow:hidden}.fill{display:block;height:100%;background:var(--acc)}
.verdict{text-align:center;border-radius:99px;font-size:12.5px;padding:1px 8px}.v-fits{background:var(--accsoft);color:var(--acc)}
.v-tight{background:var(--warn);color:var(--warnfg)}.v-too-big{background:var(--bad);color:var(--badfg)}
.cachebars li{grid-template-columns:220px 1fr 110px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:24px 0}.ccard h2{font-size:20px;margin:0}.ccard h2 a{color:var(--fg);text-decoration:none}
.legend{list-style:none;padding:0;margin:10px 0 0;display:flex;gap:16px;flex-wrap:wrap;font-size:13.5px}
.toggle{font-size:14px;color:var(--mut);display:flex;gap:8px;align-items:center}
.plainlist{font-size:18px}
.foot{border-top:1px solid var(--line);background:var(--card);color:var(--mut);font-size:14px;padding:22px 0}.foot p{margin:0;max-width:900px}
.explore>*,.pair>*,.titlerow>*{min-width:0}h1,.sub{overflow-wrap:anywhere}
[hidden]{display:none!important}
.suggest{list-style:none;padding:0;margin:20px 0 18px;display:flex;flex-wrap:wrap;gap:10px}
.chipbtn{display:inline-flex;flex-direction:column;padding:8px 14px;border:1px solid var(--line);border-radius:12px;background:var(--card);color:var(--fg);text-decoration:none;font-size:14px;font-weight:500}
.chipbtn span{font-size:12px;color:var(--mut);font-weight:400}.chipbtn:hover{border-color:var(--acc)}
.pickers{display:grid;grid-template-columns:1fr auto 1fr 1fr;gap:16px;align-items:end;margin-bottom:28px}
.slot{display:flex;flex-direction:column;gap:6px;border-left:4px solid var(--slot,var(--line));padding-left:12px}
.slot label{font-size:13px;color:var(--mut)}.slot input{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);min-width:0}
.slotmeta{font-size:12.5px;color:var(--mut);min-height:1.2em}.swap{align-self:center}
.pickact{grid-column:1/-1;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#strips{display:grid;gap:14px}.stripcard h3{font-size:18px}.stripcard h3 a{color:var(--fg);text-decoration:none}
.stripcard .cells{height:22px;margin:12px 0 8px}
#chart svg{width:100%;height:auto;display:block}#chart .grid{stroke:var(--line);stroke-width:1}
#chart .axis{fill:var(--mut);font:12px Inter,sans-serif}#chart .series{fill:none;stroke-width:2.5}
.fitrow{padding:12px 0;border-top:1px solid var(--line)}.fithead{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
@media (max-width:860px){.pickers{grid-template-columns:1fr}.swap{justify-self:start}}
@media (max-width:860px){.explore,.pair{grid-template-columns:1fr}.lastrun{display:none}.topbar{gap:16px;overflow-x:auto}
.stats>div{border-right:0;border-bottom:1px solid var(--line)}.big{font-size:34px}.mini{grid-template-columns:repeat(2,1fr);row-gap:8px}
.devices li,.cachebars li{grid-template-columns:110px 1fr 80px}.choice>span{width:auto}}
"""

JS = """
(function(){var root=document.documentElement,btn=document.querySelector('.themebtn');
  function cur(){return root.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}
  if(btn){btn.setAttribute('aria-pressed',cur()==='dark');btn.addEventListener('click',function(){
    var t=cur()==='dark'?'light':'dark';root.dataset.theme=t;btn.setAttribute('aria-pressed',t==='dark');
    try{localStorage.setItem('aw-theme',t)}catch(e){}});}
})();
(function(){var p=document.getElementById('fit');if(!p||!p.dataset.combos)return;var combos=JSON.parse(p.dataset.combos),ctx,prec;
  function sel(attr){var b=p.querySelector('['+attr+'][aria-pressed=true]');return b.getAttribute(attr)}
  function draw(){var c=combos[sel('data-ctx')+'-'+sel('data-prec')];['total','weights','cache'].forEach(function(k){
      p.querySelector('[data-out='+k+']').textContent=c[k]});
    p.querySelectorAll('.devices li').forEach(function(li,i){var d=c.devices[i];li.querySelector('.fill').style.width=d[0]+'%';
      var v=li.querySelector('.verdict');v.textContent=d[1];v.className='verdict v-'+d[1].replace(' ','-')});}
  p.querySelectorAll('.seg').forEach(function(g){g.addEventListener('click',function(ev){var b=ev.target.closest('button');if(!b)return;
    g.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});draw();});});
})();
(function(){var t=document.getElementById('onlydiff');if(!t)return;t.addEventListener('change',function(){
  document.querySelectorAll('tr[data-diff="0"]').forEach(function(r){r.hidden=t.checked})});})();
(function(){var f=document.getElementById('filters');if(!f)return;
  var q=document.getElementById('q'),sort=document.getElementById('sort'),list=document.querySelector('.cards'),
      cards=[].slice.call(list.children),rows=[].slice.call(document.querySelectorAll('.tableview tbody tr')),
      shown=document.querySelector('.shown'),empty=document.querySelector('.empty'),now=new Date();
  function vals(n){return [].slice.call(f.querySelectorAll('input[name='+n+']:checked')).map(function(x){return x.value})}
  function ok(c){var d=c.dataset,des=vals('design'),lay=vals('layers'),siz=vals('size'),rel=f.querySelector('input[name=released]:checked').value;
    var dm=(des.indexOf(d.design==='moe'?'moe':'dense')>=0)||(d.hybrid==='1'&&des.indexOf('hybrid')>=0);
    var lm=!lay.length||lay.some(function(l){return d.layers.split(' ').indexOf(l)>=0});
    var p=+d.params,sm=!siz.length||siz.some(function(s){return s==='s'?p<3e9:s==='m'?p>=3e9&&p<30e9:p>=30e9});
    var age=(now-new Date(d.date))/864e5,rm=rel==='any'||(rel==='6m'?age<=183:d.date.slice(0,4)==String(now.getFullYear()));
    var t=q.value.trim().toLowerCase();return dm&&lm&&sm&&rm&&(!t||d.search.indexOf(t)>=0)}
  function apply(){var key=sort.value,n=0;cards.sort(function(a,b){var x=a.dataset,y=b.dataset;
      return key==='big'?y.params-x.params:key==='small'?x.params-y.params:key==='name'?x.name.localeCompare(y.name):y.date.localeCompare(x.date)});
    cards.forEach(function(c,i){list.appendChild(c);var m=ok(c);c.hidden=!m;if(rows[i]){}if(m)n++});
    rows.forEach(function(r){var name=r.querySelector('a').getAttribute('href');var c=cards.filter(function(c){return c.querySelector('h3 a').getAttribute('href')===name})[0];r.hidden=c.hidden});
    shown.textContent=n;empty.hidden=n>0}
  f.addEventListener('change',apply);f.addEventListener('reset',function(){setTimeout(apply)});q.addEventListener('input',apply);sort.addEventListener('change',apply);
  document.querySelectorAll('[data-view]').forEach(function(b){b.addEventListener('click',function(){var t=b.dataset.view==='table';
    document.querySelectorAll('[data-view]').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});
    list.hidden=t;document.querySelector('.tableview').hidden=!t;})});
})();
"""

COMPARE_JS = r"""
/* ArchWatch compare tool. Plain JS; reads ../data/models.json (published values only). */
var AW = (function () {
  function parseState(search, stems, preset) {
    /* ?a=&b=&c= -> up to three distinct known stems; falls back to the page preset. */
    var q = new URLSearchParams(search || ''), out = [];
    ['a', 'b', 'c'].forEach(function (k) {
      var v = q.get(k);
      if (v && stems.indexOf(v) >= 0 && out.indexOf(v) < 0) out.push(v);
    });
    if (out.length < 2 && preset && preset.length === 2) return preset.slice();
    return out;
  }
  function stateQuery(sel) {
    var keys = ['a', 'b', 'c'], parts = [];
    sel.forEach(function (s, i) { if (s) parts.push(keys[i] + '=' + encodeURIComponent(s)); });
    return parts.length ? '?' + parts.join('&') : '';
  }
  function cacheBytes(c, n) {
    var windowed = c.window ? Math.floor(c.window_full * Math.min(n, c.window) / c.window) : 0;
    return c.grow * n + windowed + c.fixed;
  }
  return { parseState: parseState, stateQuery: stateQuery, cacheBytes: cacheBytes };
})();
if (typeof module !== 'undefined') module.exports = AW;

(function () {
  if (typeof document === 'undefined') return;
  var tool = document.getElementById('tool');
  if (!tool) return;
  var D, byStem = {}, byName = {}, sel = [];
  var ctxPick, precPick = '16';

  function el(tag, attrs, text) {
    var x = document.createElement(tag);
    for (var k in (attrs || {})) x.setAttribute(k, attrs[k]);
    if (text !== undefined) x.textContent = text;
    return x;
  }
  function sig(x) { var t = Number(x).toPrecision(3); return String(parseFloat(t)); }
  function params(n) {
    if (n === null || n === undefined) return 'n/a';
    if (n >= 1e12) return sig(n / 1e12) + 'T';
    if (n >= 1e9) return sig(n / 1e9) + 'B';
    return sig(n / 1e6) + 'M';
  }
  function bytes(b) {
    if (b >= Math.pow(2, 30)) return (b / Math.pow(2, 30)).toFixed(2) + ' GiB';
    if (b >= Math.pow(2, 20)) return sig(b / Math.pow(2, 20)) + ' MiB';
    if (b >= 1024) return sig(b / 1024) + ' KiB';
    return b + ' B';
  }
  function tokens(n) {
    if (n >= 1048576 && n % 1048576 === 0) return (n / 1048576) + 'M';
    if (n % 1024 === 0) return (n / 1024) + 'K';
    return n.toLocaleString('en-US');
  }
  function models() { return sel.map(function (s) { return byStem[s]; }).filter(Boolean); }

  function setUrl() { history.replaceState(null, '', location.pathname + AW.stateQuery(sel)); }

  function fillPickers() {
    ['a', 'b', 'c'].forEach(function (k, i) {
      var box = tool.querySelector('[data-slot=' + k + ']'), m = byStem[sel[i]];
      box.querySelector('input').value = m ? m.name : '';
      box.querySelector('.slotmeta').textContent = m ? m.publisher + ' · ' + m.date + ' · ' + m.design : '';
      box.style.setProperty('--slot', D.slots[i]);
    });
    document.getElementById('clear-c').hidden = !sel[2];
  }

  function value(m, f) {
    if (m.fields[f]) return m.fields[f];
    if (m.derived[f]) return m.derived[f];
    var imp = (D.implied[m.stem] || {})[f];
    return [imp || 'not stated', null];
  }
  function spread(nums) {
    var xs = nums.filter(function (x) { return x; });
    if (xs.length < 2) return 0;
    return Math.log(Math.max.apply(null, xs) / Math.min.apply(null, xs));
  }
  function list(ms, fn) { return ms.map(function (m) { return m.name + ' ' + fn(m); }).join(', '); }
  function renderKeyDiff(ms) {
    /* one sentence per field, covering every selected model; ranked by the size of the difference */
    var box = document.getElementById('keydiff'); box.textContent = '';
    var items = [], n = Math.min.apply(null, [131072].concat(ms.map(function (m) { return m.context; })));
    function same(fn) { var v = ms.map(fn); return v.every(function (x) { return x === v[0]; }); }
    if (!same(function (m) { return m.design + m.hybrid; }))
      items.push([2.5, 'Design: ' + list(ms, function (m) {
        return m.design === 'MoE' ? 'MoE (' + params(m.params.active) + ' of ' + params(m.params.total) + ' active per token)' : m.design + ' (every parameter active)';
      }) + '.']);
    var tot = spread(ms.map(function (m) { return m.params.total; }));
    if (tot > Math.log(1.1)) items.push([tot, 'Total parameters: ' + list(ms, function (m) { return params(m.params.total); }) + '.']);
    if (ms.every(function (m) { return m.params.active; })) {
      var act = spread(ms.map(function (m) { return m.params.active; }));
      if (act > Math.log(1.1)) items.push([act, 'Active parameters per token: ' + list(ms, function (m) { return params(m.params.active); }) + '.']);
    }
    if (!same(function (m) { return Object.keys(m.counts).sort().join(); }))
      items.push([2, 'Layer types: ' + ms.map(function (m) {
        return m.name + ' ' + Object.keys(m.counts).map(function (k) { return m.counts[k] + ' ' + D.kinds[k]; }).join(' + ');
      }).join('; ') + '.']);
    var lay = spread(ms.map(function (m) { return m.layers.length; }));
    if (lay > Math.log(1.1)) items.push([lay, 'Layers: ' + list(ms, function (m) { return m.layers.length; }) + '.']);
    var ctx = spread(ms.map(function (m) { return m.context; }));
    if (ctx > 0) items.push([ctx, 'Context: ' + list(ms, function (m) { return tokens(m.context); }) + '.']);
    var cs = ms.map(function (m) { return AW.cacheBytes(m.cache, n); }), cache = spread(cs);
    if (cache > Math.log(1.1)) items.push([cache, 'Cache at ' + tokens(n) + ' tokens: ' + ms.map(function (m, i) {
      return m.name + ' ' + bytes(cs[i]) + (!m.cache.grow && m.cache.window ? ' (stops growing at ' + m.cache.window.toLocaleString('en-US') + ' tokens)' : '');
    }).join(', ') + '.']);
    if (!same(function (m) { return value(m, 'attention')[0]; }))
      items.push([1, 'Attention: ' + list(ms, function (m) { return value(m, 'attention')[0]; }) + '.']);
    ['hidden_size', 'vocab_size'].forEach(function (f) {
      var r = spread(ms.map(function (m) { return value(m, f)[1]; }));
      if (r > Math.log(1.2)) items.push([r * 0.6, D.labels[f] + ': ' + list(ms, function (m) { return value(m, f)[0]; }) + '.']);
    });
    items.sort(function (x, y) { return y[0] - x[0]; });
    var ul = el('ul', { 'class': 'changes' });
    items.forEach(function (it) { ul.appendChild(el('li', {}, it[1])); });
    if (!items.length) ul.appendChild(el('li', {}, 'No difference of 10% or more in design, size, layers, context or cache.'));
    box.appendChild(ul);
    var pred = ms.length === 2 && (ms[1].predecessor === ms[0].stem ? ms[1] : ms[0].predecessor === ms[1].stem ? ms[0] : null);
    if (pred && pred.note) {
      box.appendChild(el('p', { 'class': 'prose' }, pred.note));
      box.appendChild(el('p', { 'class': 'byline' }, 'Written by Gemini from the verified diff; every number checked by code.'));
    }
    box.appendChild(el('p', { 'class': 'byline' }, 'One sentence per field, written by code from published values; largest differences first.'));
  }

  function renderStrips(ms) {
    var box = document.getElementById('strips'); box.textContent = '';
    ms.forEach(function (m, i) {
      var card = el('div', { 'class': 'panel stripcard' });
      var head = el('div', { 'class': 'stackhead' });
      var h = el('h3'); var a = el('a', { href: '../models/' + m.stem + '.html' }, m.name); h.appendChild(a); head.appendChild(h);
      head.appendChild(el('span', { 'class': 'mono' }, m.layers.length + ' layers'));
      card.appendChild(head);
      var strip = el('div', { 'class': 'cells', role: 'img', 'aria-label': m.name + ' layer order, layer 1 first' });
      m.layers.forEach(function (k) { strip.appendChild(el('i', { style: 'background:' + D.colors[k] })); });
      card.appendChild(strip);
      var lg = el('ul', { 'class': 'legend' });
      Object.keys(m.counts).forEach(function (k) {
        var li = el('li'); li.appendChild(el('span', { 'class': 'sw', style: 'background:' + D.colors[k], 'aria-hidden': 'true' }));
        li.appendChild(document.createTextNode(D.kinds[k] + ' ' + m.counts[k])); lg.appendChild(li);
      });
      card.appendChild(lg); box.appendChild(card);
    });
  }

  function renderChart(ms) {
    var box = document.getElementById('chart'), dataBox = document.getElementById('chartdata');
    box.textContent = ''; dataBox.textContent = '';
    var maxX = Math.max.apply(null, ms.map(function (m) { return m.context; }));
    var maxY = 0;
    ms.forEach(function (m) { m.series.forEach(function (p) { if (p[0] <= maxX && p[1] > maxY) maxY = p[1]; }); });
    maxY = maxY || 1;
    var W = Math.max(300, Math.min(900, box.clientWidth || 760)), H = W < 500 ? 260 : 300, L = 64, R = 14, T = 16, B = 40, NS = 'http://www.w3.org/2000/svg';
    var lo = Math.log(1024), hi = Math.log(maxX);
    function X(n) { return L + (W - L - R) * (Math.log(n) - lo) / (hi - lo || 1); }
    function Y(v) { return T + (H - T - B) * (1 - v / maxY); }
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H); svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Cache memory against context length for ' + ms.map(function (m) { return m.name; }).join(', '));
    function line(x1, y1, x2, y2, cls) { var l = document.createElementNS(NS, 'line'); l.setAttribute('x1', x1); l.setAttribute('y1', y1); l.setAttribute('x2', x2); l.setAttribute('y2', y2); l.setAttribute('class', cls); svg.appendChild(l); }
    function text(x, y, s, anchor) { var t = document.createElementNS(NS, 'text'); t.setAttribute('x', x); t.setAttribute('y', y); t.setAttribute('text-anchor', anchor || 'middle'); t.setAttribute('class', 'axis'); t.textContent = s; svg.appendChild(t); }
    for (var g = 0; g <= 4; g++) { var v = maxY * g / 4; line(L, Y(v), W - R, Y(v), 'grid'); text(L - 8, Y(v) + 4, bytes(v), 'end'); }
    [1024, 8192, 32768, 131072, 1048576].filter(function (n) { return n <= maxX; }).forEach(function (n) { line(X(n), T, X(n), H - B, 'grid'); text(X(n), H - B + 18, tokens(n)); });
    text((L + W - R) / 2, H - 4, 'context length (tokens, log scale)');
    ms.forEach(function (m, i) {
      var pts = m.series.filter(function (p) { return p[0] <= maxX; }).map(function (p) { return X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1); });
      var pl = document.createElementNS(NS, 'polyline'); pl.setAttribute('points', pts.join(' ')); pl.setAttribute('class', 'series');
      pl.setAttribute('stroke', D.slots[i]); svg.appendChild(pl);
      var last = m.series[m.series.length - 1], dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('cx', X(last[0])); dot.setAttribute('cy', Y(last[1])); dot.setAttribute('r', 4); dot.setAttribute('fill', D.slots[i]); svg.appendChild(dot);
    });
    box.appendChild(svg);
    var lg = el('ul', { 'class': 'legend' });
    ms.forEach(function (m, i) {
      var li = el('li'); li.appendChild(el('span', { 'class': 'sw', style: 'background:' + D.slots[i], 'aria-hidden': 'true' }));
      li.appendChild(document.createTextNode(m.name + ' (max ' + tokens(m.context) + ', ' + m.cache_label + ')')); lg.appendChild(li);
    });
    box.appendChild(lg);
    var t = el('table', { 'class': 'facts' }), hd = el('tr'); hd.appendChild(el('th', { scope: 'col' }, 'Context'));
    ms.forEach(function (m) { hd.appendChild(el('th', { scope: 'col' }, m.name)); });
    var th = el('thead'); th.appendChild(hd); t.appendChild(th); var tb = el('tbody');
    var xs = []; ms.forEach(function (m) { m.series.forEach(function (p) { if (p[0] <= maxX && xs.indexOf(p[0]) < 0) xs.push(p[0]); }); });
    xs.sort(function (a, b) { return a - b; }).forEach(function (n) {
      var tr = el('tr'); tr.appendChild(el('th', { scope: 'row' }, tokens(n)));
      ms.forEach(function (m) { var p = m.series.filter(function (q) { return q[0] === n; })[0]; tr.appendChild(el('td', { 'class': 'mono' }, p ? bytes(p[1]) : (n > m.context ? 'beyond max' : bytes(AW.cacheBytes(m.cache, n))))); });
      tb.appendChild(tr);
    });
    t.appendChild(tb); dataBox.appendChild(t);
  }

  function renderFit(ms) {
    var maxX = Math.max.apply(null, ms.map(function (m) { return m.context; }));
    var ctxs = D.contexts.filter(function (n) { return n <= maxX; });
    if (ctxs.indexOf(ctxPick) < 0) ctxPick = ctxs.indexOf(32768) >= 0 ? 32768 : ctxs[ctxs.length - 1];
    function seg(id, items, current, onpick) {
      var g = document.getElementById(id); g.textContent = '';
      items.forEach(function (it) {
        var b = el('button', { type: 'button', 'aria-pressed': String(it[0] === current) }, it[1]);
        b.addEventListener('click', function () { onpick(it[0]); renderFit(models()); }); g.appendChild(b);
      });
    }
    seg('ctxseg', ctxs.map(function (n) { return [n, tokens(n)]; }), ctxPick, function (v) { ctxPick = v; });
    seg('precseg', D.precision.map(function (p) { return [p.key, p.label]; }), precPick, function (v) { precPick = v; });
    var per = D.precision.filter(function (p) { return p.key === precPick; })[0].bytes;
    var box = document.getElementById('fit-compare'); box.textContent = '';
    ms.forEach(function (m, i) {
      var row = el('div', { 'class': 'fitrow' }), head = el('div', { 'class': 'fithead' });
      head.appendChild(el('b', { style: 'color:' + D.slots[i] }, m.name));
      if (ctxPick > m.context) {
        head.appendChild(el('span', { 'class': 'sub' }, 'beyond its maximum context (' + tokens(m.context) + ')'));
        row.appendChild(head); box.appendChild(row); return;
      }
      var w = m.params.total * per, c = AW.cacheBytes(m.cache, ctxPick), total = w + c;
      head.appendChild(el('span', { 'class': 'mono' }, bytes(total) + '  (weights ' + bytes(w) + ', cache ' + bytes(c) + ')'));
      row.appendChild(head);
      var ul = el('ul', { 'class': 'devices' });
      D.devices.forEach(function (d) {
        var word = total <= 0.75 * d.bytes ? 'fits' : total <= d.bytes ? 'tight' : 'too big';
        var li = el('li'); li.appendChild(el('span', {}, d.label));
        var tr = el('span', { 'class': 'track' }); tr.appendChild(el('span', { 'class': 'fill', style: 'width:' + Math.min(100, 100 * total / d.bytes).toFixed(1) + '%;background:' + D.slots[i] }));
        li.appendChild(tr); li.appendChild(el('span', { 'class': 'verdict v-' + word.replace(' ', '-') }, word)); ul.appendChild(li);
      });
      row.appendChild(ul); box.appendChild(row);
    });
  }

  function chip(a, b) {
    if (a === b) return ['same', 'same'];
    return ['differs', 'up'];
  }
  function chipNum(ra, rb, da, db) {
    if (ra && rb && ra === rb) return ['same', 'same'];
    if (!(ra && rb) && da === db) return ['same', 'same'];
    if (ra && rb) {
      if (rb === 2 * ra) return ['doubled', 'up'];
      if (2 * rb === ra) return ['halved', 'warn'];
      var pct = 100 * (rb - ra) / ra;
      if (Math.abs(pct) < 0.1) return [(pct > 0 ? '<+0.1%' : '<-0.1%'), 'same'];
      return [(pct > 0 ? '+' : '') + pct.toFixed(Math.abs(pct) >= 1 ? 0 : 1) + '%', rb > ra ? 'up' : 'warn'];
    }
    return ['differs', 'up'];
  }
  function table(ms, fieldsList) {
    var t = el('table', { 'class': 'facts cmp' }), hd = el('tr');
    hd.appendChild(el('th', { scope: 'col' }, 'Field'));
    ms.forEach(function (m) { hd.appendChild(el('th', { scope: 'col' }, m.name)); });
    hd.appendChild(el('th', { scope: 'col' }, 'Change'));
    var th = el('thead'); th.appendChild(hd); t.appendChild(th);
    var tb = el('tbody');
    fieldsList.forEach(function (f) {
      if (!ms.some(function (m) { return m.fields[f] || m.derived[f]; })) return;
      var vals = ms.map(function (m) { return value(m, f); });
      var shown = vals.map(function (v) { return v[0]; }), raws = vals.map(function (v) { return v[1]; });
      var c = ms.length === 2 ? chipNum(raws[0], raws[1], shown[0], shown[1]) : chip(shown.every(function (x) { return x === shown[0]; }) ? 1 : 0, 1);
      var tr = el('tr', { 'data-diff': c[0] === 'same' ? '0' : '1' });
      tr.appendChild(el('th', { scope: 'row' }, D.labels[f]));
      shown.forEach(function (x) { tr.appendChild(el('td', { 'class': 'mono' }, x)); });
      var td = el('td'); td.appendChild(el('span', { 'class': 'chip chip-' + c[1] }, c[0])); tr.appendChild(td);
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    return t;
  }
  function renderFields(ms) {
    var box = document.getElementById('fields'); box.textContent = '';
    box.appendChild(table(ms, D.order));
    var more = el('details', { 'class': 'allfields' }), sum = el('summary', {}, 'All fields');
    more.appendChild(sum); more.appendChild(table(ms, D.rest)); box.appendChild(more);
    applyOnlyDiff();
  }
  function applyOnlyDiff() {
    var on = document.getElementById('onlydiff').checked;
    document.querySelectorAll('#fields tr[data-diff="0"]').forEach(function (r) { r.hidden = on; });
  }

  function render() {
    var ms = models();
    fillPickers(); setUrl();
    if (ms.length < 2) return;
    renderKeyDiff(ms); renderStrips(ms); renderChart(ms); renderFit(ms); renderFields(ms);
  }

  function start(data) {
    D = data;
    D.models.forEach(function (m) { byStem[m.stem] = m; byName[m.name.toLowerCase()] = m.stem; });
    var stems = D.models.map(function (m) { return m.stem; });
    var preset = tool.dataset.a ? [tool.dataset.a, tool.dataset.b] : null;
    if (!preset && D.suggested.length) preset = [D.suggested[0].a, D.suggested[0].b];
    sel = AW.parseState(location.search, stems, preset);
    tool.hidden = false;
    ['a', 'b', 'c'].forEach(function (k, i) {
      var input = tool.querySelector('#pick-' + k);
      input.addEventListener('change', function () {
        var s = byName[input.value.trim().toLowerCase()];
        if (s && sel.indexOf(s) < 0) { sel[i] = s; } else if (!input.value.trim() && i === 2) { sel.splice(2, 1); }
        render();
      });
    });
    document.getElementById('swap').addEventListener('click', function () { var t = sel[0]; sel[0] = sel[1]; sel[1] = t; render(); });
    document.getElementById('clear-c').addEventListener('click', function () { sel = sel.slice(0, 2); render(); });
    document.getElementById('onlydiff').addEventListener('change', applyOnlyDiff);
    document.getElementById('copylink').addEventListener('click', function () {
      var url = location.origin + location.pathname + AW.stateQuery(sel);
      var done = function () { document.getElementById('copied').textContent = 'Link copied'; };
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(done, done); else done();
    });
    tool.querySelectorAll('.chipbtn').forEach(function (a) {
      a.addEventListener('click', function (ev) { ev.preventDefault(); sel = [a.dataset.a, a.dataset.b]; render(); });
    });
    render();
  }
  fetch('../data/models.json').then(function (r) { return r.json(); }).then(start).catch(function () {
    tool.hidden = true;
  });
})();
"""


def main():
    published, held, seen = inventory()
    site = ROOT / "site"
    write_if_changed(site / "style.css", CSS.lstrip())
    write_if_changed(site / "site.js", JS.lstrip())
    write_if_changed(site / "index.html", home(published, held, seen))
    data = compare_data(published)
    write_if_changed(site / "data" / "models.json", json.dumps(data, indent=1, sort_keys=True) + "\n")
    write_if_changed(site / "compare.js", COMPARE_JS.lstrip())
    pairs, expected = [], {"index.html"}
    for repo, (stem, rec, pub) in published.items():
        write_if_changed(site / "models" / f"{stem}.html", model_page(stem, rec, pub, published))
        before = published_predecessor(rec, published)
        if before:
            write_if_changed(site / "compare" / f"{stem}.html", compare_tool(data, (before[0], stem)))
            pairs.append(stem)
            expected.add(f"{stem}.html")
    write_if_changed(site / "compare" / "index.html", compare_tool(data))
    # pages of records that are no longer published are removed, so held numbers never stay online
    stems = {stem for stem, _, _ in published.values()}
    for folder, keep in (("models", {f"{s}.html" for s in stems}), ("compare", expected)):
        for path in (site / folder).glob("*.html"):
            if path.name not in keep:
                path.unlink()
    approved = {f"{s}.svg" for s, rec, _ in published.values() if generated.select(rec)}
    for path in (site / "diagrams").glob("*.svg") if (site / "diagrams").exists() else []:
        if path.name not in approved:
            path.unlink()
    print(f"site: {len(published)} model pages, {len(pairs)} compare pages, {len(held)} held")


if __name__ == "__main__":
    main()
