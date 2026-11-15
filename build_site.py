"""Build the static site from published records only.

    uv run --frozen python build_site.py

Pages: site/index.html (home), site/models/<stem>.html (one per published model), site/compare/<stem>.html (each
published model whose predecessor is also published) and site/compare/index.html. Every number comes from
published, verifier-approved record fields or from the cache calculator run on them. Held records appear by name
with a plain reason only. Plain HTML/CSS; a small script runs the cache-memory picker and the home-page filter.
"""

import datetime as dt
import html
import json
from pathlib import Path

import yaml

from archwatch import diagram, generated, pipeline
from archwatch.artifacts import write_if_changed
from archwatch.explainer import diff_input, input_hash
from archwatch.labels import FIELD_NAMES, KIND_LABELS
from archwatch.publication import PublicationError, plain_hold, render_cache, render_values

ROOT = Path(__file__).parent
CODE = Path(__file__).parent  # site settings live with the code; ROOT is where records are read and pages written
LABEL = KIND_LABELS
COLOR = {
    "kda_linear": "#2a9d8f",
    "gated_dsa_mla": "#e07a5f",
    "full_attention_mla": "#f2a65a",
    "short_conv": "#6a994e",
    "sliding_attention": "#4f7cc3",
    "full_attention": "#e07a5f",
    "mamba2": "#8e6cc4",
    "mlp": "#a3a9b1",
}
LINEAR = {"kda_linear"}
NON_ATTENTION = {"kda_linear", "short_conv", "mamba2"}
MLA = {"gated_dsa_mla", "full_attention_mla"}
# Fields shown elsewhere on the page (layer mix bar, diagram) or internal to the renderer.
NOT_IN_TABLE = {
    "block_shapes",
    "parameter_components",
    "weight_header_summary",
    "layer_types",
    "layer_pattern_per_layer",
    "layer_index_base_in_config",
}
KEY_FACTS = ("params_total", "params_active", "num_layers", "layer_type_counts", "context_length")
FIRST = (
    "params_total",
    "params_active",
    "params_active_non_embedding",
    "params_mtp",
    "num_layers",
    "layer_type_counts",
    "context_length",
    "hidden_size",
    "vocab_size",
    "attention_type",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
)
SOURCE_WORDS = {
    "config": "config.json",
    "weights": "safetensors headers",
    "code": "modeling code",
    "card": "model card",
    "derived": "computed from published fields",
}
STATUS_WORDS = {"verified": "verified", "single source": "one source", "differs from card": "differs from card"}


def e(x):
    return html.escape(str(x))


def settings():
    return yaml.safe_load((CODE / "config/site.yaml").read_text())


# ---------------------------------------------------------------- formatting


def human(n):
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= d:
            return f"{n / d:.2f}{unit}"
    return f"{n:,}"


def tokens(n):
    n = int(n)
    if n >= 2**20 and n % 2**20 == 0:
        return f"{n // 2**20}M"
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1024 == 0:
        return f"{n // 1024}K"
    if n % 1000 == 0:
        return f"{n // 1000}K"
    return f"{n:,}"


def size(b):
    for unit, d in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if b >= d:
            return f"{b / d:.2f} {unit}" if b / d < 100 else f"{b / d:,.0f} {unit}"
    return f"{b:,} B"


def date(iso):
    d = dt.date.fromisoformat(iso[:10])
    return f"{d.day} {d:%b %Y}"


def show(field, v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, dict):
        if all(isinstance(n, int) for n in v.values()):
            return ", ".join(f"{LABEL.get(k, k)} {n}" for k, n in sorted(v.items(), key=lambda kv: -kv[1]))
        return ", ".join(f"{k.replace('_', ' ')}: {show(k, n)}" for k, n in v.items())
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int) and (field.startswith("params") or field == "vision_params"):
        return f"{human(v)} ({v:,})"
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


def tags(pub):
    kinds = set(pub["layer_types"]) - {"mlp"}
    out = ["MoE" if "moe_num_experts" in pub else "Dense"]
    if kinds & NON_ATTENTION and kinds - NON_ATTENTION:
        out.append("Hybrid")
    if kinds & LINEAR:
        out.append("Linear attention")
    if kinds & MLA or "MLA" in str(pub.get("attention_type", "")):
        out.append("MLA")
    return out


def legend(pub, *, counts=True):
    c = pub["layer_type_counts"]
    return "".join(
        f'<li><span class="sw" style="background:{COLOR.get(k, "#999")}" aria-hidden="true"></span>'
        f"{e(LABEL.get(k, k))}{f' <span class=num>{c[k]}</span>' if counts else ''}</li>"
        for k in kinds_in_order(pub)
    )


def mix_bar(pub, label):
    """Layer mix: one segment per layer kind, sized by its layer count, in order of first appearance."""
    runs = [[k, pub["layer_type_counts"][k]] for k in kinds_in_order(pub)]
    segs = "".join(
        f'<span style="flex:{n};background:{COLOR.get(k, "#999")}" title="{e(LABEL.get(k, k))} x{n}"></span>'
        for k, n in runs
    )
    return f'<div class="bar" role="img" aria-label="{e(label)}">{segs}</div>'


def mix_text(pub):
    return ", ".join(f"{LABEL.get(k, k)} {pub['layer_type_counts'][k]}" for k in kinds_in_order(pub))


def pattern_text(pub):
    """The repeating unit of the layer pattern, if there is one ('3 KDA linear, then 1 Gated DSA on MLA')."""
    types = pub["layer_types"]
    for period in range(2, 9):
        unit = types[-period:]
        body = types[len(types) % period :]
        if len(set(unit)) > 1 and all(body[i] == unit[i % period] for i in range(len(body))):
            runs = []
            for kind in unit:
                if runs and runs[-1][0] == kind:
                    runs[-1][1] += 1
                else:
                    runs.append([kind, 1])
            return ", then ".join(f"{n} {LABEL.get(k, k)}" for k, n in runs)
    return None


def summary_line(pub, before=None):
    """One plain sentence written by code from published fields (no model text)."""
    n = pub["num_layers"]
    if before:
        a, b = before["layer_type_counts"], pub["layer_type_counts"]
        gone, new = [k for k in a if k not in b], [k for k in b if k not in a]
        parts = []
        if len(gone) == 1 and len(new) == 1 and a[gone[0]] == b[new[0]]:
            parts.append(f"{LABEL[new[0]]} replaces {LABEL[gone[0]]} in {b[new[0]]} of {n} layers")
        else:
            if n != before["num_layers"]:
                parts.append(
                    f"{abs(n - before['num_layers'])} {'more' if n > before['num_layers'] else 'fewer'} layers"
                )
            parts += [f"new {LABEL.get(k, k)} layers ({b[k]})" for k in new]
            parts += [f"no {LABEL.get(k, k)} layers" for k in gone]
        if parts:
            text = "; ".join(parts)
            return text[0].upper() + text[1:] + "."
        return "Same layer mix as its predecessor; see the changed fields."
    counts = pub["layer_type_counts"]
    moe = (
        f"MoE with {pub['moe_num_experts']} experts, {pub['moe_active_experts']} per token"
        if "moe_num_experts" in pub
        else "Dense"
    )
    if len(counts) == 1:
        kind = next(iter(counts))
        what = pub.get("attention_type") if kind == "full_attention" and pub.get("attention_type") else LABEL[kind]
        return f"{moe}: {what} in all {n} layers."
    top = max(counts, key=counts.get)
    rest = ", ".join(f"{LABEL.get(k, k)} {v}" for k, v in counts.items() if k != top)
    return f"{moe}: {LABEL[top]} in {counts[top]} of {n} layers, plus {rest}."


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


def change_label(field, before, after):
    if before is None:
        return "now stated", "new"
    if after is None:
        return "not published", "warn"
    if before == after:
        return "same", "same"
    if isinstance(before, str) or isinstance(after, str):
        return "changed", "up"
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool):
        if after == 2 * before:
            return "doubled", "up"
        if 2 * after == before:
            return "halved", "warn"
        if field.startswith("params") or field in ("vocab_size", "context_length"):
            pct = 100 * (after - before) / before
            return (f"{pct:+.0f}%" if abs(pct) >= 1 else f"{pct:+.2f}%"), "up" if after > before else "warn"
        return f"{after - before:+,}", "up" if after > before else "warn"
    return "changed", "up"


# ---------------------------------------------------------------- page frame


def frame(title, body, *, depth=0, active="", description=""):
    up = "../" * depth
    repo = settings()["repo_url"]
    nav = "".join(
        f'<a href="{href}"{" aria-current=page" if key == active else ""}>{text}</a>'
        for key, href, text in (
            ("models", f"{up}index.html#models", "Models"),
            ("compare", f"{up}compare/index.html", "Compare"),
            ("method", f"{up}index.html#method", "Method"),
            ("github", repo, "GitHub"),
        )
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><meta name="description" content="{e(description or title)}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap">
<link rel="stylesheet" href="{up}style.css"></head>
<body><a class="skip" href="#main">Skip to content</a>
<header class="top"><div class="wrap topbar"><a class="brand" href="{up}index.html">ArchWatch</a><nav aria-label="Main">{nav}</nav>
<span class="lastrun">{e(last_run())}</span></div></header>
<main id="main" class="wrap">{body}</main>
<footer id="method" class="foot"><div class="wrap cols2">
<p>Facts come from config.json, safetensors headers, modeling code and model cards at pinned commits. A fact is verified
when two independent sources agree; a fact with one source is labelled as such, and a disagreement is held for a
person to resolve. Cache sizes are computed by ArchWatch's calculator from published fields. Diagrams and
explanations are written by Gemini from verified fields only, and every number they contain is checked by code.</p>
<p>Open source, rebuilt daily by GitHub Actions on free tiers. <a href="{e(repo)}">View the code</a>.</p>
</div></footer>
<script src="{up}site.js" defer></script></body></html>"""


_LAST = {}


def last_run():
    if ROOT not in _LAST:
        stamps = [json.loads(p.read_text()).get("analyzed_at", "") for p in (ROOT / "data/models").glob("*.json")]
        latest = max((s for s in stamps if s), default="")
        _LAST[ROOT] = f"Last analysis {date(latest)}, {latest[11:16]} UTC" if latest else ""
    return _LAST[ROOT]


# ---------------------------------------------------------------- inventory


def inventory():
    """Published records (all checks pass, not held) and held repos with plain reasons; no numbers for held."""
    seen_path = ROOT / "data/seen.json"
    seen = json.loads(seen_path.read_text()).get("models", {}) if seen_path.exists() else {}
    published, held = {}, {}
    records = {}
    for path in sorted((ROOT / "data/models").glob("*.json")):
        rec = json.loads(path.read_text())
        records[rec["id"]] = (path.stem, rec)
    for repo, (stem, rec) in sorted(records.items()):
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
        if repo not in records:
            held[repo] = plain_reason(f"{state.get('status', 'unprocessed')}: {state.get('reason', '')}", {})
    return published, held, seen


def plain_reason(reason, values):
    """A short, number-free reason for the public held list."""
    configured = plain_hold(values)
    if configured:
        return configured
    text = reason.lower()
    if "finetune" in text or "derived" in text:
        return "Fine-tuned from another model: ArchWatch publishes base architectures."
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


# ---------------------------------------------------------------- home


def home(published, held, seen):
    cards = []
    order = sorted(published.values(), key=lambda x: (x[1]["hub_created"] or "", x[1]["id"]), reverse=True)
    pubs = settings()["publishers"]
    for stem, rec, pub in order:
        org, name = rec["id"].split("/")
        before = published_predecessor(rec, published)
        line = summary_line(pub, before[2] if before else None)
        tag_list = tags(pub)
        search = " ".join(
            [rec["id"], name, pubs.get(org, org), *tag_list, mix_text(pub), pub.get("attention_type", "")]
        )
        cards.append(f"""<li class="card" data-tags="{e("|".join(tag_list))}" data-search="{e(search.lower())}">
<a class="cardlink" href="models/{e(stem)}.html">
<div class="cardhead"><span>{e(pubs.get(org, org))}</span><time datetime="{e(rec["hub_created"][:10])}">{e(date(rec["hub_created"]))}</time></div>
<h3>{e(name)}</h3><p class="line">{e(line)}</p>
{mix_bar(pub, "Layer mix: " + mix_text(pub))}<ul class="legend">{legend(pub)}</ul>
<dl class="mini"><div><dt>parameters</dt><dd>{e(human(pub["params_total"]))}</dd></div>
<div><dt>layers</dt><dd>{pub["num_layers"]}</dd></div><div><dt>context</dt><dd>{e(tokens(pub["context_length"]))}</dd></div></dl>
</a></li>""")
    filters = ["All", "MoE", "Hybrid", "Linear attention", "MLA", "Dense"]
    pills = "".join(
        f'<button type="button" class="pill" data-filter="{e(f)}" aria-pressed="{"true" if f == "All" else "false"}">{e(f)}</button>'
        for f in filters
        if f == "All" or any(f in tags(p) for _, _, p in published.values())
    )
    held_rows = "".join(
        f"<tr><th scope=row>{e(repo.split('/')[-1])}</th><td>{e(reason)}</td></tr>" for repo, reason in held.items()
    )
    shown_facts = linked = 0
    for _, rec, pub in published.values():
        for c in rec["checks"]:
            if c["field"] in pub and c["field"] not in NOT_IN_TABLE:
                shown_facts += 1
                linked += any(
                    x["evidence"].split(":", 1)[0] in ("config", "weights", "code", "card") for x in c["candidates"]
                )
    share = f"{100 * linked / shown_facts:.0f}%" if shown_facts else "-"
    body = f"""<section class="hero"><h1>Every new open model, explained from its own files.</h1>
<p class="lead">ArchWatch reads each model's config, weights and code when it lands on Hugging Face, checks every
number against a second source, and shows what changed.</p></section>
<div class="controls" hidden><label class="sr" for="q">Search models</label>
<input id="q" type="search" placeholder="Search {len(published)} models, e.g. MLA, Mamba, Qwen" autocomplete="off">
<div class="pills" role="group" aria-label="Filter by architecture">{pills}</div></div>
<dl class="stats"><div><dt>architectures published</dt><dd>{len(published)}</dd></div>
<div><dt>repositories tracked</dt><dd>{len(seen)}</dd></div>
<div><dt>of published facts read from a source file</dt><dd>{share}</dd></div>
<div><dt>a month to run (free tiers only)</dt><dd>$0</dd></div></dl>
<section id="models" aria-labelledby="latest"><div class="sechead"><h2 id="latest">Latest architectures</h2>
<p class="hint">Bar = layer mix by count; the compare pages show every layer in order</p></div>
<ul class="cards">{"".join(cards)}</ul><p class="empty" hidden>No model matches.</p></section>
<section aria-labelledby="held"><h2 id="held">Held for review</h2>
<p class="sub">Found and analysed, but not published until every rule for them is checked.</p>
<div class="tablebox"><table class="held"><thead class="sr"><tr><th scope=col>Model</th><th scope=col>Reason</th></tr></thead><tbody>{held_rows or "<tr><td colspan=2>Nothing held.</td></tr>"}</tbody></table></div></section>"""
    return frame(
        "ArchWatch: open model architectures from their own files",
        body,
        active="models",
        description="Architectures of new open-weight models, read from config, weights and code and checked.",
    )


# ---------------------------------------------------------------- model page


def sources(c):
    words = []
    for x in c["candidates"]:
        word = SOURCE_WORDS.get(x["evidence"].split(":", 1)[0], x["evidence"].split(":", 1)[0])
        if word not in words:
            words.append(word)
    detail = "; ".join(x["evidence"].split(":", 1)[-1] for x in c["candidates"])
    return ", ".join(words), detail


def facts_table(rec, pub, cache):
    checks = {c["field"]: c for c in rec["checks"]}
    order = [f for f in FIRST if f in checks] + sorted(f for f in checks if f not in FIRST)
    rows = []
    for field in order:
        c = checks[field]
        if field in NOT_IN_TABLE:
            continue
        status, note = card_status(c)
        word, detail = sources(c)
        if field in pub:
            value = show(field, pub[field])
            label = STATUS_WORDS.get(status, status)
        else:
            value, label = "held for review", "held for review"
        extra = f'<span class="note">{e(note)}; recorded, not a gate</span>' if note else ""
        rows.append(
            f"<tr><th scope=row>{e(FIELD_NAMES.get(field, field.replace('_', ' ').capitalize()))}</th>"
            f'<td class="mono">{e(value)}</td><td>{e(word)}<span class="detail">{e(detail)}</span>{extra}</td>'
            f'<td><span class="st st-{label.replace(" ", "-")}">{e(label)}</span></td></tr>'
        )
    rows.append(
        f"<tr><th scope=row>Cache per token</th><td class=mono>{cache['grows_per_token_bytes']:,} B</td>"
        f"<td>ArchWatch calculator<span class=detail>bf16; growing part per token (sliding-window and constant states excluded), from published fields</span></td>"
        f'<td><span class="st st-computed">computed</span></td></tr>'
    )
    return "".join(rows)


def sentence(text):
    return text[:1].upper() + text[1:] + "."


def memory_picker(pub, cache):
    ctx = int(pub["context_length"])
    points = [n for n in settings()["memory_contexts"] if n < ctx] + [ctx]
    default = 131072 if 131072 in points else points[-1]
    options = []
    for n in points:
        total = cache["grows_per_token_bytes"] * n + cache["constant_bytes"]
        options.append(
            f'<button type="button" class="pill" aria-pressed="{"true" if n == default else "false"}" '
            f'data-bytes="{e(size(total))}" data-tokens="{n:,}">{e(tokens(n))} tokens{" (max)" if n == ctx else ""}</button>'
        )
    first = cache["grows_per_token_bytes"] * default + cache["constant_bytes"]
    grow = [r for r in cache["breakdown"] if r["grows_per_token_bytes"]]
    fixed = [r for r in cache["breakdown"] if not r["grows_per_token_bytes"]]
    parts = [
        f"{LABEL.get(r['layer_type'], r['layer_type'])} layers ({r['layers']}) add {r['grows_per_token_bytes']:,} B per token"
        for r in grow
    ]
    if fixed:
        parts.append(
            "; ".join(
                f"the {r['layers']} {LABEL.get(r['layer_type'], r['layer_type'])} layers keep a fixed-size state "
                f"({size(r['constant_bytes'])}) that does not grow with context"
                for r in fixed
                if r["constant_bytes"]
            )
            or "the other layers keep no cache"
        )
    capped = [r for r in cache["breakdown"] if r["layer_type"] == "sliding_attention"]
    if capped:
        parts.append("sliding-window layers stop growing at their window")
    return f"""<section class="panel" aria-labelledby="mem"><h2 id="mem">How much memory does the cache need?</h2>
<p class="sub">One sequence, bf16, weights not included. Pick a context length.</p>
<div class="pills" role="group" aria-label="Context length">{"".join(options)}</div>
<p class="big mono" aria-live="polite"><span class="membytes">{e(size(first))}</span>
<span class="sr"> at <span class="memtokens">{default:,}</span> tokens</span></p>
<p class="sub">{e(sentence("; ".join(parts)))}</p></section>"""


def model_page(stem, rec, pub, published):
    cache = render_cache(pub)
    org, name = rec["id"].split("/")
    pubs = settings()["publishers"]
    before = published_predecessor(rec, published)
    checks = {c["field"]: c for c in rec["checks"]}
    key = [f for f in KEY_FACTS if f in pub]
    verified = [f for f in key if card_status(checks[f])[0] == "verified"]
    badge = (
        "All key facts verified" if len(verified) == len(key) else f"{len(verified)} of {len(key)} key facts verified"
    )
    active = (
        (human(pub["params_active"]), "active per token")
        if "params_active" in pub
        else ("all", "active per token (dense)")
    )
    tiles = [
        (human(pub["params_total"]), "total parameters"),
        active,
        (str(pub["num_layers"]), "layers"),
        (tokens(pub["context_length"]), "context tokens"),
        (size(cache["grows_per_token_bytes"]), "cache per token"),
    ]
    t_html = "".join(f"<div><dt>{e(b)}</dt><dd class=mono>{e(a)}</dd></div>" for a, b in tiles)
    pattern = pattern_text(pub)
    mix_rows = "".join(
        f'<li><span class="sw" style="background:{COLOR.get(k, "#999")}" aria-hidden="true"></span>{e(LABEL.get(k, k))}'
        f'<span class="mono">{pub["layer_type_counts"][k]} layers</span></li>'
        for k in kinds_in_order(pub)
    )
    mix_box = f"""<aside class="panel mixbox" aria-labelledby="mix-{e(stem)}"><h3 id="mix-{e(stem)}">Layer mix</h3>
{mix_bar(pub, "Layer mix, layer 1 on the left: " + mix_text(pub))}<ul class="mixlist">{mix_rows}
{f"<li>Pattern<span>{e(pattern)}</span></li>" if pattern else ""}</ul></aside>"""
    if before:
        pred_name = before[1]["id"].split("/")[-1]
        note = explainer_note(rec)
        if note:
            text = f'<p>{e(note)}</p><p class="byline">Written by Gemini from the verified diff; every number checked by code.</p>'
        else:
            changed = pipeline.diff(rec, before[1])
            items = "".join(
                f"<li>{e(FIELD_NAMES.get(r['field'], r['field']))}: "
                f"{e(show(r['field'], r['before']) if r['before'] is not None else 'not published')} &rarr; "
                f"{e(show(r['field'], r['after']) if r['after'] is not None else 'not published')}</li>"
                for r in changed
                if r["field"] in FIELD_NAMES
            )
            text = (
                f"<p>{e(summary_line(pub, before[2]))}</p><ul class=changes>{items}</ul>"
                f'<p class="byline">Listed by code from the verified diff; no written explanation has passed the checks yet.</p>'
            )
        left = f"<div><h2>What changed since {e(pred_name)}</h2>{text}</div>"
        compare = f'<a class="button" href="../compare/{e(stem)}.html">Compare with {e(pred_name)}</a>'
    else:
        left = f"<div><h2>Overview</h2><p>{e(summary_line(pub))}</p><p class=byline>Written by code from published fields.</p></div>"
        compare = ""
    chosen = generated.select(rec)
    if chosen:
        svg_path = ROOT / "site/diagrams" / f"{stem}.svg"
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(svg_path, chosen[0])
        n = len(chosen[1]["checks_passed"])
        art = f'<img src="../diagrams/{e(stem)}.svg" alt="Architecture diagram of {e(name)}: {e(mix_text(pub))}">'
        art_label = f"Drawn by Gemini from verified fields, passed {n} of {n} checks"
    else:
        art = diagram.render(rec)
        art_label = "Drawn by code from verified fields; no generated diagram has passed every check yet"
    commit = rec["source_commit"]
    body = f"""<nav class="crumbs" aria-label="Breadcrumb"><a href="../index.html#models">Models</a> / {e(pubs.get(org, org))}</nav>
<div class="titlerow"><div><h1>{e(name)}</h1>
<p class="sub"><a href="https://huggingface.co/{e(rec["id"])}">{e(rec["id"])}</a> · released {e(date(rec["hub_created"]))} ·
analysed at commit <a class="mono" href="https://huggingface.co/{e(rec["id"])}/tree/{e(commit)}">{e(commit[:10])}</a></p></div>
<div class="actions"><span class="badge">{e(badge)}</span>{compare}</div></div>
<dl class="stats tiles">{t_html}</dl>
<section class="split">{left}{mix_box}</section>
<section aria-labelledby="arch"><div class="sechead"><h2 id="arch">Architecture</h2><p class="hint">{e(art_label)}</p></div>
<figure class="panel diagram">{art}</figure></section>
<section aria-labelledby="facts"><h2 id="facts">Where every number comes from</h2>
<p class="sub">Each fact is read from the model's own files. Verified means two independent sources agree.</p>
<div class="tablebox"><table class="facts"><thead><tr><th scope=col>Fact</th><th scope=col>Value</th><th scope=col>Sources</th><th scope=col>Status</th></tr></thead>
<tbody>{facts_table(rec, pub, cache)}</tbody></table></div></section>
{memory_picker(pub, cache)}"""
    return frame(f"{name} architecture · ArchWatch", body, depth=1, active="models", description=summary_line(pub))


def model_section(name, anchor):
    """Body of one model page from data/models/<name>.json (raises PublicationError when it is not publishable)."""
    rec = json.loads((ROOT / "data/models" / f"{name}.json").read_text())
    pub = render_values(rec)
    return model_page(name, rec, pub, {})


# ---------------------------------------------------------------- compare


def stack(rec, pub):
    cells = "".join(
        f'<span style="background:{COLOR.get(k, "#999")}" title="Layer {i + 1}: {e(LABEL.get(k, k))}"></span>'
        for i, k in enumerate(pub["layer_types"])
    )
    name = rec["id"].split("/")[-1]
    return f"""<div class="panel stack"><div class="stackhead"><h2>{e(name)}</h2><span class="mono">{pub["num_layers"]} layers</span></div>
<div class="cells" role="img" aria-label="{e(name)} layer stack, layer 1 first: {e(mix_text(pub))}">{cells}</div>
<p class="sub">{e(" · ".join(f"{LABEL.get(k, k)} {pub['layer_type_counts'][k]}" for k in kinds_in_order(pub)))}</p></div>"""


def dense_word(field, value, pub):
    """Dense models publish no active count or expert fields: say so instead of 'not published'."""
    if value is None and "moe_num_experts" not in pub and (field == "params_active" or field.startswith("moe_")):
        return "all (dense)" if field == "params_active" else "none (dense)"
    return value


def compare_page(stem, rec, pub, before):
    _, old, old_pub = before
    old_name, new_name = old["id"].split("/")[-1], rec["id"].split("/")[-1]
    changed = {r["field"]: r for r in pipeline.diff(rec, old)}
    fields = [f for f in ("params_total", "params_active", "num_layers", "context_length") if f in pub or f in old_pub]
    fields += [f for f in changed if f not in fields and f in FIELD_NAMES]
    old_cache, new_cache = render_cache(old_pub), render_cache(pub)
    rows = []
    for f in fields:
        a, b = old_pub.get(f), pub.get(f)
        if a is None and b is None:
            continue
        held_old = a is None and f in {r["field"] for r in old["review_queue"]}
        a, b = dense_word(f, a, old_pub), dense_word(f, b, pub)
        word, cls = ("held", "warn") if held_old else change_label(f, a, b)
        rows.append(
            f"<tr><th scope=row>{e(FIELD_NAMES.get(f, f))}</th>"
            f"<td class=mono>{e('held for review' if held_old else show(f, a) if a is not None else 'not published')}</td>"
            f"<td class=mono>{e(show(f, b) if b is not None else 'not published')}</td>"
            f'<td><span class="chip chip-{cls}">{e(word)}</span></td></tr>'
        )
    ga, gb = old_cache["grows_per_token_bytes"], new_cache["grows_per_token_bytes"]
    word, cls = change_label("kv", ga, gb)
    rows.append(
        f"<tr><th scope=row>Cache per token</th><td class=mono>{ga:,} B</td><td class=mono>{gb:,} B</td>"
        f'<td><span class="chip chip-{cls}">{e(word)}</span></td></tr>'
    )
    body = f"""<h1>{e(old_name)} &rarr; {e(new_name)}</h1>
<p class="lead">{e(summary_line(pub, old_pub))}</p>
<div class="pair">{stack(old, old_pub)}{stack(rec, pub)}</div>
<div class="tablebox"><table class="facts"><caption class="sr">Changed fields</caption><thead><tr><th scope=col>Field</th>
<th scope=col>{e(old_name)}</th><th scope=col>{e(new_name)}</th><th scope=col>Change</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="sub">Only published values are compared. A field held for review in either model is marked, not shown.
<a href="../models/{e(stem)}.html">Back to {e(new_name)}</a></p>"""
    return frame(f"{old_name} to {new_name} · ArchWatch", body, depth=1, active="compare")


def compare_index(pairs):
    items = (
        "".join(f'<li><a href="{e(stem)}.html">{e(old)} &rarr; {e(new)}</a></li>' for stem, old, new in pairs)
        or "<li>No published model has a published predecessor yet.</li>"
    )
    body = f"""<h1>Compare</h1><p class="lead">Each model next to its predecessor, block by block.</p>
<ul class="plainlist">{items}</ul>"""
    return frame("Compare · ArchWatch", body, depth=1, active="compare")


# ---------------------------------------------------------------- assets

CSS = """
:root{--bg:#F7F7F5;--fg:#111418;--mut:#5b6169;--line:#e3e3df;--card:#fff;--acc:#0B6E66;--soft:#efefeb}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 Inter,system-ui,sans-serif}
h1,h2,h3{font-family:"Space Grotesk",Inter,sans-serif;letter-spacing:-.01em;line-height:1.15;margin:0}
h1{font-size:clamp(34px,5vw,56px);font-weight:600}h2{font-size:26px;font-weight:600}h3{font-size:21px;font-weight:600}
a{color:var(--acc)}a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.mono,.num,.stats dd,.mini dd{font-family:"IBM Plex Mono",ui-monospace,monospace}
.wrap{max-width:1200px;margin:0 auto;padding:0 20px}
.skip{position:absolute;left:-999px}.skip:focus{left:12px;top:12px;background:#fff;padding:8px;z-index:9}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.top{background:#fff;border-bottom:1px solid var(--line)}
.topbar{display:flex;align-items:center;gap:28px;height:62px}
.brand{font:600 20px "Space Grotesk",sans-serif;color:var(--fg);text-decoration:none}
.top nav{display:flex;gap:22px}.top nav a{color:var(--mut);text-decoration:none;font-size:15px}
.top nav a[aria-current]{color:var(--fg);font-weight:500}.top nav a:hover{color:var(--fg)}
.lastrun{margin-left:auto;font-size:13px;color:var(--mut)}
main.wrap{padding:40px 20px 72px}
.hero h1{max-width:820px}.lead{font-size:19px;color:var(--mut);max-width:720px;margin:20px 0 32px}
.sub{color:var(--mut);margin:6px 0 16px}.hint{color:var(--mut);font-size:14px;margin:0}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:36px}
.controls input{flex:1 1 360px;max-width:520px;font:inherit;padding:12px 16px;border:1px solid var(--line);border-radius:8px;background:#fff}
.pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{font:inherit;font-size:15px;padding:7px 16px;border-radius:99px;border:1px solid var(--line);background:#fff;color:var(--fg);cursor:pointer}
.pill[aria-pressed=true]{background:var(--fg);color:#fff;border-color:var(--fg)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin:0 0 56px;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.stats>div{padding:20px 22px;border-right:1px solid var(--line);display:flex;flex-direction:column-reverse}
.stats>div:last-child{border-right:0}.stats dd{margin:0;font-size:28px;font-weight:500}.stats dt{color:var(--mut);font-size:14px}
.tiles{margin:28px 0 48px}
.sechead{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin:0 0 20px}
.cards{list-style:none;padding:0;margin:0 0 64px;display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px}
.cardlink{display:block;padding:22px;color:inherit;text-decoration:none;height:100%}
.cardlink:hover{background:#fcfcfa}.cardlink:hover h3{color:var(--acc)}
.cardhead{display:flex;justify-content:space-between;font-size:13px;color:var(--mut)}
.card h3{margin:10px 0 12px}.line{margin:0 0 16px;color:#2b3036;font-size:15.5px}
.bar{display:flex;gap:3px;height:9px;border-radius:99px;overflow:hidden}.bar span{display:block;min-width:2px}
.legend{list-style:none;padding:0;margin:10px 0 16px;display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13.5px}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}
.mini{display:grid;grid-template-columns:repeat(3,1fr);margin:0;padding-top:14px;border-top:1px solid var(--line)}
.mini div{display:flex;flex-direction:column-reverse}.mini dd{margin:0;font-size:16px}.mini dt{font-size:12.5px;color:var(--mut)}
.empty{color:var(--mut)}
.tablebox{background:#fff;border:1px solid var(--line);border-radius:12px;overflow-x:auto;margin-bottom:56px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:13px 20px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child>*{border-bottom:0}thead th{background:var(--soft);font-size:14px;font-weight:600}
tbody th{font-weight:400}.held tbody th{font-weight:500;width:30%}.held td{color:#3a4047;font-size:15px}
.facts td.mono{overflow-wrap:anywhere}.detail,.note{display:block;font-size:12.5px;color:var(--mut);word-break:break-word}
.note{color:#6b46c1}
.st,.chip{display:inline-block;font-size:13px;padding:1px 10px;border-radius:99px;white-space:nowrap}
.st-verified,.chip-up{background:#0B6E6618;color:#0B6E66}.st-one-source,.st-computed,.chip-warn{background:#f3e6c4;color:#7a5a00}
.st-differs-from-card{background:#ece4fb;color:#5b34b0}.st-held-for-review{background:#f8dcd7;color:#9b2c1f}
.chip-same{background:var(--soft);color:var(--mut)}.chip-new{background:#dde8f7;color:#244c86}
.crumbs{font-size:14px;color:var(--mut);margin-bottom:10px}
.titlerow{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap}
.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.badge{background:#0B6E6618;color:var(--acc);padding:6px 14px;border-radius:99px;font-size:14px;font-weight:500}
.badge::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--acc);margin-right:8px;vertical-align:1px}
.button{border:1px solid var(--line);background:#fff;color:var(--fg);padding:6px 14px;border-radius:99px;font-size:14px;text-decoration:none}
.split{display:grid;grid-template-columns:1.4fr 1fr;gap:40px;margin-bottom:56px;align-items:start}
.split p{color:#2b3036}.split h2{margin-bottom:12px}.byline{font-size:13px;color:var(--mut)}
.changes{padding-left:20px;color:#2b3036}
.panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px 24px}
.mixbox h3{font-size:17px;margin-bottom:14px}.mixlist{list-style:none;padding:0;margin:16px 0 0}
.mixlist li{display:flex;align-items:center;padding:4px 0}.mixlist li>span:last-child{margin-left:auto;color:var(--mut)}
.diagram{margin:0 0 56px;overflow-x:auto}.diagram img,.diagram svg{display:block;width:100%;height:auto;min-width:720px}
.big{font-size:44px;margin:18px 0 8px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:32px 0}
.stackhead{display:flex;justify-content:space-between;align-items:baseline}.stack h2{font-size:20px}
.cells{display:flex;flex-wrap:wrap;gap:4px;margin:16px 0 8px}.cells span{width:28px;height:18px;border-radius:3px}
.plainlist{font-size:18px}
.foot{border-top:1px solid var(--line);background:#fff;color:var(--mut);font-size:14px;padding:28px 0}
.cols2{display:grid;grid-template-columns:1.2fr 1fr;gap:48px}
@media (max-width:800px){.split,.pair,.cols2{grid-template-columns:1fr}.lastrun{display:none}.topbar{gap:16px;overflow-x:auto}
.stats>div{border-right:0;border-bottom:1px solid var(--line)}.big{font-size:34px}}
.split>*,.pair>*,.cols2>*,.titlerow>*{min-width:0}h1,.sub{overflow-wrap:anywhere}.diagram{max-width:100%}
@media (prefers-reduced-motion:no-preference){.cardlink{transition:background .15s}}
"""

JS = """
document.querySelectorAll('.panel .pills').forEach(function(group){
  var panel=group.closest('.panel'),out=panel.querySelector('.membytes'),tok=panel.querySelector('.memtokens');
  group.addEventListener('click',function(ev){var b=ev.target.closest('button');if(!b)return;
    group.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});
    out.textContent=b.dataset.bytes;tok.textContent=b.dataset.tokens;});
});
(function(){var box=document.querySelector('.controls');if(!box)return;box.hidden=false;
  var q=document.getElementById('q'),pills=box.querySelectorAll('.pill'),cards=document.querySelectorAll('.card'),
      empty=document.querySelector('.empty'),filter='All';
  function apply(){var t=q.value.trim().toLowerCase(),n=0;cards.forEach(function(c){
      var ok=(filter==='All'||c.dataset.tags.split('|').indexOf(filter)>=0)&&(!t||c.dataset.search.indexOf(t)>=0);
      c.hidden=!ok;if(ok)n++;});empty.hidden=n>0;}
  q.addEventListener('input',apply);
  pills.forEach(function(p){p.addEventListener('click',function(){filter=p.dataset.filter;
    pills.forEach(function(x){x.setAttribute('aria-pressed',x===p?'true':'false')});apply();});});
})();
"""


def main():
    published, held, seen = inventory()
    site = ROOT / "site"
    write_if_changed(site / "style.css", CSS.lstrip())
    write_if_changed(site / "site.js", JS.lstrip())
    write_if_changed(site / "index.html", home(published, held, seen))
    pairs = []
    expected = {"index.html"}
    for repo, (stem, rec, pub) in published.items():
        write_if_changed(site / "models" / f"{stem}.html", model_page(stem, rec, pub, published))
        before = published_predecessor(rec, published)
        if before:
            write_if_changed(site / "compare" / f"{stem}.html", compare_page(stem, rec, pub, before))
            pairs.append((stem, before[1]["id"].split("/")[-1], rec["id"].split("/")[-1]))
            expected.add(f"{stem}.html")
    write_if_changed(site / "compare" / "index.html", compare_index(sorted(pairs)))
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
