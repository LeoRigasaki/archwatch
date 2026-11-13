"""Architecture diagrams generated from a verified record (no hand drawing, no LLM).

Two parts per model:
  1. layer map  - one cell per decoder layer, coloured by mixer type, read from the verified pattern
  2. block view - the decoder block with one box per mixer type and the FFN, labelled from verified fields
Output is a standalone SVG; colours come from CSS classes so the page can theme it.
"""

from __future__ import annotations

from html import escape

from .publication import render_values

KIND = {
    "kda_linear": ("KDA linear attention", "k-kda"),
    "gated_dsa_mla": ("Gated DSA (sparse, on MLA)", "k-dsa"),
    "full_attention_mla": ("MLA (full)", "k-mla"),
    "short_conv": ("Gated short conv", "k-conv"),
    "sliding_attention": ("Sliding-window attention", "k-swa"),
    "full_attention": ("Global attention", "k-full"),
}
STYLE = """
.t{font:600 17px system-ui,-apple-system,Segoe UI,sans-serif;fill:#1f2328}
.h{font:600 13px system-ui,-apple-system,Segoe UI,sans-serif;fill:#1f2328}
.b{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:#1f2328}
.q{font:11.5px system-ui,-apple-system,Segoe UI,sans-serif;fill:#59636e}
.edge{stroke:#8c959f;stroke-width:1.25;fill:none}
.box{fill:#ffffff;stroke:#8c959f;stroke-width:1.25}
.cont{fill:#f6f8fa;stroke:#8c959f;stroke-width:1;stroke-dasharray:4 4}
.k-kda{fill:#2a9d8f}.k-dsa{fill:#e76f51}.k-mla{fill:#f4a261}.k-conv{fill:#5b7fbf}.k-swa{fill:#8ab17d}.k-full{fill:#e76f51}
.mix.k-kda{fill:#2a9d8f22;stroke:#2a9d8f}.mix.k-dsa{fill:#e76f5122;stroke:#e76f51}.mix.k-mla{fill:#f4a26122;stroke:#f4a261}
.mix.k-conv{fill:#5b7fbf22;stroke:#5b7fbf}.mix.k-swa{fill:#8ab17d22;stroke:#8ab17d}.mix.k-full{fill:#e76f5122;stroke:#e76f51}
.mix{stroke-width:2}
"""


def _t(x, y, s, cls="b", anchor="middle"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(str(s))}</text>'


def _fmt_int(n):
    return f"{n:,}"


def _human(n):
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= d:
            return f"{n / d:.2f}{unit}".replace(".00", "")
    return str(n)


def _mixer_lines(kind: str, v: dict) -> list[str]:
    gate = "output gate" if v.get("attention_output_gate") else "gate status unverified"
    additions = " + ".join(
        label for field, label in (("qk_norm", "QK-norm"), ("attention_output_gate", "output gate")) if v.get(field)
    )
    if kind == "kda_linear":
        return [
            f"{v['linear_attn_num_heads']} heads x {v['linear_attn_head_dim']} dims",
            f"fixed {v['linear_attn_head_dim']}x{v['linear_attn_head_dim']} state per head",
            f"short conv k={v.get('linear_attn_conv_kernel')} on q, k, v",
            "cache does not grow with context",
        ]
    if kind == "gated_dsa_mla":
        return [
            f"MLA: caches {v['mla_kv_lora_rank']} latent + {v['mla_qk_rope_head_dim']} RoPE",
            f"indexer {v['dsa_index_num_heads']} heads x {v['dsa_index_head_dim']}",
            f"each query reads top-{v['dsa_index_topk']} tokens",
            gate,
        ]
    if kind == "full_attention_mla":
        return [
            f"MLA: caches {v['mla_kv_lora_rank']} latent + {v['mla_qk_rope_head_dim']} RoPE",
            f"{v['num_attention_heads']} heads, q/k {v['head_dim']}, v {v.get('v_head_dim')}",
            "attends to every token",
            gate,
        ]
    rope = v.get("rope") or {}
    if kind == "short_conv":
        return [
            f"depthwise conv, kernel {v.get('short_conv_kernel')}",
            "gated: in_proj x3, out_proj",
            "keeps last few inputs",
            "cache does not grow",
        ]
    if kind == "sliding_attention":
        th = (rope.get("sliding_attention") or {}).get("rope_theta")
        return [
            f"{v.get('attention_type', 'Attention')} {v['num_attention_heads']} q / {v['num_kv_heads']} kv, head {v['head_dim']}",
            f"window {v.get('sliding_window')} tokens",
            f"RoPE theta {int(th):,}" if th else "RoPE",
            additions,
        ]
    if kind == "full_attention":
        r = rope.get("full_attention") or {}
        pr = r.get("partial_rotary_factor")
        return [
            f"{v.get('attention_type', 'Attention')} {v['num_attention_heads']} q / {v['num_kv_heads']} kv, head {v['head_dim']}",
            "attends to every token",
            (f"partial RoPE {pr:g}, theta {int(r.get('rope_theta')):,}" if pr else "RoPE"),
            additions,
        ]
    return ["no rule for this layer type"]


def render(rec: dict) -> str:
    v = render_values(rec)
    types: list[str] = v["layer_types"]
    n = len(types)
    W = 960
    out = []
    name = rec["id"].split("/")[-1]
    total = v["params_total"]
    act = "params_active" in v
    sub = (
        f"{_human(total)} parameters"
        + (f", {_human(v['params_active'])} active per token" if act else "")
        + f" · {n} layers · context {_fmt_int(v.get('context_length', 0))}"
    )
    out.append(_t(24, 32, f"{name}: architecture", "t", "start"))
    out.append(_t(24, 52, sub, "q", "start"))

    # 1. layer map
    y0, x0, wmap = 84, 24, W - 48
    out.append(_t(24, y0 - 8, "Layer map (layer 1 on the left)", "h", "start"))
    cw = wmap / n
    for i, kind in enumerate(types):
        cls = KIND.get(kind, ("?", "k-full"))[1]
        out.append(
            f'<rect x="{x0 + i * cw + 0.6:.2f}" y="{y0}" width="{max(cw - 1.2, 1):.2f}" height="34" rx="2" class="{cls}"/>'
        )
    step = 1 if n <= 30 else (4 if n <= 60 else 12)
    for i in range(0, n, step):
        out.append(_t(x0 + i * cw + cw / 2, y0 + 50, i + 1, "q"))
    if n % step != 1 and step > 1:
        out.append(_t(x0 + (n - 1) * cw + cw / 2, y0 + 50, n, "q"))
    dense = v.get("dense_layers") or 0
    if dense and "moe_num_experts" in v:
        out.append(f'<path d="M{x0 + 0.6} {y0 + 38}H{x0 + dense * cw - 0.6}" stroke="#1f2328" stroke-width="3"/>')
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    lx, ly = 24, y0 + 72
    for kind, c in counts.items():
        label, cls = KIND.get(kind, (kind, "k-full"))
        out.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" class="{cls}"/>')
        txt = f"{label} x {c}"
        out.append(_t(lx + 18, ly, txt, "b", "start"))
        lx += 18 + len(txt) * 6.6 + 24
    if dense and "moe_num_experts" in v:
        out.append(_t(lx, ly, f"(black bar: layer 1 has a dense FFN; layers {dense + 1}-{n} use MoE)", "q", "start"))

    # 2. block view
    kinds = list(counts)
    top = ly + 34
    out.append(_t(24, top, "Decoder block", "h", "start"))
    cx = W / 2
    y = top + 18

    def box(x, yy, w, h, title, lines, cls="box"):
        s = [
            f'<rect x="{x:.1f}" y="{yy:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" class="{cls}"/>',
            _t(x + w / 2, yy + (h / 2 + 4.5 if not lines else 20), title, "h"),
        ]
        for j, ln in enumerate(lines):
            s.append(_t(x + w / 2, yy + 38 + 16 * j, ln, "q"))
        return s

    # LM head on top
    lm = "LM head (tied to embedding)" if v.get("tied_embeddings") else "LM head"
    out += box(cx - 170, y, 340, 44, f"Final norm + {lm}", [])
    y_lm_bottom = y + 44
    y += 44 + 28

    # container
    mix_h = 40 + 16 * 4 + 10
    ffn_lines = []
    if "moe_num_experts" in v:
        ffn_title = (
            f"MoE FFN: {v['moe_num_experts']} experts, top-{v['moe_active_experts']} + {v['moe_shared_experts']} shared"
        )
        lat = v.get("moe_latent_dim")
        if lat:
            ffn_lines.append(
                f"routed experts run in a {lat}-dim latent: {v['hidden_size']} -> {lat} -> "
                f"{v['moe_expert_intermediate']} -> {lat} -> {v['hidden_size']}"
            )
        card_name = f" (card: {v['activation_card_name']})" if v.get("activation_card_name") else ""
        ffn_lines.append(f"{v.get('moe_router', '')} router · activation {v.get('activation', '')}" + card_name)
        if v.get("dense_layers", 0):
            ffn_lines.append(
                f"dense prefix: FFN {v['hidden_size']} -> {v['dense_ffn_intermediate']} -> {v['hidden_size']}"
            )
    else:
        ffn_title = f"Gated FFN: {v['hidden_size']} -> {v.get('ffn_intermediate')} -> {v['hidden_size']}"
        ffn_lines.append("w1 and w3 in parallel, gated, then w2")
    ffn_h = 40 + 16 * len(ffn_lines)
    sandwich = bool(v.get("sandwich_norm"))
    norm_h = 30
    inner = [norm_h, mix_h, norm_h if sandwich else 0, norm_h, ffn_h, norm_h if sandwich else 0]
    gaps = 18
    cont_h = 36 + sum(h for h in inner if h) + gaps * (len([h for h in inner if h]) - 1) + 24 + 26
    cont_y = y
    out.append(f'<rect x="24" y="{cont_y}" width="{W - 48}" height="{cont_h}" rx="10" class="cont"/>')
    attn_res = v.get("parameter_components", {}).get("attention_residuals", 0)
    title = (
        (f"repeated {n} times; learned residual-weight tensors present; mechanism unverified")
        if attn_res
        else f"repeated {n} times; residual connections around the mixer and the FFN"
    )
    out.append(_t(40, cont_y + 22, title, "h", "start"))
    yy = cont_y + 36

    stack = []  # (y_top, y_bottom) of each element, bottom of block first in data flow

    def norm(label):
        nonlocal yy
        s = box(cx - 120, yy, 240, norm_h, label, [])
        s[1] = _t(cx, yy + 19, label, "b")
        stack.append((yy, yy + norm_h))
        yy += norm_h + gaps
        return s

    # drawn top-down = output side first
    if sandwich:
        out += norm("Norm (post-FFN)")
    out += box(cx - 330, yy, 660, ffn_h, ffn_title, ffn_lines)
    stack.append((yy, yy + ffn_h))
    yy += ffn_h + gaps
    out += norm("Norm")
    if sandwich:
        out += norm("Norm (post-mixer)")
    k = len(kinds)
    yy += 26  # room for the merge line and the caption above the mixers
    mw = min(300, (W - 48 - 32 - 16 * (k - 1)) / k)
    mx0 = cx - (mw * k + 16 * (k - 1)) / 2
    for j, kind in enumerate(kinds):
        label, cls = KIND.get(kind, (kind, "k-full"))
        out += box(
            mx0 + j * (mw + 16), yy, mw, mix_h, f"{label} ({counts[kind]} layers)", _mixer_lines(kind, v), f"mix {cls}"
        )
    out.append(_t(mx0, yy - 6, "one mixer per layer", "q", "start"))
    mixer_centers = [mx0 + j * (mw + 16) + mw / 2 for j in range(k)]
    stack.append((yy, yy + mix_h))
    yy += mix_h + gaps
    out += norm("Norm (pre-mixer)")

    # arrows between stacked elements (data flows upward)
    mix_i = len(stack) - 2  # the mixer row is second from the bottom
    for i in range(len(stack) - 1):
        upper_bottom = stack[i][1]
        lower_top = stack[i + 1][0]
        if i == mix_i - 1 and k > 1:  # mixers -> norm above: one line from each mixer
            mid = (upper_bottom + lower_top) / 2
            for mcx in mixer_centers:
                out.append(
                    f'<path d="M{mcx:.1f} {lower_top}V{mid}H{cx}V{upper_bottom + 2}" class="edge" marker-end="url(#ar)"/>'
                )
        elif i == mix_i and k > 1:  # norm below -> each mixer
            mid = (upper_bottom + lower_top) / 2
            for mcx in mixer_centers:
                out.append(
                    f'<path d="M{cx} {lower_top}V{mid}H{mcx:.1f}V{upper_bottom + 2}" class="edge" marker-end="url(#ar)"/>'
                )
        else:
            out.append(f'<path d="M{cx} {lower_top}V{upper_bottom + 2}" class="edge" marker-end="url(#ar)"/>')
    # input to block and block to LM head
    out.append(f'<path d="M{cx} {cont_y}V{y_lm_bottom + 2}" class="edge" marker-end="url(#ar)"/>')
    y = cont_y + cont_h + 28
    out.append(f'<path d="M{cx} {y}V{stack[-1][1] + 2}" class="edge" marker-end="url(#ar)"/>')

    emb_rows = v.get("vocab_size")
    out += box(cx - 170, y, 340, 44, f"Token embedding {_fmt_int(emb_rows)} x {v['hidden_size']}", [])
    y += 44
    if v.get("vision_encoder_layers"):
        vp = v.get("parameter_components", {})
        out.append(f'<path d="M{cx} {y + 32}V{y + 2}" class="edge" marker-end="url(#ar)"/>')
        out.append(_t(cx + 8, y + 20, "image tokens join the token sequence", "q", "start"))
        out += box(
            cx - 170,
            y + 32,
            340,
            60,
            "Vision encoder + projector",
            [
                f"{v['vision_encoder_layers']} layers, {_human(vp.get('vision', 0))} + {_human(vp.get('vision_projector', 0))} params"
            ],
        )
        y += 32 + 60
    y += 30
    for note in rec.get("unchecked_notes", [])[:1]:
        words, line = ("Note: " + note).split(), ""
        for wd in words:
            if len(line) + len(wd) > 140:
                out.append(_t(24, y, line, "q", "start"))
                y += 16
                line = ""
            line += wd + " "
        out.append(_t(24, y, line, "q", "start"))
        y += 16
    H = y + 12
    defs = (
        '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#8c959f"/></marker></defs>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}" role="img" '
        f'aria-label="{escape(name)} architecture"><style>{STYLE}</style>{defs}'
        f'<rect width="{W}" height="{H:.0f}" fill="#ffffff"/>' + "".join(out) + "</svg>"
    )
