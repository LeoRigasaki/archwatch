"""Rich diagram specs built from published record fields only (no LLM).

The spec is what Gemini draws from. Every number in it comes from a published, verifier-approved
field: scalar config facts, the published per-layer-type tensor shapes (`block_shapes`) or the cache
estimate recomputed from published values. Words describing a mechanism are fixed per layer type
in this module; they never introduce a number.
"""

from __future__ import annotations

import re

import yaml

from .publication import PublicationError, render_cache, render_values


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------- formatting


def _commas(n) -> str:
    return f"{int(n):,}"


def _short_count(n: int) -> str:
    """1,048,576 -> 1M, 262,144 -> 256K, 128,000 -> 128K, 163,840 -> 160K (exact units only)."""
    for base, units in ((1000, ("K", "M")), (1024, ("K", "M"))):
        for power, unit in ((2, units[1]), (1, units[0])):
            div = base**power
            if n % div == 0 and n >= div:
                return f"{n // div}{unit}"
    return _commas(n)


def _params(n: float, digits=None) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if n >= div:
            x = n / div
            if digits is not None:
                return f"{x:.{digits}f}{unit}"
            return f"{x:.0f}{unit}" if x >= 10 else f"{x:.2f}{unit}"
    return _commas(n)


def _bytes(n: int) -> str:
    for unit, div in (("MB", 1024**2), ("KB", 1024)):
        if n >= div:
            x = n / div
            return f"{x:.0f} {unit}" if abs(x - round(x)) < 0.05 or x >= 100 else f"{x:.1f} {unit}"
    return f"{n} B"


def _layers(ranges) -> str:
    parts = [f"{a + 1}" if a == b else f"{a + 1}-{b + 1}" for a, b in ranges]
    single = len(ranges) == 1 and ranges[0][0] == ranges[0][1]
    return ("layer " if single else "layers ") + ", ".join(parts)


def _linear(shape) -> str:
    out, inp = shape[0], shape[1]
    return f"Linear {inp} -> {out}"


# ---------------------------------------------------------------- layer pattern


def compress(sequence: list, labels: list | None = None) -> list[dict]:
    """Groups with repeat counts covering `sequence` exactly once, bottom to top.

    The repeating unit is found on `sequence` (the mixer types). Among decompositions
    prefix + unit x k + suffix, take the largest coverage, then the smallest unit, then the most
    balanced prefix/suffix. Prefix and suffix are compressed recursively. When `labels` (the full
    per-layer labels) distinguish one repeat from the others, that repeat becomes its own group.
    """
    labels = labels or list(sequence)
    n = len(sequence)
    best = None
    for p in range(1, n // 2 + 1):
        for s in range(0, n - 2 * p + 1):
            unit = sequence[s : s + p]
            k = 1
            while sequence[s + k * p : s + (k + 1) * p] == unit:
                k += 1
            if k < 2:
                continue
            prefix, suffix = s, n - s - k * p
            score = (k * p, -p, -max(prefix, suffix), -prefix)
            if best is None or score > best[0]:
                best = (score, s, p, k)
    if best is None:
        return [dict(repeat=1, start=0, layers=list(labels))] if n else []
    _, s, p, k = best
    groups = [dict(g, start=g["start"]) for g in compress(sequence[:s], labels[:s])]
    for i in range(k):
        block = labels[s + i * p : s + (i + 1) * p]
        if groups and groups[-1]["layers"] == block and groups[-1]["start"] + groups[-1]["repeat"] * p == s + i * p:
            groups[-1]["repeat"] += 1
        else:
            groups.append(dict(repeat=1, start=s + i * p, layers=block))
    tail = s + k * p
    groups += [dict(g, start=g["start"] + tail) for g in compress(sequence[tail:], labels[tail:])]
    return groups


def _rule(sequence, names):
    """'Every 4th layer is X (layers 4, 8 ... 92), and layer 93 is also X. All other layers are Y.'"""
    kinds = list(dict.fromkeys(sequence))
    if len(kinds) != 2:
        return None
    minority = min(kinds, key=sequence.count)
    other = kinds[1 - kinds.index(minority)]
    pos = [i + 1 for i, k in enumerate(sequence) if k == minority]
    step = pos[0]
    run = 1
    while run < len(pos) and pos[run] == pos[0] + run * step:
        run += 1
    if step < 2 or run < 4:
        return None
    suffix = {2: "nd", 3: "rd"}.get(step, "th")
    text = f"Every {step}{suffix} layer is {names[minority]} (layers {pos[0]}, {pos[1]}, {pos[2]} ... {pos[run - 1]})"
    extra = pos[run:]
    if extra:
        text += f", and layer{'s' if len(extra) > 1 else ''} {', '.join(map(str, extra))} {'are' if len(extra) > 1 else 'is'} also {names[minority]}"
    return text + f". All other layers are {names[other]}."


# ---------------------------------------------------------------- names


def _attention_name(v):
    return v.get("attention_type") or "attention"


def _mixer_name(kind, v, kinds):
    if kind == "kda_linear":
        return "KDA"
    if kind == "gated_dsa_mla":
        return "Gated DSA"
    if kind == "full_attention_mla":
        return "MLA"
    if kind == "short_conv":
        return "ShortConv"
    if kind == "full_attention":
        return ("Global " if "sliding_attention" in kinds else "") + _attention_name(v)
    if kind == "sliding_attention":
        return "Sliding-window " + _attention_name(v)
    if kind == "mamba2":
        return "Mamba-2"
    if kind == "mlp":
        return "MLP"
    raise SpecError(f"No diagram-spec rule for layer type {kind!r}")


def _is_moe(tensors):
    return any(k.startswith("block_sparse_moe.") or ".gate.weight" in k or "router" in k for k in tensors)


def _activation(v):
    if v.get("activation_card_name"):
        return v["activation_card_name"]
    act, gated = v.get("activation"), v.get("ffn_gated")
    if act == "silu" and gated:
        return "SwiGLU"
    if act == "gelu" and gated:
        return "GeGLU"
    if act == "relu2":
        return "squared ReLU"
    return act


def _ffn_name(group, v, moe_model):
    if _is_moe(group["tensors"]):
        return "LatentMoE" if v.get("moe_latent_dim") else "MoE"
    return "dense FFN" if moe_model else "FFN"


# ---------------------------------------------------------------- tensors


def _find(tensors, *suffixes):
    for suffix in suffixes:
        for name, shape in tensors.items():
            if name.endswith(suffix):
                return shape
    return None


def _need(tensors, *suffixes):
    shape = _find(tensors, *suffixes)
    if shape is None:
        raise SpecError("Published tensor shapes lack " + " / ".join(suffixes))
    return shape


# ---------------------------------------------------------------- zoom panels


def _kda_panel(t, v, count, cache_rows):
    q, k, val = (_need(t, f"self_attn.{x}_proj.weight") for x in "qkv")
    heads, hd = v["linear_attn_num_heads"], v["linear_attn_head_dim"]
    parts = []
    if q == k == val:
        parts.append(f"q, k, v projections: {_linear(q)} each")
    conv = _find(t, "q_conv1d.weight")
    if conv and _find(t, "k_conv1d.weight") and _find(t, "v_conv1d.weight"):
        parts.append(f"short depthwise Conv1d (kernel {conv[-1]}) on q, k and v separately")
    beta = _find(t, "self_attn.b_proj.weight")
    if beta:
        parts.append(f"beta: {_linear(beta)}" + (" (one per head)" if beta[0] == heads else ""))
    fa, fb = _find(t, "self_attn.f_a_proj.weight"), _find(t, "self_attn.f_b_proj.weight")
    if fa and fb:
        parts.append(f"decay gate: low-rank Linear {fa[1]} -> {fa[0]} -> {fb[0]}")
    grows = cache_rows.get("kda_linear", {}).get("grows_per_token_bytes")
    parts.append(
        f"gated delta rule (fixed {hd} x {hd} state per head"
        + ("; cache does not grow with context)" if grows == 0 else ")")
    )
    onorm = _find(t, "self_attn.o_norm.weight")
    if onorm:
        parts.append(f"output norm (per head, {onorm[0]})")
    gate = _find(t, "self_attn.g_proj.weight")
    if gate:
        parts.append(f"output gate: {_linear(gate)}, multiplied in")
    parts.append(f"output projection: {_linear(_need(t, 'self_attn.o_proj.weight'))}")
    return dict(title=f"Kimi Delta Attention (KDA), {count} layers", heads=f"{heads} heads x {hd} dims", parts=parts)


def _mla_parts(t, v):
    heads, hd, rope = v["num_attention_heads"], v["head_dim"], v["mla_qk_rope_head_dim"]
    vdim, latent = v.get("v_head_dim"), v["mla_kv_lora_rank"]
    parts = []
    qa, qb = _find(t, "self_attn.q_a_proj.weight"), _find(t, "self_attn.q_b_proj.weight")
    if qa and qb and qb[0] == heads * hd:
        norm = ", norm" if _find(t, "self_attn.q_a_layernorm.weight") else ""
        parts.append(f"query path: {_linear(qa)}{norm}, Linear {qb[1]} -> {heads} heads x {hd} ({hd - rope} + {rope})")
    kva = _need(t, "self_attn.kv_a_proj_with_mqa.weight")
    if kva[0] != latent + rope:
        raise SpecError("MLA latent tensor disagrees with published latent/rope widths")
    kv = f"KV path: {_linear(kva)} = {latent} latent + {rope} key dims"
    if _find(t, "self_attn.kv_a_layernorm.weight"):
        kv += f"; norm on the {latent} latent"
    kvb = _find(t, "self_attn.kv_b_proj.weight")
    if kvb and vdim and kvb[0] == heads * (hd - rope + vdim):
        kv += f"; Linear {kvb[1]} -> {heads} heads x ({hd - rope} key + {vdim} value)"
    parts.append(kv)
    if v.get("mla_use_nope") is True:
        parts.append("position encoding: NoPE (config mla_use_nope = true)")
    return parts


def _gate_and_out(t, parts):
    gate = _find(t, "self_attn.g_proj.weight")
    if gate:
        parts.append(f"output gate: {_linear(gate)}, multiplied in")
    parts.append(f"output projection: {_linear(_need(t, 'self_attn.o_proj.weight'))}")
    return parts


def _dsa_panel(t, v, count, cache_rows):
    parts = _mla_parts(t, v)
    ih, idim, top = v["dsa_index_num_heads"], v["dsa_index_head_dim"], v["dsa_index_topk"]
    wq, wk, wp = (_find(t, f"indexer.{x}.weight") for x in ("wq_b", "wk", "weights_proj"))
    detail = []
    if wq and wq[0] == ih * idim:
        detail.append(f"query from the {wq[1]} latent")
    if wk:
        detail.append(f"key {_linear(wk)}" + (" with norm" if _find(t, "indexer.k_norm.weight") else ""))
    if wp:
        detail.append(f"per-head weights {_linear(wp)}")
    parts.append(f"indexer: {ih} heads x {idim} dims" + (f" ({', '.join(detail)})" if detail else ""))
    parts.append(f"indexer picks the top {top} tokens; attention runs only on those")
    _gate_and_out(t, parts)
    kva = _need(t, "self_attn.kv_a_proj_with_mqa.weight")
    panel = dict(title=f"Gated DeepSeek Sparse Attention (DSA), {count} layers", parts=parts)
    if v.get("dsa_index_key_width"):
        panel["cache_per_token"] = f"{kva[0]} MLA values + {v['dsa_index_key_width']} indexer key values per layer"
    return panel


def _mla_panel(t, v, count, cache_rows):
    parts = _gate_and_out(t, _mla_parts(t, v))
    return dict(title=f"Multi-head Latent Attention (MLA), {count} layers", parts=parts)


def _attention_panel(t, v, counts, kinds):
    heads, kvh, hd = v["num_attention_heads"], v["num_kv_heads"], v["head_dim"]
    name = _attention_name(v)
    pre = "self_attn." if _find(t, "self_attn.q_proj.weight") else "mixer."
    q, k, val = (_need(t, f"{pre}{x}_proj.weight") for x in "qkv")
    parts = [f"{heads} query heads, {kvh} key/value heads, head size {hd}"]
    parts.append(
        f"q: {_linear(q)}; k and v: {_linear(k)} each"
        if k == val
        else f"q: {_linear(q)}; k: {_linear(k)}; v: {_linear(val)}"
    )
    if v.get("qk_norm") and _find(t, f"{pre}q_norm.weight"):
        parts.append(f"QK-norm on q and k (per head, {_find(t, f'{pre}q_norm.weight')[0]})")
    rope = v.get("rope") or {}
    full, sliding = counts.get("full_attention", 0), counts.get("sliding_attention", 0)
    if "sliding_attention" in kinds and "full_attention" in kinds:
        g, s = rope.get("full_attention") or {}, rope.get("sliding_attention") or {}
        pr = g.get("partial_rotary_factor")
        gl = "global layers: "
        if pr:
            gl += "partial rotary (half of each head)" if pr == 0.5 else f"partial rotary ({pr:g} of each head)"
        elif g:
            gl += "rotary"
        if g.get("rope_theta"):
            gl += f", theta {_commas(g['rope_theta'])}"
        parts.append(gl.replace(": , ", ": ") + ", attend to all tokens")
        sl = "sliding-window layers: "
        if s:
            sl += "full rotary" if not s.get("partial_rotary_factor") else "partial rotary"
            if s.get("rope_theta"):
                sl += f", theta {_commas(s['rope_theta'])}"
            sl += ", "
        parts.append(sl + f"attend to the last {v['sliding_window']} tokens only")
    elif rope.get("rope_theta"):
        text = f"rotary position encoding, theta {_commas(rope['rope_theta'])}"
        scaling = rope.get("scaling") or {}
        if scaling.get("rope_type") and scaling.get("factor"):
            text += f" ({scaling['rope_type']} scaling, factor {scaling['factor']:g})"
        parts.append(text)
    gate = _find(t, f"{pre}gate_proj.weight")
    if gate and v.get("attention_output_gate"):
        parts.append(f"head-wise output gate: {_linear(gate)}" + (" (one gate per head)" if gate[0] == heads else ""))
    parts.append(f"output projection: {_linear(_need(t, f'{pre}o_proj.weight'))}")
    if full and sliding:
        title = f"{name} attention ({full} global + {sliding} sliding-window layers)"
    else:
        title = f"{name} attention ({full or sliding} layers)"
    return dict(title=title, parts=parts)


def _short_conv_panel(t, v, count, cache_rows):
    inp, conv, out = _need(t, "conv.in_proj.weight"), _need(t, "conv.conv.weight"), _need(t, "conv.out_proj.weight")
    hidden = v["hidden_size"]
    if inp[0] != 3 * hidden:
        raise SpecError("ShortConv in_proj is not the three-way gated split this rule describes")
    parts = [
        f"in_proj: {_linear(inp)}, split into three parts B, C and x ({hidden} each)",
        f"B multiplied with x, then depthwise Conv1d (kernel {conv[-1]}), then multiplied with C",
        f"out_proj: {_linear(out)}",
    ]
    if cache_rows.get("short_conv", {}).get("grows_per_token_bytes") == 0:
        parts.append("keeps only the last few inputs; cache does not grow with context")
    return dict(title=f"Gated ShortConv ({count} layers)", parts=parts)


def _mamba_panel(t, v, count, cache_rows):
    inp, out = _need(t, "mixer.in_proj.weight"), _need(t, "mixer.out_proj.weight")
    conv = _need(t, "mixer.conv1d.weight")
    heads, hd, state, groups = v["mamba_num_heads"], v["mamba_head_dim"], v["mamba_state_size"], v["mamba_groups"]
    inner = heads * hd
    if inp[0] != 2 * inner + 2 * groups * state + heads or conv[0] != inner + 2 * groups * state:
        raise SpecError("Mamba-2 in_proj/conv widths disagree with the published head/state sizes")
    parts = [
        f"in_proj: {_linear(inp)}, split into z ({inner}), x ({inner}), B and C ({groups} groups x {state} each)"
        f" and dt ({heads})",
        f"depthwise Conv1d (kernel {conv[-1]}) on x, B and C ({conv[0]} channels)",
        f"selective state-space scan: {heads} heads x {hd} dims, state size {state}"
        + (
            "; cache does not grow with context"
            if cache_rows.get("mamba2", {}).get("grows_per_token_bytes") == 0
            else ""
        ),
    ]
    norm = _find(t, "mixer.norm.weight")
    if norm:
        parts.append(f"norm ({norm[0]}) gated by z")
    parts.append(f"out_proj: {_linear(out)}")
    return dict(title=f"Mamba-2 ({count} layers)", parts=parts)


def _dense_ffn_parts(t, v, act):
    gate = _find(t, "gate_proj.weight", "w1.weight")
    up = _find(t, "up_proj.weight", "w3.weight")
    down = _need(t, "down_proj.weight", "w2.weight")
    if gate and up:
        if gate != up:
            raise SpecError("Gate and up projections disagree")
        names = ("w1", "w3") if _find(t, "w1.weight") else ("gate", "up")
        act_text = f"; {act} on {names[0]}" if act and act not in ("SwiGLU", "GeGLU") else ""
        if act and act.endswith("GLU") and act not in ("SwiGLU", "GeGLU"):
            act_text = f"; combined by {act}"
        if act in ("SwiGLU", "GeGLU"):
            act_text = f"; {'SiLU' if act == 'SwiGLU' else 'GELU'} on {names[0]}"
        first = f"{names[0]} and {names[1]}: {_linear(gate)} in parallel{act_text}; multiply"
    else:
        first = f"up projection: {_linear(up)}" + (f"; {act}" if act else "")
    return [first, f"{'w2' if _find(t, 'w2.weight') else 'down'}: {_linear(down)}"]


def _moe_parts(t, v, act):
    parts = []
    router = _find(t, "gate.weight")
    experts, top = v["moe_num_experts"], v["moe_active_experts"]
    score = f", {v['moe_router']} scores" if v.get("moe_router") else ""
    if router and router[0] == experts:
        parts.append(f"router: {experts} routed experts{score}, top {top} per token")
    hidden, inter = v["hidden_size"], v["moe_expert_intermediate"]
    latent = v.get("moe_latent_dim")
    if latent:
        down = _need(t, "routed_expert_down_proj.weight")
        up = _need(t, "routed_expert_up_proj.weight")
        if down != [latent, hidden] or up != [hidden, latent]:
            raise SpecError("Latent projection shapes disagree with the published latent width")
        norm = " (with norm)" if _find(t, "routed_expert_norm.weight") else ""
        parts.append(
            f"routed experts work in a smaller latent: {hidden} -> {latent}{norm} -> expert {inter} -> {latent} -> {hidden}"
        )
    else:
        parts.append(f"each routed expert: {hidden} -> {inter} -> {hidden}")
    shared = v.get("moe_shared_experts")
    width = _find(t, "shared_experts.gate_proj.weight", "shared_expert.gate_proj.weight")
    if shared and width:
        parts.append(
            f"{shared} shared expert{'s' if shared > 1 else ''} (combined width {width[0]}), applied to every token"
            f" in the full {hidden} space"
        )
    if act:
        parts.append(f"activation: {act}" + (" (gated)" if v.get("ffn_gated") else ""))
    return parts


# ---------------------------------------------------------------- the spec


def _title(repo, v):
    name = repo.split("/", 1)[1]
    if re.search(r"\d(\.\d+)?[BTM]\b", name):
        return name
    return f"{name} ({_params(v['params_total'], 1)})"


def build_spec(record: dict) -> dict:
    try:
        v = render_values(record)
        cache = render_cache(v)
    except PublicationError as error:
        raise SpecError(str(error)) from error
    blocks = v.get("block_shapes")
    if not blocks:
        raise SpecError("Record has no published tensor shapes (block_shapes); reprocess it first")
    kinds_seq = v["layer_types"]
    kinds = list(dict.fromkeys(kinds_seq))
    counts = v["layer_type_counts"]
    cache_rows = {row["layer_type"]: row for row in cache["breakdown"]}
    names = {k: _mixer_name(k, v, kinds) for k in kinds}
    standalone = v.get("block_layout") == "standalone"
    ffns = blocks["ffns"]
    moe_model = any(_is_moe(g["tensors"]) for g in ffns)
    ffn_of_layer = {}
    for group in ffns:
        for a, b in group["layer_ranges"]:
            for i in range(a, b + 1):
                ffn_of_layer[i] = group
    if not standalone and len(ffn_of_layer) != len(kinds_seq):
        raise SpecError("Published FFN shape groups do not cover every layer")
    labels = [
        names[k] + ("" if standalone else " + " + _ffn_name(ffn_of_layer[i], v, moe_model))
        for i, k in enumerate(kinds_seq)
    ]
    groups = compress(kinds_seq, labels)
    total = sum(g["repeat"] * len(g["layers"]) for g in groups)
    if total != v["num_layers"]:
        raise SpecError("Compressed layer groups do not add up to the published layer count")
    compressed = []
    for g in groups:
        end = g["start"] + g["repeat"] * len(g["layers"]) - 1
        compressed.append(dict(repeat=g["repeat"], layers=g["layers"], note=_layers([[g["start"], end]])))

    act = _activation(v)
    vocab, ctx, hidden = v["vocab_size"], v["context_length"], v["hidden_size"]
    model = dict(
        name=record["id"].split("/", 1)[1],
        title_label=_title(record["id"], v),
        total_params=_params(v["params_total"]),
    )
    if v.get("params_active"):
        model["active_params_per_token"] = _params(v["params_active"])
    model.update(
        layers=v["num_layers"],
        hidden_size=hidden,
        vocabulary=f"{_commas(vocab)} ({_short_count(vocab)})"
        if _short_count(vocab) != _commas(vocab)
        else _commas(vocab),
        context_length=f"{_short_count(ctx)} tokens ({_commas(ctx)})",
    )
    if v.get("tied_embeddings"):
        model["embedding"] = "tied to the LM head"
    if v.get("vision_params"):
        model["modality"] = "text + image (vision encoder not shown in the main diagram)"

    composition = " + ".join(f"{counts[k]} {names[k]}" for k in kinds)
    pattern = dict(summary=composition + " layers")
    rule = _rule(kinds_seq, names)
    if rule:
        pattern["rule"] = rule
    pattern["compressed_view"] = compressed

    # ---------- one decoder layer, bottom to top, from the published tensor names
    sample = next(iter(blocks["mixers"].values()))["tensors"]
    has = lambda name: any(k == name or k.endswith("." + name) or k == name for k in sample)  # noqa: E731
    learned_res = has("self_attention_res_proj.weight")
    if standalone:
        layer = ["norm", "operator: " + ", ".join(names[k] for k in kinds) + " (see zoom panels)", "residual add"]
    else:
        mixer_text = "Token mixer: " + " or ".join(names[k] for k in kinds) + " (see zoom panels)"
        if len(kinds) == 1:
            mixer_text = f"Token mixer: {names[kinds[0]]} (see zoom panels)"
        ffn_names = list(dict.fromkeys(_ffn_name(g, v, moe_model) for g in ffns))
        ffn_text = "FFN: " + " or ".join(
            f"{n} ({_layers(next(g for g in ffns if _ffn_name(g, v, moe_model) == n)['layer_ranges'])})"
            if len(ffn_names) > 1
            else n
            for n in ffn_names
        )
        layer = ["norm (pre-mixer)", mixer_text]
        if v.get("sandwich_norm"):
            layer.append("norm (post-mixer)")
        if learned_res:
            res = "Attention residual mix (learned"
            res += (
                f"; config attn_res_block_size = {v['attn_res_block_size']})" if v.get("attn_res_block_size") else ")"
            )
            layer.append(res)
        else:
            layer.append("residual add")
        if ffn_text == "FFN: FFN":
            ffn_text = "FFN (see zoom panels)"
        layer += ["norm (pre-FFN)", ffn_text]
        if v.get("sandwich_norm"):
            layer.append("norm (post-FFN)")
        layer.append("FFN residual mix (learned)" if learned_res else "residual add")

    # ---------- zoom panels
    panels = {}
    attention_done = False
    for kind in kinds:
        t = blocks["mixers"][kind]["tensors"]
        count = counts[kind]
        if kind == "kda_linear":
            panels["kda"] = _kda_panel(t, v, count, cache_rows)
        elif kind == "gated_dsa_mla":
            panels["gated_dsa"] = _dsa_panel(t, v, count, cache_rows)
        elif kind == "full_attention_mla":
            panels["mla"] = _mla_panel(t, v, count, cache_rows)
        elif kind in ("full_attention", "sliding_attention"):
            if not attention_done:
                key = _attention_name(v).lower()
                panels[key] = _attention_panel(t, v, counts, kinds)
                attention_done = True
            elif blocks["mixers"].get("full_attention", {}).get("tensors") != blocks["mixers"].get(
                "sliding_attention", {}
            ).get("tensors"):
                raise SpecError("Global and sliding attention weights differ; one shared panel would misdescribe them")
        elif kind == "short_conv":
            panels["short_conv"] = _short_conv_panel(t, v, count, cache_rows)
        elif kind == "mamba2":
            panels["mamba2"] = _mamba_panel(t, v, count, cache_rows)
        elif kind == "mlp":
            panels["mlp"] = dict(title=f"MLP ({count} layers)", parts=_dense_ffn_parts(t, v, act))
    for group in ffns:
        name = _ffn_name(group, v, moe_model)
        where = _layers(group["layer_ranges"])
        if _is_moe(group["tensors"]):
            panels["latent_moe" if v.get("moe_latent_dim") else "moe"] = dict(
                title=f"{name} FFN ({where})", parts=_moe_parts(group["tensors"], v, act)
            )
        elif moe_model:
            title = f"Dense FFN ({where} only)" if len(group["layer_ranges"]) == 1 else f"Dense FFN ({where})"
            panels["dense_ffn"] = dict(title=title, parts=_dense_ffn_parts(group["tensors"], v, act))
        else:
            title = f"{act} FFN" if act in ("SwiGLU", "GeGLU") else "Gated FFN" if v.get("ffn_gated") else "FFN"
            panels["ffn"] = dict(title=title, parts=_dense_ffn_parts(group["tensors"], v, act))
    for panel in panels.values():
        panel["flow"] = "bottom to top, in the order listed"

    # ---------- input / output
    g = blocks["globals"]
    embed = _find(g, "embed_tokens.weight", "embeddings.weight", "tok_embeddings.weight")
    if not embed or embed != [vocab, hidden]:
        raise SpecError("Embedding tensor disagrees with the published vocabulary/hidden size")
    io = dict(bottom=["Tokenized text", f"Token embedding {_commas(vocab)} x {hidden}"], top=[])
    if _find(g, "output_attn_res_proj.weight"):
        io["top"].append("Final AttnRes mix")
    io["top"].append("Final norm")
    if v.get("tied_embeddings"):
        io["top"].append("LM head (shares weights with the token embedding)")
    elif _find(g, "lm_head.weight") == [vocab, hidden]:
        io["top"].append(f"LM head {hidden} -> {_commas(vocab)} (not tied to the embedding)")
    else:
        raise SpecError("LM head shape is not published")
    if v.get("vision_params") and v.get("vision_encoder_layers"):
        projector = (v.get("parameter_components") or {}).get("vision_projector")
        io["vision"] = (
            f"{v['vision_encoder_layers']}-layer vision encoder ({_params(v['vision_params'])})"
            + (f" + projector ({_params(projector)})" if projector else "")
            + " feeds image tokens into the embedding sequence; show as a small side note only"
        )

    # ---------- callouts (margin labels, at most five)
    callouts = [f"Context length: {_short_count(ctx)} tokens"]
    if len(kinds) > 1:
        callouts.append(composition)
    dense = [g for g in ffns if not _is_moe(g["tensors"])]
    sparse = [g for g in ffns if _is_moe(g["tensors"])]
    if dense and sparse:
        d, s = _layers(dense[0]["layer_ranges"]), _layers(sparse[0]["layer_ranges"])
        callouts.append(f"{d[0].upper() + d[1:]} dense FFN; {s} {_ffn_name(sparse[0], v, True)}")
    if sparse:
        shared = v.get("moe_shared_experts")
        callouts.append(
            f"{v['moe_num_experts']} experts, {v['moe_active_experts']} active"
            + (f" + {shared} shared" if shared else "")
        )
    if v.get("attention_type") in ("GQA", "MQA"):
        callouts.append(f"{v['num_attention_heads']} Q / {v['num_kv_heads']} KV heads")
    callouts += [f"Vocabulary: {_short_count(vocab)}", f"Embedding dimension: {hidden}"]
    grows = cache["grows_per_token_bytes"]
    if grows:
        about = "about " if cache.get("grows_per_token_bytes_low", grows) != grows else ""
        text = f"KV cache: {about}{_bytes(grows)} per token"
        growing = [r for r in cache["breakdown"] if r["grows_per_token_bytes"]]
        naive = cache.get("naive_mha_per_token_bytes")
        if v.get("attention_type") == "MLA" and naive:
            text += f" (vs {_bytes(naive)} for plain multi-head attention)"
        elif len(growing) < len(cache["breakdown"]):
            n = sum(r["layers"] for r in growing)
            label = " and ".join(
                "global"
                if r["layer_type"] == "full_attention" and "sliding_attention" in kinds
                else names[r["layer_type"]]
                for r in growing
            )
            text += f" (only the {n} {label} layers grow)"
        callouts = callouts[:4] + [text]
    else:
        callouts = callouts[:5]

    dont = ["any benchmark scores", "any number not in this file", "logos or watermarks"]
    if v.get("mla_use_nope") is True:
        dont.append("the word RoPE")
    elif not v.get("rope"):
        dont.append("any position-encoding label (none is published for this model)")

    spec = dict(model=model, layer_pattern=pattern, decoder_layer=layer)
    if learned_res:
        spec["residuals"] = (
            "Residuals are not plain additions. Each layer has learned residual-mixing weights"
            " (self_attention_res_proj and mlp_res_proj, plus their norms)"
            + (", and a final output residual mix before the last norm" if "Final AttnRes mix" in io["top"] else "")
            + '. Show this as a small "AttnRes mix" node on each residual path. Do not invent its internal math.'
        )
    spec.update(zoom_panels=panels, io=io, callouts=callouts, do_not_include=dont)
    return spec


def dump(spec: dict, record: dict) -> str:
    header = (
        f"# Diagram spec: {spec['model']['name']}\n"
        f"# Generated by archwatch.diagram_spec from the published, verifier-approved fields of\n"
        f"# {record['id']} at commit {record['source_commit'][:10]}. No value here comes from an LLM.\n"
        "# The diagram must use ONLY what is written here. If something is not here, leave it out.\n\n"
    )
    return header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=10_000)
