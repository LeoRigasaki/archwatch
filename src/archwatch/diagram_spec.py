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


# ---------------------------------------------------------------- zoom panels as small graphs


MAX_WORDS = 6


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z0-9]*|\d[\d,]*(?:\.\d+)?", text))


def _fit(text: str, short: str) -> str:
    return text if _words(text) <= MAX_WORDS else short


class _Graph:
    """Nodes, directed edges (data flows bottom to top) and captions for notes that are not parts."""

    def __init__(self, title):
        self.title, self.nodes, self.edges, self.caption = title, [], [], []

    def node(self, key, label):
        if _words(label) > MAX_WORDS:
            raise SpecError(f"Node label over {MAX_WORDS} words: {label!r}")
        self.nodes.append(dict(id=key, label=label))
        return key

    def chain(self, *keys):
        keys = [k for k in keys if k]
        self.edges += [f"{a} -> {b}" for a, b in zip(keys, keys[1:])]

    def spec(self):
        ids = [n["id"] for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise SpecError("Duplicate node id in " + self.title)
        for edge in self.edges:
            a, b = edge.split(" -> ")
            if a not in ids or b not in ids:
                raise SpecError(f"Edge {edge} names an unknown node in {self.title}")
        return dict(title=self.title, nodes=self.nodes, edges=self.edges, caption=self.caption)


def _io(g, hidden):
    return g.node("in", f"input ({hidden})"), g.node("out", f"output ({hidden})")


def _short(shape) -> str:
    return f"{shape[1]} -> {shape[0]}"


def _kda_panel(t, v, count, cache_rows):
    q, k, val = (_need(t, f"self_attn.{x}_proj.weight") for x in "qkv")
    if not q == k == val:
        raise SpecError("KDA q/k/v projections differ; the shared-node rule does not apply")
    heads, hd, hidden = v["linear_attn_num_heads"], v["linear_attn_head_dim"], v["hidden_size"]
    g = _Graph(f"Kimi Delta Attention (KDA), {count} layers")
    inp, out = _io(g, hidden)
    qkv = g.node("qkv", f"q, k, v: {_linear(q)}")
    conv = _find(t, "q_conv1d.weight")
    c = g.node("conv", f"Conv1d, kernel {conv[-1]}") if conv else None
    delta = g.node("delta", "gated delta rule")
    g.chain(inp, qkv, c, delta)
    beta = _find(t, "self_attn.b_proj.weight")
    if beta:
        g.chain(inp, g.node("beta", f"beta: {_linear(beta)}"), delta)
    fa, fb = _find(t, "self_attn.f_a_proj.weight"), _find(t, "self_attn.f_b_proj.weight")
    if fa and fb:
        g.chain(inp, g.node("decay", f"decay: Linear {fa[1]} -> {fa[0]} -> {fb[0]}"), delta)
    onorm = _find(t, "self_attn.o_norm.weight")
    norm = g.node("onorm", f"output norm ({onorm[0]})") if onorm else None
    gate = _find(t, "self_attn.g_proj.weight")
    mul = g.node("mul", "multiply") if gate else None
    oproj = g.node("oproj", f"output projection: {_short(_need(t, 'self_attn.o_proj.weight'))}")
    g.chain(delta, norm, mul, oproj, out)
    if gate:
        g.chain(inp, g.node("gate", f"output gate: {_linear(gate)}"), mul)
    g.caption.append(f"{heads} heads x {hd} dims")
    if conv:
        g.caption.append("short depthwise Conv1d on q, k and v separately")
    if beta and beta[0] == heads:
        g.caption.append("beta: one per head")
    if fa and fb:
        g.caption.append("decay gate is low-rank")
    grows = cache_rows.get("kda_linear", {}).get("grows_per_token_bytes")
    g.caption.append(f"fixed {hd} x {hd} state per head" + ("; cache does not grow with context" if grows == 0 else ""))
    return g.spec()


def _mla_graph(t, v, title, *, indexer):
    heads, hd, rope = v["num_attention_heads"], v["head_dim"], v["mla_qk_rope_head_dim"]
    vdim, latent, hidden = v.get("v_head_dim"), v["mla_kv_lora_rank"], v["hidden_size"]
    g = _Graph(title)
    inp, out = _io(g, hidden)
    qa, qb = _need(t, "self_attn.q_a_proj.weight"), _need(t, "self_attn.q_b_proj.weight")
    if qb[0] != heads * hd:
        raise SpecError("MLA query projection disagrees with published heads x head_dim")
    qnorm = ", norm" if _find(t, "self_attn.q_a_layernorm.weight") else ""
    kva = _need(t, "self_attn.kv_a_proj_with_mqa.weight")
    if kva[0] != latent + rope:
        raise SpecError("MLA latent tensor disagrees with published latent/rope widths")
    kvb = _need(t, "self_attn.kv_b_proj.weight")
    if not vdim or kvb[0] != heads * (hd - rope + vdim):
        raise SpecError("MLA kv_b projection disagrees with published head widths")
    nope = v.get("mla_use_nope") is True
    attn = g.node("attn", ("sparse attention" if indexer else "attention") + (", NoPE" if nope else ""))
    g.chain(inp, g.node("qa", f"q: {_linear(qa)}{qnorm}"), g.node("qb", f"Linear {qb[1]} -> {heads} x {hd}"), attn)
    kv = g.node("kva", f"kv: {_linear(kva)}")
    norm = g.node("kvnorm", f"norm on {latent} latent") if _find(t, "self_attn.kv_a_layernorm.weight") else None
    g.chain(inp, kv, norm, g.node("kvb", f"Linear {kvb[1]} -> {heads} x ({hd - rope} + {vdim})"), attn)
    g.chain(kv, attn)  # the decoupled key dims bypass the latent norm and kv_b
    if indexer:
        ih, idim, top = v["dsa_index_num_heads"], v["dsa_index_head_dim"], v["dsa_index_topk"]
        wq, wk, wp = (_find(t, f"indexer.{x}.weight") for x in ("wq_b", "wk", "weights_proj"))
        idx = g.node("indexer", f"indexer: {ih} heads x {idim}")
        if wq and wq[1] == qa[0]:
            g.chain("qa", idx)
        g.chain(inp, idx, g.node("topk", f"select top {top} tokens"), attn)
        detail = []
        if wq and wq[0] == ih * idim:
            detail.append(f"query from the {wq[1]} latent")
        if wk:
            detail.append(f"key {_linear(wk)}" + (" with norm" if _find(t, "indexer.k_norm.weight") else ""))
        if wp:
            detail.append(f"per-head weights {_linear(wp)}")
        g.caption.append("indexer: " + ", ".join(detail))
        g.caption.append(f"indexer picks the top {top} tokens; attention runs only on those")
    gate = _find(t, "self_attn.g_proj.weight")
    mul = g.node("mul", "multiply") if gate else None
    g.chain(attn, mul, g.node("oproj", f"output projection: {_short(_need(t, 'self_attn.o_proj.weight'))}"), out)
    if gate:
        g.chain(inp, g.node("gate", f"output gate: {_linear(gate)}"), mul)
    g.caption.append(f"{heads} heads; query/key {hd} = {hd - rope} + {rope}, value {vdim}")
    g.caption.append(f"kv: {kva[0]} = {latent} latent + {rope} key dims; the {rope} key dims skip the latent norm")
    if nope:
        g.caption.append("position encoding: NoPE (config mla_use_nope = true)")
    if indexer and v.get("dsa_index_key_width"):
        g.caption.append(
            f"cache per token: {kva[0]} MLA values + {v['dsa_index_key_width']} indexer key values per layer"
        )
    return g.spec()


def _dsa_panel(t, v, count, cache_rows):
    return _mla_graph(t, v, f"Gated DeepSeek Sparse Attention (DSA), {count} layers", indexer=True)


def _mla_panel(t, v, count, cache_rows):
    return _mla_graph(t, v, f"Multi-head Latent Attention (MLA), {count} layers", indexer=False)


def _attention_panel(t, v, counts, kinds):
    heads, kvh, hd, hidden = v["num_attention_heads"], v["num_kv_heads"], v["head_dim"], v["hidden_size"]
    name = _attention_name(v)
    pre = "self_attn." if _find(t, "self_attn.q_proj.weight") else "mixer."
    q, k, val = (_need(t, f"{pre}{x}_proj.weight") for x in "qkv")
    full, sliding = counts.get("full_attention", 0), counts.get("sliding_attention", 0)
    if full and sliding:
        title = f"{name} attention ({full} global + {sliding} sliding-window layers)"
    else:
        title = f"{name} attention ({full or sliding} layers)"
    g = _Graph(title)
    inp, out = _io(g, hidden)
    qn = kn = None
    if v.get("qk_norm") and _find(t, f"{pre}q_norm.weight") and _find(t, f"{pre}k_norm.weight"):
        qn = g.node("qnorm", f"q norm ({_find(t, f'{pre}q_norm.weight')[0]})")
        kn = g.node("knorm", f"k norm ({_find(t, f'{pre}k_norm.weight')[0]})")
    rope = v.get("rope") or {}
    rot = g.node("rotary", "rotary on q and k") if rope else None
    attn = g.node("attn", f"attention: {heads} q / {kvh} kv heads")
    g.chain(inp, g.node("q", f"q: {_linear(q)}"), qn, rot, attn)
    g.chain(inp, g.node("k", f"k: {_linear(k)}"), kn, rot)
    if not rot:
        g.chain(kn or "k", attn)
    g.chain(inp, g.node("v", f"v: {_linear(val)}"), attn)
    gate = _find(t, f"{pre}gate_proj.weight") if v.get("attention_output_gate") else None
    mul = g.node("mul", "multiply") if gate else None
    g.chain(attn, mul, g.node("oproj", f"output projection: {_short(_need(t, f'{pre}o_proj.weight'))}"), out)
    if gate:
        g.chain(inp, g.node("gate", f"head gate: {_linear(gate)}"), mul)
        if gate[0] == heads:
            g.caption.append("head gate: one gate per head")
    g.caption.append(f"{heads} query heads, {kvh} key/value heads, head size {hd}")
    if "sliding_attention" in kinds and "full_attention" in kinds:
        gl, sl = rope.get("full_attention") or {}, rope.get("sliding_attention") or {}
        pr = gl.get("partial_rotary_factor")
        text = "global layers: "
        if pr:
            text += (
                "partial rotary (half of each head)" if pr == 0.5 else f"partial rotary ({pr:g} of each head)"
            ) + ", "
        if gl.get("rope_theta"):
            text += f"theta {_commas(gl['rope_theta'])}, "
        g.caption.append(text + "attend to all tokens")
        text = "sliding-window layers: "
        if sl:
            text += "full rotary, " if not sl.get("partial_rotary_factor") else "partial rotary, "
            if sl.get("rope_theta"):
                text += f"theta {_commas(sl['rope_theta'])}, "
        g.caption.append(text + f"attend to the last {v['sliding_window']} tokens only")
    elif rope.get("rope_theta"):
        text = f"rotary position encoding, theta {_commas(rope['rope_theta'])}"
        scaling = rope.get("scaling") or {}
        if scaling.get("rope_type") and scaling.get("factor"):
            text += f" ({scaling['rope_type']} scaling, factor {scaling['factor']:g})"
        g.caption.append(text)
    return g.spec()


def _short_conv_panel(t, v, count, cache_rows):
    inp_w, conv, out_w = _need(t, "conv.in_proj.weight"), _need(t, "conv.conv.weight"), _need(t, "conv.out_proj.weight")
    hidden = v["hidden_size"]
    if inp_w[0] != 3 * hidden:
        raise SpecError("ShortConv in_proj is not the three-way gated split this rule describes")
    g = _Graph(f"Gated ShortConv ({count} layers)")
    inp, out = _io(g, hidden)
    ip = g.node("in_proj", f"in_proj: {_linear(inp_w)}")
    cmul = g.node("mul_c", "multiply with C")
    g.chain(inp, ip, g.node("mul_bx", "multiply B and x"), g.node("conv", f"depthwise Conv1d, kernel {conv[-1]}"), cmul)
    g.chain(ip, cmul)
    g.chain(cmul, g.node("out_proj", f"out_proj: {_linear(out_w)}"), out)
    g.caption.append(f"in_proj output splits into B, C and x ({hidden} each)")
    if cache_rows.get("short_conv", {}).get("grows_per_token_bytes") == 0:
        g.caption.append("keeps only the last few inputs; cache does not grow with context")
    return g.spec()


def _mamba_panel(t, v, count, cache_rows):
    inp_w, out_w = _need(t, "mixer.in_proj.weight"), _need(t, "mixer.out_proj.weight")
    conv = _need(t, "mixer.conv1d.weight")
    heads, hd, state, groups = v["mamba_num_heads"], v["mamba_head_dim"], v["mamba_state_size"], v["mamba_groups"]
    inner = heads * hd
    if inp_w[0] != 2 * inner + 2 * groups * state + heads or conv[0] != inner + 2 * groups * state:
        raise SpecError("Mamba-2 in_proj/conv widths disagree with the published head/state sizes")
    g = _Graph(f"Mamba-2 ({count} layers)")
    inp, out = _io(g, v["hidden_size"])
    ip = g.node("in_proj", f"in_proj: {_linear(inp_w)}")
    ssm = g.node("scan", f"scan: {heads} heads x {hd}")
    g.chain(inp, ip, g.node("conv", f"Conv1d kernel {conv[-1]} ({conv[0]} channels)"), ssm)
    g.chain(ip, ssm)  # dt
    norm = _find(t, "mixer.norm.weight")
    gn = g.node("norm", f"norm ({norm[0]}) gated by z") if norm else None
    if gn:
        g.chain(ip, gn)  # z
    g.chain(ssm, gn, g.node("out_proj", f"out_proj: {_linear(out_w)}"), out)
    g.caption.append(
        f"in_proj splits into z ({inner}), x ({inner}), B and C ({groups} groups x {state} each) and dt ({heads})"
    )
    g.caption.append("selective state-space scan; Conv1d runs on x, B and C; dt goes to the scan; z gates the norm")
    g.caption.append(
        f"state size {state}"
        + (
            "; cache does not grow with context"
            if cache_rows.get("mamba2", {}).get("grows_per_token_bytes") == 0
            else ""
        )
    )
    return g.spec()


ACT_NODE = {"silu": "SiLU", "gelu": "GELU", "situ": "SiTU", "relu2": "squared ReLU"}


def _ffn_graph(title, t, v):
    g = _Graph(title)
    inp, out = _io(g, v["hidden_size"])
    gate_w = _find(t, "gate_proj.weight", "w1.weight")
    up_w = _find(t, "up_proj.weight", "w3.weight")
    down_w = _need(t, "down_proj.weight", "w2.weight")
    names = ("w1", "w3", "w2") if _find(t, "w1.weight") else ("gate", "up", "down")
    act = ACT_NODE.get(v.get("activation"), v.get("activation"))
    down = g.node("down", f"{names[2]}: {_linear(down_w)}")
    if gate_w and up_w:
        if gate_w != up_w:
            raise SpecError("Gate and up projections disagree")
        mul = g.node("mul", "multiply")
        g.chain(inp, g.node("gate", f"{names[0]}: {_linear(gate_w)}"), g.node("act", act) if act else None, mul)
        g.chain(inp, g.node("up", f"{names[1]}: {_linear(up_w)}"), mul)
        g.chain(mul, down, out)
    else:
        g.chain(inp, g.node("up", f"up: {_linear(up_w)}"), g.node("act", act) if act else None, down, out)
    if v.get("activation_card_name"):
        g.caption.append(f"activation: {v['activation_card_name']} (gated)")
    return g


def _moe_panel(title, t, v):
    g = _Graph(title)
    hidden, inter = v["hidden_size"], v["moe_expert_intermediate"]
    experts, top = v["moe_num_experts"], v["moe_active_experts"]
    inp, out = _io(g, hidden)
    router = _need(t, "gate.weight")
    if router[0] != experts:
        raise SpecError("Router width disagrees with the published expert count")
    total = g.node("sum", "sum")
    r = g.node("router", f"router: {experts} experts, top {top}")
    latent = v.get("moe_latent_dim")
    if latent:
        down = _need(t, "routed_expert_down_proj.weight")
        up = _need(t, "routed_expert_up_proj.weight")
        if down != [latent, hidden] or up != [hidden, latent]:
            raise SpecError("Latent projection shapes disagree with the published latent width")
        norm = ", norm" if _find(t, "routed_expert_norm.weight") else ""
        ex = g.node("experts", f"{top} experts: {latent} -> {inter} -> {latent}")
        g.chain(inp, g.node("latent_down", f"Linear {hidden} -> {latent}{norm}"), ex)
        g.chain(ex, g.node("latent_up", f"Linear {latent} -> {hidden}"), total)
        g.caption.append(f"routed experts work in a smaller latent: {hidden} -> {latent} -> {hidden}")
    else:
        ex = g.node("experts", f"{top} experts: {hidden} -> {inter} -> {hidden}")
        g.chain(inp, ex, total)
    g.chain(inp, r, ex)
    shared = v.get("moe_shared_experts")
    width = _find(t, "shared_experts.gate_proj.weight", "shared_expert.gate_proj.weight")
    if shared and width:
        g.chain(inp, g.node("shared", f"{shared} shared experts ({width[0]})"), total)
        g.caption.append(f"shared experts: combined width {width[0]}, every token, full {hidden} space")
    g.chain(total, out)
    if v.get("moe_router"):
        g.caption.append(f"router uses {v['moe_router']} gating")
    if v.get("activation_card_name"):
        g.caption.append(f"activation: {v['activation_card_name']} (gated)")
    return g.spec()


def _each_layer(v, names, kinds, ffn_names, standalone, learned_res, ffn_where):
    g = _Graph("Each layer")
    inp, out = g.node("in", "layer input"), g.node("out", "layer output")
    if standalone:
        res = g.node("res", "residual add")
        g.chain(
            inp,
            g.node("norm", "norm"),
            g.node("op", _fit("operator: " + " or ".join(names[k] for k in kinds), "operator")),
            res,
        )
        g.chain(inp, res)
        g.chain(res, out)
        g.caption.append("each layer is one operator: " + ", ".join(names[k] for k in kinds))
        return g.spec()
    res_label = "AttnRes mix" if learned_res else "residual add"
    r1, r2 = g.node("res1", res_label), g.node("res2", res_label)
    mixer_text = "token mixer: " + " or ".join(names[k] for k in kinds)
    mixer = g.node("mixer", _fit(mixer_text, "token mixer"))
    if _words(mixer_text) > MAX_WORDS:
        g.caption.append(mixer_text)
    post1 = g.node("post_mixer_norm", "norm (post-mixer)") if v.get("sandwich_norm") else None
    g.chain(inp, g.node("pre_mixer_norm", "norm (pre-mixer)"), mixer, post1, r1)
    g.chain(inp, r1)
    ffn = g.node("ffn", "FFN: " + " or ".join(ffn_names) if ffn_names != ["FFN"] else "FFN")
    post2 = g.node("post_ffn_norm", "norm (post-FFN)") if v.get("sandwich_norm") else None
    g.chain(r1, g.node("pre_ffn_norm", "norm (pre-FFN)"), ffn, post2, r2)
    g.chain(r1, r2)
    g.chain(r2, out)
    if learned_res:
        g.caption.append(
            "learned residual-mixing weights (self_attention_res_proj, mlp_res_proj)"
            + (f"; config attn_res_block_size = {v['attn_res_block_size']}" if v.get("attn_res_block_size") else "")
        )
    if len(ffn_where) > 1:
        g.caption.append("; ".join(f"{n}: {w}" for n, w in ffn_where))
    return g.spec()


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
    if sum(g["repeat"] * len(g["layers"]) for g in groups) != v["num_layers"]:
        raise SpecError("Compressed layer groups do not add up to the published layer count")
    compressed = []
    for g in groups:
        end = g["start"] + g["repeat"] * len(g["layers"]) - 1
        compressed.append(dict(repeat=g["repeat"], layers=g["layers"], note=_layers([[g["start"], end]])))

    vocab, ctx, hidden = v["vocab_size"], v["context_length"], v["hidden_size"]
    model = dict(
        name=record["id"].split("/", 1)[1],
        title_label=_title(record["id"], v),
        summary_line=", ".join(
            [f"{_params(v['params_total'])} parameters"]
            + ([f"{_params(v['params_active'])} active"] if v.get("params_active") else [])
            + [f"{v['num_layers']} layers", f"{_short_count(ctx)} context"]
        ),
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

    sample = next(iter(blocks["mixers"].values()))["tensors"]
    learned_res = any(k.endswith("self_attention_res_proj.weight") for k in sample)
    ffn_names = list(dict.fromkeys(_ffn_name(g, v, moe_model) for g in ffns))
    ffn_where = [(_ffn_name(g, v, moe_model), _layers(g["layer_ranges"])) for g in ffns]
    each_layer = _each_layer(v, names, kinds, ffn_names, standalone, learned_res, ffn_where)

    # ---------- zoom panels
    panels = {}
    attention_key = None
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
            if attention_key is None:
                attention_key = _attention_name(v).lower()
                panels[attention_key] = _attention_panel(t, v, counts, kinds)
            elif blocks["mixers"].get("full_attention", {}).get("tensors") != blocks["mixers"].get(
                "sliding_attention", {}
            ).get("tensors"):
                raise SpecError("Global and sliding attention weights differ; one shared panel would misdescribe them")
        elif kind == "short_conv":
            panels["short_conv"] = _short_conv_panel(t, v, count, cache_rows)
        elif kind == "mamba2":
            panels["mamba2"] = _mamba_panel(t, v, count, cache_rows)
        elif kind == "mlp":
            panels["mlp"] = _ffn_graph(f"MLP ({count} layers)", t, v).spec()
    moe_key = None
    for group in ffns:
        name = _ffn_name(group, v, moe_model)
        where = _layers(group["layer_ranges"])
        if _is_moe(group["tensors"]):
            moe_key = "latent_moe" if v.get("moe_latent_dim") else "moe"
            panels[moe_key] = _moe_panel(f"{name} FFN ({where})", group["tensors"], v)
        elif moe_model:
            title = f"Dense FFN ({where} only)" if len(group["layer_ranges"]) == 1 else f"Dense FFN ({where})"
            panels["dense_ffn"] = _ffn_graph(title, group["tensors"], v).spec()
        else:
            act = _activation(v)
            title = f"{act} FFN" if act in ("SwiGLU", "GeGLU") else "Gated FFN" if v.get("ffn_gated") else "FFN"
            panels["ffn"] = _ffn_graph(title, group["tensors"], v).spec()

    # ---------- input / output (the main stack, bottom to top)
    g = blocks["globals"]
    embed = _find(g, "embed_tokens.weight", "embeddings.weight", "tok_embeddings.weight")
    if not embed or embed != [vocab, hidden]:
        raise SpecError("Embedding tensor disagrees with the published vocabulary/hidden size")
    bottom = [
        dict(id="text", label="Tokenized text"),
        dict(id="embedding", label=f"Token embedding {_commas(vocab)} x {hidden}"),
    ]
    top = []
    if _find(g, "output_attn_res_proj.weight"):
        top.append(dict(id="final_res", label="Final AttnRes mix"))
    top.append(dict(id="final_norm", label="Final norm"))
    if v.get("tied_embeddings"):
        top.append(dict(id="lm_head", label="LM head (tied to the token embedding)"))
    elif _find(g, "lm_head.weight") == [vocab, hidden]:
        top.append(dict(id="lm_head", label=f"LM head {hidden} -> {_commas(vocab)}"))
    else:
        raise SpecError("LM head shape is not published")
    io = dict(bottom=bottom, top=top)
    if not v.get("tied_embeddings"):
        io["note"] = "LM head is not tied to the embedding"
    if v.get("vision_params") and v.get("vision_encoder_layers"):
        projector = (v.get("parameter_components") or {}).get("vision_projector")
        io["vision"] = (
            f"{v['vision_encoder_layers']}-layer vision encoder ({_params(v['vision_params'])})"
            + (f" + projector ({_params(projector)})" if projector else "")
            + " feeds image tokens into the embedding sequence"
        )

    # ---------- callouts: each sits beside the part it describes
    callouts = [dict(text=f"Context length: {_short_count(ctx)} tokens", beside="stack")]
    if len(kinds) > 1:
        callouts.append(dict(text=composition, beside="stack"))
    dense = [x for x in ffns if not _is_moe(x["tensors"])]
    sparse = [x for x in ffns if _is_moe(x["tensors"])]
    if dense and sparse:
        d, s = _layers(dense[0]["layer_ranges"]), _layers(sparse[0]["layer_ranges"])
        callouts.append(
            dict(text=f"{d[0].upper() + d[1:]} dense FFN; {s} {_ffn_name(sparse[0], v, True)}", beside=moe_key)
        )
    if sparse:
        shared = v.get("moe_shared_experts")
        callouts.append(
            dict(
                text=f"{v['moe_num_experts']} experts, {v['moe_active_experts']} active"
                + (f" + {shared} shared" if shared else ""),
                beside=moe_key,
            )
        )
    if v.get("attention_type") in ("GQA", "MQA") and attention_key:
        callouts.append(dict(text=f"{v['num_attention_heads']} Q / {v['num_kv_heads']} KV heads", beside=attention_key))
    callouts += [
        dict(text=f"Vocabulary: {_short_count(vocab)}", beside="embedding"),
        dict(text=f"Embedding dimension: {hidden}", beside="embedding"),
    ]
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
        target = {"gated_dsa_mla": "gated_dsa", "full_attention_mla": "mla"}.get(
            growing[0]["layer_type"], attention_key
        )
        callouts = callouts[:4] + [dict(text=text, beside=target or "stack")]
    else:
        callouts = callouts[:5]

    dont = ["any benchmark scores", "any number not in this file", "logos or watermarks"]
    if v.get("mla_use_nope") is True:
        dont.append("the word RoPE")
    elif not v.get("rope"):
        dont.append("any position-encoding label (none is published for this model)")

    return dict(
        model=model,
        layer_pattern=pattern,
        io=io,
        each_layer=each_layer,
        zoom_panels=panels,
        callouts=callouts,
        do_not_include=dont,
    )


def dump(spec: dict, record: dict) -> str:
    header = (
        f"# Diagram spec: {spec['model']['name']}\n"
        f"# Generated by archwatch.diagram_spec from the published, verifier-approved fields of\n"
        f"# {record['id']} at commit {record['source_commit'][:10]}. No value here comes from an LLM.\n"
        "# The diagram must use ONLY what is written here. If something is not here, leave it out.\n\n"
    )
    return header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=10_000)
