"""Worked examples for concept guides, computed at build time from a published record.

Which model a guide uses, and the sentences around the numbers, live in config/concept-lessons.yaml
(`example.model`, `example.compute`, templates). This module only reads approved fields and does the
arithmetic. Every result is also returned as a (label, ours, reference) pair so a test can hold it to
the cache calculator or the record.
"""

from __future__ import annotations

import json

from .publication import render_cache, render_values

BF16 = 2
FP32 = 4
APPROVED = {"verified", "single source"}


class ExampleError(ValueError):
    pass


def sig(x: float) -> str:
    text = f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def size(b: float) -> str:
    for unit, power in (("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if b >= 2**power:
            return f"{sig(b / 2**power)} {unit}"
    return f"{int(b):,} B"


def params(n: float) -> str:
    return f"{sig(n / 1e12)}T" if n >= 1e12 else f"{sig(n / 1e9)}B"


def approved(record: dict, names: list[str]) -> tuple[dict, list[tuple[str, object, str]]]:
    """Published values for `names`, each backed by a current, passing verifier decision."""
    values = render_values(record)
    checks = {c["field"]: c for c in record.get("checks", [])}
    used = []
    for name in names:
        check = checks.get(name, {})
        if name not in values or not check.get("published") or check.get("status") not in APPROVED:
            raise ExampleError(f"{record['id']}: {name} is not an approved published field")
        if json.dumps(check.get("value"), sort_keys=True) != json.dumps(values[name], sort_keys=True):
            raise ExampleError(f"{record['id']}: {name} check does not match the published value")
        used.append((name, values[name], check["status"]))
    return values, used


def _row(cache: dict, kind: str) -> dict:
    return next(row for row in cache["breakdown"] if row["layer_type"] == kind)


def kv_cache(record):
    v, used = approved(record, ["num_layers", "num_kv_heads", "head_dim", "context_length", "params_total"])
    layers, kv, hd, ctx = v["num_layers"], v["num_kv_heads"], v["head_dim"], v["context_length"]
    per_layer = 2 * kv * hd * BF16
    per_token = per_layer * layers
    weights = v["params_total"] * BF16
    cache = render_cache(v)
    out = dict(
        layers=layers,
        kv=kv,
        hd=hd,
        ctx=f"{ctx:,}",
        per_layer=f"{per_layer:,} B",
        per_token=size(per_token),
        at_8k=size(per_token * 8192),
        at_full=size(per_token * ctx),
        params=params(v["params_total"]),
        weights=size(weights),
        ratio=sig(per_token * ctx / weights),
        per_token_bytes=per_token,
    )
    checks = [("cache per token", per_token, cache["grows_per_token_bytes"])]
    checks.append(("cache at 8,192 tokens", per_token * 8192, cache["memory_at_context"]["8192"]["bytes"]))
    return out, used, checks


def gqa(record):
    v, used = approved(record, ["num_layers", "num_attention_heads", "num_kv_heads", "head_dim", "context_length"])
    layers, hq, kv, hd, ctx = (
        v["num_layers"],
        v["num_attention_heads"],
        v["num_kv_heads"],
        v["head_dim"],
        v["context_length"],
    )
    ours = 2 * kv * hd * BF16 * layers
    mha = 2 * hq * hd * BF16 * layers
    cache = render_cache(v)
    out = dict(
        layers=layers,
        hq=hq,
        kv=kv,
        hd=hd,
        group=hq // kv,
        ctx=f"{ctx:,}",
        gqa=size(ours),
        mha=size(mha),
        gqa_full=size(ours * ctx),
        mha_full=size(mha * ctx),
        per_token_bytes=ours,
    )
    checks = [("GQA cache per token", ours, cache["grows_per_token_bytes"])]
    checks.append(("MHA-equivalent cache per token", mha, cache["naive_mha_per_token_bytes"]))
    return out, used, checks


def mla(record):
    v, used = approved(
        record,
        ["num_layers", "num_attention_heads", "head_dim", "v_head_dim", "mla_kv_lora_rank", "mla_qk_rope_head_dim"],
    )
    layers, heads, dk, dv = v["num_layers"], v["num_attention_heads"], v["head_dim"], v["v_head_dim"]
    rank, rope = v["mla_kv_lora_rank"], v["mla_qk_rope_head_dim"]
    per_layer = (rank + rope) * BF16
    per_token = per_layer * layers
    expanded = heads * (dk + dv) * BF16 * layers
    cache = render_cache(v)
    out = dict(
        layers=layers,
        heads=heads,
        dk=dk,
        dv=dv,
        rank=rank,
        rope=rope,
        per_layer=f"{per_layer:,} B",
        per_token=size(per_token),
        per_token_b=f"{per_token:,} B",
        expanded=size(expanded),
        ratio=sig(expanded / per_token),
        per_token_bytes=per_token,
    )
    checks = [("MLA cache per token", per_token, cache["grows_per_token_bytes"])]
    checks.append(("expanded keys and values per token", expanded, cache["naive_mha_per_token_bytes"]))
    return out, used, checks


def sliding_window(record):
    v, used = approved(record, ["num_layers", "num_kv_heads", "head_dim", "sliding_window", "context_length"])
    layers, kv, hd, window, ctx = (
        v["num_layers"],
        v["num_kv_heads"],
        v["head_dim"],
        v["sliding_window"],
        v["context_length"],
    )
    per_token = 2 * kv * hd * BF16 * layers
    cap = per_token * window
    cache = render_cache(v)
    out = dict(
        layers=layers,
        kv=kv,
        hd=hd,
        window=f"{window:,}",
        ctx=f"{ctx:,}",
        per_token=size(per_token),
        cap=size(cap),
        full=size(per_token * ctx),
        ratio=sig(ctx / window),
        per_token_bytes=per_token,
    )
    checks = [("capped cache", cap, _row(cache, "sliding_attention")["constant_bytes"])]
    return out, used, checks


def moe(record):
    v, used = approved(
        record, ["moe_num_experts", "moe_active_experts", "moe_shared_experts", "params_total", "params_active"]
    )
    total, active = v["params_total"], v["params_active"]
    out = dict(
        experts=v["moe_num_experts"],
        k=v["moe_active_experts"],
        shared=v["moe_shared_experts"],
        routed_share=sig(100 * v["moe_active_experts"] / v["moe_num_experts"]),
        total=params(total),
        active=params(active),
        active_share=sig(100 * active / total),
        weights=size(total * BF16),
        active_weights=size(active * BF16),
    )
    record_values = render_values(record)
    checks = [("total parameters", total, record_values["params_total"])]
    checks.append(("active parameters", active, record_values["params_active"]))
    return out, used, checks


def linear_attention(record):
    v, used = approved(
        record,
        [
            "layer_type_counts",
            "linear_attn_num_heads",
            "linear_attn_head_dim",
            "linear_attn_conv_kernel",
            "mla_kv_lora_rank",
            "mla_qk_rope_head_dim",
            "context_length",
        ],
    )
    counts = v["layer_type_counts"]
    linear, full = counts["kda_linear"], counts.get("full_attention_mla", 0)
    heads, width, kernel, ctx = (
        v["linear_attn_num_heads"],
        v["linear_attn_head_dim"],
        v["linear_attn_conv_kernel"],
        v["context_length"],
    )
    state = heads * width * width * FP32
    conv = 3 * heads * width * (kernel - 1) * BF16
    fixed = (state + conv) * linear
    mla_layer = (v["mla_kv_lora_rank"] + v["mla_qk_rope_head_dim"]) * BF16
    grows = mla_layer * full
    cache = render_cache(v)
    out = dict(
        linear=linear,
        full=full,
        heads=heads,
        width=width,
        taps=kernel - 1,
        ctx=f"{ctx:,}",
        state=size(state),
        conv=size(conv),
        fixed=size(fixed),
        mla_layer=f"{mla_layer:,} B",
        grows=size(grows),
        grows_full=size(grows * ctx),
        avoided=size(mla_layer * linear * ctx),
    )
    checks = [("fixed linear-attention state", fixed, _row(cache, "kda_linear")["constant_bytes"])]
    checks.append(("growing cache per token", grows, cache["grows_per_token_bytes"]))
    return out, used, checks


def _prod(shape):
    out = 1
    for n in shape:
        out *= n
    return out


def _tensor(values, name):
    """The shape of the first decoder tensor whose name ends with `name`."""
    for group in values["block_shapes"]["mixers"].values():
        for tensor, shape in group["tensors"].items():
            if tensor.endswith(name):
                return shape
    for group in values["block_shapes"]["ffns"]:
        for tensor, shape in group["tensors"].items():
            if tensor.endswith(name):
                return shape
    raise ExampleError(f"no tensor ending in {name}")


def mha(record):
    v, used = approved(record, ["num_layers", "num_attention_heads", "num_kv_heads", "head_dim", "context_length"])
    layers, hq, kv, hd, ctx = (
        v["num_layers"],
        v["num_attention_heads"],
        v["num_kv_heads"],
        v["head_dim"],
        v["context_length"],
    )
    per_layer = 2 * kv * hd * BF16
    per_token = per_layer * layers
    cache = render_cache(v)
    out = dict(
        layers=layers,
        hq=hq,
        kv=kv,
        hd=hd,
        ctx=f"{ctx:,}",
        per_layer=f"{per_layer:,} B",
        per_token=size(per_token),
        at_full=size(per_token * ctx),
        shared4=size(per_token // 4),
        per_token_bytes=per_token,
    )
    checks = [("cache per token", per_token, cache["grows_per_token_bytes"])]
    checks.append(("no saving against multi-head attention", per_token, cache["naive_mha_per_token_bytes"]))
    return out, used, checks


def sparse_attention(record):
    v, used = approved(
        record,
        [
            "layer_type_counts",
            "dsa_index_topk",
            "context_length",
            "mla_kv_lora_rank",
            "mla_qk_rope_head_dim",
            "dsa_index_head_dim",
        ],
    )
    layers, topk, ctx = v["layer_type_counts"]["gated_dsa_mla"], v["dsa_index_topk"], v["context_length"]
    per_layer = (v["mla_kv_lora_rank"] + v["mla_qk_rope_head_dim"] + v["dsa_index_head_dim"]) * BF16
    grows = per_layer * layers
    cache = render_cache(v)
    out = dict(
        layers=layers,
        topk=f"{topk:,}",
        ctx=f"{ctx:,}",
        share_full=sig(100 * topk / ctx),
        share_8k=sig(100 * topk / 8192),
        grows=size(grows),
        grows_full=size(grows * ctx),
    )
    checks = [("cache per token, every position kept", grows, _row(cache, "gated_dsa_mla")["grows_per_token_bytes"])]
    return out, used, checks


def dsa(record):
    v, used = approved(
        record,
        [
            "layer_type_counts",
            "dsa_index_num_heads",
            "dsa_index_head_dim",
            "dsa_index_topk",
            "mla_kv_lora_rank",
            "mla_qk_rope_head_dim",
            "block_shapes",
        ],
    )
    layers = v["layer_type_counts"]["gated_dsa_mla"]
    rank, rope, idx = v["mla_kv_lora_rank"], v["mla_qk_rope_head_dim"], v["dsa_index_head_dim"]
    per_layer = (rank + rope + idx) * BF16
    cache = render_cache(v)
    out = dict(
        layers=layers,
        heads=v["dsa_index_num_heads"],
        idx=idx,
        topk=f"{v['dsa_index_topk']:,}",
        rank=rank,
        rope=rope,
        per_layer=f"{per_layer:,} B",
        per_token=size(per_layer * layers),
        index_part=size(idx * BF16 * layers),
        index_share=sig(100 * idx / (rank + rope)),
    )
    checks = [("cache per token", per_layer * layers, _row(cache, "gated_dsa_mla")["grows_per_token_bytes"])]
    checks.append(("index key width", idx, _tensor(v, "indexer.wk.weight")[0]))
    return out, used, checks


def mamba2(record):
    v, used = approved(
        record,
        [
            "layer_type_counts",
            "mamba_num_heads",
            "mamba_head_dim",
            "mamba_state_size",
            "mamba_conv_channels",
            "mamba_conv_kernel",
            "num_kv_heads",
            "head_dim",
            "context_length",
        ],
    )
    counts = v["layer_type_counts"]
    blocks, attn = counts["mamba2"], counts.get("full_attention", 0)
    state = v["mamba_num_heads"] * v["mamba_head_dim"] * v["mamba_state_size"] * FP32
    conv = v["mamba_conv_channels"] * v["mamba_conv_kernel"] * BF16
    fixed = (state + conv) * blocks
    grows = 2 * v["num_kv_heads"] * v["head_dim"] * BF16 * attn
    cache = render_cache(v)
    out = dict(
        blocks=blocks,
        attn=attn,
        mlp=counts.get("mlp", 0),
        heads=v["mamba_num_heads"],
        dim=v["mamba_head_dim"],
        n=v["mamba_state_size"],
        channels=f"{v['mamba_conv_channels']:,}",
        kernel=v["mamba_conv_kernel"],
        state=size(state),
        conv=size(conv),
        fixed=size(fixed),
        grows=size(grows),
        ctx=f"{v['context_length']:,}",
        grows_full=size(grows * v["context_length"]),
    )
    checks = [("fixed Mamba-2 state", fixed, _row(cache, "mamba2")["constant_bytes"])]
    checks.append(("attention cache per token", grows, cache["grows_per_token_bytes"]))
    return out, used, checks


def short_convolution(record):
    v, used = approved(record, ["layer_type_counts", "hidden_size", "short_conv_kernel", "block_shapes"])
    layers, hidden, kernel = v["layer_type_counts"]["short_conv"], v["hidden_size"], v["short_conv_kernel"]
    weights = hidden * kernel
    state = hidden * kernel * BF16
    cache = render_cache(v)
    out = dict(
        layers=layers,
        hidden=f"{hidden:,}",
        kernel=kernel,
        past=kernel - 1,
        weights=f"{weights:,}",
        state=size(state),
        fixed=size(state * layers),
        strict=size(hidden * (kernel - 1) * BF16 * layers),
    )
    checks = [("convolution weights per layer", weights, _prod(_tensor(v, "conv.conv.weight")))]
    checks.append(("fixed convolution state", state * layers, _row(cache, "short_conv")["constant_bytes"]))
    return out, used, checks


def hybrid_attention(record):
    v, used = approved(
        record, ["layer_type_counts", "num_kv_heads", "head_dim", "hidden_size", "short_conv_kernel", "context_length"]
    )
    counts = v["layer_type_counts"]
    attn, conv_layers = counts["full_attention"], counts["short_conv"]
    per_layer = 2 * v["num_kv_heads"] * v["head_dim"] * BF16
    grows = per_layer * attn
    everywhere = per_layer * (attn + conv_layers)
    conv = v["hidden_size"] * v["short_conv_kernel"] * BF16 * conv_layers
    ctx = v["context_length"]
    cache = render_cache(v)
    out = dict(
        attn=attn,
        conv_layers=conv_layers,
        total=attn + conv_layers,
        per_layer=f"{per_layer:,} B",
        grows=size(grows),
        everywhere=size(everywhere),
        ctx=f"{ctx:,}",
        grows_full=size(grows * ctx),
        everywhere_full=size(everywhere * ctx),
        conv=size(conv),
    )
    checks = [("attention cache per token", grows, cache["grows_per_token_bytes"])]
    checks.append(("fixed convolution state", conv, cache["constant_bytes"]))
    return out, used, checks


def rmsnorm(record):
    v, used = approved(
        record,
        ["normalization_type", "hidden_size", "num_layers", "params_total", "block_shapes", "parameter_components"],
    )
    hidden = v["hidden_size"]
    per_layer = {}
    for group in v["block_shapes"]["mixers"].values():
        norms = [t for t, shape in group["tensors"].items() if t.endswith("norm.weight") and shape == [hidden]]
        norms = [t for t in norms if not t.startswith("self_attn")]
        per_layer[len(norms)] = per_layer.get(len(norms), 0) + group["layers"]
    if len(per_layer) != 1:
        raise ExampleError("layers carry different numbers of decoder norms")
    ((count, layers),) = per_layer.items()
    final = sum(
        1 for t, shape in v["block_shapes"]["globals"].items() if t.endswith("norm.weight") and shape == [hidden]
    )
    total = (count * layers + final) * hidden
    out = dict(
        hidden=f"{hidden:,}",
        count=count,
        layers=layers,
        final=final,
        norms=count * layers + final,
        total=f"{total:,}",
        share=f"{100 * total / v['params_total']:.3f}",
    )
    checks = [("norm parameters", total, v["parameter_components"]["norms"])]
    return out, used, checks


def qk_norm(record):
    v, used = approved(
        record, ["qk_norm", "head_dim", "num_attention_heads", "num_kv_heads", "num_layers", "block_shapes"]
    )
    hd, layers = v["head_dim"], v["num_layers"]
    out = dict(
        hd=hd,
        hq=v["num_attention_heads"],
        kv=v["num_kv_heads"],
        layers=layers,
        per_layer=2 * hd,
        total=f"{2 * hd * layers:,}",
    )
    checks = [
        ("q_norm width", hd, _tensor(v, "q_norm.weight")[0]),
        ("k_norm width", hd, _tensor(v, "k_norm.weight")[0]),
    ]
    return out, used, checks


def rope(record):
    import math

    v, used = approved(record, ["rope", "head_dim", "context_length"])
    theta, hd, ctx = v["rope"]["rope_theta"], v["head_dim"], v["context_length"]
    slowest = 2 * math.pi * theta ** ((hd - 2) / hd)
    out = dict(
        theta=f"{theta:,.0f}",
        hd=hd,
        pairs=hd // 2,
        fastest=sig(2 * math.pi),
        slowest=f"{slowest / 1e6:.2f} million",
        ctx=f"{ctx:,}",
        ratio=sig(slowest / ctx),
        turn=sig(100 * ctx / slowest),
        exponent=f"({hd} − 2) / {hd}",
    )
    checks = [
        ("rotation pairs per head", hd // 2, v["head_dim"] // 2),
        ("base", theta, render_values(record)["rope"]["rope_theta"]),
    ]
    return out, used, checks


def silu(record):
    v, used = approved(
        record,
        [
            "activation",
            "ffn_gated",
            "hidden_size",
            "ffn_intermediate",
            "num_layers",
            "params_total",
            "parameter_components",
        ],
    )
    if v["activation"] != "silu" or not v["ffn_gated"]:
        raise ExampleError("the SiLU example needs a gated SiLU feed-forward network")
    hidden, inter, layers = v["hidden_size"], v["ffn_intermediate"], v["num_layers"]
    per_layer = 3 * hidden * inter
    total = per_layer * layers
    out = dict(
        hidden=f"{hidden:,}",
        inter=f"{inter:,}",
        layers=layers,
        per_layer=f"{per_layer:,}",
        total=params(total),
        share=sig(100 * total / v["params_total"]),
    )
    checks = [("feed-forward parameters", total, v["parameter_components"]["dense_ffn"])]
    return out, used, checks


def nope(record):
    v, used = approved(
        record, ["mla_use_nope", "layer_type_counts", "mla_kv_lora_rank", "mla_qk_rope_head_dim", "block_shapes"]
    )
    counts = v["layer_type_counts"]
    mla_layers = counts.get("full_attention_mla", 0)
    rank, rope_w = v["mla_kv_lora_rank"], v["mla_qk_rope_head_dim"]
    per_layer = (rank + rope_w) * BF16
    cache = render_cache(v)
    out = dict(
        mla_layers=mla_layers,
        linear=counts.get("kda_linear", 0),
        rank=rank,
        rope=rope_w,
        width=rank + rope_w,
        per_layer=f"{per_layer:,} B",
        per_token=size(per_layer * mla_layers),
    )
    checks = [("compressed projection width", rank + rope_w, _tensor(v, "kv_a_proj_with_mqa.weight")[0])]
    checks.append(("cache per token", per_layer * mla_layers, cache["grows_per_token_bytes"]))
    return out, used, checks


def gated_attention(record):
    v, used = approved(
        record,
        [
            "attention_output_gate",
            "num_attention_heads",
            "v_head_dim",
            "hidden_size",
            "layer_type_counts",
            "block_shapes",
        ],
    )
    heads, dv, hidden = v["num_attention_heads"], v["v_head_dim"], v["hidden_size"]
    layers = v["layer_type_counts"].get("full_attention_mla", 0)
    width = heads * dv
    per_layer = width * hidden
    out = dict(
        heads=heads,
        dv=dv,
        width=f"{width:,}",
        hidden=f"{hidden:,}",
        layers=layers,
        per_layer=params(per_layer) if per_layer >= 1e9 else f"{per_layer / 1e6:.1f} million",
        total=params(per_layer * layers),
    )
    gate = _tensor(v, "g_proj.weight")
    checks = [("gate width", width, gate[0]), ("gate parameters per layer", per_layer, _prod(gate))]
    return out, used, checks


def attention_residuals(record):
    v, used = approved(
        record, ["attn_res_block_size", "hidden_size", "num_layers", "block_shapes", "parameter_components"]
    )
    hidden, layers = v["hidden_size"], v["num_layers"]
    per_layer = 0
    for group in v["block_shapes"]["mixers"].values():
        per_layer = sum(_prod(shape) for t, shape in group["tensors"].items() if "_res_" in t)
    output = sum(_prod(shape) for t, shape in v["block_shapes"]["globals"].items() if "_res_" in t)
    total = per_layer * layers + output
    out = dict(
        block=v["attn_res_block_size"],
        hidden=f"{hidden:,}",
        layers=layers,
        per_layer=f"{per_layer:,}",
        total=f"{total / 1e6:.2f} million",
    )
    checks = [("attention-residual parameters", total, v["parameter_components"]["attention_residuals"])]
    return out, used, checks


def latent_moe(record):
    v, used = approved(
        record,
        [
            "moe_latent_dim",
            "hidden_size",
            "moe_expert_intermediate",
            "moe_num_experts",
            "num_layers",
            "dense_layers",
            "ffn_gated",
            "parameter_components",
        ],
    )
    latent, hidden, inter, experts = (
        v["moe_latent_dim"],
        v["hidden_size"],
        v["moe_expert_intermediate"],
        v["moe_num_experts"],
    )
    mats = 3 if v["ffn_gated"] else 2
    layers = v["num_layers"] - v["dense_layers"]
    per_expert = mats * latent * inter
    full_width = mats * hidden * inter
    total = per_expert * experts * layers
    out = dict(
        latent=f"{latent:,}",
        hidden=f"{hidden:,}",
        inter=f"{inter:,}",
        experts=experts,
        layers=layers,
        per_expert=f"{per_expert / 1e6:.1f} million",
        full_width=f"{full_width / 1e6:.1f} million",
        total=params(total),
        saved=params((full_width - per_expert) * experts * layers),
        projection=f"{2 * hidden * latent / 1e6:.1f} million",
        ratio=sig(hidden / latent),
        full_total=params(full_width * experts * layers),
    )
    checks = [("routed expert parameters", total, v["parameter_components"]["moe_routed_experts"])]
    return out, used, checks


def lab_preset(record):
    """Starting values for an attention-cache calculator, from the same approved fields."""
    values = render_values(record)
    names = ["num_layers", "num_attention_heads", "num_kv_heads", "head_dim", "context_length"]
    if values.get("sliding_window"):
        names.append("sliding_window")
    v, _ = approved(record, names)
    return dict(
        model=record["id"].split("/")[-1],
        layers=v["num_layers"],
        hq=v["num_attention_heads"],
        kv=v["num_kv_heads"],
        hd=v["head_dim"],
        context=v["context_length"],
        window=v.get("sliding_window"),
        bytes=render_cache(v),
    )


COMPUTE = {
    "kv_cache": kv_cache,
    "gqa": gqa,
    "mla": mla,
    "sliding_window": sliding_window,
    "moe": moe,
    "linear_attention": linear_attention,
    "mha": mha,
    "sparse_attention": sparse_attention,
    "dsa": dsa,
    "mamba2": mamba2,
    "short_convolution": short_convolution,
    "hybrid_attention": hybrid_attention,
    "rmsnorm": rmsnorm,
    "qk_norm": qk_norm,
    "rope": rope,
    "silu": silu,
    "nope": nope,
    "gated_attention": gated_attention,
    "attention_residuals": attention_residuals,
    "latent_moe": latent_moe,
}


def worked(example: dict, published: dict) -> dict:
    """Fill an example's templates from its model's record. `published` maps repo -> (stem, record, values)."""
    repo = example["model"]
    if repo not in published:
        raise ExampleError(f"{repo} is not a published model")
    stem, record, _ = published[repo]
    out, used, checks = COMPUTE[example["compute"]](record)
    out["model"] = repo.split("/")[-1]

    def fill(text):
        return text.format(**out)

    return dict(
        repo=repo,
        stem=stem,
        commit=record["source_commit"],
        fields=used,
        checks=checks,
        values=out,
        setup=fill(example["setup"]),
        steps=[fill(step) for step in example["steps"]],
        result=fill(example["result"]),
        code=fill(example["code"]) if example.get("code") else None,
    )
