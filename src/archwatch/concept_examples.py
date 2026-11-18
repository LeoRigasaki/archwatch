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


COMPUTE = {
    "kv_cache": kv_cache,
    "gqa": gqa,
    "mla": mla,
    "sliding_window": sliding_window,
    "moe": moe,
    "linear_attention": linear_attention,
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
