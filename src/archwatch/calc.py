"""Inference-memory calculator: what the cache costs per token and what it costs regardless of length.

Layer types differ in kind, not just size:
  - full attention keeps K and V for every token          -> grows with context
  - MLA keeps one compressed latent (+ RoPE key) per token  -> grows, but much slower
  - DSA adds a small indexer key per token                  -> grows
  - sliding-window attention keeps at most `window` tokens  -> constant
  - short convolutions keep the last few inputs             -> constant
  - linear attention (KDA) keeps a fixed matrix state       -> constant
Every assumption that is not read from the config is listed in the output.
"""

from __future__ import annotations

BF16 = 2
FP32 = 4


def cache_model(v: dict) -> dict:
    """v: plain {field: value} facts. Returns per-token and constant cache bytes with a breakdown."""
    counts: dict = v["layer_type_counts"]
    kv, hd, heads = v.get("num_kv_heads"), v.get("head_dim"), v.get("num_attention_heads")
    per_token, per_token_low, constant = 0, 0, 0
    rows, assumptions = [], ["cache values stored in bf16 (2 bytes) unless noted"]

    for kind, n in counts.items():
        if kind == "full_attention":
            b = 2 * kv * hd * BF16
            per_token += b * n
            per_token_low += b * n
            rows.append(
                {
                    "layer_type": kind,
                    "layers": n,
                    "grows_per_token_bytes": b * n,
                    "constant_bytes": 0,
                    "formula": f"2 x {kv} KV heads x {hd} head_dim x 2 B x {n} layers",
                }
            )
        elif kind == "sliding_attention":
            w = v["sliding_window"]
            b = 2 * kv * hd * BF16 * w
            constant += b * n
            rows.append(
                {
                    "layer_type": kind,
                    "layers": n,
                    "grows_per_token_bytes": 0,
                    "constant_bytes": b * n,
                    "formula": f"window {w} x 2 x {kv} x {hd} x 2 B x {n} layers (capped)",
                }
            )
        elif kind == "short_conv":
            k, h = v.get("short_conv_kernel", 3), v["hidden_size"]
            b = h * k * BF16
            constant += b * n
            rows.append(
                {
                    "layer_type": kind,
                    "layers": n,
                    "grows_per_token_bytes": 0,
                    "constant_bytes": b * n,
                    "formula": f"{h} channels x {k} taps x 2 B x {n} layers (upper bound)",
                }
            )
            assumptions.append(f"short-conv state taken as hidden x kernel ({k}); real kernels may keep kernel-1")
        elif kind in ("gated_dsa_mla", "full_attention_mla"):
            lat, rope = v["mla_kv_lora_rank"], v.get("mla_qk_rope_head_dim", 0)
            idx = v.get("dsa_index_head_dim", 0) if kind == "gated_dsa_mla" else 0
            high = (lat + rope + idx) * BF16
            low = (lat + rope) * BF16 + idx * 1
            per_token += high * n
            per_token_low += low * n
            parts = f"({lat} latent + {rope} rope key" + (f" + {idx} DSA index key" if idx else "") + ") x 2 B"
            rows.append(
                {
                    "layer_type": kind,
                    "layers": n,
                    "grows_per_token_bytes": high * n,
                    "grows_per_token_bytes_low": low * n,
                    "constant_bytes": 0,
                    "formula": f"{parts} x {n} layers",
                }
            )
            if idx:
                assumptions.append(
                    "DSA indexer keys counted at bf16; low estimate uses fp8 (1 B), as DeepSeek-style indexers often do"
                )
        elif kind == "kda_linear":
            lh, ld, ck = v["linear_attn_num_heads"], v["linear_attn_head_dim"], v.get("linear_attn_conv_kernel", 4)
            state = lh * ld * ld * FP32
            conv = 3 * lh * ld * (ck - 1) * BF16
            constant += (state + conv) * n
            rows.append(
                {
                    "layer_type": kind,
                    "layers": n,
                    "grows_per_token_bytes": 0,
                    "constant_bytes": (state + conv) * n,
                    "formula": f"({lh} heads x {ld}x{ld} state x 4 B + q/k/v conv {ck - 1} taps) x {n} layers",
                }
            )
            assumptions.append("KDA recurrent state counted in fp32 (4 B), the usual choice for delta-rule states")
        else:
            rows.append(
                {
                    "layer_type": kind,
                    "layers": n,
                    "grows_per_token_bytes": None,
                    "constant_bytes": None,
                    "formula": "unknown layer type: not counted",
                }
            )
            assumptions.append(f"layer type '{kind}' has no memory rule yet; sent to review")

    # baseline: the same model with plain multi-head attention caching full K and V in every layer
    vd = v.get("v_head_dim") or hd
    naive = v["num_layers"] * heads * (hd + vd) * BF16 if heads and hd else None
    ctx = v.get("context_length") or 0
    points = [n for n in (8192, 32768, 131072, 262144, 1048576) if n <= ctx] or [ctx]
    at = {
        str(n): {"bytes": per_token * n + constant, "gib": round((per_token * n + constant) / 2**30, 3)} for n in points
    }
    return {
        "grows_per_token_bytes": per_token,
        "grows_per_token_bytes_low": per_token_low,
        "constant_bytes": constant,
        "naive_mha_per_token_bytes": naive,
        "reduction_vs_naive_mha": round(naive / per_token, 1) if naive and per_token else None,
        "memory_at_context": at,
        "breakdown": rows,
        "assumptions": assumptions,
    }
