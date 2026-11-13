"""Exact parameter counts from safetensors headers (tensor shapes), with quantization awareness.

Packed low-bit weights (e.g. MXFP4 stored as uint8) hold 8/bits parameters per stored element;
quantization scales are metadata, not parameters, and are skipped.
"""

from __future__ import annotations

import re
from math import prod

SCALE_SUFFIXES = ("weight_scale", "weight_global_scale", "input_scale", "weight_scale_2", "weight_zero_point")


def _component(name: str) -> str:
    n = name
    if "mm_projector" in n:
        return "vision_projector"
    if "vision_tower" in n or "visual." in n:
        return "vision"
    if "embed_tokens" in n:
        return "embedding"
    if "lm_head" in n:
        return "lm_head"
    if ".experts." in n:
        return "moe_routed_experts"
    if "shared_experts" in n:
        return "moe_shared_experts"
    if "routed_expert_" in n:
        return "moe_latent_projection"
    if re.search(r"block_sparse_moe\.gate|mlp\.gate\.|router", n):
        return "moe_router"
    if "self_attn" in n:
        return "attention_or_linear_mixer"
    if ".conv." in n:
        return "short_conv_mixer"
    if "feed_forward" in n or ".mlp." in n:
        return "dense_ffn"
    if "_res_" in n:
        return "attention_residuals"
    if "norm" in n:
        return "norms"
    return "other"


def _layer_idx(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def count(tensors: dict[str, dict], quant_bits: int | None) -> dict:
    by_comp: dict[str, int] = {}
    per_layer_routed: dict[int, int] = {}
    experts_per_layer: dict[int, set] = {}
    skipped_scales = 0
    for name, t in sorted(tensors.items()):
        if name.endswith(SCALE_SUFFIXES):
            skipped_scales += 1
            continue
        n = prod(t["shape"])
        if name.endswith("weight_packed") and t["dtype"] in ("U8", "I8"):
            n = n * (8 // (quant_bits or 4))
        c = _component(name)
        by_comp[c] = by_comp.get(c, 0) + n
        li = _layer_idx(name)
        if c == "moe_routed_experts" and li is not None:
            per_layer_routed[li] = per_layer_routed.get(li, 0) + n
            m = re.search(r"experts\.(\d+)\.", name)
            if m:
                experts_per_layer.setdefault(li, set()).add(int(m.group(1)))
    total = sum(by_comp.values())
    layers_in_weights = sorted({i for i in (_layer_idx(k) for k in tensors) if i is not None})
    return {
        "total": total,
        "text_total": total - by_comp.get("vision", 0) - by_comp.get("vision_projector", 0),
        "by_component": dict(sorted(by_comp.items(), key=lambda kv: (-kv[1], kv[0]))),
        "routed_expert_params": by_comp.get("moe_routed_experts", 0),
        "moe_layers": len(per_layer_routed),
        "experts_per_moe_layer": sorted({len(v) for v in experts_per_layer.values()}),
        "layer_indices_in_weights": [layers_in_weights[0], layers_in_weights[-1]] if layers_in_weights else None,
        "tensor_count": len(tensors),
        "skipped_scale_tensors": skipped_scales,
    }


def active_params(counts: dict, num_experts: int | None, k: int | None) -> dict | None:
    """Parameters used per token: everything except the routed experts that are not selected."""
    if not num_experts or not k:
        return None
    routed = counts["routed_expert_params"]
    inactive = routed * (1 - k / num_experts)
    text_active = counts["text_total"] - inactive
    return {
        "text_active": round(text_active),
        "text_active_without_embeddings": round(text_active - counts["by_component"].get("embedding", 0)),
        "with_vision": round(counts["total"] - inactive),
    }


def layer_kinds(tensors: dict[str, dict]) -> list[str]:
    """Mixer kind of each decoder layer (0-based), read from which tensors the layer has.

    Sliding and full attention share the same weights, so both read as 'attention'."""
    names: dict[int, set] = {}
    for k in tensors:
        if "vision" in k:
            continue
        i = _layer_idx(k)
        if i is not None:
            names.setdefault(i, set()).add(k)
    out = []
    for i in sorted(names):
        s = " ".join(sorted(names[i]))
        if "indexer." in s:
            out.append("dsa_mla")
        elif "kv_a_proj_with_mqa" in s:
            out.append("mla")
        elif "self_attn.A_log" in s or "dt_bias" in s:
            out.append("linear_delta")
        elif ".conv.conv." in s or ".conv.in_proj" in s:
            out.append("short_conv")
        elif "self_attn.q_proj" in s:
            out.append("attention")
        else:
            out.append("unknown")
    return out


def shape_facts(tensors: dict[str, dict], quant_bits: int | None) -> dict:
    """Facts read straight from weight shapes, used to cross-check config values."""
    out: dict = {}
    mult = 8 // (quant_bits or 4)
    for name, t in sorted(tensors.items()):
        if re.search(r"experts\.0\.w1\.weight(_packed)?$", name) and "expert_intermediate" not in out:
            rows, cols = t["shape"]
            out["expert_intermediate"] = rows
            out["expert_input_dim"] = cols * (mult if name.endswith("_packed") else 1)
        if name.endswith("routed_expert_up_proj.weight") and "latent_dim" not in out:
            out["latent_dim"] = t["shape"][1] if t["shape"][0] > t["shape"][1] else t["shape"][0]
        if name.endswith("embed_tokens.weight") and "vocab_rows" not in out:
            out["vocab_rows"], out["hidden_size"] = t["shape"]
        if name.endswith("feed_forward.w1.weight") and "ffn_intermediate" not in out:
            out["ffn_intermediate"] = t["shape"][0]
        if name.endswith("conv.conv.weight") and "conv_kernel" not in out:
            out["conv_kernel"] = t["shape"][-1]
        if name.endswith("self_attn.q_norm.weight") or name.endswith("self_attn.k_norm.weight"):
            out["qk_norm"] = True
        if name.endswith("self_attn.gate_proj.weight") or name.endswith("self_attn.g_proj.weight"):
            out["attention_output_gate_tensor"] = True
        if name.endswith("lm_head.weight"):
            out["has_lm_head"] = True
        if name.endswith("kv_a_proj_with_mqa.weight") and "mla_cached_width" not in out:
            out["mla_cached_width"] = t["shape"][0]  # kv_lora_rank + qk_rope_head_dim
        if name.endswith("indexer.wk.weight") and "dsa_index_key_width" not in out:
            out["dsa_index_key_width"] = t["shape"][0]
    return out
