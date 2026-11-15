"""Exact parameter counts from safetensors headers (tensor shapes), with quantization awareness.

Packed low-bit weights (e.g. MXFP4 stored as uint8) hold 8/bits parameters per stored element;
quantization scales are metadata, not parameters, and are skipped.
"""

from __future__ import annotations

import re
from math import prod

SCALE_SUFFIXES = (
    "weight_scale",
    "weight_global_scale",
    "input_scale",
    "weight_scale_2",
    "weight_zero_point",
    "qscale_weight",
)


def annotate_packed_fp4(tensors: dict[str, dict]) -> dict[str, dict]:
    """Mark 4-bit weights stored two per byte (NVFP4, MXFP4, native FP4) from their block scales.

    A U8/I8 `<x>.weight` whose sibling `<x>.weight_scale` has one scale per 16 or 32 values along the last
    dimension, covering twice the stored columns, holds two 4-bit values per byte.
    """
    for name, t in tensors.items():
        if not name.endswith(".weight") or t["dtype"] not in ("U8", "I8") or len(t["shape"]) < 2:
            continue
        scale = tensors.get(name + "_scale")
        if not scale or len(scale["shape"]) != len(t["shape"]) or scale["shape"][:-1] != t["shape"][:-1]:
            continue
        if any(scale["shape"][-1] * block == 2 * t["shape"][-1] for block in (16, 32)):
            t["fp4_packed"] = True
    return tensors


def logical_elements(name, tensor, quant_bits=None, native_expert_bits=None):
    """Exclude recognized storage metadata; unpack only an explicitly supported format."""
    if name.endswith(SCALE_SUFFIXES) or (name.endswith(".scale") and tensor["dtype"] == "F8_E8M0"):
        return None
    if tensor["dtype"] == "I64" and not name.endswith(".weight"):
        return None  # e.g. n-gram hash multipliers, offsets and vocabulary sizes
    if name.endswith("_scales") and tensor["dtype"] == "U8":
        return None  # MXFP4 block scales are metadata
    n = prod(tensor["shape"]) * tensor.get("tp_parts", 1)
    if tensor.get("fp4_packed"):
        return n * 2
    if name.endswith("_blocks") and tensor["dtype"] == "U8":
        return n * 2  # MXFP4: two 4-bit values per stored byte
    if name.endswith("weight_packed") and tensor["dtype"] in ("U8", "I8"):
        return n * (8 // (quant_bits or 4))
    if native_expert_bits == 4 and ".experts." in name and name.endswith(".weight") and tensor["dtype"] in ("U8", "I8"):
        return n * 2
    return n


MTP = re.compile(r"(^|\.)mtp(\.|_)")  # multi-token-prediction / speculative-decoding heads, e.g. mtp.*, mtp_dspark.*


def _is_mtp(name: str) -> bool:
    return bool(MTP.search(name))


def main_language_tensor(name, num_layers=None):
    if any(s in name for s in ("vision", "visual.", "cls_net.", "nextn_predict")) or _is_mtp(name):
        return False
    index = _layer_idx(name)
    return num_layers is None or index is None or index < num_layers


LOOKUP_TABLE = re.compile(r"engram\.embed|ngram_embed|ple_embedding|embed_tokens_per_layer|per_layer_embed")


def _component(name: str) -> str:
    n = name
    if LOOKUP_TABLE.search(n):
        return "lookup_embedding"  # per-token row lookups (Engram, n-gram, per-layer embeddings), not matmul weights
    if "cls_net." in n or _is_mtp(n):
        return "auxiliary"
    if any(x in n for x in ("mm_projector", "multi_modal_projector", "patch_merge", "pre_mm_projector_norm")):
        return "vision_projector"  # image-to-text projectors are not text weights
    if "vision" in n or "visual." in n:
        return "vision"
    if (
        "embed_tokens" in n
        or "word_embeddings" in n
        or n in ("backbone.embeddings.weight", "embed.weight", "tok_embeddings.weight")
    ):
        return "embedding"
    if "lm_head" in n or n in ("head.weight", "output.weight", "model.output.weight"):
        return "lm_head"
    if ".experts." in n or ".experts.gate_up_proj" in n or ".experts.down_proj" in n:
        return "moe_routed_experts"
    if "shared_expert" in n:
        return "moe_shared_experts"
    if "routed_expert_" in n:
        return "moe_latent_projection"
    if re.search(r"block_sparse_moe\.gate|mlp\.gate\.|router", n):
        return "moe_router"
    if "self_attn" in n:
        return "attention_or_linear_mixer"
    if ".mixer." in n and any(s in n for s in ("in_proj", "conv1d", "A_log", "dt_bias", ".D", "out_proj")):
        return "state_space_mixer"
    if ".mixer." in n and any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention_or_linear_mixer"
    if ".mixer." in n and any(s in n for s in ("up_proj", "down_proj")):
        return "dense_ffn"
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


def count(tensors: dict[str, dict], quant_bits: int | None, *, native_expert_bits=None, num_layers=None) -> dict:
    annotate_packed_fp4(tensors)
    by_comp: dict[str, int] = {}
    per_layer_routed: dict[int, int] = {}
    experts_per_layer: dict[int, set] = {}
    skipped_scales = 0
    skipped_buffers = 0
    for name, t in sorted(tensors.items()):
        n = logical_elements(name, t, quant_bits, native_expert_bits)
        if n is None:
            if t["dtype"] == "I64":
                skipped_buffers += 1
            else:
                skipped_scales += 1
            continue
        c = _component(name)
        li = _layer_idx(name)
        if li is not None and num_layers is not None and li >= num_layers and c not in ("vision", "vision_projector"):
            c = "auxiliary"  # MTP / next-token predictor layers stored after the decoder stack
        by_comp[c] = by_comp.get(c, 0) + n
        if c == "moe_routed_experts" and li is not None and main_language_tensor(name, num_layers):
            per_layer_routed[li] = per_layer_routed.get(li, 0) + n
            m = re.search(r"experts\.(\d+)\.", name)
            if m:
                experts_per_layer.setdefault(li, set()).add(int(m.group(1)))
            elif len(t["shape"]) == 3:
                experts_per_layer.setdefault(li, set()).update(range(t["shape"][0]))
    total = sum(by_comp.values())
    layers_in_weights = sorted(
        {_layer_idx(k) for k in tensors if main_language_tensor(k, num_layers) and _layer_idx(k) is not None}
    )
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
        "skipped_buffer_tensors": skipped_buffers,
    }


def active_params(counts: dict, num_experts: int | None, k: int | None) -> dict | None:
    """Parameters used per token: everything except unselected routed experts and auxiliary heads."""
    if not num_experts or not k:
        return None
    routed = counts["routed_expert_params"]
    inactive = routed * (1 - k / num_experts)
    comp = counts["by_component"]
    # Auxiliary heads (MTP, speculative decoding) are not part of the standard forward pass, and lookup tables
    # (Engram memory, n-gram / per-layer embeddings) are read a few rows per token, so they are not active compute.
    text_active = counts["text_total"] - inactive - comp.get("auxiliary", 0) - comp.get("lookup_embedding", 0)
    vision = comp.get("vision", 0) + comp.get("vision_projector", 0)
    return {
        "text_active": round(text_active),
        "text_active_without_embeddings": round(text_active - comp.get("embedding", 0)),
        # "Non-embedding" as model cards use it: input embedding and output head both excluded.
        "text_active_non_embedding": round(text_active - comp.get("embedding", 0) - comp.get("lm_head", 0)),
        "with_vision": round(text_active + vision),  # auxiliary heads and lookup tables stay excluded
    }


def layer_kinds(tensors: dict[str, dict], *, num_layers=None) -> list[str]:
    """Mixer kind of each decoder layer (0-based), read from which tensors the layer has.

    Sliding and full attention share the same weights, so both read as 'attention'."""
    names: dict[int, set] = {}
    for k in tensors:
        if not main_language_tensor(k, num_layers):
            continue
        i = _layer_idx(k)
        if i is not None:
            names.setdefault(i, set()).add(k)
    out = []
    for i in sorted(names):
        s = " ".join(sorted(names[i]))
        if "indexer.index_qk_proj" in s:
            out.append("qsa")
        elif ".attn.wkv.weight" in s:
            out.append("compressed_attention")
        elif "indexer." in s:
            out.append("dsa_mla")
        elif "kv_a_proj_with_mqa" in s or "wkv_a_with_mqa" in s:
            out.append("mla")
        elif ".mixer.A_log" in s and ".mixer.conv1d.weight" in s:
            out.append("mamba2")
        elif "self_attn.A_log" in s or "linear_attn.A_log" in s or "dt_bias" in s:
            out.append("linear_delta")
        elif ".conv.conv." in s or ".conv.in_proj" in s:
            out.append("short_conv")
        elif "self_attn.q_proj" in s or ".mixer.q_proj" in s:
            out.append("attention")
        elif ".mixer.up_proj" in s and ".mixer.down_proj" in s:
            out.append("mlp")
        else:
            out.append("unknown")
    return out


def shape_facts(tensors: dict[str, dict], quant_bits: int | None, *, native_expert_bits=None, num_layers=None) -> dict:
    """Facts read straight from weight shapes, used to cross-check config values."""
    annotate_packed_fp4(tensors)
    out: dict = {}
    mult = 8 // (quant_bits or 4)
    for name, t in sorted(tensors.items()):
        if not main_language_tensor(name, num_layers):
            continue
        if re.search(r"experts\.0\.(w1|gate_proj)\.weight(_packed)?$", name) and "expert_intermediate" not in out:
            rows, cols = t["shape"]
            out["expert_intermediate"] = rows
            out["expert_input_dim"] = cols * (mult if name.endswith("_packed") else 2 if t.get("fp4_packed") else 1)
            if native_expert_bits == 4 and t["dtype"] in ("U8", "I8"):
                out["expert_input_dim"] = cols * 2
        if name.endswith("experts.down_proj") and len(t["shape"]) == 3:
            # Llama4 stores [expert, intermediate, hidden]; Qwen/Gemma use [expert, hidden, intermediate].
            gate = tensors.get(name.removesuffix("down_proj") + "gate_up_proj")
            if gate:
                _, a, b = t["shape"]
                if gate["shape"][1:] == [b, 2 * a]:
                    out.setdefault("expert_intermediate", a)
                    out.setdefault("expert_input_dim", b)
                elif gate["shape"][1:] == [2 * b, a]:
                    out.setdefault("expert_intermediate", b)
                    out.setdefault("expert_input_dim", a)
                out["ffn_gated"] = True
        if name.endswith(("shared_expert.gate_proj.weight", "shared_experts.gate_proj.weight")):
            out["shared_experts"] = 1
            out["shared_expert_intermediate"] = t["shape"][0]
        if ".router.proj.weight" in name and any(
            k.startswith(name.split(".router.")[0] + ".experts.") for k in tensors
        ):
            # Gemma MoE keeps a dense MLP alongside the routed experts.
            out["shared_experts"] = 1
            dense_gate = tensors.get(name.split(".router.")[0] + ".mlp.gate_proj.weight")
            if dense_gate:
                out["shared_expert_intermediate"] = dense_gate["shape"][0]
        if name.endswith("routed_expert_up_proj.weight") and "latent_dim" not in out:
            out["latent_dim"] = t["shape"][1] if t["shape"][0] > t["shape"][1] else t["shape"][0]
        if (
            name.endswith("embed_tokens.weight")
            or name in ("backbone.embeddings.weight", "embed.weight", "tok_embeddings.weight")
        ) and "vocab_rows" not in out:
            out["vocab_rows"], out["hidden_size"] = t["shape"]
        if name.endswith("feed_forward.w1.weight") and "ffn_intermediate" not in out:
            out["ffn_intermediate"] = t["shape"][0]
        if not any(s in name for s in (".experts.", ".shared_experts.")) and "ffn_intermediate" not in out:
            if name.endswith(
                ("mlp.gate_proj.weight", "mlp.fc1.weight", "mixer.up_proj.weight", "feed_forward.gate_proj.weight")
            ):
                out["ffn_intermediate"] = t["shape"][0]
            elif name.endswith("mlp.gate_up_proj.weight"):
                out["ffn_intermediate"] = t["shape"][0] // 2
        if name.endswith("conv.conv.weight") and "conv_kernel" not in out:
            out["conv_kernel"] = t["shape"][-1]
        if name.endswith("self_attn.q_norm.weight") or name.endswith("self_attn.k_norm.weight"):
            out["qk_norm"] = True
        if name.endswith("self_attn.gate_proj.weight") or name.endswith("self_attn.g_proj.weight"):
            out["attention_output_gate_tensor"] = True
        if name.endswith("lm_head.weight") or name in ("head.weight", "output.weight", "model.output.weight"):
            out["has_lm_head"] = True  # separate output head (HF lm_head, DeepSeek-native head, Mistral-native output)
        if name.endswith("kv_a_proj_with_mqa.weight") and "mla_cached_width" not in out:
            out["mla_cached_width"] = t["shape"][0]  # kv_lora_rank + qk_rope_head_dim
        if name.endswith("indexer.wk.weight") and "dsa_index_key_width" not in out:
            out["dsa_index_key_width"] = t["shape"][0]
        if name.endswith("mixer.conv1d.weight"):
            out["mamba_conv_channels"] = t["shape"][0]
    language = [n for n in tensors if main_language_tensor(n, num_layers)]
    if any(
        n.endswith(
            (
                "feed_forward.w3.weight",
                "mlp.gate_proj.weight",
                "mlp.gate_up_proj.weight",
                "experts.0.w3.weight_packed",
                "experts.0.w3.weight",
                "mixer.gate_proj.weight",
                "feed_forward.gate_proj.weight",
            )
        )
        for n in language
    ):
        out["ffn_gated"] = True
    elif any(n.endswith(("mlp.fc1.weight", "mixer.up_proj.weight")) for n in language):
        out["ffn_gated"] = False
    if any("post_feedforward_layernorm" in n or "post_ffn_norm" in n or "post_mlp_layernorm" in n for n in language):
        out["sandwich_norm"] = True
    return out


LAYER_TENSOR = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(.+)$")
FFN_PREFIXES = ("mlp.", "feed_forward.", "block_sparse_moe.")
GLOBAL_PREFIXES = ("language_model.", "model.", "backbone.")


def _ranges(indices):
    out = []
    for i in sorted(indices):
        if out and out[-1][1] == i - 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def block_shapes(tensors: dict[str, dict], layer_types: list[str] | None, *, num_layers=None, quant_bits=None):
    """Representative logical tensor shapes per config layer type, read from weight headers.

    Mixer/norm/residual tensors are grouped by the config layer type and must be identical across
    every layer of that type; FFN tensors are grouped by their own shape signature. Routed-expert
    tensors are excluded (their widths are published separately). Returns None when the headers and
    the config pattern cannot be aligned, so the field stays unpublished rather than guessed.
    """
    annotate_packed_fp4(tensors)
    if not layer_types or any(t.get("tp_parts") for t in tensors.values()):
        return None  # tensor-parallel slices are per-rank shapes, not logical ones
    per_layer: dict[int, dict] = {}
    global_shapes = {}
    for name, t in tensors.items():
        if (
            not main_language_tensor(name, num_layers)
            or ".experts." in name
            or _component(name) in ("vision", "vision_projector")
        ):
            continue
        if logical_elements(name, t, quant_bits) is None:
            continue
        shape = list(t["shape"])
        if t.get("fp4_packed"):
            shape[-1] *= 2
        if name.endswith("weight_packed") and t["dtype"] in ("U8", "I8"):
            shape[-1] *= 8 // (quant_bits or 4)
            name = name.removesuffix("_packed")
        match = LAYER_TENSOR.match(name)
        if match:
            per_layer.setdefault(int(match.group(1)), {})[match.group(2)] = shape
        else:
            short = name
            while short.startswith(GLOBAL_PREFIXES):
                short = short.split(".", 1)[1]
            global_shapes[short] = shape
    if sorted(per_layer) != list(range(len(layer_types))):
        return None
    mixers: dict[str, dict] = {}
    ffn_groups: dict[str, dict] = {}
    for index, kind in enumerate(layer_types):
        layer = per_layer[index]
        mixer = {k: v for k, v in sorted(layer.items()) if not k.startswith(FFN_PREFIXES)}
        ffn = {k: v for k, v in sorted(layer.items()) if k.startswith(FFN_PREFIXES)}
        if kind in mixers:
            if mixers[kind]["tensors"] != mixer:
                return None  # same config type, different weights: needs review, not a representative shape
            mixers[kind]["layers"] += 1
        else:
            mixers[kind] = {"layers": 1, "tensors": mixer}
        if ffn:
            key = repr(sorted(ffn.items()))
            ffn_groups.setdefault(key, {"layer_ranges": [], "tensors": ffn})["layer_ranges"].append(index)
    ffns = [dict(group, layer_ranges=_ranges(group["layer_ranges"])) for group in ffn_groups.values()]
    ffns.sort(key=lambda g: g["layer_ranges"][0][0])
    return {"mixers": mixers, "ffns": ffns, "globals": dict(sorted(global_shapes.items()))}
