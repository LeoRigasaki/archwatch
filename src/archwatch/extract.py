"""Deterministic config.json -> architecture facts. No LLM.

Every fact is returned as (value, evidence) where evidence names the config key it came from,
so the verifier and the published page can show where each number was read.
"""
from __future__ import annotations

from typing import Any

Fact = tuple[Any, str]


def _first(cfg: dict, keys: list[str], prefix: str) -> Fact | None:
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k], f"config:{prefix}{k}"
    return None


def _layer_types(text: dict, prefix: str, n_layers: int) -> tuple[list[str], str, int | None]:
    """Return a per-layer mixer list (0-based), the evidence, and the index base the config used."""
    if isinstance(text.get("layer_types"), list):
        mapping = {"conv": "short_conv", "full_attention": "full_attention",
                   "sliding_attention": "sliding_attention", "linear_attention": "linear_attention"}
        return [mapping.get(t, t) for t in text["layer_types"]], f"config:{prefix}layer_types", 0

    lac = text.get("linear_attn_config")
    if isinstance(lac, dict) and lac.get("kda_layers") is not None:
        kda, full = set(lac["kda_layers"]), set(lac["full_attn_layers"])
        allidx = kda | full
        # 1-based if the lists never use 0 and do use n_layers; the code scan confirms this separately
        base = 1 if (0 not in allidx and n_layers in allidx) else 0
        dsa = set((text.get("dsa_config") or {}).get("dsa_layers") or [])
        out = []
        for i in range(n_layers):
            j = i + base
            if j in kda:
                out.append("kda_linear")
            elif j in full:
                out.append("gated_dsa_mla" if j in dsa else "full_attention_mla")
            else:
                out.append("unknown")
        return out, f"config:{prefix}linear_attn_config", base

    window = text.get("sliding_window")
    pattern = text.get("sliding_window_pattern")
    if window and pattern:
        return (["full_attention" if (i + 1) % pattern == 0 else "sliding_attention" for i in range(n_layers)],
                f"config:{prefix}sliding_window_pattern", None)
    kind = "full_attention_mla" if text.get("kv_lora_rank") else "full_attention"
    return [kind] * n_layers, "config:(no per-layer pattern; all layers assumed identical)", None


def extract(config: dict) -> dict[str, Fact]:
    text, prefix = (config["text_config"], "text_config.") if "text_config" in config else (config, "")
    f: dict[str, Fact] = {}

    f["architecture_class"] = ((config.get("architectures") or ["?"])[0], "config:architectures")
    f["model_type"] = (config.get("model_type"), "config:model_type")

    n_layers = text["num_hidden_layers"]
    f["num_layers"] = (n_layers, f"config:{prefix}num_hidden_layers")
    f["hidden_size"] = (text["hidden_size"], f"config:{prefix}hidden_size")
    f["vocab_size"] = (text.get("vocab_size"), f"config:{prefix}vocab_size")
    ctx = _first(text, ["max_position_embeddings", "max_seq_len", "seq_length"], prefix)
    if ctx:
        f["context_length"] = ctx
    tie = config.get("tie_word_embeddings", text.get("tie_word_embeddings"))
    f["tied_embeddings"] = (bool(tie), "config:tie_word_embeddings")

    types, ev, base = _layer_types(text, prefix, n_layers)
    f["layer_types"] = (types, ev)
    if base is not None:
        f["layer_index_base_in_config"] = (base, ev + " (inferred from index range)")
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    f["layer_type_counts"] = (counts, ev)

    # attention
    heads = text.get("num_attention_heads")
    kv = text.get("num_key_value_heads", heads)
    f["num_attention_heads"] = (heads, f"config:{prefix}num_attention_heads")
    f["num_kv_heads"] = (kv, f"config:{prefix}num_key_value_heads")
    if text.get("kv_lora_rank") and text.get("qk_nope_head_dim") is not None:
        # MLA: a query/key head is the no-RoPE part plus the RoPE part; values have their own width
        f["head_dim"] = (text["qk_nope_head_dim"] + (text.get("qk_rope_head_dim") or 0),
                         f"derived:{prefix}qk_nope_head_dim + qk_rope_head_dim")
        f["v_head_dim"] = (text.get("v_head_dim"), f"config:{prefix}v_head_dim")
    else:
        head_dim = text.get("head_dim") or (text["hidden_size"] // heads if heads else None)
        f["head_dim"] = (head_dim, f"config:{prefix}head_dim" if text.get("head_dim") else "derived:hidden_size/num_attention_heads")
    if text.get("kv_lora_rank"):
        f["attention_type"] = ("MLA", f"config:{prefix}kv_lora_rank")
        for k in ["kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"]:
            if text.get(k) is not None:
                f[f"mla_{k}"] = (text[k], f"config:{prefix}{k}")
        if text.get("mla_use_nope") is not None:
            f["mla_use_nope"] = (text["mla_use_nope"], f"config:{prefix}mla_use_nope")
        if text.get("mla_use_output_gate"):
            f["attention_output_gate"] = (True, f"config:{prefix}mla_use_output_gate")
    elif heads and kv:
        kind = "MHA" if kv == heads else ("MQA" if kv == 1 else "GQA")
        f["attention_type"] = (kind, f"derived:num_key_value_heads({kv}) vs num_attention_heads({heads})")
    if text.get("sliding_window"):
        f["sliding_window"] = (text["sliding_window"], f"config:{prefix}sliding_window")

    dsa = text.get("dsa_config")
    if dsa:
        for k in ["index_topk", "index_num_heads", "index_head_dim"]:
            f[f"dsa_{k}"] = (dsa[k], f"config:{prefix}dsa_config.{k}")
    lac = text.get("linear_attn_config")
    if lac:
        f["linear_attn_num_heads"] = (lac.get("num_heads"), f"config:{prefix}linear_attn_config.num_heads")
        f["linear_attn_head_dim"] = (lac.get("head_dim"), f"config:{prefix}linear_attn_config.head_dim")
        f["linear_attn_conv_kernel"] = (lac.get("short_conv_kernel_size"), f"config:{prefix}linear_attn_config.short_conv_kernel_size")
    if text.get("conv_L_cache"):
        f["short_conv_kernel"] = (text["conv_L_cache"], f"config:{prefix}conv_L_cache")

    # feed-forward / MoE
    n_exp = _first(text, ["num_experts", "n_routed_experts", "num_local_experts"], prefix)
    if n_exp:
        f["decoder_type"] = ("MoE", n_exp[1])
        f["moe_num_experts"] = n_exp
        k = _first(text, ["num_experts_per_token", "num_experts_per_tok", "moe_top_k"], prefix)
        if k:
            f["moe_active_experts"] = k
        sh = _first(text, ["num_shared_experts", "n_shared_experts"], prefix)
        if sh:
            f["moe_shared_experts"] = sh
        mi = _first(text, ["moe_intermediate_size"], prefix)
        if mi:
            f["moe_expert_intermediate"] = mi
        lat = _first(text, ["routed_expert_hidden_size"], prefix)
        if lat:
            f["moe_latent_dim"] = lat
        fk = _first(text, ["first_k_dense_replace"], prefix)
        f["dense_layers"] = (fk[0] if fk else 0, fk[1] if fk else "default:0")
        f["dense_ffn_intermediate"] = (text.get("intermediate_size"), f"config:{prefix}intermediate_size")
        if text.get("moe_router_activation_func"):
            f["moe_router"] = (text["moe_router_activation_func"], f"config:{prefix}moe_router_activation_func")
    else:
        mixers = set(types)
        hybrid = len(mixers) > 1 and any(m in mixers for m in ("short_conv", "kda_linear", "linear_attention"))
        f["decoder_type"] = ("Hybrid (dense)" if hybrid else "Dense", ev if hybrid else "derived:no expert keys")
        f["ffn_intermediate"] = (text.get("intermediate_size"), f"config:{prefix}intermediate_size")
    act = _first(text, ["hidden_act", "hidden_activation", "activation"], prefix)
    if act:
        f["activation"] = act

    # norms and positions
    norm_bits = ["RMSNorm"]
    if text.get("use_sandwich_norm"):
        norm_bits.append("sandwich (post-mixer + post-FFN)")
        f["sandwich_norm"] = (True, f"config:{prefix}use_sandwich_norm")
    f["norm"] = (", ".join(norm_bits), f"config:{prefix}{'norm_eps' if 'norm_eps' in text else 'rms_norm_eps'}")
    rp = text.get("rope_parameters")
    if isinstance(rp, dict):
        f["rope"] = ({k: {kk: vv for kk, vv in v.items()} for k, v in rp.items()} if all(isinstance(v, dict) for v in rp.values()) else rp,
                     f"config:{prefix}rope_parameters")
    elif text.get("rope_theta"):
        f["rope"] = ({"rope_theta": text["rope_theta"]}, f"config:{prefix}rope_theta")

    q = text.get("quantization_config") or config.get("quantization_config")
    if q:
        fmt = q.get("format") or q.get("quant_method")
        bits = None
        for g in (q.get("config_groups") or {}).values():
            bits = (g.get("weights") or {}).get("num_bits", bits)
        f["released_weights_format"] = (f"{fmt}" + (f" ({bits}-bit)" if bits else ""), f"config:{prefix}quantization_config")
        f["_quant_bits"] = (bits, f"config:{prefix}quantization_config")
    else:
        f["released_weights_format"] = (str(config.get("dtype") or config.get("torch_dtype") or text.get("dtype")), "config:dtype")

    vc = config.get("vision_config")
    if vc:
        f["vision_encoder_layers"] = (vc.get("vt_num_hidden_layers") or vc.get("num_hidden_layers"), "config:vision_config")
        f["vision_hidden_size"] = (vc.get("vt_hidden_size") or vc.get("hidden_size"), "config:vision_config")
    return f
