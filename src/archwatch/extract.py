"""Deterministic config.json -> architecture facts. No LLM.

Every fact is returned as (value, evidence) where evidence names the config key it came from,
so the verifier and the published page can show where each number was read.
"""

from __future__ import annotations

import re
from typing import Any

from .concept_extract import config_facts

Fact = tuple[Any, str]


class _Config(dict):
    """Track actual value reads, including nested dictionaries, without mutating input."""

    def __init__(self, value, consumed, prefix=""):
        self.consumed, self.prefix = consumed, prefix

        def wrap(value, path):
            if isinstance(value, dict):
                return _Config(value, consumed, path + ".")
            if isinstance(value, list):
                return [wrap(item, f"{path}[{i}]") for i, item in enumerate(value)]
            return value

        super().__init__({k: wrap(v, prefix + k) for k, v in value.items()})

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.consumed.add(self.prefix + key)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return ((key, self[key]) for key in self)

    def values(self):
        return (self[key] for key in self)


BULK_LEAVES = ("quantization_config.quantized_layers", "quantization_config.ignore")


def _config_paths(value, prefix=""):
    for key, child in value.items():
        path = prefix + key
        if path.endswith(BULK_LEAVES):
            yield path  # a per-tensor quantization map: one setting, read as a whole
        elif isinstance(child, dict) and child:
            yield from _config_paths(child, path + ".")
        elif isinstance(child, list) and any(isinstance(item, dict) for item in child):
            for i, item in enumerate(child):
                if isinstance(item, dict):
                    yield from _config_paths(item, f"{path}[{i}].")
        else:
            yield path


def extract_with_usage(config):
    consumed = set()

    def plain(value):
        # Forwarded structured facts (e.g. RoPE scaling) consume their leaves,
        # and must not leak tracking wrappers into safe YAML serialization.
        if isinstance(value, dict):
            return {key: plain(child) for key, child in value.items()}
        if isinstance(value, list):
            return [plain(child) for child in value]
        if isinstance(value, tuple):
            return tuple(plain(child) for child in value)
        return value

    facts = plain(_extract(_Config(config, consumed)))
    paths = set(_config_paths(config))
    return facts, {"consumed": sorted(paths & consumed), "unconsumed": sorted(paths - consumed)}


def extract(config: dict) -> dict[str, Fact]:
    return extract_with_usage(config)[0]


def _first(cfg: dict, keys: list[str], prefix: str) -> Fact | None:
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k], f"config:{prefix}{k}"
    return None


# Nemotron-H blocks hold exactly one operator each: Mamba-2, attention, a dense MLP or an MoE layer.
NEMOTRON_PATTERN = {"M": "mamba2", "*": "full_attention", "-": "mlp", "E": "moe"}
NEMOTRON_BLOCKS = {"mamba": "mamba2", "attention": "full_attention", "mlp": "mlp", "moe": "moe"}


def _num_layers(text: dict, prefix: str) -> Fact:
    """Decoder depth under the config's own naming: HF, GPT-2 legacy, Mistral-native, or a Nemotron block list."""
    found = _first(text, ["num_hidden_layers", "n_layer", "n_layers"], prefix)
    if found:
        return found
    if text.get("model_type") == "nemotron_h" and isinstance(text.get("layers_block_type"), list):
        return len(text["layers_block_type"]), f"derived:len({prefix}layers_block_type)"
    raise KeyError("num_hidden_layers")


def _layer_types(text: dict, prefix: str, n_layers: int) -> tuple[list[str], str, int | None]:
    """Return a per-layer mixer list (0-based), the evidence, and the index base the config used."""
    model_type = str(text.get("model_type") or "")
    if model_type.startswith("deepseek_v4"):
        return ["compressed_attention"] * n_layers, f"config:{prefix}model_type/compress_ratios", None
    if text.get("model_type") == "nemotron_h" and text.get("hybrid_override_pattern"):
        pattern = text["hybrid_override_pattern"]
        if len(pattern) != n_layers or set(pattern) - set(NEMOTRON_PATTERN):
            raise ValueError("Invalid Nemotron-H block pattern")
        return [NEMOTRON_PATTERN[p] for p in pattern], f"config:{prefix}hybrid_override_pattern", None
    if text.get("model_type") == "nemotron_h" and isinstance(text.get("layers_block_type"), list):
        blocks = text["layers_block_type"]
        if len(blocks) != n_layers or set(blocks) - set(NEMOTRON_BLOCKS):
            raise ValueError("Invalid Nemotron-H layers_block_type")
        return [NEMOTRON_BLOCKS[b] for b in blocks], f"config:{prefix}layers_block_type", None
    if isinstance(text.get("layer_types"), list):
        mapping = {
            "conv": "short_conv",
            "full_attention": "full_attention",
            "sliding_attention": "sliding_attention",
            "linear_attention": "linear_attention",
        }
        if model_type in ("qwen4_exp_text", "glm5_next_text"):
            mapping.update(
                linear_attention="gated_delta_net" if model_type == "qwen4_exp_text" else "kda_linear",
                full_attention="qwen_sparse_attention" if model_type == "qwen4_exp_text" else "full_attention",
                deepseek_sparse_attention="indexed_mla",
            )
        if model_type == "mistral_large4_text":
            mapping["indexed_attention"] = "indexed_mla"
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
        return (
            ["full_attention" if (i + 1) % pattern == 0 else "sliding_attention" for i in range(n_layers)],
            f"config:{prefix}sliding_window_pattern",
            None,
        )
    if window and text.get("model_type") == "mistral":
        return ["sliding_attention"] * n_layers, f"config:{prefix}sliding_window (Mistral all-layer rule)", None
    kind = "full_attention_mla" if text.get("kv_lora_rank") else "full_attention"
    return [kind] * n_layers, "config:(no per-layer pattern; all layers assumed identical)", None


def _extract(config: dict) -> dict[str, Fact]:
    text, prefix = config, ""
    for nested in ("text_config", "llm_config"):
        if isinstance(config.get(nested), dict):
            text, prefix = config[nested], nested + "."
            break
    f: dict[str, Fact] = {}

    f["architecture_class"] = ((config.get("architectures") or ["?"])[0], "config:architectures")
    f["model_type"] = (config.get("model_type"), "config:model_type")

    f["num_layers"] = _num_layers(text, prefix)
    n_layers = f["num_layers"][0]
    f["hidden_size"] = _first(text, ["hidden_size", "n_embd", "dim"], prefix) or (None, f"config:{prefix}hidden_size")
    hidden = f["hidden_size"][0]
    # Structured mechanisms remain source facts even when the graph/cache
    # renderer does not yet qualify their topology or sharing semantics.
    for field, source in (
        ("global_head_dim", "global_head_dim"),
        ("num_global_kv_heads", "num_global_key_value_heads"),
        ("linear_num_key_heads", "linear_num_key_heads"),
        ("linear_num_value_heads", "linear_num_value_heads"),
        ("linear_key_head_dim", "linear_key_head_dim"),
        ("linear_value_head_dim", "linear_value_head_dim"),
        ("linear_conv_kernel", "linear_conv_kernel_dim"),
        ("ngram_layer_ids", "engram_layer_ids"),
        ("recurrent_loops", "n_recurrent_loops"),
        ("recurrent_layers", "n_recurrent_layers"),
        ("input_layers", "n_input_layers"),
        ("output_layers", "n_output_layers"),
        ("recurrent_kv_cache_mode", "recurrent_kv_cache_mode"),
    ):
        if text.get(source) is not None:
            f[field] = (text[source], f"config:{prefix}{source}")
    hc = _first(text, ["hc_mult", "hc_count"], prefix)
    if hc:
        f["hyper_connection_multiplier"] = hc
    mtp = _first(text, ["num_nextn_predict_layers", "mtp_num_hidden_layers"], prefix)
    if mtp:
        f["num_mtp_layers"] = mtp
    # Nemotron-H MTP: the blocks of each prediction module (counted as MTP, not as decoder layers)
    if isinstance(text.get("mtp_layers_block_type"), list):
        f["mtp_block_types"] = (
            [NEMOTRON_BLOCKS.get(b, b) for b in text["mtp_layers_block_type"]],
            f"config:{prefix}mtp_layers_block_type",
        )
    elif text.get("mtp_hybrid_override_pattern"):
        f["mtp_block_types"] = (
            [NEMOTRON_PATTERN.get(b, b) for b in text["mtp_hybrid_override_pattern"]],
            f"config:{prefix}mtp_hybrid_override_pattern",
        )
    if text.get("kv_source_layers"):
        f["shared_kv_layers"] = (
            sum(i != source for i, source in enumerate(text["kv_source_layers"])),
            f"derived:{prefix}kv_source_layers differs from layer index",
        )
    if config.get("model_type") == "diffusion_gemma":
        f["generation_type"] = ("block_diffusion", "config:model_type (DiffusionGemma architecture)")
        f["canvas_length"] = (config.get("canvas_length"), "config:canvas_length")
    if text.get("model_type") == "dory":
        f["generation_type"] = ("latent_recurrent", f"config:{prefix}n_recurrent_loops/recurrent_kv_cache_mode")
    if text.get("model_type") == "nemotron_h":
        pattern_key = "hybrid_override_pattern" if text.get("hybrid_override_pattern") else "layers_block_type"
        f["block_layout"] = ("standalone", f"config:{prefix}{pattern_key} (one operator per block)")
        for key, source in (
            ("mamba_num_heads", "mamba_num_heads"),
            ("mamba_head_dim", "mamba_head_dim"),
            ("mamba_state_size", "ssm_state_size"),
            ("mamba_conv_kernel", "conv_kernel"),
            ("mamba_groups", "n_groups"),
            ("mamba_activation", "mamba_hidden_act"),
        ):
            f[key] = (text.get(source), f"config:{prefix}{source}")
    if text.get("model_type") == "phi":
        f["block_layout"] = (
            "parallel",
            "derived:PhiDecoderLayer parallel attention/MLP residual rule (Transformers v4.37.0)",
        )
    f["vocab_size"] = (text.get("vocab_size"), f"config:{prefix}vocab_size")
    ctx = _first(text, ["max_position_embeddings", "max_seq_len", "seq_length", "n_positions"], prefix)
    if ctx:
        f["context_length"] = ctx
    tie = config.get("tie_word_embeddings") if "tie_word_embeddings" in config else text.get("tie_word_embeddings")
    if tie is not None:  # an absent key is not evidence either way; the weights decide (see verify)
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
    heads_fact = _first(text, ["num_attention_heads", "n_head", "n_heads"], prefix)
    heads = heads_fact[0] if heads_fact else None
    kv_fact = _first(text, ["num_key_value_heads", "n_kv_heads"], prefix)
    kv = kv_fact[0] if kv_fact else heads
    f["num_attention_heads"] = heads_fact or (None, f"config:{prefix}num_attention_heads")
    if kv_fact:
        f["num_kv_heads"] = kv_fact
    elif heads_fact and not heads_fact[1].endswith("num_attention_heads"):
        f["num_kv_heads"] = (kv, f"derived:{heads_fact[1].split(':', 1)[1]} (no separate key-value head key)")
    else:
        f["num_kv_heads"] = (kv, f"config:{prefix}num_key_value_heads")
    if text.get("kv_lora_rank") and text.get("qk_nope_head_dim") is not None:
        # MLA: a query/key head is the no-RoPE part plus the RoPE part; values have their own width
        f["head_dim"] = (
            text["qk_nope_head_dim"] + (text.get("qk_rope_head_dim") or 0),
            f"derived:{prefix}qk_nope_head_dim + qk_rope_head_dim",
        )
        f["v_head_dim"] = (text.get("v_head_dim"), f"config:{prefix}v_head_dim")
    else:
        f["head_dim"] = _first(text, ["head_dim", "attention_head_dim"], prefix) or (
            hidden // heads if heads and hidden else None,
            "derived:hidden_size/num_attention_heads",
        )
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
        f["linear_attn_conv_kernel"] = (
            lac.get("short_conv_kernel_size"),
            f"config:{prefix}linear_attn_config.short_conv_kernel_size",
        )
    if text.get("conv_L_cache"):
        f["short_conv_kernel"] = (text["conv_L_cache"], f"config:{prefix}conv_L_cache")

    # feed-forward / MoE
    n_exp = _first(text, ["num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"], prefix)
    if n_exp:
        f["decoder_type"] = ("MoE", n_exp[1])
        f["moe_num_experts"] = n_exp
        k = _first(text, ["num_experts_per_token", "num_experts_per_tok", "moe_top_k", "experts_per_token"], prefix)
        if k:
            f["moe_active_experts"] = k
        sh = _first(text, ["num_shared_experts", "n_shared_experts"], prefix)
        if sh:
            f["moe_shared_experts"] = sh
        # Private flag: does the config name shared experts at all? (Zero is only published with weight evidence.)
        f["_config_names_shared_experts"] = (
            any("shared_expert" in key or key in ("num_shared_experts", "n_shared_experts") for key in list(text)),
            f"config:{prefix}* keys",
        )
        mi = _first(text, ["moe_intermediate_size", "expert_intermediate_size"], prefix)
        if mi is None and text.get("model_type") == "llama4_text":
            mi = _first(text, ["intermediate_size"], prefix)
        if mi:
            f["moe_expert_intermediate"] = mi
        lat = _first(text, ["routed_expert_hidden_size", "moe_latent_size"], prefix)
        if lat:
            f["moe_latent_dim"] = lat
        fk = _first(text, ["first_k_dense_replace"], prefix)
        f["dense_layers"] = (fk[0] if fk else 0, fk[1] if fk else "default:0")
        if text.get("model_type") == "nemotron_h":
            f["dense_layers"] = (types.count("mlp"), ev + " (MLP blocks)")
        f["dense_ffn_intermediate"] = (text.get("intermediate_size"), f"config:{prefix}intermediate_size")
        if text.get("model_type") == "llama4_text":
            step = text["interleave_moe_layer_step"]
            f["dense_layers"] = (n_layers - n_layers // step, f"derived:{prefix}interleave_moe_layer_step")
            f["dense_ffn_intermediate"] = (text["intermediate_size_mlp"], f"config:{prefix}intermediate_size_mlp")
        top = _first(text, ["top_k_experts"], prefix)
        if k is None and top:
            f["moe_active_experts"] = top
        if text.get("moe_router_activation_func"):
            f["moe_router"] = (text["moe_router_activation_func"], f"config:{prefix}moe_router_activation_func")
    else:
        mixers = set(types)
        hybrid = len(mixers) > 1 and any(
            m in mixers for m in ("short_conv", "kda_linear", "linear_attention", "mamba2")
        )
        f["decoder_type"] = ("Hybrid (dense)" if hybrid else "Dense", ev if hybrid else "derived:no expert keys")
        width_key = next(
            (key for key in ("block_ff_dim", "intermediate_size", "n_inner") if key in text), "intermediate_size"
        )
        width = text.get(width_key)
        f["ffn_intermediate"] = (width, f"config:{prefix}{width_key}")
        if width is None and text.get("model_type") == "gpt2" and hidden:
            # GPT2MLP: inner_dim = n_inner if n_inner is not None else 4 * hidden_size
            f["ffn_intermediate"] = (4 * hidden, f"derived:4 * {prefix}n_embd (GPT-2 default when n_inner is null)")
        if width is not None and text.get("block_auto_adjust_ff_dim") is True:
            # Lfm2MLP applies these operations only inside the auto-adjust branch.
            multiplier = text.get("block_ffn_dim_multiplier", 1.0)
            multiple = text.get("block_multiple_of", 256)
            effective = int(2 * width / 3)
            if multiplier is not None:
                effective = int(multiplier * effective)
                effective = multiple * ((effective + multiple - 1) // multiple)
            f["ffn_intermediate_raw"] = (width, f"config:{prefix}{width_key}")
            f["ffn_intermediate"] = (
                effective,
                f"derived:Lfm2MLP({prefix}{width_key}={width}, block_auto_adjust_ff_dim=true, "
                f"block_ffn_dim_multiplier={multiplier}, block_multiple_of={multiple})",
            )
    act = _first(
        text, ["hidden_act", "hidden_activation", "mlp_hidden_act", "activation", "activation_function"], prefix
    )
    if act:
        f["activation"] = act

    # norms and positions
    if text.get("attn_res_block_size") is not None:
        f["attn_res_block_size"] = (text["attn_res_block_size"], f"config:{prefix}attn_res_block_size")
    if text.get("use_sandwich_norm") is not None:
        f["sandwich_norm"] = (bool(text["use_sandwich_norm"]), f"config:{prefix}use_sandwich_norm")
    rp = text.get("rope_parameters")
    if isinstance(rp, dict):
        f["rope"] = (
            {k: {kk: vv for kk, vv in v.items()} for k, v in rp.items()}
            if all(isinstance(v, dict) for v in rp.values())
            else rp,
            f"config:{prefix}rope_parameters",
        )
    elif text.get("rope_theta"):
        rope = {"rope_theta": text["rope_theta"]}
        if text.get("partial_rotary_factor") is not None:
            rope["partial_rotary_factor"] = text["partial_rotary_factor"]
        if text.get("rope_scaling"):
            rope["scaling"] = text["rope_scaling"]
        if text.get("rope_local_base_freq"):
            rope = {"full_attention": rope, "sliding_attention": {"rope_theta": text["rope_local_base_freq"]}}
        f["rope"] = (
            rope,
            f"config:{prefix}rope_theta/rope_scaling/partial_rotary_factor/rope_local_base_freq (present keys)",
        )

    q = text.get("quantization_config") or config.get("quantization_config")
    q_where = prefix if text.get("quantization_config") else ""
    if q:
        fmt = q.get("format") or q.get("quant_method")
        bits = None
        for g in (q.get("config_groups") or {}).values():
            bits = (g.get("weights") or {}).get("num_bits", bits)
        f["released_weights_format"] = (
            f"{fmt}" + (f" ({bits}-bit)" if bits else ""),
            f"config:{q_where}quantization_config",
        )
        f["_quant_bits"] = (bits, f"config:{q_where}quantization_config")
    else:
        f["released_weights_format"] = (
            str(config.get("dtype") or config.get("torch_dtype") or text.get("dtype")),
            "config:dtype",
        )

    vc = config.get("vision_config")
    if vc:
        f["vision_encoder_layers"] = (
            vc.get("vt_num_hidden_layers") or vc.get("num_hidden_layers"),
            "config:vision_config",
        )
        f["vision_hidden_size"] = (vc.get("vt_hidden_size") or vc.get("hidden_size"), "config:vision_config")
    f.update(config_facts(config))
    return f


# Mistral-native params.json -> the HF-style keys the extractor reads. Each entry: (HF key, params.json path).
MISTRAL_PARAMS = (
    ("hidden_size", "dim"),
    ("num_hidden_layers", "n_layers"),
    ("num_attention_heads", "n_heads"),
    ("num_key_value_heads", "n_kv_heads"),
    ("intermediate_size", "hidden_dim"),
    ("vocab_size", "vocab_size"),
    ("tie_word_embeddings", "tied_embeddings"),
    ("rope_theta", "rope_theta"),
    ("rms_norm_eps", "norm_eps"),
    ("q_lora_rank", "q_lora_rank"),
    ("kv_lora_rank", "kv_lora_rank"),
    ("qk_rope_head_dim", "qk_rope_head_dim"),
    ("qk_nope_head_dim", "qk_nope_head_dim"),
    ("v_head_dim", "v_head_dim"),
    ("head_dim", "head_dim"),
    ("n_routed_experts", "moe.num_experts"),
    ("num_experts_per_tok", "moe.num_experts_per_tok"),
    ("first_k_dense_replace", "moe.first_k_dense_replace"),
    ("moe_intermediate_size", "moe.expert_hidden_dim"),
    ("n_shared_experts", "moe.num_shared_experts"),
    ("rope_scaling", "yarn"),
    ("quantization_config", "quantization_config"),
)


def _dig(value, path):
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def extract_mistral_params(params: dict):
    """Facts from a Mistral-native params.json; every evidence string names params.json and the native key."""
    cfg, origin = {}, {}
    for hf_key, native in MISTRAL_PARAMS:
        value = _dig(params, native)
        if value is not None:
            cfg[hf_key], origin[hf_key] = value, native
    # Context: max_seq_len is the served length; max_position_embeddings is the RoPE table it can address.
    for native in ("max_seq_len", "max_position_embeddings"):
        if params.get(native) is not None:
            cfg["max_position_embeddings"], origin["max_position_embeddings"] = params[native], native
            break
    vision = params.get("vision_encoder")
    if isinstance(vision, dict):
        cfg["vision_config"] = {k: vision[k] for k in ("num_hidden_layers", "hidden_size") if k in vision}
        origin.update({f"vision_config.{k}": f"vision_encoder.{k}" for k in cfg["vision_config"]})
    facts, coverage = extract_with_usage(cfg)

    def native(evidence):
        if not isinstance(evidence, str):
            return evidence
        for hf_key in sorted(origin, key=len, reverse=True):
            evidence = re.sub(rf"config:{re.escape(hf_key)}\b", f"params.json:{origin[hf_key]}", evidence)
            evidence = re.sub(rf"(derived:[^ ]*?)\b{re.escape(hf_key)}\b", rf"\g<1>{origin[hf_key]}", evidence)
        evidence = evidence.replace("config:vision_config", "params.json:vision_encoder")
        evidence = evidence.replace("config:architectures", "params.json:(no architectures key)")
        evidence = evidence.replace("config:model_type", "params.json:(no model_type key)")
        return evidence.replace("config:", "params.json:")  # anything else was read from the same file

    facts = {k: (v[0], native(v[1])) for k, v in facts.items()}
    consumed_native = {origin.get(p.split(".")[0], p) if p.split(".")[0] in origin else p for p in coverage["consumed"]}
    consumed_native |= {origin[p] for p in coverage["consumed"] if p in origin}
    every = set(_config_paths(params))
    used = {p for p in every if any(p == c or p.startswith(c + ".") for c in consumed_native)}
    return facts, {"consumed": sorted(used), "unconsumed": sorted(every - used), "source": "params.json"}
