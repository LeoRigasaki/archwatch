"""Cross-check every field across independent sources and assign confidence.

Sources: config.json, weight shapes (safetensors headers), the repo's own code, and the model card.
  verified       >= 2 independent sources agree, none disagree          -> confidence 1.0
  single source  only one source states it                               -> confidence 0.7
  conflict       sources disagree beyond rounding tolerance              -> confidence 0.3, sent to review
Only fields with confidence >= 0.7 are published; conflicts are listed for a human.
"""

from __future__ import annotations

PUBLISH_THRESHOLD = 0.7

CARD_LAYER_NAMES = {
    "kda": "kda_linear",
    "gated dsa": "gated_dsa_mla",
    "gated mla": "full_attention_mla",
    "mla": "full_attention_mla",
    "gated short convolutions": "short_conv",
    "short convolutions": "short_conv",
    "sliding-window attention": "sliding_attention",
    "global attention": "full_attention",
    "full attention": "full_attention",
}


def _src(evidence: str) -> str:
    return evidence.split(":", 1)[0]


def _agree(a, b, tol: float, k_suffix: bool) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if k_suffix:  # "160K" may mean 160x1000 or 160x1024
            return abs(a - b) <= max(tol, 0.03 * max(abs(a), abs(b)))
        return abs(a - b) <= tol
    return a == b


def check(field: str, cands: list[dict]) -> dict:
    cands = [c for c in cands if c.get("value") is not None]
    if not cands:
        return {"field": field, "status": "missing", "confidence": 0.0, "candidates": []}
    ref = next((c for c in cands if _src(c["evidence"]) in ("config", "weights", "code", "derived")), cands[0])
    conflicts = [c for c in cands if not _agree(ref["value"], c["value"], c.get("tol", 0), c.get("k_suffix", False))]
    sources = {_src(c["evidence"]) for c in cands}
    if conflicts:
        status, conf = "conflict", 0.3
    elif len(sources) >= 2:
        status, conf = "verified", 1.0
    else:
        status, conf = "single source", 0.7
    return {
        "field": field,
        "value": ref["value"],
        "status": status,
        "confidence": conf,
        "published": conf >= PUBLISH_THRESHOLD,
        "candidates": [
            {"value": c["value"], "evidence": c["evidence"]} | ({"raw": c["raw"]} if "raw" in c else {}) for c in cands
        ],
    }


TO_WEIGHT_KIND = {
    "compressed_attention": "compressed_attention",
    "gated_delta_net": "linear_delta",
    "qwen_sparse_attention": "qsa",
    "indexed_mla": "dsa_mla",
    "mamba2": "mamba2",
    "mlp": "mlp",
    "kda_linear": "linear_delta",
    "gated_dsa_mla": "dsa_mla",
    "full_attention_mla": "mla",
    "short_conv": "short_conv",
    "full_attention": "attention",
    "sliding_attention": "attention",
}


def verify(
    facts: dict,
    wcount: dict,
    wshape: dict,
    active: dict | None,
    code: dict,
    card: dict,
    wkinds: list[str] | None = None,
) -> list[dict]:
    V = {k: v for k, v in facts.items()}
    cf = code.get("facts", {})

    def c(key):
        return {"value": V[key][0], "evidence": V[key][1]} if key in V and V[key][0] is not None else {}

    def k(key):
        e = card.get(key)
        if not e or "value" not in e:
            return {}
        return {
            "value": e["value"],
            "evidence": f"card:{key}",
            "tol": e.get("tolerance", 0),
            "k_suffix": e.get("k_suffix", False),
            "raw": e["raw"],
        }

    def w(val, what):
        return {"value": val, "evidence": f"weights:{what}"} if val is not None else {}

    def cd(key):
        return {"value": cf[key][0], "evidence": cf[key][1]} if key in cf else {}

    out = []
    li = wcount.get("layer_indices_in_weights")
    out.append(
        check("num_layers", [c("num_layers"), k("num_layers"), w(li[1] + 1 if li else None, "highest layer index + 1")])
    )
    out.append(
        check("hidden_size", [c("hidden_size"), k("hidden_size"), w(wshape.get("hidden_size"), "embed_tokens columns")])
    )
    out.append(
        check("vocab_size", [c("vocab_size"), k("vocab_size"), w(wshape.get("vocab_rows"), "embed_tokens rows")])
    )
    out.append(check("context_length", [c("context_length"), k("context_length")]))
    out.append(check("num_attention_heads", [c("num_attention_heads"), k("num_attention_heads")]))
    out.append(check("num_kv_heads", [c("num_kv_heads")]))
    out.append(check("head_dim", [c("head_dim")]))
    out.append(check("attention_type", [c("attention_type")]))
    out.append(check("ffn_gated", [w(wshape.get("ffn_gated"), "FFN gate/up/down tensor layout")]))
    out.append(
        check("sandwich_norm", [c("sandwich_norm"), w(wshape.get("sandwich_norm"), "post-mixer/post-FFN norm tensors")])
    )
    if "mamba_num_heads" in V:
        channels = (
            V["mamba_num_heads"][0] * V["mamba_head_dim"][0] + 2 * V["mamba_groups"][0] * V["mamba_state_size"][0]
        )
        out.append(
            check(
                "mamba_conv_channels",
                [
                    {"value": channels, "evidence": "derived:mamba heads*head_dim + 2*groups*state_size"},
                    w(wshape.get("mamba_conv_channels"), "mixer.conv1d channels"),
                ],
            )
        )

    if "moe_num_experts" in V:
        epl = wcount.get("experts_per_moe_layer") or []
        out.append(
            check(
                "moe_num_experts",
                [
                    c("moe_num_experts"),
                    k("moe_num_experts"),
                    w(epl[0] if len(epl) == 1 else None, "expert tensors per MoE layer"),
                ],
            )
        )
        out.append(check("moe_active_experts", [c("moe_active_experts"), k("moe_active_experts")]))
        out.append(
            check(
                "moe_shared_experts",
                [
                    c("moe_shared_experts"),
                    k("moe_shared_experts"),
                    w(
                        wshape.get("shared_experts") if not c("moe_shared_experts") else None,
                        "one shared expert module (config count absent; fused widths do not independently prove logical count)",
                    ),
                ],
            )
        )
        if wshape.get("shared_expert_intermediate") is not None:
            out.append(
                check(
                    "moe_shared_expert_intermediate",
                    [w(wshape["shared_expert_intermediate"], "shared expert gate rows")],
                )
            )
        out.append(
            check(
                "moe_expert_intermediate",
                [
                    c("moe_expert_intermediate"),
                    k("moe_expert_intermediate"),
                    w(wshape.get("expert_intermediate"), "experts.0.w1 rows"),
                ],
            )
        )
        if "moe_latent_dim" in V:
            out.append(
                check(
                    "moe_latent_dim",
                    [
                        c("moe_latent_dim"),
                        k("moe_latent_dim"),
                        w(wshape.get("latent_dim"), "routed_expert_up_proj shape"),
                        w(wshape.get("expert_input_dim"), "experts.0.w1 input width"),
                    ],
                )
            )
        out.append(
            check(
                "dense_layers",
                [
                    c("dense_layers"),
                    k("dense_layers"),
                    w(V["num_layers"][0] - wcount["moe_layers"], "layers without expert tensors"),
                ],
            )
        )
    else:
        out.append(
            check(
                "ffn_intermediate", [c("ffn_intermediate"), w(wshape.get("ffn_intermediate"), "feed_forward.w1 rows")]
            )
        )

    for key in ("dsa_index_topk", "dsa_index_num_heads", "dsa_index_head_dim"):
        if key in V:
            out.append(check(key, [c(key), k(key)]))
    if "short_conv_kernel" in V:
        out.append(
            check(
                "short_conv_kernel", [c("short_conv_kernel"), w(wshape.get("conv_kernel"), "conv.conv.weight last dim")]
            )
        )
    if "sliding_window" in V:
        out.append(check("sliding_window", [c("sliding_window")]))

    # layer composition: config pattern vs card text
    comp = (card.get("layer_composition") or {}).get("parts") or {}
    mapped = {}
    for name, n in comp.items():
        key = next((v for kname, v in CARD_LAYER_NAMES.items() if kname in name.lower()), None)
        if key:
            mapped[key] = n
    cands = [c("layer_type_counts")]
    if mapped:
        cands.append({"value": mapped, "evidence": "card:layer_composition", "raw": card["layer_composition"]["raw"]})
    out.append(check("layer_type_counts", cands))

    # the per-layer pattern, read independently from which tensors each layer has
    if wkinds and "layer_types" in V:
        from_config = [TO_WEIGHT_KIND.get(t, t) for t in V["layer_types"][0]]
        out.append(
            check(
                "layer_pattern_per_layer",
                [
                    {"value": from_config, "evidence": V["layer_types"][1] + " (mapped to weight-level kinds)"},
                    {"value": wkinds, "evidence": "weights:tensor names per layer"},
                ],
            )
        )
    if "mla_kv_lora_rank" in V and wshape.get("mla_cached_width"):
        out.append(
            check(
                "mla_cached_width_per_token",
                [
                    {
                        "value": V["mla_kv_lora_rank"][0] + (V.get("mla_qk_rope_head_dim", (0,))[0] or 0),
                        "evidence": "derived:kv_lora_rank + qk_rope_head_dim",
                    },
                    w(wshape["mla_cached_width"], "kv_a_proj_with_mqa output rows"),
                ],
            )
        )
    if "dsa_index_head_dim" in V and wshape.get("dsa_index_key_width"):
        out.append(
            check("dsa_index_key_width", [c("dsa_index_head_dim"), w(wshape["dsa_index_key_width"], "indexer.wk rows")])
        )
    if "activation" in V and "activation" in card:
        cv, raw = str(V["activation"][0]).lower(), card["activation"]["raw"]
        out.append(
            check(
                "activation",
                [
                    c("activation"),
                    {
                        "value": V["activation"][0] if cv in raw.lower() else raw,
                        "evidence": "card:activation",
                        "raw": raw,
                    },
                ],
            )
        )
        if out[-1]["published"]:
            out.append(check("activation_card_name", [{"value": raw, "evidence": "card:activation"}]))

    if "layer_index_base_in_config" in V or "layer_index_base_in_config" in cf:
        out.append(
            check("layer_index_base_in_config", [c("layer_index_base_in_config"), cd("layer_index_base_in_config")])
        )
    out.append(check("qk_norm", [cd("qk_norm"), w(wshape.get("qk_norm"), "q_norm/k_norm tensors")]))
    out.append(
        check(
            "attention_output_gate",
            [
                c("attention_output_gate"),
                cd("attention_output_gate"),
                w(wshape.get("attention_output_gate_tensor"), "gate_proj/g_proj tensors"),
            ],
        )
    )

    out.append(check("params_total", [w(wcount["total"], "sum of tensor shapes"), k("params_total")]))
    if active:
        # labs count "active" differently; try each convention and say which one the card used
        conventions = [
            ("text_active_without_embeddings", "text model, input embedding excluded"),
            ("text_active", "text model, all non-expert weights included"),
            ("with_vision", "text + vision"),
        ]
        kc = k("params_active")
        pick = conventions[1]
        if kc:
            for key, label in conventions:
                if _agree(active[key], kc["value"], kc["tol"], False):
                    pick = (key, label)
                    break
        out.append(check("params_active", [w(active[pick[0]], f"total - unselected routed experts ({pick[1]})"), kc]))
    if "vision_params" in card:
        out.append(
            check(
                "vision_params",
                [
                    w(wcount["by_component"].get("vision"), "vision_tower tensors (projector excluded)"),
                    k("vision_params"),
                ],
            )
        )
    # Explicit rendering/calculator inputs get the same publication decision as
    # every other fact. Do not promote the raw facts dictionary in consumers.
    rendering_fields = (
        "model_type",
        "global_head_dim",
        "num_global_kv_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel",
        "ngram_layer_ids",
        "hyper_connection_multiplier",
        "num_mtp_layers",
        "shared_kv_layers",
        "generation_type",
        "canvas_length",
        "recurrent_loops",
        "recurrent_layers",
        "input_layers",
        "output_layers",
        "recurrent_kv_cache_mode",
        "layer_types",
        "decoder_type",
        "tied_embeddings",
        "rope",
        "sandwich_norm",
        "linear_attn_num_heads",
        "linear_attn_head_dim",
        "linear_attn_conv_kernel",
        "mla_kv_lora_rank",
        "mla_qk_rope_head_dim",
        "v_head_dim",
        "dense_ffn_intermediate",
        "moe_router",
        "vision_encoder_layers",
        "vision_hidden_size",
        "activation",
        "released_weights_format",
        "block_layout",
        "mamba_num_heads",
        "mamba_head_dim",
        "mamba_state_size",
        "mamba_conv_kernel",
        "mamba_groups",
        "mamba_activation",
    )
    checked = {r["field"] for r in out}
    for field in rendering_fields:
        if (field in V or field in cf) and field not in checked:
            out.append(check(field, [c(field), cd(field)]))
    if wcount:
        out.append(check("parameter_components", [w(wcount.get("by_component"), "parameter totals by tensor group")]))
        out.append(
            check(
                "weight_header_summary",
                [
                    w(
                        {
                            "tensor_count": wcount.get("tensor_count"),
                            "skipped_scale_tensors": wcount.get("skipped_scale_tensors"),
                            "skipped_buffer_tensors": wcount.get("skipped_buffer_tensors", 0),
                        },
                        "tensor header inventory",
                    )
                ],
            )
        )
    return [r for r in out if r["status"] != "missing"]
