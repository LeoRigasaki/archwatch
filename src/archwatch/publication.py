"""Validated inputs for public architecture views. Raw facts stay diagnostic."""

from collections import Counter

from . import calc
from .verify import TO_WEIGHT_KIND


class PublicationError(ValueError):
    """A record lacks the approved inputs required by a public renderer."""


def render_values(record: dict) -> dict:
    values = dict(record["published"])
    checks = {check["field"]: check for check in record["checks"]}
    pattern = checks.get("layer_pattern_per_layer", {})
    if pattern.get("status") != "verified" or not pattern.get("published"):
        raise PublicationError("Layer pattern is not verified; architecture rendering is withheld")
    for field in values:
        check = checks.get(field, {})
        if not check.get("published") or check.get("status") not in ("verified", "single source"):
            raise PublicationError(f"Published input {field!r} has no passing verifier decision")
    required = {
        "layer_types",
        "layer_type_counts",
        "num_layers",
        "hidden_size",
        "vocab_size",
        "params_total",
        "context_length",
        "num_attention_heads",
        "head_dim",
        "decoder_type",
    }
    kinds = set(values.get("layer_types", []))
    if kinds & {"full_attention", "sliding_attention"}:
        required.add("num_kv_heads")
    if "sliding_attention" in kinds:
        required.add("sliding_window")
    if "short_conv" in kinds:
        required.add("short_conv_kernel")
    if kinds & {"gated_dsa_mla", "full_attention_mla"}:
        required.update(("mla_kv_lora_rank", "mla_qk_rope_head_dim"))
    if "gated_dsa_mla" in kinds:
        required.update(("dsa_index_num_heads", "dsa_index_head_dim", "dsa_index_topk"))
    if "kda_linear" in kinds:
        required.update(("linear_attn_num_heads", "linear_attn_head_dim", "linear_attn_conv_kernel"))
    if "moe_num_experts" in values:
        required.update(("moe_active_experts", "moe_shared_experts", "moe_expert_intermediate"))
        if values.get("dense_layers", 0):
            required.add("dense_ffn_intermediate")
    else:
        required.add("ffn_intermediate")
    missing = sorted(field for field in required if values.get(field) is None)
    if missing:
        raise PublicationError("Required published inputs are unavailable: " + ", ".join(missing))
    if len(values["layer_types"]) != values["num_layers"]:
        raise PublicationError("Published layer count and pattern disagree")
    if dict(Counter(values["layer_types"])) != values["layer_type_counts"]:
        raise PublicationError("Published layer composition and pattern disagree")
    projected = [TO_WEIGHT_KIND.get(kind, kind) for kind in values["layer_types"]]
    if projected != values.get("layer_pattern_per_layer"):
        raise PublicationError("Published layer pattern differs from the verified weight-kind pattern")
    return values


def render_cache(values: dict) -> dict:
    memory = calc.cache_model(values)
    if any(row["grows_per_token_bytes"] is None or row["constant_bytes"] is None for row in memory["breakdown"]):
        raise PublicationError("Cache calculation contains an unsupported layer type")
    return memory
