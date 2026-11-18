"""One model in, one verified record out: fetch -> extract -> weights -> code -> card -> calc -> verify."""

from __future__ import annotations

import datetime as dt

from . import calc, card, codescan, extract, hub, verify, weights

DIFF_FIELDS = [
    "num_layers",
    "hidden_size",
    "layer_type_counts",
    "attention_type",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
    "context_length",
    "vocab_size",
    "moe_num_experts",
    "moe_active_experts",
    "moe_shared_experts",
    "moe_expert_intermediate",
    "moe_latent_dim",
    "dense_layers",
    "dsa_index_topk",
    "sliding_window",
    "short_conv_kernel",
    "activation",
    "norm",
    "ffn_intermediate",
    "released_weights_format",
    "params_total",
    "params_active",
    "kv_cache_per_token_bytes",
    "model_type",
    "ffn_gated",
    "generation_type",
    "canvas_length",
    "hyper_connection_multiplier",
    "global_head_dim",
    "num_global_kv_heads",
    "shared_kv_layers",
    "ngram_layer_ids",
    "num_mtp_layers",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel",
    "recurrent_loops",
    "recurrent_layers",
    "input_layers",
    "output_layers",
    "recurrent_kv_cache_mode",
    "moe_shared_expert_intermediate",
]


def native_expert_bits_of(cfg: dict) -> int | None:
    """Native FP4 expert storage, flagged in quantization_config or on the (text) model config itself."""
    quantization = cfg.get("quantization_config") or {}
    text = cfg.get("text_config") or {}
    dtype = quantization.get("expert_dtype") or cfg.get("expert_dtype") or text.get("expert_dtype")
    return 4 if dtype == "fp4" else None


def analyze(repo: str, with_weights: bool = True, *, info: dict | None = None) -> dict:
    # Refresh once per analysis; every subsequent file/header read uses this snapshot.
    info = info or hub.repo_info(repo, refresh=True)
    cfg = hub.get_json(repo, "config.json", revision=info["sha"])
    if not isinstance(cfg, dict):
        raise ValueError("No readable root config.json; this source format requires a separate adapter")
    facts, config_coverage = extract.extract_with_usage(cfg)
    bits = facts.get("_quant_bits", (None, ""))[0]
    native_expert_bits = native_expert_bits_of(cfg)
    num_layers = facts["num_layers"][0]

    py = {
        f: hub.get_text(repo, f, revision=info["sha"])
        for f in sorted(info["files"])
        if f.endswith(".py") and (f.startswith("modeling_") or f.startswith("configuration_"))
    }
    code = codescan.scan({k: v for k, v in py.items() if v})

    readme = hub.get_text(repo, "README.md", revision=info["sha"]) or ""
    claims = card.parse_card(readme)

    wcount = wshape = active = wkinds = blocks = None
    if with_weights:
        tensors = hub.safetensors_headers(repo, info=info)
        wcount = weights.count(tensors, bits, native_expert_bits=native_expert_bits, num_layers=num_layers)
        wshape = weights.shape_facts(tensors, bits, native_expert_bits=native_expert_bits, num_layers=num_layers)
        wkinds = weights.layer_kinds(tensors, num_layers=num_layers)
        blocks = weights.block_shapes(
            tensors, facts.get("layer_types", (None,))[0], num_layers=num_layers, quant_bits=bits
        )
        active = weights.active_params(
            wcount, facts.get("moe_num_experts", (None,))[0], facts.get("moe_active_experts", (None,))[0]
        )

    values = {k: v[0] for k, v in facts.items()}
    mem = calc.cache_model(values)
    checks = verify.verify(facts, wcount or {}, wshape or {}, active, code, claims, wkinds, blocks)

    published = {r["field"]: r["value"] for r in checks if r["published"]}
    review = [r for r in checks if not r["published"]]
    # things the pipeline could not check at all are reported too, so nobody mistakes silence for a pass
    notes = []
    if not code["facts"]["modeling_code_in_repo"][0]:
        notes.append(
            "No modeling_*.py in the repo: mechanism details (layer internals, residual scheme, "
            "activation) are read from config, card and tensor names only, not from code."
        )
    unknown_cfg = config_coverage["unconsumed"]
    if unknown_cfg:
        notes.append(
            "Config keys not consumed by the extractor (including metadata/defaults): " + ", ".join(unknown_cfg)
        )
    return {
        "id": repo,
        "source_commit": info["sha"],
        "hub_created_at": info["created_at"],
        "license": info.get("license_name") or info.get("license"),
        "hub_metadata": {k: info.get(k) for k in ("base_model", "base_model_relation", "tags")},
        "analyzed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "facts": {k: {"value": v[0], "evidence": v[1]} for k, v in facts.items() if not k.startswith("_")},
        "config_coverage": config_coverage,
        "source_files": sorted(
            f
            for f in info["files"]
            if f in ("config.json", "README.md") or f in py or f.endswith((".safetensors", ".safetensors.index.json"))
        ),
        "params": {"counted_from_weights": wcount, "active": active},
        "cache": mem,
        "code_scan": {
            "files": code["files"],
            "classes": code["classes"],
            "facts": {k: {"value": v[0], "evidence": v[1]} for k, v in code["facts"].items()},
        },
        "card_claims": claims,
        "checks": checks,
        "published": published,
        "review_queue": review,
        "unchecked_notes": notes,
        "weight_layer_kinds": wkinds,
    }


def summary_values(rec: dict) -> dict:
    v = dict(rec["published"])
    try:
        memory = calc.cache_model(v)
    except (KeyError, TypeError):
        memory = None
    if memory and all(row["grows_per_token_bytes"] is not None for row in memory["breakdown"]):
        v["kv_cache_per_token_bytes"] = memory["grows_per_token_bytes"]
    return v


def tensor_diff(new_repo: str, old_repo: str, top: int = 6, *, new_info=None, old_info=None) -> dict:
    """Which kinds of tensors appeared or disappeared, with their parameter totals.

    Layer numbers and expert numbers are folded (layers.N -> layers.*) so a pattern names a kind of weight."""
    import re

    def groups(repo, info):
        g: dict[str, int] = {}
        cfg = (hub.get_json(repo, "config.json", revision=info["sha"]) or {}) if info else {}
        native_bits = native_expert_bits_of(cfg)
        tensors = hub.safetensors_headers(repo, info=info) if info is not None else hub.safetensors_headers(repo)
        weights.annotate_packed_fp4(tensors)
        for k, t in tensors.items():
            n = weights.logical_elements(k, t, native_expert_bits=native_bits)
            if n is None:
                continue
            pat = re.sub(r"\.(\d+)\.", ".*.", re.sub(r"^(language_model\.)?model\.", "", k))
            pat = re.sub(r"\.weight(_packed)?$", "", pat)
            g[pat] = g.get(pat, 0) + n
        return g

    a, b = groups(old_repo, old_info), groups(new_repo, new_info)
    added = sorted(((p, b[p]) for p in b.keys() - a.keys()), key=lambda x: (-x[1], x[0]))
    removed = sorted(((p, a[p]) for p in a.keys() - b.keys()), key=lambda x: (-x[1], x[0]))
    return {
        "added": [{"pattern": p, "params": n} for p, n in added[:top]],
        "added_total": sum(n for _, n in added),
        "added_count": len(added),
        "removed": [{"pattern": p, "params": n} for p, n in removed[:top]],
        "removed_total": sum(n for _, n in removed),
        "removed_count": len(removed),
    }


def diff(new: dict, old: dict) -> list[dict]:
    a, b = summary_values(old), summary_values(new)
    rows = []
    for f in DIFF_FIELDS:
        if a.get(f) != b.get(f) and (a.get(f) is not None or b.get(f) is not None):
            rows.append({"field": f, "before": a.get(f), "after": b.get(f)})
    return rows
