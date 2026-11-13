"""One model in, one verified record out: fetch -> extract -> weights -> code -> card -> calc -> verify."""
from __future__ import annotations

import datetime as dt

from . import calc, card, codescan, extract, hub, verify, weights

DIFF_FIELDS = ["num_layers", "hidden_size", "layer_type_counts", "attention_type", "num_attention_heads",
               "num_kv_heads", "head_dim", "context_length", "vocab_size", "moe_num_experts", "moe_active_experts",
               "moe_shared_experts", "moe_expert_intermediate", "moe_latent_dim", "dense_layers", "dsa_index_topk",
               "sliding_window", "short_conv_kernel", "activation", "norm", "ffn_intermediate",
               "released_weights_format", "params_total", "params_active", "kv_cache_per_token_bytes"]


def analyze(repo: str, with_weights: bool = True) -> dict:
    info = hub.repo_info(repo)
    cfg = hub.get_json(repo, "config.json")
    facts = extract.extract(cfg)
    bits = facts.get("_quant_bits", (None, ""))[0]

    py = {f: hub.get_text(repo, f) for f in info["files"]
          if f.endswith(".py") and (f.startswith("modeling_") or f.startswith("configuration_"))}
    code = codescan.scan({k: v for k, v in py.items() if v})

    readme = hub.get_text(repo, "README.md") or ""
    claims = card.parse_card(readme)

    wcount = wshape = active = wkinds = None
    if with_weights:
        tensors = hub.safetensors_headers(repo)
        wcount = weights.count(tensors, bits)
        wshape = weights.shape_facts(tensors, bits)
        wkinds = weights.layer_kinds(tensors)
        active = weights.active_params(wcount, facts.get("moe_num_experts", (None,))[0],
                                       facts.get("moe_active_experts", (None,))[0])

    values = {k: v[0] for k, v in facts.items()}
    mem = calc.cache_model(values)
    checks = verify.verify(facts, wcount or {}, wshape or {}, active, code, claims, wkinds)

    published = {r["field"]: r["value"] for r in checks if r["published"]}
    review = [r for r in checks if not r["published"]]
    # things the pipeline could not check at all are reported too, so nobody mistakes silence for a pass
    notes = []
    if not code["facts"]["modeling_code_in_repo"][0]:
        notes.append("No modeling_*.py in the repo: mechanism details (layer internals, residual scheme, "
                     "activation) are read from config, card and tensor names only, not from code.")
    unknown_cfg = sorted(k for k in (cfg.get("text_config") or cfg) if k in
                         ("attn_res_block_size", "activation_situ_beta", "latent_moe_use_norm"))
    if unknown_cfg:
        notes.append("Config keys with no extraction rule yet (need a human or code to interpret): "
                     + ", ".join(unknown_cfg))
    return {
        "id": repo,
        "source_commit": info["sha"],
        "hub_created": info["created_at"],
        "license": info.get("license_name") or info.get("license"),
        "analyzed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "facts": {k: {"value": v[0], "evidence": v[1]} for k, v in facts.items() if not k.startswith("_")},
        "params": {"counted_from_weights": wcount, "active": active},
        "cache": mem,
        "code_scan": {"files": code["files"], "classes": code["classes"],
                      "facts": {k: {"value": v[0], "evidence": v[1]} for k, v in code["facts"].items()}},
        "card_claims": claims,
        "checks": checks,
        "published": published,
        "review_queue": review,
        "unchecked_notes": notes,
        "weight_layer_kinds": wkinds,
    }


def summary_values(rec: dict) -> dict:
    v = {k: x["value"] for k, x in rec["facts"].items()}
    v.update(rec["published"])
    if rec["params"]["counted_from_weights"] and "params_total" not in rec["published"]:
        v["params_total"] = rec["params"]["counted_from_weights"]["total"]
    if rec["params"]["active"] and "params_active" not in rec["published"]:
        v["params_active"] = rec["params"]["active"]["text_active"]
    # a field in review is shown with its weight-derived value: weights are what actually runs
    for r in rec["review_queue"]:
        w = [c for c in r["candidates"] if c["evidence"].startswith("weights:")]
        if w:
            v[r["field"]] = w[0]["value"]
    v["kv_cache_per_token_bytes"] = rec["cache"]["grows_per_token_bytes"]
    return v


def tensor_diff(new_repo: str, old_repo: str, top: int = 6) -> dict:
    """Which kinds of tensors appeared or disappeared, with their parameter totals.

    Layer numbers and expert numbers are folded (layers.N -> layers.*) so a pattern names a kind of weight."""
    import re
    from math import prod

    def groups(repo):
        g: dict[str, int] = {}
        for k, t in hub.safetensors_headers(repo).items():
            if k.endswith(weights.SCALE_SUFFIXES):
                continue
            pat = re.sub(r"\.(\d+)\.", ".*.", re.sub(r"^(language_model\.)?model\.", "", k))
            pat = re.sub(r"\.weight(_packed)?$", "", pat)
            g[pat] = g.get(pat, 0) + prod(t["shape"]) * (2 if k.endswith("_packed") else 1)
        return g

    a, b = groups(old_repo), groups(new_repo)
    added = sorted(((p, b[p]) for p in b.keys() - a.keys()), key=lambda x: -x[1])
    removed = sorted(((p, a[p]) for p in a.keys() - b.keys()), key=lambda x: -x[1])
    return {"added": [{"pattern": p, "params": n} for p, n in added[:top]],
            "added_total": sum(n for _, n in added), "added_count": len(added),
            "removed": [{"pattern": p, "params": n} for p, n in removed[:top]],
            "removed_total": sum(n for _, n in removed), "removed_count": len(removed)}


def diff(new: dict, old: dict) -> list[dict]:
    a, b = summary_values(old), summary_values(new)
    rows = []
    for f in DIFF_FIELDS:
        if a.get(f) != b.get(f) and (a.get(f) is not None or b.get(f) is not None):
            rows.append({"field": f, "before": a.get(f), "after": b.get(f)})
    return rows
