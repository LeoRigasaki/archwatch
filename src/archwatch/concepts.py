"""Deterministic concept membership with field-level, commit-pinned provenance.

The registry explains concepts; it never supplies model facts. Absent or rejected
facts produce no positive match. Public consumers must call ``for_record``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

from .publication import PublicationError, render_cache, render_values
from .sources import source_url
from .watcher import derivative_relation

REGISTRY = Path(__file__).resolve().parents[2] / "config/concepts.yaml"
EVIDENCE_VERSION = "archwatch-concept-evidence/1"
ATTENTION = {
    "full_attention",
    "sliding_attention",
    "full_attention_mla",
    "gated_dsa_mla",
    "indexed_mla",
    "qwen_sparse_attention",
}
RECURRENT = {"short_conv", "kda_linear", "mamba2", "linear_attention", "gated_delta_net"}


def registry():
    return _registry(REGISTRY, REGISTRY.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _registry(path, modified):
    data = yaml.safe_load(path.read_text())
    aliases = {}
    for key, item in data.items():
        for alias in [key, item["title"], item["short"], *item.get("aliases", [])]:
            normalized = alias.casefold()
            if normalized in aliases and aliases[normalized] != key:
                raise ValueError(f"Ambiguous concept alias: {alias}")
            aliases[normalized] = key
    return data


def canonical(name):
    name = name.casefold()
    return next(
        (
            key
            for key, item in registry().items()
            if name in {s.casefold() for s in [key, item["title"], item["short"], *item.get("aliases", [])]}
        ),
        None,
    )


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _path(value, path):
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _condition(condition, values):
    if "all" in condition:
        return all(_condition(c, values) for c in condition["all"])
    if "any" in condition:
        return any(_condition(c, values) for c in condition["any"])
    value = _path(values, condition["field"])
    if "equals" in condition:
        return type(value) is type(condition["equals"]) and value == condition["equals"]
    if "one_of" in condition:
        return any(type(value) is type(v) and value == v for v in condition["one_of"])
    if "positive" in condition:
        return positive(value)
    if "greater_than" in condition:
        return positive(value) and value > condition["greater_than"]
    if "nonempty" in condition:
        return isinstance(value, (dict, list)) and bool(value)
    raise ValueError("Unrecognized concept rule")


def _fields(condition, values):
    if "all" in condition or "any" in condition:
        children = condition.get("all", condition.get("any"))
        return {field for child in children if _condition(child, values) for field in _fields(child, values)}
    return {condition["field"].split(".")[0]}


def from_values(values):
    """Structural matches for already approved values; this does not assert evidence status."""
    hits = []
    kinds = {k for k, n in values.get("layer_type_counts", {}).items() if positive(n)}
    for key, item in registry().items():
        rule = item["rule"]
        if not _condition(rule, values):
            continue
        fields = _fields(rule, values)
        scope = item["scope"]
        if key in {"mha", "gqa"}:
            heads, kv = values.get("num_attention_heads"), values.get("num_kv_heads")
            if (
                not positive(heads)
                or not positive(kv)
                or kinds & {"full_attention_mla", "gated_dsa_mla", "indexed_mla"}
            ):
                continue
            if key == "mha" and heads != kv:
                continue
            if key == "gqa" and not (1 <= kv < heads and heads % kv == 0):
                continue
            fields |= {"num_attention_heads", "num_kv_heads"}
            if key == "gqa" and kv == 1:
                scope = "MQA subtype: one key-value head shared by the query heads."
        if key == "hybrid-attention" and not (kinds & ATTENTION and kinds & RECURRENT):
            continue
        if key == "short-convolution":
            scopes = []
            if "short_conv" in kinds:
                scopes.append("Standalone short-convolution decoder mixer layers.")
            if "kda_linear" in kinds and positive(values.get("linear_attn_conv_kernel")):
                fields.add("linear_attn_conv_kernel")
                scopes.append("Short convolution inside KDA; it adds no decoder layers.")
            if not scopes:
                continue
            scope = " ".join(scopes)
        if key == "rope":
            rope = values["rope"]
            paths = rope if rope and all(isinstance(v, dict) for v in rope.values()) else {"attention": rope}
            active = [
                name
                for name, settings in paths.items()
                if isinstance(settings, dict)
                and positive(settings.get("rope_theta"))
                and positive(settings.get("partial_rotary_factor", 1))
                and (name not in ATTENTION or name in kinds)
            ]
            if not active:
                continue
            scope = "Configured rotary position encoding: " + ", ".join(sorted(active)) + ". Not an all-layer claim."
        if key == "rmsnorm":
            modules = [
                m
                for m in values.get("normalization_modules", [])
                if m.get("kind") == "RMSNorm" and m.get("scope") == "decoder"
            ]
            if modules:
                fields.add("normalization_modules")
                scope = "Decoder normalization modules: " + ", ".join(m["module"] for m in modules) + "."
            else:
                scope = "Explicit configured decoder normalization; implementation is not cross-checked."
        if key == "silu":
            scope = "Published feed-forward activation; not a claim about every activation in the model."
        hits.append({"key": key, "scope": scope, "fields": sorted(fields)})
    return hits


def for_record(record):
    """Only publish concepts supported by passing, current-value verifier decisions.

    The whole-model publication gate runs first, including layer/cache holds.
    Derived relationships retain their input evidence rather than counting the
    calculation as another independent source.
    """
    if derivative_relation(record.get("hub_metadata") or {}):
        raise PublicationError("Derivative models are outside the architecture catalog")
    values = render_values(record)
    render_cache(values)
    checks = {c["field"]: c for c in record.get("checks", [])}
    matches = []
    for hit in from_values(values):
        fields = []
        for name in hit["fields"]:
            check = checks.get(name, {})
            if not check.get("published") or check.get("status") not in {"verified", "single source"}:
                break
            if json.dumps(check.get("value"), sort_keys=True) != json.dumps(values.get(name), sort_keys=True):
                break
            evidence = [
                {"claim": candidate["evidence"], "url": source_url(record, candidate["evidence"])}
                for candidate in check.get("candidates", [])
            ]
            fields.append({"field": name, "value": values[name], "status": check["status"], "sources": evidence})
        else:
            if not fields:
                continue
            matches.append(
                {
                    "key": hit["key"],
                    "scope": hit["scope"],
                    "status": "verified" if all(f["status"] == "verified" for f in fields) else "single source",
                    "fields": fields,
                }
            )
    return matches


def evidence_spec(record):
    """Versioned companion to a diagram, never silently appended to an approved SVG."""
    return {
        "schema_version": EVIDENCE_VERSION,
        "model": record["id"],
        "source_commit": record["source_commit"],
        "concepts": for_record(record),
    }
