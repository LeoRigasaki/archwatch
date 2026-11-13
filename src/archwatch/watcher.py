"""Bounded Hub discovery, pinned admission, verified dedupe and restartable analysis.

This queues work for analysis or diagram eligibility; it does not call an LLM or
schedule generation. Observation is separate from successful processing.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from . import hub, pipeline
from .artifacts import save_record, write_if_changed
from .publication import PublicationError, render_cache, render_values

# These describe release packaging/training provenance, not architecture.
PACKAGING_KEYS = {
    "_name_or_path",
    "name_or_path",
    "transformers_version",
    "torch_dtype",
    "dtype",
    "quantization_config",
    "_commit_hash",
}
ARCH_FIELDS = set(pipeline.DIFF_FIELDS) - {"released_weights_format", "kv_cache_per_token_bytes", "norm"}
ARCH_FIELDS |= {
    "layer_types",
    "layer_pattern_per_layer",
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
    "dsa_index_num_heads",
    "dsa_index_head_dim",
    "qk_norm",
    "attention_output_gate",
    "vision_encoder_layers",
    "vision_hidden_size",
}
DONE = {"recorded", "duplicate", "ready_for_diagram", "rejected", "review"}


def error_status(error):
    return getattr(getattr(error, "response", None), "status_code", getattr(error, "code", None))


def timestamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def retry_seconds(error, now):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(error, "headers", {}) or {}
    value = headers.get("retry-after", headers.get("Retry-After"))
    if value is None:
        return None
    try:
        return max(float(value), 0)
    except ValueError:
        try:
            return max((parsedate_to_datetime(value).astimezone(timezone.utc) - now).total_seconds(), 0)
        except (ValueError, TypeError):
            return None


def load_policy(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if value.get("schema_version") != 1 or not value.get("core_publishers") or not value.get("tasks"):
        raise ValueError("Invalid watcher policy")
    for key in (
        "per_publisher_task",
        "trending_limit",
        "recent_limit",
        "max_candidates",
        "max_process",
        "core_slots",
        "outside_slots",
        "request_attempts",
        "max_retry_wait_seconds",
        "max_attempts_per_commit",
    ):
        if not isinstance(value.get(key), int) or value[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for rule in value.get("family_rules", []):
        re.compile(rule["pattern"])
    return value


def valid_repo(repo: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo))


def family(repo: str, policy: dict) -> str:
    if repo in policy.get("family_overrides", {}):
        return policy["family_overrides"][repo]
    name = repo.split("/")[-1].lower()
    root = next(
        (rule["family"] for rule in policy.get("family_rules", []) if re.search(rule["pattern"], name, re.I)), None
    )
    root = root or re.split(r"\d", name, maxsplit=1)[0].rstrip("-_.")
    # Keep visibly separate product streams apart; do not conflate Qwen-VL/Omni/Coder.
    stream = [s for s in ("vl", "omni", "coder") if re.search(rf"(?:^|[-_]){s}(?:$|[-_])", name)]
    return root + (":" + ":".join(stream) if stream else "")


def predecessor(candidate: dict, catalog: dict[str, dict]) -> tuple[dict | None, str]:
    earlier = [
        m
        for key, m in catalog.items()
        if key != candidate["id"]
        and key.split("/")[0].casefold() == candidate["id"].split("/")[0].casefold()
        and m.get("family") == candidate["family"]
        and timestamp(m.get("created_at"))
        and timestamp(candidate.get("created_at"))
        and timestamp(m["created_at"]) < timestamp(candidate["created_at"])
    ]
    if not earlier:
        return None, "No earlier same-org family release in the inspected catalog"
    newest = max(timestamp(m["created_at"]) for m in earlier)
    matches = sorted((m for m in earlier if timestamp(m["created_at"]) == newest), key=lambda m: m["id"])
    if len(matches) != 1:
        return None, "Ambiguous previous release: " + ", ".join(m["id"] for m in matches)
    return matches[0], "Latest earlier Hub creation date in the inspected same-org family catalog"


def _structural(value):
    if isinstance(value, dict):
        return {k: _structural(v) for k, v in value.items() if k not in PACKAGING_KEYS}
    if isinstance(value, list):
        return [_structural(v) for v in value]
    return value


def architecture_hash(record: dict, config: dict, code: dict[str, str]) -> str:
    # Keep unknown config fields and exact custom code in the signature. Matching
    # extracted dimensions alone must not hide an unimplemented mechanism change.
    fields = {k: v for k, v in record["published"].items() if k in ARCH_FIELDS}
    data = {"published": fields, "config": _structural(config), "custom_code": code}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def published_hash(record):
    data = {k: v for k, v in record["published"].items() if k in ARCH_FIELDS}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class HubSource:
    def __init__(self, policy: dict, api=None, wait=time.sleep):
        self.api, self.policy, self.wait = api or HfApi(), policy, wait

    def request(self, operation):
        for attempt in range(self.policy["request_attempts"]):
            try:
                return operation()
            except Exception as error:
                status = error_status(error)
                if status not in (429, 502, 503, 504) or attempt + 1 == self.policy["request_attempts"]:
                    raise
                delay = retry_seconds(error, datetime.now(timezone.utc))
                delay = delay if delay is not None else 2**attempt
                if delay > self.policy["max_retry_wait_seconds"]:
                    raise error
                self.wait(delay)

    def listing(self, **kwargs):
        return self.request(lambda: list(self.api.list_models(**kwargs)))

    def metadata(self, repo):
        return self.request(lambda: hub.repo_info(repo, refresh=True, api=self.api))

    def config(self, repo, commit):
        return self.request(lambda: hub.get_json(repo, "config.json", revision=commit))

    def code(self, repo, commit, files):
        return {
            f: self.request(lambda f=f: hub.get_text(repo, f, revision=commit)) or ""
            for f in sorted(files)
            if f.endswith(".py") and "/" not in f and f.startswith(("modeling_", "configuration_"))
        }

    def analyze(self, repo, info):
        return self.request(lambda: pipeline.analyze(repo, info=info))

    def tensor_diff(self, new, old, new_info, old_info):
        return self.request(lambda: pipeline.tensor_diff(new, old, new_info=new_info, old_info=old_info))


def discover(source: HubSource, policy: dict, repos: list[str] | None = None) -> tuple[list[dict], list[str]]:
    core = {s.casefold() for s in policy["core_publishers"]}
    candidates, errors = {}, []

    def add(repo, origin, rank=None):
        if not valid_repo(repo):
            errors.append("Invalid repository ID in discovery")
            return
        entry = candidates.setdefault(repo, {"id": repo, "explicit": False, "trending_rank": None})
        entry["pool"] = "core" if repo.split("/")[0].casefold() in core else "outside"
        entry["explicit"] |= origin == "explicit"
        if origin == "trending":
            entry["trending_rank"] = min(entry["trending_rank"] or rank, rank)

    if repos:
        for repo in sorted(set(repos)):
            if not valid_repo(repo):
                raise ValueError(f"Invalid repository ID: {repo!r}")
            add(repo, "explicit")
    else:
        for task in policy["tasks"]:
            for org in policy["core_publishers"]:
                try:
                    for model in source.listing(
                        author=org, pipeline_tag=task, sort="created_at", limit=policy["per_publisher_task"], full=True
                    ):
                        add(model.id, "core")
                except Exception as error:
                    errors.append(f"{org}/{task}: {type(error).__name__}")
            for sort, limit in (("trending_score", policy["trending_limit"]), ("created_at", policy["recent_limit"])):
                try:
                    for rank, model in enumerate(
                        source.listing(pipeline_tag=task, sort=sort, limit=limit, full=True), 1
                    ):
                        add(model.id, "trending" if sort == "trending_score" else "recent", rank)
                except Exception as error:
                    errors.append(f"global/{task}/{sort}: {type(error).__name__}")
    # Round-robin publishers before taking the bound; a busy lab cannot crowd out all other core labs.
    ordered, pending = (
        [],
        sorted(
            candidates.values(),
            key=lambda c: (not c["explicit"], c["pool"] != "core", c["trending_rank"] or 10**9, c["id"]),
        ),
    )
    while pending and len(ordered) < policy["max_candidates"]:
        used, rest = set(), []
        for candidate in pending:
            org = candidate["id"].split("/")[0].casefold()
            if org in used or len(ordered) >= policy["max_candidates"]:
                rest.append(candidate)
            else:
                ordered.append(candidate)
                used.add(org)
        pending = rest
    return ordered, errors


def _load_records(root):
    return {
        rec["id"]: rec
        for path in sorted((root / "data/models").glob("*.json"))
        if (rec := json.loads(path.read_text())).get("id")
    }


def run(root: Path, policy: dict, *, source=None, repos=None, process=False, retry_review=False) -> dict:
    source = source or HubSource(policy)
    path = root / "data/seen.json"
    state = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "models": {}}
    if state.get("schema_version") != 1 or not isinstance(state.get("models"), dict):
        raise ValueError("Unsupported seen.json schema")
    entries, records, catalog, signatures = state["models"], _load_records(root), {}, {}
    published_signatures = {}
    report = {"discovery_errors": [], "outcomes": {}, "processed": 0}
    for repo, record in records.items():
        catalog[repo] = {
            "id": repo,
            "source_commit": record["source_commit"],
            "created_at": record.get("hub_created"),
            "family": family(repo, policy),
        }
        entry = entries.setdefault(
            repo,
            {
                "last_observed_commit": record["source_commit"],
                "last_successful_commit": record["source_commit"],
                "status": "recorded",
                "family": family(repo, policy),
                "created_at": record.get("hub_created"),
                "pool": "core"
                if repo.split("/")[0].casefold() in {o.casefold() for o in policy["core_publishers"]}
                else "outside",
            },
        )
        try:
            render_cache(render_values(record))
            config = source.config(repo, record["source_commit"])
            signature = architecture_hash(
                record, config, source.code(repo, record["source_commit"], record["code_scan"]["files"])
            )
            signatures.setdefault(signature, repo)
            published_signatures.setdefault(published_hash(record), repo)
            entry.setdefault("architecture_hash", signature)
        except Exception:
            # A saved record without inspectable primary evidence cannot be a dedupe authority.
            pass
    candidates, report["discovery_errors"] = discover(source, policy, repos)
    if report["discovery_errors"]:
        state["discovery_errors"] = report["discovery_errors"]
    else:
        state.pop("discovery_errors", None)
    if not repos and entries:
        # Retain pending work and revisit old repositories even after they leave
        # the Hub's newest/trending pages. Rotation is computed, not persisted as
        # a timestamp/cursor-only change to tracked state.
        reserve = max(1, policy["max_candidates"] // 2)
        pending_ids = sorted(
            (r for r, e in entries.items() if e.get("status") in ("pending", "error")),
            key=lambda r: (entries[r].get("attempts", 0), r),
        )
        others = sorted(set(entries) - set(pending_ids))
        if others:
            offset = (datetime.now(timezone.utc).date().toordinal() * reserve) % len(others)
            others = others[offset:] + others[:offset]
        retained = (pending_ids + others)[:reserve]
        previous = [
            {
                "id": r,
                "pool": "core"
                if r.split("/")[0].casefold() in {o.casefold() for o in policy["core_publishers"]}
                else "outside",
                "explicit": False,
                "trending_rank": None,
                "tracked": entries[r].get("last_successful_commit") is not None,
            }
            for r in retained
        ]
        candidates = (previous + [c for c in candidates if c["id"] not in retained])[: policy["max_candidates"]]
    admitted = []
    for candidate in candidates:
        repo = candidate["id"]
        previous = dict(entries.get(repo, {}))
        entry = entries.setdefault(repo, {})
        if timestamp(entry.get("retry_after")) and timestamp(entry["retry_after"]) > datetime.now(timezone.utc):
            report["outcomes"][repo] = "rate_limit_deferred"
            continue
        try:
            info = source.metadata(repo)  # Existence and current commit, never trust the listing alone.
            if not re.fullmatch(r"[a-f0-9]{40}", info["sha"]):
                raise ValueError("Hub returned an invalid commit")
            entry["last_observed_commit"] = info["sha"]
            entry.setdefault("last_successful_commit", None)
            if previous.get("last_observed_commit") != info["sha"] or (
                retry_review and previous.get("status") == "review"
            ):
                entry["attempts"] = 0
            if entry.get("attempts", 0) >= policy["max_attempts_per_commit"]:
                entry.update(
                    status="review", reason="Automatic retry budget exhausted; use --retry-review after investigation"
                )
                report["outcomes"][repo] = "review"
                continue
            if previous.get("last_observed_commit") == info["sha"] and previous.get("status") in DONE:
                if not (retry_review and previous["status"] == "review"):
                    report["outcomes"][repo] = "unchanged"
                    continue
            files = info["files"]
            if "config.json" not in files or not any(f.endswith(".safetensors") and "/" not in f for f in files):
                entry.update(status="rejected", reason="Requires root config.json and safetensors files")
                report["outcomes"][repo] = entry["status"]
                continue
            if info.get("pipeline_tag") not in (None, *policy["tasks"]):
                entry.update(status="rejected", reason="Task outside language/text-backbone scope")
                report["outcomes"][repo] = entry["status"]
                continue
            popular = (info.get("downloads") or 0) >= policy["outside_min_downloads"]
            trending = (candidate["trending_rank"] or 10**9) <= policy["outside_top_trending"]
            if candidate["pool"] == "outside" and not (
                candidate["explicit"] or candidate.get("tracked") or popular or trending
            ):
                # Not terminal: it can become popular without a new source commit.
                entry.update(status="deferred", reason="Below outside discovery popularity threshold")
                report["outcomes"][repo] = entry["status"]
                continue
            config = source.config(repo, info["sha"])
            if not isinstance(config, dict) or not isinstance(config.get("text_config", config), dict):
                raise ValueError("Invalid config.json")
            meta = {
                "id": repo,
                "source_commit": info["sha"],
                "created_at": info.get("created_at"),
                "family": family(repo, policy),
            }
            catalog[repo] = meta
            if previous.get("last_observed_commit") != info["sha"] or previous.get("status") != "pending":
                entry.update(
                    status="pending",
                    reason="Hub existence, config and safetensors checked",
                    pool=candidate["pool"],
                    family=meta["family"],
                    created_at=meta["created_at"],
                )
            admitted.append((candidate, info, config))
            entry.pop("retry_after", None)
            report["outcomes"][repo] = entry["status"]
        except Exception as error:
            attempts = min(previous.get("attempts", 0) + 1, policy["max_attempts_per_commit"])
            status = error_status(error)
            entry.update(
                status="rejected"
                if status == 404
                else "review"
                if status in (401, 403) or attempts >= policy["max_attempts_per_commit"]
                else "error",
                reason=f"{type(error).__name__}" + (f" (HTTP {status})" if status is not None else ""),
                attempts=attempts,
            )
            delay = retry_seconds(error, datetime.now(timezone.utc))
            if delay is not None:
                entry["retry_after"] = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
            report["outcomes"][repo] = entry["status"]
        write_if_changed(path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    # Include previously discovered family history even when it fell outside today's list.
    for repo, entry in entries.items():
        if (
            entry.get("created_at")
            and entry.get("family")
            and entry.get("status") not in ("rejected", "duplicate", "error")
        ):
            catalog.setdefault(
                repo,
                {
                    "id": repo,
                    "source_commit": entry["last_observed_commit"],
                    "created_at": entry["created_at"],
                    "family": entry["family"],
                },
            )
    if process:
        selected = []
        for pool, slots in (("core", policy["core_slots"]), ("outside", policy["outside_slots"])):
            selected.extend([row for row in admitted if row[0]["pool"] == pool][:slots])
        selected.extend(row for row in admitted if row not in selected)
        for candidate, info, config in selected[: policy["max_process"]]:
            repo, entry = candidate["id"], entries[candidate["id"]]
            entry["attempts"] = entry.get("attempts", 0) + 1
            try:
                record = source.analyze(repo, info)
                if record["id"] != repo or record["source_commit"] != info["sha"]:
                    raise ValueError("Analysis did not use the admitted source commit")
                render_cache(render_values(record))
                signature = architecture_hash(record, config, source.code(repo, info["sha"], info["files"]))
                match = signatures.get(signature)
                same_fields = published_signatures.get(published_hash(record))
                if match and match != repo:
                    entry.update(
                        status="duplicate",
                        reason="Verified architecture matches an existing record",
                        architecture_match=match,
                        architecture_hash=signature,
                        last_successful_commit=info["sha"],
                    )
                    catalog.pop(repo, None)
                elif same_fields and match is None:
                    entry.update(
                        status="review",
                        reason="Published dimensions match but config/custom code differs; verify mechanism before deduplication or a new diagram",
                    )
                else:
                    pred, reason = predecessor(catalog[repo], catalog)
                    if pred:
                        old = records.get(pred["id"])
                        old_info = source.metadata(pred["id"])
                        if old_info["sha"] != pred["source_commit"]:
                            raise ValueError("Predecessor changed since selection; retry discovery")
                        if old is None or old["source_commit"] != old_info["sha"]:
                            old = source.analyze(pred["id"], old_info)
                        record["predecessor"] = {
                            "id": pred["id"],
                            "source_commit": old["source_commit"],
                            "selection": reason,
                        }
                        record["diff_vs_predecessor"] = pipeline.diff(record, old)
                        record["tensor_diff_vs_predecessor"] = source.tensor_diff(repo, pred["id"], info, old_info)
                        if pred["id"] not in records:
                            save_record(old, root / "data/models")
                            records[pred["id"]] = old
                    changed_architecture = (
                        entry.get("architecture_hash") != signature or entry.get("last_successful_commit") is None
                    )
                    save_record(record, root / "data/models", with_yaml=True, diagrams=root / "diagrams")
                    records[repo], signatures[signature] = record, repo
                    published_signatures[published_hash(record)] = repo
                    entry.update(
                        status="ready_for_diagram" if changed_architecture else "recorded",
                        reason=reason,
                        architecture_hash=signature,
                        last_successful_commit=info["sha"],
                        predecessor=pred["id"] if pred else None,
                    )
                report["processed"] += 1
            except PublicationError as error:
                entry.update(status="review", reason=str(error))
            except Exception as error:
                entry.update(status="error", reason=type(error).__name__)
                delay = retry_seconds(error, datetime.now(timezone.utc))
                if delay is not None:
                    entry["retry_after"] = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
            report["outcomes"][repo] = entry["status"]
            write_if_changed(path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    write_if_changed(path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    return report
