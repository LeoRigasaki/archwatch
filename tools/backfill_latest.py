"""Process the frozen newest-release cohort and its explicitly sourced predecessors.

This is a bounded, manual backfill, not discovery, generation or a scheduled loop.
Review records retain source facts without acquiring public YAML/SVG artifacts.
"""

import argparse
import json
from pathlib import Path

import yaml

from archwatch import pipeline
from archwatch.artifacts import save_record, write_if_changed
from archwatch.publication import PublicationError, render_cache, render_values
from archwatch.watcher import HubSource, architecture_hash, derivative_relation, family, load_policy

ROOT = Path(__file__).resolve().parents[1]


def run(root, manifest, policy, *, batch=0, source=None):
    source = source or HubSource(policy)
    state_path = root / "data/seen.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"schema_version": 1, "models": {}}
    results, analyzed, infos = {}, {}, {}
    limit = policy["max_process"]
    selected = manifest["models"][batch * limit : (batch + 1) * limit]
    if not selected:
        raise ValueError("Empty cohort batch")

    def analyze(repo, revision):
        if repo in analyzed:
            return analyzed[repo]
        entry = state["models"].setdefault(repo, {})
        info = source.metadata(repo)
        if info["sha"] != revision:
            raise ValueError("Upstream commit changed after cohort freeze")
        infos[repo] = info
        entry["last_observed_commit"] = revision
        if derivative_relation(info):
            raise ValueError("Explicit Hub derivative relation; not admitted")
        if "config.json" not in info["files"] or not any(
            f.endswith(".safetensors") and "/" not in f for f in info["files"]
        ):
            raise ValueError("Missing root config.json or safetensors; source-format adapter/access required")
        record = source.analyze(repo, info)
        if record["id"] != repo or record["source_commit"] != revision:
            raise ValueError("Analysis did not use the frozen revision")
        analyzed[repo] = record
        return record

    def error(repo, revision, exc):
        status = getattr(exc, "code", None)
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        # Never serialize HTTP exception URLs/headers or authentication material.
        item = {"source_commit": revision, "error": type(exc).__name__, "http_status": status, "reason": message}
        write_if_changed(
            root / "data/source-review" / (repo.replace("/", "__") + ".json"), json.dumps(item, indent=2) + "\n"
        )
        state["models"].setdefault(repo, {}).update(
            status="review",
            last_observed_commit=revision,
            reason=message + (f" (HTTP {status})" if status else ""),
        )
        results[repo] = dict(status="review", **item)

    def persist(record):
        repo = record["id"]
        try:
            render_cache(render_values(record))
            eligible, reason = True, "Frozen primary-source backfill passed publication checks"
        except (PublicationError, KeyError, TypeError) as exc:
            eligible, reason = False, str(exc)
        # Preserve lineage if this is a refresh of a previously stored comparison.
        target = root / "data/models" / (repo.replace("/", "__") + ".json")
        if target.exists() and "predecessor" not in record:
            old = json.loads(target.read_text())
            if old["source_commit"] == record["source_commit"]:
                for key in ("predecessor", "diff_vs_predecessor", "tensor_diff_vs_predecessor"):
                    if key in old:
                        record[key] = old[key]
        save_record(record, root / "data/models", with_yaml=eligible, diagrams=root / "diagrams" if eligible else None)
        if not eligible:
            # An old standalone artifact must not keep serving a now-held record.
            for path in (target.with_suffix(".yaml"), root / "diagrams" / (target.stem + ".svg")):
                path.unlink(missing_ok=True)
        entry = state["models"].setdefault(repo, {})
        entry.update(
            status="ready_for_diagram" if eligible else "review",
            reason=reason,
            last_observed_commit=record["source_commit"],
            analyzed_commit=record["source_commit"],
            family=family(repo, policy),
            created_at=record.get("hub_created_at"),
            pool="core",
        )
        if eligible:
            info = infos[repo]
            entry["architecture_hash"] = architecture_hash(
                record, source.config(repo, info["sha"]), source.code(repo, info["sha"], info["files"])
            )
            entry["last_successful_commit"] = record["source_commit"]
        else:
            entry.setdefault("last_successful_commit", None)
        results[repo] = dict(status=entry["status"], source_commit=record["source_commit"], reason=reason)

    for selection in selected:
        repo, revision = selection["id"], selection["source_commit"]
        pred, old = selection.get("predecessor"), None
        if pred:
            try:
                old = analyze(pred, selection["predecessor_commit"])
                persist(old)
            except Exception as exc:
                error(pred, selection["predecessor_commit"], exc)
        try:
            record = analyze(repo, revision)
            if pred:
                record["predecessor"] = {
                    "id": pred,
                    "source_commit": selection["predecessor_commit"],
                    "selection": selection["selection"],
                }
                try:
                    if old:
                        record["diff_vs_predecessor"] = pipeline.diff(record, old)
                        record["tensor_diff_vs_predecessor"] = source.tensor_diff(repo, pred, infos[repo], infos[pred])
                    else:
                        record["predecessor_review"] = results[pred]
                        if pred in infos and results[pred].get("http_status") is None:
                            # Native headers may remain readable without config.
                            record["tensor_diff_vs_predecessor"] = source.tensor_diff(
                                repo, pred, infos[repo], infos[pred]
                            )
                except Exception as comparison_error:
                    record["predecessor_review"] = {
                        "status": "review",
                        "error": type(comparison_error).__name__,
                        "reason": "Predecessor comparison unavailable; primary analysis retained",
                    }
                    diagnostic = dict(
                        field="predecessor_comparison", status="missing", confidence=0.0, published=False, candidates=[]
                    )
                    record["checks"].append(diagnostic)
                    record["review_queue"].append(diagnostic)
            persist(record)
        except Exception as exc:
            error(repo, revision, exc)
        write_if_changed(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    return {"batch": batch, "primary_models": len(selected), "outcomes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/newest-backfill.yaml")
    parser.add_argument("--batch", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    report = run(
        args.root,
        yaml.safe_load(args.manifest.read_text()),
        load_policy(ROOT / "config/watcher.yaml"),
        batch=args.batch,
    )
    write_if_changed(
        args.root / f"results/newest-backfill-batch-{args.batch}.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
