"""Scheduled diagram queue: changed architectures only, bounded per run and by the daily limit.

    uv run --frozen python tools/diagram_queue.py --plan     # show what the next run would do, no calls
    uv run --frozen python tools/diagram_queue.py            # run (needs GEMINI_API_KEY)

Candidates are records the watcher marked ready_for_diagram or recorded (both passed the publication gate)
whose spec hash has no passing diagram yet.
At most `schedule.max_models_per_run` models and `schedule.max_calls_per_run` calls per run; every call
also counts against the configured daily limit. A model is started only if its full allowance
(1 + repairs calls) still fits. Failed or unreached models stay queued for the next slot. A spec the
builder cannot produce (unsupported family, unpublished shapes) is recorded and costs no call.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archwatch.diagram_checks import spec_dimension_problems  # noqa: E402
from archwatch.diagram_spec import CURRENT_TEMPLATE, SpecError, build_spec, dump  # noqa: E402
from archwatch.generated import publish, select  # noqa: E402


def published_for(root, record):
    """The published generated diagram if it still matches this record's spec and passes the static checks."""
    return select(record, root)


_gen_spec = importlib.util.spec_from_file_location("diagram_gen", ROOT / "tools/diagram_gen.py")
diagram_gen = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(diagram_gen)


def spec_hash(spec_text: str) -> str:
    body = spec_text.split("\n\n", 1)[1]  # the header carries the commit; the body is what is drawn
    return hashlib.sha256(body.encode()).hexdigest()


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"models": {}, "runs": []}


def plan(root: Path, state: dict, limits: dict) -> tuple[list, list]:
    """Return (queue in run order, skipped) without making any call."""
    seen = json.loads((root / "data/seen.json").read_text())["models"]
    queue, skipped = [], []
    for repo, entry in sorted(seen.items()):
        if entry.get("status") not in ("ready_for_diagram", "recorded"):  # both passed the publication gate
            continue
        path = root / "data/models" / (repo.replace("/", "__") + ".json")
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        try:
            spec = build_spec(record)
        except SpecError as error:
            skipped.append(dict(model=repo, reason=str(error)))
            continue
        mismatched = spec_dimension_problems(spec)
        if mismatched:  # the drawing could never pass "dimensions connect": fix the spec, spend no call
            skipped.append(dict(model=repo, reason="spec widths do not connect: " + "; ".join(mismatched)))
            continue
        digest = spec_hash(dump(spec, record))
        done = state["models"].get(repo, {})
        if done.get("status") == "passed":
            drawn = done.get("template", "v1")  # entries written before templates were versioned are v1
            try:
                kept = spec_hash(dump(build_spec(record, drawn), record))
            except SpecError:
                kept = None
            if done.get("spec_sha256") == kept:
                continue  # approved under its own template: never requeued for a wording change
        if published_for(root, record) is not None:
            continue  # an approved/published generated diagram for this exact spec: never redraw it
        retry = done.get("spec_sha256") == digest  # same spec, failed before: waits behind new work
        queue.append(dict(model=repo, record=record, spec=spec, spec_sha256=digest, retry=retry))
    priority = limits.get("priority") or []
    rank = {repo: i for i, repo in enumerate(priority)}  # configured order first, then new work, then retries
    queue.sort(key=lambda q: (rank.get(q["model"], len(priority)), q["retry"], q["model"]))
    return queue[: limits["max_models_per_run"]], skipped


def run(root: Path, config: dict, client, *, today=None) -> dict:
    limits = config["schedule"]["diagrams"]
    per_model = 1 + config["diagram_gen"]["repairs"]
    state_path = root / limits["state"]
    state = load_state(state_path)
    queue, skipped = plan(root, state, limits)
    budget, rows, stopped = limits["max_calls_per_run"], [], None
    for item in queue:
        if budget < per_model:
            stopped = "run call budget"
            break
        folder = root / limits["output"] / item["model"].replace("/", "__") / item["spec_sha256"][:12]
        row = diagram_gen.run_model(item["model"], config, client, out=folder, spec=item["spec"], record=item["record"])
        budget -= row["calls"]
        status = "passed" if row["passed"] else ("error" if row["errors"] else "failed")
        if row["passed"]:
            publish(root, item["record"], folder / f"v{row['final_version']}.svg")  # gallery shows it after review
        state["models"][item["model"]] = dict(
            spec_sha256=item["spec_sha256"],
            template=CURRENT_TEMPLATE,
            status=status,
            version=row["final_version"],
            folder=str(folder.relative_to(root)),
            calls=row["calls"],
            updated=(today or dt.date.today()).isoformat(),
        )
        rows.append(dict(model=item["model"], status=status, calls=row["calls"], responses=row.get("responses", 0)))
        if row["errors"]:  # a failed retry, a 429, the daily cap or any other provider error: stop the whole run
            stopped = "daily limit" if "daily" in row["errors"][-1] else "provider error: " + row["errors"][-1]
            break
    summary = dict(
        date=(today or dt.date.today()).isoformat(),
        processed=rows,
        skipped=skipped,
        calls=limits["max_calls_per_run"] - budget,
        responses=sum(r["responses"] for r in rows),
        queued=len(queue),
        stopped=stopped,
        waiting=[q["model"] for q in queue if q["model"] not in {r["model"] for r in rows}],
    )
    state["runs"] = (state["runs"] + [summary])[-50:]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="list the next run's queue; no model calls")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
    if args.plan:
        limits = config["schedule"]["diagrams"]
        queue, skipped = plan(ROOT, load_state(ROOT / limits["state"]), limits)
        print(json.dumps(dict(queue=[q["model"] for q in queue], skipped=skipped), indent=2))
        return
    from archwatch.gemini import Gemini, local_env

    local_env(ROOT / ".env")
    client = Gemini(config, ROOT / config["diagram_gen"]["ledger"])
    summary = run(ROOT, config, client)
    print(json.dumps(summary, indent=2))
    failure = failed(summary)
    if failure:
        from archwatch.notifications import send_text

        run_url = os.environ.get("ARCHWATCH_RUN_URL", "")
        send_text(f"ArchWatch diagram run failed: {failure}" + (f"\nRun: <{run_url}>" if run_url else ""))
        raise SystemExit(f"diagram run failed: {failure}")


def failed(summary: dict) -> str | None:
    """A run succeeds only if at least one diagram attempt got a response, whenever the queue had work."""
    if not summary["queued"]:
        return None
    if summary["calls"] == 0:
        return f"{summary['queued']} model(s) queued but no call was made ({summary['stopped'] or 'nothing started'})"
    if summary["responses"] == 0:
        return f"all {summary['calls']} call(s) failed ({summary['stopped'] or 'no response'})"
    return None


if __name__ == "__main__":
    main()
