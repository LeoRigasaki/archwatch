"""Scheduled diagram queue: changed architectures only, bounded per run and by the daily limit.

    uv run --frozen python tools/diagram_queue.py --plan     # show what the next run would do, no calls
    uv run --frozen python tools/diagram_queue.py            # run (needs GEMINI_API_KEY)

Candidates are records the watcher marked ready_for_diagram whose spec hash has no passing diagram yet.
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
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archwatch.diagram_spec import SpecError, build_spec, dump  # noqa: E402

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
        if entry.get("status") != "ready_for_diagram":
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
        digest = spec_hash(dump(spec, record))
        done = state["models"].get(repo, {})
        if done.get("spec_sha256") == digest and done.get("status") == "passed":
            continue
        retry = done.get("spec_sha256") == digest  # same spec, failed before: waits behind new work
        queue.append(dict(model=repo, record=record, spec=spec, spec_sha256=digest, retry=retry))
    queue.sort(key=lambda q: (q["retry"], q["model"]))
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
        state["models"][item["model"]] = dict(
            spec_sha256=item["spec_sha256"],
            status=status,
            version=row["final_version"],
            folder=str(folder.relative_to(root)),
            calls=row["calls"],
            updated=(today or dt.date.today()).isoformat(),
        )
        rows.append(dict(model=item["model"], status=status, calls=row["calls"]))
        if any("daily" in e for e in row["errors"]):
            stopped = "daily limit"
            break
    summary = dict(
        date=(today or dt.date.today()).isoformat(),
        processed=rows,
        skipped=skipped,
        calls=limits["max_calls_per_run"] - budget,
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
    print(json.dumps(run(ROOT, config, client), indent=2))


if __name__ == "__main__":
    main()
