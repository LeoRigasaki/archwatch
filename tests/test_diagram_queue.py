import copy
import importlib.util
import json
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
spec = importlib.util.spec_from_file_location("diagram_queue", ROOT / "tools/diagram_queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)

READY = ["meta-llama/Llama-3.1-8B", "moonshotai/Kimi-K3.1-Pro", "nvidia/Nemotron-H-8B-Base-8K", "LiquidAI/LFM3-1.7B"]


def root_with(tmp_path, repos, extra_status=None):
    (tmp_path / "data/models").mkdir(parents=True)
    models = {}
    for repo in repos:
        stem = repo.replace("/", "__")
        shutil.copy(ROOT / "data/models" / f"{stem}.json", tmp_path / "data/models" / f"{stem}.json")
        models[repo] = {"status": "ready_for_diagram"}
    models.update(extra_status or {})
    (tmp_path / "data/seen.json").write_text(json.dumps({"models": models}))
    return tmp_path


@pytest.fixture(autouse=True)
def no_publish(monkeypatch):
    monkeypatch.setattr(queue, "publish", lambda root, record, svg: None)
    monkeypatch.setattr(queue, "published_for", lambda root, record: None)


def fake_runner(results, log):
    def run_model(repo, config, client, *, out, spec, record):
        log.append(repo)
        passed, calls, errors = results.get(repo, (True, 1, []))
        return dict(passed=passed, calls=calls, errors=errors, final_version=calls)

    return run_model


def test_queue_processes_changed_architectures_only(tmp_path, monkeypatch):
    root = root_with(tmp_path, READY[:3], {"x/held": {"status": "review"}})
    log = []
    monkeypatch.setattr(queue.diagram_gen, "run_model", fake_runner({}, log))
    first = queue.run(root, CONFIG, client=None, today=date(2026, 11, 16))
    assert sorted(log) == sorted(READY[:3]) and first["calls"] == 3
    log.clear()
    second = queue.run(root, CONFIG, client=None, today=date(2026, 11, 18))
    assert log == [] and second["processed"] == []  # unchanged specs with passing diagrams: no calls
    record = json.loads((root / "data/models/meta-llama__Llama-3.1-8B.json").read_text())
    record["published"]["context_length"] = 65536  # a changed architecture spec
    (root / "data/models/meta-llama__Llama-3.1-8B.json").write_text(json.dumps(record))
    queue.run(root, CONFIG, client=None, today=date(2026, 11, 20))
    assert log == ["meta-llama/Llama-3.1-8B"]


def test_failures_wait_for_the_next_slot_behind_new_work(tmp_path, monkeypatch):
    root = root_with(tmp_path, READY[:2])
    log = []
    monkeypatch.setattr(queue.diagram_gen, "run_model", fake_runner({"meta-llama/Llama-3.1-8B": (False, 3, [])}, log))
    queue.run(root, CONFIG, client=None, today=date(2026, 11, 16))
    state = json.loads((root / "data/diagram-queue.json").read_text())
    assert state["models"]["meta-llama/Llama-3.1-8B"]["status"] == "failed"
    log.clear()
    shutil.copy(
        ROOT / "data/models/nvidia__Nemotron-H-8B-Base-8K.json", root / "data/models/nvidia__Nemotron-H-8B-Base-8K.json"
    )
    seen = json.loads((root / "data/seen.json").read_text())
    seen["models"]["nvidia/Nemotron-H-8B-Base-8K"] = {"status": "ready_for_diagram"}
    (root / "data/seen.json").write_text(json.dumps(seen))
    queue.run(root, CONFIG, client=None, today=date(2026, 11, 18))
    assert log == ["nvidia/Nemotron-H-8B-Base-8K", "meta-llama/Llama-3.1-8B"]  # retried, after new work


def test_run_respects_model_and_call_caps(tmp_path, monkeypatch):
    root = root_with(tmp_path, READY[:3])
    config = copy.deepcopy(CONFIG)
    config["schedule"]["diagrams"]["max_calls_per_run"] = 7  # room for two full allowances of 3 calls
    log = []
    monkeypatch.setattr(queue.diagram_gen, "run_model", fake_runner({r: (False, 3, []) for r in READY}, log))
    summary = queue.run(root, config, client=None, today=date(2026, 11, 16))
    assert len(log) == 2 and summary["stopped"] == "run call budget" and len(summary["waiting"]) == 1
    config["schedule"]["diagrams"]["max_models_per_run"] = 1
    assert (
        len(queue.plan(root, queue.load_state(root / "data/diagram-queue.json"), config["schedule"]["diagrams"])[0])
        == 1
    )


def test_daily_limit_stops_the_run_and_keeps_the_rest_queued(tmp_path, monkeypatch):
    root = root_with(tmp_path, READY[:3])
    log = []
    error = ["Configured free daily request cap reached"]
    monkeypatch.setattr(
        queue.diagram_gen, "run_model", fake_runner({"meta-llama/Llama-3.1-8B": (False, 1, error)}, log)
    )
    summary = queue.run(root, CONFIG, client=None, today=date(2026, 11, 16))
    assert log == ["meta-llama/Llama-3.1-8B"] and summary["stopped"] == "daily limit"
    assert set(summary["waiting"]) == {"moonshotai/Kimi-K3.1-Pro", "nvidia/Nemotron-H-8B-Base-8K"}


def test_unsupported_specs_cost_no_call(tmp_path, monkeypatch):
    root = root_with(tmp_path, ["LiquidAI/LFM3-1.7B"])
    record = json.loads((root / "data/models/LiquidAI__LFM3-1.7B.json").read_text())
    record["published"].pop("block_shapes")
    (root / "data/models/LiquidAI__LFM3-1.7B.json").write_text(json.dumps(record))
    log = []
    monkeypatch.setattr(queue.diagram_gen, "run_model", fake_runner({}, log))
    summary = queue.run(root, CONFIG, client=None, today=date(2026, 11, 16))
    assert log == [] and "block_shapes" in summary["skipped"][0]["reason"]


def test_crons_stay_disabled_and_daily_job_makes_no_llm_calls():
    for name in ("daily.yml", "diagrams.yml"):
        flow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        assert "schedule" not in flow[True], name  # PyYAML reads the `on:` key as True
    daily = (ROOT / ".github/workflows/daily.yml").read_text()
    assert "GEMINI" not in daily and "diagram_queue" not in daily
    assert CONFIG["schedule"]["diagrams"]["max_models_per_run"] == 5
    assert CONFIG["schedule"]["diagrams"]["max_calls_per_run"] == 15


def test_published_diagrams_are_never_redrawn(monkeypatch):
    # The real repository: Llama, Kimi and Nemotron have approved, published diagrams for their current specs.
    monkeypatch.setattr(queue, "published_for", lambda root, record: queue.select(record, root))
    planned, _ = queue.plan(
        ROOT, {"models": {}, "runs": []}, dict(CONFIG["schedule"]["diagrams"], max_models_per_run=50)
    )
    names = [q["model"] for q in planned]
    for repo in ("meta-llama/Llama-3.1-8B", "nvidia/Nemotron-H-8B-Base-8K"):
        assert repo not in names
