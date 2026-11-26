"""Diagram pipeline safety: one logged retry on 5xx only, a run stops on provider errors, a run with work but no
response fails loudly, the configured priority goes first, and an open review PR's state is carried, not erased."""

import importlib.util
import json
import urllib.error
from datetime import date
from pathlib import Path

import pytest
import yaml

from archwatch import gemini
from archwatch.gemini import Gemini, ProviderError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config/providers.yaml").read_text())


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load("diagram_queue")
carry = load("carry_pending")


class Ok:
    def __init__(self):
        self.body = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "<svg/>"}]}, "finishReason": "STOP"}]}
        ).encode()

    def read(self, n):
        return self.body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def client(tmp_path, monkeypatch, outcomes):
    """A Gemini client whose transport answers with `outcomes` in order: an int is an HTTP error, None is 200."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    waits, sent = [], []

    def opener(request, timeout):
        sent.append(request)
        code = outcomes.pop(0)
        if code is None:
            return Ok()
        raise urllib.error.HTTPError(request.full_url, code, "err", {}, None)

    clock = iter(range(1000, 100_000, 100))
    c = Gemini(CONFIG, tmp_path / "ledger.json", opener=opener, clock=lambda: next(clock), wait=waits.append)
    return c, waits, sent


def ledger(tmp_path):
    return json.loads((tmp_path / "ledger.json").read_text())


def test_one_retry_after_a_server_error_logged_as_its_own_call(tmp_path, monkeypatch):
    c, waits, sent = client(tmp_path, monkeypatch, [503, None])
    assert c.generate("S", "P", tag={"for": "org/m"})[0] == "<svg/>"
    entries = ledger(tmp_path)
    assert len(sent) == 2 and len(entries) == 2 and gemini.RETRY_WAIT_SECONDS in waits
    assert entries[0]["for"] == entries[1]["for"] == "org/m" and entries[1]["retry"] == "after HTTP 503"


@pytest.mark.parametrize("outcomes, requests", [([500, 502], 2), ([429], 1), ([400], 1)])
def test_no_other_retries(tmp_path, monkeypatch, outcomes, requests):
    c, waits, sent = client(tmp_path, monkeypatch, list(outcomes))
    with pytest.raises(ProviderError) as error:
        c.generate("S", "P")
    assert len(sent) == requests and len(ledger(tmp_path)) == requests and error.value.status == outcomes[-1]
    assert waits.count(gemini.RETRY_WAIT_SECONDS) == (1 if outcomes[0] >= 500 else 0)


READY = ["meta-llama/Llama-3.1-8B", "moonshotai/Kimi-K3", "nvidia/Nemotron-H-8B-Base-8K"]


def root_with(tmp_path, repos):
    (tmp_path / "data/models").mkdir(parents=True)
    for repo in repos:
        stem = repo.replace("/", "__")
        (tmp_path / "data/models" / f"{stem}.json").write_text((ROOT / "data/models" / f"{stem}.json").read_text())
    (tmp_path / "data/seen.json").write_text(json.dumps({"models": {r: {"status": "recorded"} for r in repos}}))
    return tmp_path


@pytest.fixture(autouse=True)
def no_publish(monkeypatch):
    monkeypatch.setattr(queue, "publish", lambda root, record, svg: None)
    monkeypatch.setattr(queue, "published_for", lambda root, record: None)


def runner(results, log):
    def run_model(repo, config, client, *, out, spec, record):
        log.append(repo)
        passed, calls, responses, errors = results.get(repo, (True, 1, 1, []))
        return dict(passed=passed, calls=calls, responses=responses, errors=errors, final_version=calls)

    return run_model


def test_the_configured_priority_goes_first():
    assert CONFIG["schedule"]["diagrams"]["priority"][0] == "moonshotai/Kimi-K3"


def test_priority_order_and_a_provider_error_stops_the_run(tmp_path, monkeypatch):
    root = root_with(tmp_path, READY)
    log = []
    error = ["Gemini request failed (HTTP 503); no fallback attempted"]
    monkeypatch.setattr(queue.diagram_gen, "run_model", runner({"moonshotai/Kimi-K3": (False, 2, 0, error)}, log))
    summary = queue.run(root, CONFIG, client=None, today=date(2026, 11, 26))
    assert log == ["moonshotai/Kimi-K3"]  # first by priority, and the failed retry stops every later model
    assert summary["stopped"].startswith("provider error") and summary["responses"] == 0 and summary["calls"] == 2
    assert queue.failed(summary) == f"all 2 call(s) failed ({summary['stopped']})"


def test_success_means_at_least_one_response(tmp_path, monkeypatch):
    root = root_with(tmp_path, READY)
    monkeypatch.setattr(queue.diagram_gen, "run_model", runner({}, []))
    summary = queue.run(root, CONFIG, client=None, today=date(2026, 11, 26))
    assert summary["responses"] == 3 and queue.failed(summary) is None
    assert queue.failed(dict(queued=0, calls=0, responses=0, stopped=None)) is None  # nothing to do is fine
    assert "no call was made" in queue.failed(dict(queued=4, calls=0, responses=0, stopped="run call budget"))


def test_a_failed_run_exits_non_zero_and_notifies(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(queue, "run", lambda root, config, client: dict(queued=2, calls=2, responses=0, stopped="x"))
    monkeypatch.setattr("archwatch.gemini.Gemini.__init__", lambda self, *a, **k: None)
    monkeypatch.setattr("archwatch.notifications.send_text", lambda text, **k: sent.append(text) or True)
    monkeypatch.setattr("sys.argv", ["diagram_queue.py"])
    with pytest.raises(SystemExit, match="diagram run failed"):
        queue.main()
    assert sent and sent[0].startswith("ArchWatch diagram run failed: all 2 call(s) failed")


def test_carry_keeps_every_ledger_entry_newer_queue_state_and_missing_files(tmp_path):
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    for root in (ours, theirs):
        (root / "data/generated-diagrams/m/abc").mkdir(parents=True)
        (root / "diagrams/generated").mkdir(parents=True)
    a, b, c = (dict(time=t, day="2026-11-2" + str(t), model="g") for t in (1, 2, 3))
    (ours / carry.LEDGER).write_text(json.dumps([a, c]))
    (theirs / carry.LEDGER).write_text(json.dumps([a, b]))
    (ours / carry.QUEUE).write_text(
        json.dumps({"models": {"x": {"status": "error", "updated": "2026-11-18"}}, "runs": [{"date": "2026-11-26"}]})
    )
    (theirs / carry.QUEUE).write_text(
        json.dumps(
            {
                "models": {
                    "x": {"status": "passed", "updated": "2026-11-20"},
                    "y": {"status": "passed", "updated": "2026-11-20"},
                },
                "runs": [{"date": "2026-11-20"}],
            }
        )
    )
    (theirs / "data/generated-diagrams/m/abc/v1.svg").write_text("<svg>theirs</svg>")
    (ours / "diagrams/generated/k.svg").write_text("<svg>reviewed</svg>")
    (theirs / "diagrams/generated/k.svg").write_text("<svg>other</svg>")
    result = carry.carry(theirs, ours)
    assert json.loads((ours / carry.LEDGER).read_text()) == [a, b, c] and result["ledger_entries_added"] == 1
    state = json.loads((ours / carry.QUEUE).read_text())
    assert state["models"]["x"]["status"] == "passed" and "y" in state["models"]
    assert [r["date"] for r in state["runs"]] == ["2026-11-20", "2026-11-26"]
    assert (ours / "data/generated-diagrams/m/abc/v1.svg").read_text() == "<svg>theirs</svg>"
    assert (ours / "diagrams/generated/k.svg").read_text() == "<svg>reviewed</svg>"  # never overwritten
    assert result["files_added"] == 1


def test_the_workflow_carries_the_open_pr_before_generating_and_reports_failures():
    workflow = yaml.safe_load((ROOT / ".github/workflows/diagrams.yml").read_text())
    steps = workflow["jobs"]["run"]["steps"]
    names = [s.get("name") or s.get("uses", "") for s in steps]
    assert names.index("Carry its unmerged state") < names.index("Generate diagrams for changed architectures")
    generate = next(s for s in steps if s.get("id") == "generate")
    assert "DISCORD_WEBHOOK_URL" in generate["env"] and "ARCHWATCH_RUN_URL" in generate["env"]
    pr = next(s for s in steps if "create-pull-request" in s.get("uses", ""))
    assert pr["with"]["branch"] == "diagram-review" and "diagram-review" in json.dumps(steps)
    carry_step = next(s for s in steps if s.get("name") == "Carry its unmerged state")
    assert "rm -rf pending" in carry_step["run"]  # the side checkout never lands in the commit


def test_the_lost_runs_are_in_the_ledger():
    entries = json.loads((ROOT / "data/gemini-requests.json").read_text())
    carried = [
        e
        for e in entries
        if e.get("run_id") in (44301688691, 44733223773, 45185607805, 45623762927)
        and e.get("note", "").startswith("carried")
    ]
    assert sorted({e["day"] for e in carried}) == ["2026-11-18", "2026-11-20", "2026-11-23", "2026-11-25"]
    assert len(carried) == 22
