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


# ---------------------------------------------------------------- dimensions connect

from archwatch import diagram_checks, diagram_spec, generated  # noqa: E402


def test_glm_rejected_svg_fails_dimensions_connect():
    root = diagram_checks.parse((ROOT / "tests/fixtures/diagrams/glm-4-9b-0414-rejected.svg").read_text())
    assert diagram_checks.svg_dimension_problems(root) == [
        'ffn: "down: Linear 13696 -> 4096" takes 13696 but is fed 27392'
    ]


@pytest.mark.parametrize("svg", sorted((ROOT / "diagrams/generated").glob("*.svg")), ids=lambda p: p.stem)
def test_every_approved_diagram_passes_dimensions_connect(svg):
    assert diagram_checks.svg_dimension_problems(diagram_checks.parse(svg.read_text())) == []


def test_split_and_multiply_explain_width_changes_and_per_head_norms_keep_width():
    nodes = {
        "in": "input (4096)",
        "gu": "gate_up: Linear 4096 -> 27392",
        "split": "split: 2 x 13696",
        "act": "SiLU",
        "mul": "multiply",
        "down": "down: Linear 13696 -> 4096",
        "out": "output (4096)",
    }
    edges = [
        ("in", "gu"),
        ("gu", "split"),
        ("split", "act"),
        ("act", "mul"),
        ("split", "mul"),
        ("mul", "down"),
        ("down", "out"),
    ]
    assert diagram_checks.dimension_problems(nodes, edges) == []
    bad = dict(nodes, split="split: 2 x 12000")
    assert diagram_checks.dimension_problems(bad, edges)[0].startswith('"split: 2 x 12000" splits 24000 but is fed')
    heads = {
        "in": "input (1152)",
        "q": "q: Linear 1152 -> 1024",
        "n": "q norm (256)",
        "o": "output projection: 1024 -> 1152",
    }
    assert diagram_checks.dimension_problems(heads, [("in", "q"), ("q", "n"), ("n", "o")]) == []


def test_fused_gate_up_specs_draw_an_explicit_split():
    spec = diagram_spec.build_spec(json.loads((ROOT / "data/models/zai-org__GLM-4-9B-0414.json").read_text()))
    ffn = next(p for k, p in spec["zoom_panels"].items() if "ffn" in k)
    labels = [n["label"] for n in ffn["nodes"]]
    assert "gate_up: Linear 4096 -> 27392" in labels and "split: 2 x 13696" in labels and "multiply" in labels
    assert diagram_checks.spec_dimension_problems(spec) == []


def test_a_spec_whose_widths_do_not_connect_costs_no_call(tmp_path, monkeypatch):
    root = root_with(tmp_path, READY)
    log = []
    monkeypatch.setattr(queue.diagram_gen, "run_model", runner({}, log))
    monkeypatch.setattr(queue, "spec_dimension_problems", lambda spec: ['ffn: "down" takes 1 but is fed 2'])
    summary = queue.run(root, CONFIG, client=None, today=date(2026, 11, 26))
    assert log == [] and summary["calls"] == 0 and all("do not connect" in s["reason"] for s in summary["skipped"])


# ---------------------------------------------------------------- timeouts


def test_a_timeout_gets_the_same_single_logged_retry(tmp_path, monkeypatch):
    import socket

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    answers = [urllib.error.URLError(socket.timeout("timed out")), None]
    waits = []

    def opener(request, timeout):
        answer = answers.pop(0)
        if answer is None:
            return Ok()
        raise answer

    clock = iter(range(1000, 100_000, 100))
    c = Gemini(CONFIG, tmp_path / "ledger.json", opener=opener, clock=lambda: next(clock), wait=waits.append)
    assert c.generate("S", "P", tag={"for": "org/m"})[0] == "<svg/>"
    entries = ledger(tmp_path)
    assert len(entries) == 2 and entries[1]["retry"] == "after a timeout" and gemini.RETRY_WAIT_SECONDS in waits
    answers[:] = [TimeoutError("timed out"), TimeoutError("timed out")]
    with pytest.raises(ProviderError) as error:
        c.generate("S", "P")
    assert error.value.status == "timeout" and len(ledger(tmp_path)) == 4  # one retry, then the run stops


# ---------------------------------------------------------------- spec template versions


def test_v2_template_shows_exact_vocabulary_and_title_rounding_v1_is_unchanged():
    record = json.loads((ROOT / "data/models/deepseek-ai__DeepSeek-V2-Lite.json").read_text())
    v1, v2 = diagram_spec.build_spec(record, "v1"), diagram_spec.build_spec(record, "v2")
    assert diagram_spec.CURRENT_TEMPLATE == "v2"
    assert v1["model"]["summary_line"].startswith("16B parameters") and "Vocabulary: 100K" in diagram_spec.dump(
        v1, record
    )
    assert v2["model"]["summary_line"].startswith("15.7B parameters") and v2["model"]["title_label"].endswith("(15.7B)")
    assert "Vocabulary: 102,400" in diagram_spec.dump(v2, record) and "100K" not in v2["model"]["vocabulary"]


def test_approved_diagrams_keep_their_template_and_are_not_requeued(monkeypatch):
    monkeypatch.setattr(queue, "published_for", lambda root, record: generated.select(record, root))  # undo the stub
    for meta in sorted((ROOT / "diagrams/generated").glob("*.json")):
        record = json.loads((ROOT / "data/models" / meta.name).read_text())
        assert generated.select(record, ROOT) is not None, meta.stem  # still selected under its own template
    plan, _ = queue.plan(
        ROOT, queue.load_state(ROOT / CONFIG["schedule"]["diagrams"]["state"]), CONFIG["schedule"]["diagrams"]
    )
    approved = {json.loads(m.read_text())["model"] for m in (ROOT / "diagrams/generated").glob("*.json")}
    assert not approved & {q["model"] for q in plan}


def test_bot_prs_use_bot_pat_only_for_create_pull_request():
    for name in ("daily.yml", "diagrams.yml"):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        steps = [s for job in workflow["jobs"].values() for s in job["steps"]]
        for step in steps:
            uses_pat = "BOT_PAT" in json.dumps(step)
            if "create-pull-request" in step.get("uses", ""):
                assert step["with"]["token"] == "${{ secrets.BOT_PAT || github.token }}", name  # falls back if missing
                bot = "archwatch-bot <41898282+github-actions[bot]@users.noreply.github.com>"
                assert step["with"]["author"] == step["with"]["committer"] == bot, name
            else:
                assert not uses_pat, (name, step.get("name") or step.get("uses"))
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        if path.name not in ("daily.yml", "diagrams.yml"):
            assert "BOT_PAT" not in path.read_text(), path.name
