import json
import logging
import subprocess
import urllib.error
from pathlib import Path

import yaml

from archwatch.notifications import FAILURE_MESSAGE, annotate_report, format_message, notify, snapshot

ROOT = Path(__file__).resolve().parents[1]
WEBHOOK = "https://discord.com/api/webhooks/123/inert-test-token"
PR = "https://github.com/example/models/pull/12"


def report():
    return dict(
        changed=True,
        discovery_errors=[],
        model_events=[
            dict(id="lab/Ready", status="ready_for_diagram", processed=True, access_blocked=False),
            dict(id="lab/Gated", status="review", processed=False, access_blocked=True),
            dict(id="lab/Conflict", status="review", processed=False, access_blocked=False),
        ],
    )


def test_formats_current_models_access_links_review_and_pr_deterministically():
    value = report()
    content = format_message(value, PR)
    assert "Processed: 1 | Sent to review: 2 | Access blocked: 1" in content
    assert "lab/Gated — [request access here](<https://huggingface.co/lab/Gated>)" in content
    assert "Processed: lab/Ready" in content and "Review: lab/Conflict" in content
    assert f"PR: <{PR}>" in content
    value["model_events"].reverse()
    assert format_message(value, PR) == content


def test_no_changes_sends_nothing_even_with_old_review_items_and_pr(caplog):
    value = report() | dict(changed=False)
    calls = []
    assert format_message(value, PR) is None
    assert not notify(value, PR, environ=dict(DISCORD_WEBHOOK_URL=WEBHOOK), opener=lambda *a, **kw: calls.append(a))
    assert not calls and not caplog.records


def test_missing_secret_is_silent_and_does_not_even_format_invalid_input(caplog):
    calls = []
    assert not notify(None, environ={}, opener=lambda *a, **kw: calls.append(a))
    assert not calls and not caplog.records


def test_posts_one_text_only_message_with_mentions_and_embeds_disabled():
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def send(request, **kwargs):
        calls.append((request, kwargs))
        return Response()

    assert notify(report(), PR, environ=dict(DISCORD_WEBHOOK_URL=WEBHOOK), opener=send)
    assert len(calls) == 1
    request, options = calls[0]
    payload = json.loads(request.data)
    assert request.full_url == WEBHOOK + "?wait=true" and options["timeout"] == 10
    assert payload == dict(content=format_message(report(), PR), allowed_mentions=dict(parse=[]), flags=4)


def test_webhook_error_never_raises_or_retries_and_logs_no_url(caplog):
    calls = []

    def broken(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(WEBHOOK, 429, "private " + WEBHOOK, {}, None)

    with caplog.at_level(logging.WARNING):
        assert not notify(report(), environ=dict(DISCORD_WEBHOOK_URL=WEBHOOK), opener=broken)
    assert len(calls) == 1
    assert [r.getMessage() for r in caplog.records] == [FAILURE_MESSAGE]
    assert WEBHOOK not in caplog.text and "inert-test-token" not in caplog.text


def test_large_discovery_keeps_one_bounded_message_and_pr_link():
    value = report()
    value["model_events"] *= 200
    content = format_message(value, PR)
    assert len(content) <= 2000 and content.endswith(f"PR: <{PR}>")
    assert "more model details" in content


def test_snapshot_reports_real_changes_once_excluding_cache_report_and_mtimes(tmp_path):
    state = tmp_path / "data/seen.json"
    state.parent.mkdir()
    state.write_text(json.dumps(dict(models={"lab/New": dict(status="pending")})))
    before = snapshot(tmp_path)
    state.write_text(json.dumps(dict(models={"lab/New": dict(status="review", reason="GatedRepoError (HTTP 403)")})))
    run = dict(outcomes={"lab/New": "review"}, discovery_errors=[], processed=0)
    change = annotate_report(tmp_path, run, before)
    assert change["changed"]
    assert change["model_events"] == [dict(id="lab/New", status="review", processed=False, access_blocked=True)]
    saved = snapshot(tmp_path)
    (tmp_path / "run-report.json").write_text("{}")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/new-fetch.json").write_text("{}")
    state.touch()
    repeated = annotate_report(tmp_path, run, saved)
    assert not repeated["changed"] and repeated["model_events"] == []
    assert format_message(repeated) is None


def test_record_change_not_hidden_by_unchanged_seen_status(tmp_path):
    folder = tmp_path / "data/models"
    folder.mkdir(parents=True)
    path = folder / "lab__Ready.json"
    path.write_text('{"id":"lab/Ready","published":{"num_layers":2}}')
    before = snapshot(tmp_path)
    path.write_text('{"id":"lab/Ready","published":{"num_layers":3}}')
    change = annotate_report(tmp_path, dict(outcomes={"lab/Ready": "recorded"}), before)
    assert change["changed"] and change["model_events"][0]["processed"]


def test_cli_without_secret_succeeds_silently_even_without_report(tmp_path):
    import os

    env = dict(os.environ)
    env.pop("DISCORD_WEBHOOK_URL", None)
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/notify_discord.py")], env=env, capture_output=True
    )
    assert result.returncode == 0 and result.stdout == b"" and result.stderr == b""


def test_cli_unreadable_report_is_one_redacted_line_and_success(tmp_path):
    import os

    env = dict(os.environ, DISCORD_WEBHOOK_URL=WEBHOOK)
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/notify_discord.py"), "--report", str(tmp_path / "absent")],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0 and result.stdout == b""
    assert result.stderr.decode().strip() == "WARNING:root:" + FAILURE_MESSAGE
    assert WEBHOOK.encode() not in result.stderr


def test_watcher_cli_writes_ephemeral_change_report_then_no_change(tmp_path, monkeypatch, capsys):
    import sys

    monkeypatch.syspath_prepend(str(ROOT))
    import watch

    state = tmp_path / "data/seen.json"
    state.parent.mkdir()

    def run(*args, **kwargs):
        if not state.exists():
            state.write_text(json.dumps(dict(models={"lab/New": dict(status="ready_for_diagram")})))
            return dict(outcomes={"lab/New": "ready_for_diagram"}, discovery_errors=[], processed=1)
        return dict(outcomes={"lab/New": "unchanged"}, discovery_errors=[], processed=0)

    monkeypatch.setattr(watch, "run", run)
    path = tmp_path / "ephemeral.json"
    monkeypatch.setattr(sys, "argv", ["watch.py", "--root", str(tmp_path), "--report", str(path)])
    watch.main()
    first = json.loads(path.read_text())
    assert first["changed"] and first["model_events"][0]["processed"]
    watch.main()
    assert not json.loads(path.read_text())["changed"]
    assert len(capsys.readouterr().out) > 0


def test_workflow_secret_is_scoped_to_notification_step_and_cron_stays_disabled():
    workflow = yaml.safe_load((ROOT / ".github/workflows/daily.yml").read_text())
    assert "DISCORD_WEBHOOK_URL" not in workflow.get("env", {})
    steps = workflow["jobs"]["run"]["steps"]
    assert "DISCORD_WEBHOOK_URL" not in workflow["jobs"]["run"].get("env", {})
    holders = [s for s in steps if "DISCORD_WEBHOOK_URL" in s.get("env", {})]
    assert len(holders) == 1 and holders[0]["name"] == "Optional Discord watcher summary"
    assert holders[0]["continue-on-error"] is True
    assert steps.index(holders[0]) > next(i for i, s in enumerate(steps) if s.get("id") == "updates")
    assert "schedule" not in workflow.get("on", workflow.get(True, {}))
