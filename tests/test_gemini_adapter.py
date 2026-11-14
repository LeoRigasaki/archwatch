import copy
import io
import json
import urllib.error
from pathlib import Path

import pytest
import yaml

from archwatch.gemini import Gemini, ProviderError, local_env

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / "config/providers.yaml").read_text())


def test_api_key_is_header_only_and_failure_is_reserved_without_leaking(tmp_path, monkeypatch):
    cfg = config()
    key = "fake-private-key-for-test"
    monkeypatch.setenv(cfg["key_env"], key)
    calls = []

    def failing(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 429, key, None, None)

    ledger = tmp_path / "usage.json"
    client = Gemini(cfg, ledger, opener=failing)
    with pytest.raises(ProviderError) as error:
        client.generate("facts only")
    assert key not in str(error.value) and key not in calls[0].full_url
    assert calls[0].get_header("X-goog-api-key") == key
    assert len(json.loads(ledger.read_text())) == 1


def test_caps_stop_before_transport_and_never_switch_models(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setenv(cfg["key_env"], "fake-key")
    cfg["route_test"]["max_generation_requests"] = 1
    now = [1794600000.0]
    calls = []

    def wait(seconds):
        now[0] += seconds

    def respond(request, **kwargs):
        calls.append(request)
        return io.BytesIO(
            json.dumps(
                dict(
                    candidates=[dict(content=dict(parts=[dict(text="{}")]))], usageMetadata=dict(candidatesTokenCount=2)
                )
            ).encode()
        )

    client = Gemini(cfg, tmp_path / "usage.json", clock=lambda: now[0], wait=wait, opener=respond)
    client.generate("facts only", json_output=True)
    with pytest.raises(ProviderError, match="cap reached"):
        client.generate("again")
    assert len(calls) == 1
    payload = json.loads(calls[0].data)
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert cfg["models"]["diagram"]["id"] in calls[0].full_url


def test_nonfree_model_endpoint_missing_key_and_excess_token_budget_rejected(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.delenv(cfg["key_env"], raising=False)
    with pytest.raises(ProviderError, match="Missing"):
        Gemini(cfg, tmp_path / "usage.json")
    monkeypatch.setenv(cfg["key_env"], "fake-key")
    paid = copy.deepcopy(cfg)
    paid["models"]["diagram"]["free"] = False
    with pytest.raises(ProviderError, match="free"):
        Gemini(paid, tmp_path / "usage.json")
    other = copy.deepcopy(cfg)
    other["endpoint"] = "https://example.invalid"
    with pytest.raises(ProviderError, match="endpoint"):
        Gemini(other, tmp_path / "usage.json")
    client = Gemini(cfg, tmp_path / "usage.json")
    with pytest.raises(ProviderError, match="token bound"):
        client.generate("x" * 250000)
    assert not (tmp_path / "usage.json").exists()


def test_local_secrets_do_not_override_actions_environment(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=local-fake\n")
    monkeypatch.setenv("GEMINI_API_KEY", "environment-fake")
    local_env(path)
    assert __import__("os").environ["GEMINI_API_KEY"] == "environment-fake"


@pytest.mark.parametrize("limit", ["daily", "weekly"])
def test_daily_and_weekly_caps_are_enforced_independently(tmp_path, monkeypatch, limit):
    cfg = config()
    monkeypatch.setenv(cfg["key_env"], "fake-key")
    if limit == "daily":
        cfg["models"]["diagram"]["rpd"] = 1
    else:
        cfg["weekly_generation_request_cap"] = 1
    now = [1794600000.0]

    def wait(seconds):
        now[0] += seconds

    def response(*args, **kwargs):
        return io.BytesIO(b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}')

    client = Gemini(cfg, tmp_path / "usage.json", clock=lambda: now[0], wait=wait, opener=response)
    client.generate("one")
    with pytest.raises(ProviderError, match=limit):
        client.generate("two")
