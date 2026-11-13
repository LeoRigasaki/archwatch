"""Revision and byte-range boundaries without contacting the Hub."""

import io
import json
import struct

import pytest

from archwatch import hub


def test_text_fetch_and_cache_use_the_recorded_revision(tmp_path, monkeypatch):
    revision = "a" * 40
    calls = []
    monkeypatch.setattr(hub, "CACHE", tmp_path)
    monkeypatch.setattr(hub, "repo_info", lambda repo: {"sha": revision})
    monkeypatch.setattr(hub, "_get", lambda url: calls.append(url) or b'{"value":1}')
    assert hub.get_json("lab/model", "config.json") == {"value": 1}
    assert hub.get_text("lab/model", "config.json") == '{"value":1}'
    assert calls == [f"https://huggingface.co/lab/model/resolve/{revision}/config.json"]


def test_weight_header_fetches_use_the_recorded_revision(monkeypatch):
    revision = "b" * 40
    payload = json.dumps({"layer.weight": {"dtype": "BF16", "shape": [2, 3]}}).encode()
    calls = []
    monkeypatch.setattr(hub, "repo_info", lambda repo: {"sha": revision})

    def fetch(url, headers):
        calls.append((url, headers))
        return struct.pack("<Q", len(payload)) if headers["Range"] == "bytes=0-7" else payload

    monkeypatch.setattr(hub, "_get", fetch)
    assert hub._header("lab/model", "model.safetensors")["layer.weight"]["shape"] == [2, 3]
    assert all(f"/resolve/{revision}/" in url for url, _ in calls)
    assert calls[1][1]["Range"] == f"bytes=8-{7 + len(payload)}"


def test_ignored_range_is_rejected_before_reading_the_body(monkeypatch):
    class WholeBodyResponse(io.BytesIO):
        status = 200

        def read(self, *args):
            raise AssertionError("The full model-weight response must not be read")

    monkeypatch.setattr(hub.urllib.request, "urlopen", lambda *a, **k: WholeBodyResponse())
    with pytest.raises(ValueError, match="ignored the byte range"):
        hub._get("https://huggingface.co/lab/model/resolve/revision/model.safetensors", {"Range": "bytes=0-7"})


def test_partial_response_read_is_bounded(monkeypatch):
    class PartialResponse(io.BytesIO):
        status = 206

    monkeypatch.setattr(hub.urllib.request, "urlopen", lambda *a, **k: PartialResponse(b"12345678"))
    assert hub._get("https://huggingface.co/model.safetensors", {"Range": "bytes=0-7"}) == b"12345678"


def test_oversized_header_is_rejected_before_requesting_it(monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "repo_info", lambda repo: {"sha": "c" * 40})

    def fetch(url, headers):
        calls.append(headers)
        return struct.pack("<Q", hub.MAX_HEADER_BYTES + 1)

    monkeypatch.setattr(hub, "_get", fetch)
    with pytest.raises(ValueError, match="header length"):
        hub._header("lab/model", "model.safetensors")
    assert len(calls) == 1
