import io
from types import SimpleNamespace
from urllib.request import Request

import pytest

from archwatch import hub


@pytest.mark.parametrize("target", ["https://cdn.example/weights", "http://huggingface.co/file"])
def test_source_redirect_never_forwards_credentials_off_host_or_to_http(target):
    req = Request(
        "https://huggingface.co/lab/model", headers={"Authorization": "Bearer test-only", "Range": "bytes=0-7"}
    )
    redirected = hub._SourceRedirect().redirect_request(req, None, 302, "Found", {}, target)
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Range") == "bytes=0-7"


def test_authenticated_header_read_remains_bounded(monkeypatch):
    class Response(io.BytesIO):
        status = 206

    captured = []

    def opener(request, timeout):
        captured.append(request)
        return Response(b"12345678")

    monkeypatch.setattr(hub, "get_token", lambda: "test-only")
    monkeypatch.setattr(hub.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=opener))
    assert (
        hub._get("https://huggingface.co/lab/model/resolve/rev/file.safetensors", {"Range": "bytes=0-7"}) == b"12345678"
    )
    assert captured[0].get_header("Authorization") == "Bearer test-only"
    assert captured[0].get_header("Range") == "bytes=0-7"
