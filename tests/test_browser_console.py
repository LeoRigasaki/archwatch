"""Every page type loads in a real headless browser with no console error and no uncaught page error."""

import functools
import http.server
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def chrome():
    for candidate in (
        os.environ.get("CHROME"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        MAC_CHROME if Path(MAC_CHROME).exists() else None,
    ):
        if candidate:
            return candidate
    return None


@pytest.fixture(scope="module")
def server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    handler.log_message = lambda *a, **k: None
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def load(url, tmp_path):
    binary = chrome()
    if not binary:
        pytest.skip("no Chrome/Chromium available")
    args = [
        binary,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        f"--user-data-dir={tmp_path}",
        "--enable-logging=stderr",
        "--v=0",
        "--virtual-time-budget=8000",
        "--dump-dom",
        url,
    ]
    if sys.platform.startswith("linux"):
        args.insert(1, "--no-sandbox")
    out = subprocess.run(args, capture_output=True, text=True, timeout=120)
    console = [line for line in out.stderr.splitlines() if ":CONSOLE" in line]
    return out.stdout, console


def pages():
    model = sorted((SITE / "models").glob("*.html"))[0].name
    pair = sorted(p.name for p in (SITE / "compare").glob("*.html") if p.name != "index.html")[0]
    import json

    stems = [m["stem"] for m in json.loads((SITE / "data/models.json").read_text())["models"]]
    return [
        ("home", "/index.html"),
        ("model", f"/models/{model}"),
        ("compare tool", "/compare/index.html"),
        ("compare pair", f"/compare/{pair}"),
        ("compare three-way", f"/compare/index.html?a={stems[0]}&b={stems[1]}&c={stems[2]}"),
    ]


@pytest.mark.parametrize("name,path", pages())
def test_page_loads_without_console_or_page_errors(name, path, server, tmp_path):
    dom, console = load(server + path, tmp_path)
    assert not console, f"{name}: {console}"
    assert "</html>" in dom
    if name.startswith("compare"):
        # the tool rendered: visible, with key differences, strips, chart and both field tables
        assert '<div id="tool" class="tool"' in dom and 'id="tool" class="tool" hidden' not in dom
        assert "One sentence per field" in dom and "<svg" in dom and "All fields" in dom
    if name == "compare three-way":
        context = [line for line in dom.split("<li>") if line.startswith("Context: ")]
        assert (
            len(context) == 1 and context[0].count(" tokens") == 0 and context[0].count(",") >= 2
        )  # one sentence, all models
