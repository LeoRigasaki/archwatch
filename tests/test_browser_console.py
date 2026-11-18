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


KINDS = ("home", "model", "compare tool", "compare pair", "compare three-way")


def path_for(kind):
    """Resolved at test time: site/ is built by the session fixture, after collection."""
    import json

    if kind == "home":
        return "/index.html"
    if kind == "model":
        return "/models/" + sorted((SITE / "models").glob("*.html"))[0].name
    if kind == "compare tool":
        return "/compare/index.html"
    if kind == "compare pair":
        return "/compare/" + sorted(p.name for p in (SITE / "compare").glob("*.html") if p.name != "index.html")[0]
    stems = [m["stem"] for m in json.loads((SITE / "data/models.json").read_text())["models"]]
    return f"/compare/index.html?a={stems[0]}&b={stems[1]}&c={stems[2]}"


@pytest.mark.parametrize("name", KINDS)
def test_page_loads_without_console_or_page_errors(name, server, tmp_path):
    dom, console = load(server + path_for(name), tmp_path)
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


HARNESS = """<!doctype html><meta charset="utf-8"><pre id="out"></pre>
<iframe id="f" src="/index.html" style="width:1280px;height:800px"></iframe>
<script>
document.getElementById('f').addEventListener('load', function () {
  setTimeout(function () {
    var d = document.getElementById('f').contentDocument, sel = d.getElementById('sort'), out = {};
    ['small', 'big', 'name', 'new'].forEach(function (key) {
      sel.value = key; sel.dispatchEvent(new Event('change'));
      var names = function (q) { return [].map.call(d.querySelectorAll(q), function (a) { return a.textContent; }); };
      out[key] = {cards: names('.cards > li:not([hidden]) h3 a'), table: names('.tableview tbody tr:not([hidden]) th a')};
    });
    document.getElementById('out').textContent = JSON.stringify(out);
  }, 500);
});
</script>"""


def test_table_view_follows_the_sort_control(server, tmp_path):
    import html
    import json
    import re

    harness = SITE / "_sort_harness.html"
    harness.write_text(HARNESS)
    try:
        dom, console = load(server + "/_sort_harness.html", tmp_path)
    finally:
        harness.unlink()
    assert not console, console
    result = json.loads(html.unescape(re.search(r'<pre id="out">(.*?)</pre>', dom, re.S).group(1)))
    for key, order in result.items():
        assert order["cards"] and order["table"] == order["cards"], key
    assert result["small"]["table"] == list(reversed(result["big"]["table"]))
    assert result["name"]["table"] == sorted(result["name"]["table"], key=str.lower)
