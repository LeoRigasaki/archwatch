"""Worked examples are educational data; model facts remain an independent evidence layer."""

import contextlib
import io
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_browser_console import chrome, server  # noqa: F401
from test_site import builder

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_every_guide_has_an_original_worked_lesson_and_downloadable_figure():
    site = builder()
    lessons = site.lessons()
    assert set(lessons) == set(site.concepts())
    examples = []
    for key, lesson in lessons.items():
        assert lesson["why"] and lesson["equation"] and lesson["symbols"]
        assert len(lesson["example"]["steps"]) >= 3 and len(lesson["pitfalls"]) >= 3
        assert set(lesson["related"]) <= set(lessons)
        examples.append(json.dumps(lesson["example"], sort_keys=True))
        page = (SITE / "concepts" / f"{key}.html").read_text()
        for anchor in ["why", "worked-example", "equation", "pitfalls", "verify", "examples"]:
            assert f'id="{anchor}"' in page
        assert "not a checkpoint" in page or "not a model" in page
        svg = SITE / "concepts/figures" / f"{key}.svg"
        root = ET.parse(svg).getroot()
        assert root.tag.endswith("svg") and root.get("role") == "img"
        assert not any(node.tag.endswith(("script", "image", "foreignObject")) for node in root.iter())
    assert len(set(examples)) == len(lessons)


def test_minimal_python_examples_match_independent_hand_calculations():
    lessons = builder().lessons()
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(lessons["mha"]["code"], scope)
    assert sum(scope["weights"]) == pytest.approx(1)
    assert scope["output"] == pytest.approx([1.3395231, 1.3209538], rel=1e-6)
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(lessons["gqa"]["code"], scope)
    assert scope["cache_bytes"] == 4 * 1024 * 1024
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(lessons["rmsnorm"]["code"], scope)
    assert scope["rms"] == pytest.approx((12.5 + 1e-6) ** 0.5)


def test_compare_mockup_controls_and_teaching_labs_in_chrome(server, tmp_path):  # noqa: F811
    api = pytest.importorskip("playwright.sync_api")
    if not chrome():
        pytest.skip("Chrome not available")
    with api.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chrome(), headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, color_scheme="light")
        page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(server + "/compare/index.html")
        page.locator(".comparison-overview").wait_for()
        assert page.locator(".slot-preview .cells").count() == 2
        summary = page.locator(".comparison-overview").inner_text()
        assert "27 KiB to 33 KiB" in summary
        assert "RoPE" not in summary  # mockup text said removed, but both pinned records state NoPE
        page.get_by_role("button", name="Technical", exact=True).click()
        assert page.locator("#tool").get_attribute("data-reading") == "technical"
        assert not page.locator(".comparison-overview").is_visible()
        page.get_by_role("button", name="Plain", exact=True).click()
        page.locator("#onlydiff").check()
        assert page.locator('#fields tr[data-diff="0"]:visible').count() == 0
        page.locator("#onlydiff").uncheck()
        active = page.locator("#fields table").first.locator("tbody tr").filter(has_text="Active parameters per token")
        assert "104.2B" in active.inner_text() and "104.4B" in active.inner_text()
        page.locator("#swap").click()
        assert "33 KiB to 27 KiB" in page.locator(".comparison-overview").inner_text()
        page.locator("#add-third").click()
        page.locator("#pick-c").fill("Llama-3.1-8B")
        page.locator("#pick-c").dispatch_event("change")
        assert page.locator(".slot-preview .cells").count() == 3
        assert "c=meta-llama" in page.url
        page.locator("#clear-c").click()
        assert page.locator(".slot-preview .cells").count() == 2
        page.locator("#pick-a").fill("does-not-exist")
        page.locator("#pick-a").dispatch_event("change")
        assert "Selection unchanged" in page.locator("#selection-error").inner_text()
        assert page.locator("#cache-summarybars li").count() == 4
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.goto(server + "/concepts/gqa.html")
        assert page.locator("[data-toy=bytes]").inner_text() == "4.00 MiB"
        page.locator("[name=toy-context]").fill("32768")
        assert page.locator("[data-toy=bytes]").inner_text() == "32.00 MiB"
        page.locator("[name=toy-kv]").select_option("8")
        assert page.locator("[data-toy=bytes]").inner_text() == "128.00 MiB"
        page.locator("[name=toy-window]").select_option("512")
        assert page.locator("[data-toy=bytes]").inner_text() == "2.00 MiB"
        page.locator("[name=toy-kv]").select_option("2")
        assert page.locator("[data-toy=bytes]").inner_text() == "512.00 KiB"
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.goto(server + "/concepts/rmsnorm.html")
        assert "0.8485, 1.1314" in page.locator(".norm-lab output").inner_text()
        page.locator(".norm-lab input").fill("not a vector")
        assert page.locator(".norm-lab output").inner_text().startswith("Enter 1–16")
        assert not errors, errors
        browser.close()
