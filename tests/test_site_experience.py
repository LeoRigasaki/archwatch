"""Public downloads, guide membership and real user flows retain the publication boundary."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_browser_console import chrome, server  # noqa: F401
from test_site import builder

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_guide_membership_and_downloads_use_only_published_models():
    site = builder()
    published, held, _ = site.inventory()
    exports = {p.stem: json.loads(p.read_text()) for p in (SITE / "data/models").glob("*.json")}
    assert set(exports) == {stem for stem, _, _ in published.values()}
    for stem, rec, pub in published.values():
        data = exports[stem]
        assert data["published"] == pub
        assert set(data["evidence"]) <= set(pub)
        assert "facts" not in data and "candidates" not in json.dumps(data)
        for evidence in data["evidence"].values():
            assert all(rec["source_commit"] in url for url in evidence["sources"])
        for key in site.concept_keys(pub):
            guide = (SITE / "concepts" / (key + ".html")).read_text()
            assert f"../models/{stem}.html" in guide
    for repo in held:
        assert repo.replace("/", "__") not in exports
    gqa = (SITE / "concepts/gqa.html").read_text()
    assert "DeepSeek-V2-Lite.html" not in gqa  # MLA is not GQA despite having head counts
    assert "DeepSeek-V2-Lite.html" in (SITE / "concepts/mla.html").read_text()


def test_feed_uses_analysis_times_and_contains_only_published_models():
    site = builder()
    published, _, _ = site.inventory()
    items = ET.parse(SITE / "feed.xml").findall("./channel/item")
    assert len(items) == len(published)
    assert {item.findtext("title") for item in items} == {rec["id"].split("/")[-1] for _, rec, _ in published.values()}
    for item in items:
        assert item.findtext("pubDate") and item.findtext("guid").split("#")[-1] in {
            rec["source_commit"] for _, rec, _ in published.values()
        }


def test_real_browser_selection_mobile_controls_planner_and_downloads(server, tmp_path):  # noqa: F811
    api = pytest.importorskip("playwright.sync_api")
    binary = chrome()
    if not binary:
        pytest.skip("no Chrome/Chromium available")
    with api.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=binary, headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")
        page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(server)
        initial_count = page.locator(".shown").inner_text()
        assert not page.locator("#filter-panel").evaluate("el => el.open")
        assert page.locator(".cards>li").first.bounding_box()["y"] < 900
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.get_by_role("button", name="Menu", exact=True).click()
        assert page.locator("#main-nav").is_visible()
        page.keyboard.press("Escape")
        assert not page.locator("#main-nav").is_visible()
        page.locator("#q").fill("definitely absent model")
        assert page.locator(".empty").is_visible() and page.locator(".cards>li:visible").count() == 0
        page.locator("#q").fill("")
        page.locator("#filter-panel summary").click()
        page.locator("#publisher").select_option("LiquidAI")
        assert page.locator(".cards>li:visible").count() == 2
        page.get_by_role("button", name="Reset filters").click()
        page.wait_for_function("n => document.querySelector('.shown').textContent === n", arg=initial_count)
        for i in range(3):
            page.locator(".compare-pick").nth(i).click()
        page.locator(".compare-pick").nth(3).click()
        assert page.locator(".compare-pick[aria-pressed=true]").count() == 3
        assert "up to 3" in page.locator("#compare-selection").inner_text()
        page.locator("#compare-selection-link").click()
        page.locator("#fields table").first.wait_for()
        assert page.locator("#fields table").first.locator("thead th").count() == 5
        page.evaluate(
            "Object.defineProperty(navigator, 'clipboard', {value:{writeText:()=>Promise.reject(Error('blocked'))}, configurable:true})"
        )
        page.locator("#copylink").click()
        page.wait_for_function("document.getElementById('copied').textContent.startsWith('Could not copy')")
        with page.expect_download() as download:
            page.locator("#export-csv").click()
        csv_path = tmp_path / "comparison.csv"
        download.value.save_as(csv_path)
        csv = csv_path.read_text()
        assert "Source commit" in csv and "Pinned primary files" in csv
        page.goto(server + "/models/LiquidAI__LFM3-1.7B-Base.html")
        page.locator('[data-prec="4"]').click()
        weights = page.locator("[data-out=weights]").inner_text()
        total = page.locator("[data-out=total]").inner_text()
        page.locator("#sequences").fill("4")
        assert page.locator("[data-out=weights]").inner_text() == weights
        assert page.locator("[data-out=total]").inner_text() != total
        page.locator("#device-memory").fill("1")
        assert page.locator("#budget-result").inner_text().startswith("Exceeds")
        page.locator("#sequences").fill("-1")
        assert page.locator("#budget-result").inner_text().startswith("Enter valid")
        page.locator("#expand-diagram").click()
        assert page.locator("#diagram-dialog").is_visible()
        page.keyboard.press("Escape")
        assert not page.locator("#diagram-dialog").is_visible()
        with page.expect_download() as download:
            page.locator("#download-diagram").click()
        svg_path = tmp_path / "diagram.svg"
        download.value.save_as(svg_path)
        assert ET.parse(svg_path).getroot().tag.endswith("svg")
        for path in ["/concepts/index.html", "/concepts/gqa.html", "/review/index.html"]:
            page.goto(server + path)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), path
        page.goto(server + "/concepts/index.html")
        page.locator("#concept-search").fill("RMSNorm")
        assert page.locator(".guide-grid>li:visible").count() == 1
        page.locator("#concept-search").fill("")
        page.locator("#concept-category").select_option("Residual connections")
        assert page.locator(".guide-grid>li:visible").count() == 3
        page.goto(server + "/models/LiquidAI__LFM3-1.7B-Base.html#concept-evidence")
        page.locator("#concept-rmsnorm .proof-hint").click()
        assert page.locator("#concept-rmsnorm").evaluate("el => el.open")
        assert page.locator('#concept-rmsnorm a[href*="modeling_lfm3.py#L65"]').count() == 2
        with page.expect_download() as download:
            page.get_by_role("link", name="Download concept evidence").click()
        payload = tmp_path / "concept-evidence.json"
        download.value.save_as(payload)
        evidence = json.loads(payload.read_text())
        assert evidence["schema_version"] == "archwatch-concept-evidence/1"
        rms = next(hit for hit in evidence["concepts"] if hit["key"] == "rmsnorm")
        assert rms["status"] == "single source" and "decoder" in rms["scope"].lower()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        browser.close()
