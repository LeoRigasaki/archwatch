"""Worked examples are educational data; model facts remain an independent evidence layer."""

import contextlib
import io
import json
import re
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
        assert len(lesson["example"]["steps"]) >= 3 and 1 <= len(lesson["pitfalls"]) <= 3
        assert set(lesson["related"]) <= set(lessons)
        examples.append(json.dumps(lesson["example"], sort_keys=True))
        page = (SITE / "concepts" / f"{key}.html").read_text()
        for anchor in ["why", "worked-example", "equation", "pitfalls", "verify", "examples"]:
            assert f'id="{anchor}"' in page
        if lesson["example"].get("model"):
            assert "Computed when the site is built, from the published record of" in page
        else:
            assert "yet, so this example uses toy numbers." in page
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
    site = builder()
    published, _, _ = site.inventory()
    gqa = site.concept_examples.worked(lessons["gqa"]["example"], published)
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(gqa["code"], scope)
    assert scope["cache_per_token"] == gqa["checks"][0][2]  # the calculator's value for that model
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
        assert page.locator("[data-toy=bytes]").inner_text() == "16.00 GiB"  # Llama-3.1-8B at 131,072 tokens
        assert page.locator("[data-toy=type]").inner_text() == "GQA"
        page.get_by_role("button", name="Reset to toy values").click()
        assert page.locator("[data-lab=toy-note]").is_visible()
        assert page.locator("[data-toy=bytes]").inner_text() == "4.00 MiB"
        page.locator("[name=toy-context]").fill("32768")
        assert page.locator("[data-toy=bytes]").inner_text() == "32.00 MiB"
        page.locator("[name=toy-kv]").select_option("8")
        assert page.locator("[data-toy=bytes]").inner_text() == "128.00 MiB"
        page.locator("[name=toy-window]").select_option("512")
        assert page.locator("[data-toy=bytes]").inner_text() == "2.00 MiB"
        page.locator("[name=toy-kv]").select_option("2")
        assert page.locator("[data-toy=bytes]").inner_text() == "512.00 KiB"
        page.get_by_role("button", name="Use Llama-3.1-8B values").click()
        assert page.locator("[data-toy=bytes]").inner_text() == "16.00 GiB"
        assert not page.locator("[data-lab=toy-note]").is_visible()
        page.goto(server + "/concepts/sliding-window.html")
        assert page.locator("[data-toy=bytes]").inner_text() == "512.00 MiB"  # Mistral-7B capped by its window
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.goto(server + "/concepts/rmsnorm.html")
        assert "0.8485, 1.1314" in page.locator(".norm-lab output").inner_text()
        page.locator(".norm-lab input").fill("not a vector")
        assert page.locator(".norm-lab output").inner_text().startswith("Enter 1–16")
        assert not errors, errors
        browser.close()


HEDGES = (
    r"\b(may|might|could|perhaps|possibly)\b|depends on the implementation|according to the implementation"
    r"|implementation-specific|model-specific|not necessarily|real conventions differ|can be wrong"
)


def published_hits():
    from archwatch import concepts

    published, _, _ = builder().inventory()
    hits = {}
    for _, rec, _ in published.values():
        for match in concepts.for_record(rec):
            hits.setdefault(match["key"], set()).add(rec["id"])
    return published, hits


def test_a_guide_uses_a_real_model_exactly_when_a_published_model_has_the_concept():
    published, hits = published_hits()
    for key, lesson in builder().lessons().items():
        model = lesson["example"].get("model")
        if key in hits:
            assert model in hits[key], (key, model)  # the example model is one the guide lists
        else:
            assert model is None, key


def test_worked_examples_equal_the_calculator_and_the_record():
    site = builder()
    published, _ = published_hits()
    backed = 0
    for key, lesson in site.lessons().items():
        example = lesson["example"]
        if not example.get("model"):
            continue
        backed += 1
        worked = site.concept_examples.worked(example, published)
        assert worked["checks"], key
        for label, ours, reference in worked["checks"]:
            assert ours == reference, (key, label, ours, reference)
        page = (SITE / "concepts" / f"{key}.html").read_text()
        for text in [worked["setup"], worked["result"], *worked["steps"]]:
            assert site.e(text) in page, (key, text)
        stem, _, _ = published[example["model"]]
        assert f'href="../models/{stem}.html"' in page
        assert "{" not in worked["result"] + "".join(worked["steps"])
    assert backed == 20


def test_mla_example_matches_deepseek_v2_lite_record():
    site = builder()
    published, _, _ = site.inventory()
    worked = site.concept_examples.worked(site.lessons()["mla"]["example"], published)
    cache = site.render_cache(published["deepseek-ai/DeepSeek-V2-Lite"][2])
    assert worked["values"]["per_token"] == "30.4 KiB" and worked["values"]["per_token_bytes"] == 31104
    assert cache["grows_per_token_bytes"] == 31104


def test_calculators_start_from_the_worked_example_model():
    import html
    import re

    site = builder()
    published, _, _ = site.inventory()
    for key in sorted(site.ATTENTION_LABS):
        model = site.lessons()[key]["example"]["model"]
        values = published[model][2]
        cache = site.render_cache(values)
        page = (SITE / "concepts" / f"{key}.html").read_text()
        presets = json.loads(html.unescape(re.search(r"data-presets='([^']+)'", page).group(1)))
        start = presets["model"]
        assert start["model"] == model.split("/")[-1]
        _, nbytes = site.lab_bytes(start, presets["windowed"])
        expected = (
            cache["constant_bytes"]
            if presets["windowed"]
            else cache["grows_per_token_bytes"] * values["context_length"]
        )
        assert nbytes == expected, key
        assert presets["toy"]["model"] == "toy decoder" and "Reset to toy values" in page


def test_pitfalls_are_specific_and_no_guide_hedges():
    import yaml

    concepts = yaml.safe_load((ROOT / "config/concepts.yaml").read_text())
    lessons = builder().lessons()
    assert set(concepts) == set(lessons) and len(concepts) == 32
    for key in concepts:
        assert 1 <= len(lessons[key]["pitfalls"]) <= 3, key
        text = json.dumps([concepts[key], lessons[key]], ensure_ascii=False)
        assert not re.search(HEDGES, text, re.I), (key, re.search(HEDGES, text, re.I).group(0))
        assert concepts[key]["reviewed"] is False
