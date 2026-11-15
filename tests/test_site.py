"""The static site: published records only, held records by name and plain reason, numbers from our calculator."""

import importlib.util
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def builder():
    spec = importlib.util.spec_from_file_location("site_builder_tests", ROOT / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pages():
    return {p: p.read_text() for p in SITE.rglob("*.html")}


def test_three_page_types_exist_for_published_records_only():
    site = builder()
    published, held, _ = site.inventory()
    models = {p.stem for p in (SITE / "models").glob("*.html")}
    assert models == {stem for stem, _, _ in published.values()}
    for repo in held:
        assert not (SITE / "models" / (repo.replace("/", "__") + ".html")).exists()
    compares = {p.stem for p in (SITE / "compare").glob("*.html")} - {"index"}
    expected = {stem for stem, rec, _ in published.values() if site.published_predecessor(rec, published)}
    assert compares == expected and expected  # a compare page only when both models are published


def test_held_records_show_no_numbers():
    site = builder()
    published, held, _ = site.inventory()
    shared = {pub.get("params_total") for _, _, pub in published.values()}  # e.g. a fine-tune of a published base
    home = (SITE / "index.html").read_text()
    rows = re.findall(r"<tr><th scope=row>([^<]+)</th><td>([^<]+)</td></tr>", home)
    assert {name for name, _ in rows} == {repo.split("/")[-1] for repo in held}
    for _, reason in rows:
        assert not re.search(r"\d", reason), reason  # plain reasons carry no numbers
    for repo in held:
        path = ROOT / "data/models" / (repo.replace("/", "__") + ".json")
        if path.exists():
            total = json.loads(path.read_text())["published"].get("params_total")
            if total and total not in shared:
                for page in pages().values():
                    assert f"{total:,}" not in page


def test_site_never_mentions_the_audit_or_its_reference():
    for path, page in pages().items():
        low = page.lower()
        assert "audit" not in low and "owner" not in low and "external" not in low, path


def test_memory_picker_values_come_from_our_calculator():
    site = builder()
    published, _, _ = site.inventory()
    stem, rec, pub = published["moonshotai/Kimi-K3.1-Pro"]
    cache = site.render_cache(pub)
    page = (SITE / "models" / f"{stem}.html").read_text()
    for n in (8192, 131072, int(pub["context_length"])):
        expected = site.size(cache["grows_per_token_bytes"] * n + cache["constant_bytes"])
        assert f'data-bytes="{expected}" data-tokens="{n:,}"' in page


def test_fact_statuses_use_plain_words():
    page = (SITE / "models" / "moonshotai__Kimi-K3.1-Pro.html").read_text()
    assert "st-one-source" in page and "single source" not in page
    assert "st-computed" in page  # cache per token, from our calculator


def test_pages_have_basic_accessible_markup():
    for path, page in pages().items():
        assert '<html lang="en">' in page and 'class="skip"' in page and '<main id="main"' in page, path
        for img in re.findall(r"<img [^>]*>", page):
            assert 'alt="' in img and 'alt=""' not in img, path
        assert "<th>" not in page.split("<tbody>")[-1], path  # body header cells carry a scope


def test_style_matches_the_brief():
    css = (SITE / "style.css").read_text()
    for token in ("#F7F7F5", "#111418", "#0B6E66", "Space Grotesk", "IBM Plex Mono", "Inter"):
        assert token in css
    js = (SITE / "site.js").read_text()
    assert "import" not in js and "require(" not in js  # no framework


def test_site_settings_hold_no_numbers_for_models():
    config = yaml.safe_load((ROOT / "config/site.yaml").read_text())
    assert set(config) == {"repo_url", "publishers", "memory_contexts"}


def test_pages_workflow_publishes_site_from_main_after_the_daily_run_and_is_gated():
    flow = yaml.safe_load((ROOT / ".github/workflows/pages.yml").read_text())
    on = flow[True]
    assert on["workflow_run"]["workflows"] == ["daily"] and on["workflow_run"]["types"] == ["completed"]
    job = flow["jobs"]["deploy"]
    assert "vars.PAGES_ENABLED == 'true'" in job["if"]
    text = (ROOT / ".github/workflows/pages.yml").read_text()
    assert "actions/upload-pages-artifact" in text and "actions/deploy-pages" in text
    assert "path: site" in text and "ref: main" in text and "GEMINI" not in text
