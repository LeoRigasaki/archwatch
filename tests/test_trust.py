"""Dates are labelled for what they are, and every fact links to the file it was read from."""

import importlib.util
import json
import re
from pathlib import Path

from archwatch.card import parse_card, parse_release_date

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def builder():
    spec = importlib.util.spec_from_file_location("site_trust", ROOT / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_card_release_dates_are_explicit_dates_only():
    assert parse_release_date("2026-11-02") == "2026-11-02"
    assert parse_release_date("November 2, 2026") == "2026-11-02"
    assert parse_release_date("2 Nov 2026") == "2026-11-02"
    assert parse_release_date("November 2026") is None and parse_release_date("Q4 2026") is None
    assert parse_card("| Release date | November 2, 2026 |\n")["release_date"]["value"] == "2026-11-02"
    assert "release_date" not in parse_card("| Release date | soon |\n")
    for invalid in ("2026-99-99", "February 29, 2025", "31 April 2026", "0 Nov 2026"):
        assert parse_release_date(invalid) is None
    assert parse_release_date("February 29, 2024") == "2024-02-29"


def test_hub_creation_time_is_stored_apart_and_never_called_a_release():
    for path in (ROOT / "data/models").glob("*.json"):
        rec = json.loads(path.read_text())
        assert "hub_created_at" in rec and "hub_created" not in rec, path
    site = builder()
    published, _, _ = site.inventory()
    for stem, rec, pub in published.values():
        page = (SITE / "models" / f"{stem}.html").read_text()
        label, iso = site.when(rec, pub)
        if pub.get("release_date"):
            assert label == "released" and "(model card)" in page
        else:
            assert label == "added to Hub" and iso == rec["hub_created_at"][:10]
            assert "added to Hub" in page and "released " not in page.split("<main")[1].split("</p>")[0]


def test_a_card_release_date_is_shown_as_a_release():
    site = builder()
    published, _, _ = site.inventory()
    stem, rec, pub = next(iter(published.values()))
    assert site.when(rec, dict(pub, release_date="2026-01-05")) == ("released", "2026-01-05")


def test_every_source_links_to_its_file_at_the_pinned_commit():
    site = builder()
    published, _, _ = site.inventory()
    for stem, rec, pub in published.values():
        page = (SITE / "models" / f"{stem}.html").read_text()
        base = f"https://huggingface.co/{rec['id']}/"
        for href in re.findall(r'<a href="(https://huggingface.co/[^"]+)">', page):
            if href.startswith(base + "blob/"):
                assert f"/blob/{rec['source_commit']}/" in href, href  # pinned, never a branch
        assert f'href="{base}blob/{rec["source_commit"]}/config.json">config.json</a>' in page
    lfm = (SITE / "models/LiquidAI__LFM3-1.7B-Base.html").read_text()
    assert re.search(r'blob/[0-9a-f]{40}/modeling_lfm3\.py#L\d+">modeling_lfm3\.py line \d+</a>', lfm)
    assert site.source_url({"id": "o/m", "source_commit": "c" * 40, "source_files": []}, "derived:x") is None


def test_single_file_weights_link_to_the_pinned_file():
    site = builder()
    rec = dict(id="lab/model", source_commit="a" * 40, source_files=["config.json", "model.safetensors"])
    assert (
        site.source_url(rec, "weights:header") == f"https://huggingface.co/lab/model/blob/{'a' * 40}/model.safetensors"
    )
