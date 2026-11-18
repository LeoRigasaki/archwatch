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
    assert compares == expected == {"moonshotai__Kimi-K3.1-Pro"}


def test_lfm3_has_no_compare_page_and_its_card_describes_the_model_itself():
    home = (SITE / "index.html").read_text()
    card = home.split('id="h-LiquidAI__LFM3-1.7B-Base"')[1].split("</article>")[0]
    assert "Compare with" not in card and "LFM2.5" not in card
    assert "13 Gated short conv layers (kernel 7)" in card
    assert not (SITE / "compare" / "LiquidAI__LFM3-1.7B-Base.html").exists()


def test_held_list_is_collapsed_has_no_numbers_and_no_fine_tunes():
    site = builder()
    published, held, _ = site.inventory()
    home = (SITE / "index.html").read_text()
    box = re.search(
        r'<details class="heldbox"><summary>(\d+) models found but not yet published</summary>(.*?)</details>',
        home,
        re.S,
    )
    assert box and int(box.group(1)) == len(held)
    rows = re.findall(r"<li><b>([^<]+)</b><span>([^<]+)</span></li>", box.group(2))
    assert {name for name, _ in rows} == {repo.split("/")[-1] for repo in held}
    assert not any(re.search(r"\d", reason) for _, reason in rows)
    for name in ("LFM2.5-1.2B-Instruct", "LFM3-1.7B<"):  # fine-tunes are out of scope, not held
        assert name not in home
    shared = {pub.get("params_total") for _, _, pub in published.values()}
    for repo in held:
        path = ROOT / "data/models" / (repo.replace("/", "__") + ".json")
        total = json.loads(path.read_text())["published"].get("params_total") if path.exists() else None
        if total and total not in shared:
            assert all(f"{total:,}" not in page for page in pages().values())


def test_site_never_mentions_the_audit_or_its_reference():
    for path, page in pages().items():
        low = page.lower()
        assert "audit" not in low and "answer sheet" not in low and "reference sheet" not in low, path


def test_memory_planner_values_come_from_our_calculator():
    site = builder()
    published, _, _ = site.inventory()
    stem, rec, pub = published["moonshotai/Kimi-K3.1-Pro"]
    cache = site.render_cache(pub)
    page = (SITE / "models" / f"{stem}.html").read_text()
    combos = json.loads(re.search(r"data-combos='([^']+)'", page).group(1).replace("&quot;", '"'))
    for n in (8192, 32768, int(pub["context_length"])):
        kv = cache["grows_per_token_bytes"] * n + cache["constant_bytes"]
        for key, _, per in site.PRECISION:
            c = combos[f"{n}-{key}"]
            assert c["cache"] == site.size(kv) and c["total"] == site.size(pub["params_total"] * per + kv)


def test_key_facts_have_plain_labels_and_the_rest_are_collapsed():
    page = (SITE / "models" / "LiquidAI__LFM3-1.7B-Base.html").read_text()
    key, rest = page.split('<details class="allfields">')
    for label in (
        "Total parameters",
        "Layers",
        "Layer mix",
        "Context length",
        "Hidden size (width)",
        "Attention type and heads",
    ):
        assert f"<th scope=row>{label}<span" in key
    assert '<span class="key">max_position_embeddings</span>' in key  # the config key sits under the label
    labels = re.findall(r"<th scope=row>([^<]+)<span", rest)
    assert labels and not [x for x in labels if "_" in x or x[0].islower()]
    assert "Mla kv lora rank" not in page and ">Model type<" not in page


def test_what_changed_skips_small_changes_and_bookkeeping():
    site = builder()
    published, _, _ = site.inventory()
    _, rec, _ = published["moonshotai/Kimi-K3.1-Pro"]
    _, old, _ = published["moonshotai/Kimi-K3"]
    labels = [label for label, _, _ in site.changes(rec, old)]
    assert "Layer mix" in labels and "Cache per token" in labels
    assert "Total parameters" not in labels and "Active parameters per token" not in labels  # under 1%
    page = (SITE / "models" / "moonshotai__Kimi-K3.1-Pro.html").read_text()
    what = page.split('id="chg"')[1].split("</section>")[0]
    assert "weight group" not in what and "2.7799T" not in what and "27 KiB" in what and "33 KiB" in what
    kimi = next(
        m
        for m in json.loads((SITE / "data/models.json").read_text())["models"]
        if m["stem"] == "moonshotai__Kimi-K3.1-Pro"
    )
    assert kimi["note"] is None  # the saved note states sub-1% changes and weight groups, so it is not shown


def test_sig_figures():
    site = builder()
    assert site.sig(27.0) == "27" and site.sig(1.7288) == "1.73" and site.sig(104.37) == "104"
    assert site.human(999_890_000) == "1.00B"


def test_fact_statuses_use_plain_words():
    page = (SITE / "models" / "moonshotai__Kimi-K3.1-Pro.html").read_text()
    assert "st-one-source" in page and "single source" not in page and "st-computed" in page


def test_pages_have_basic_accessible_markup():
    for path, page in pages().items():
        assert '<html lang="en">' in page and 'class="skip"' in page and '<main id="main"' in page, path
        for img in re.findall(r"<img [^>]*>", page):
            assert 'alt="' in img and 'alt=""' not in img, path
        assert "<th>" not in page, path  # header cells carry a scope


def test_style_matches_the_brief():
    css = (SITE / "style.css").read_text()
    for token in (
        "#f5f2e9",
        "#141d1a",
        "#1f6552",
        "Bricolage Grotesque",
        "IBM Plex Mono",
        "Manrope",
        "prefers-color-scheme:dark",
    ):
        assert token in css
    js = (SITE / "site.js").read_text()
    assert "import" not in js and "require(" not in js  # no framework


def test_site_settings_hold_no_numbers_for_models():
    config = yaml.safe_load((ROOT / "config/site.yaml").read_text())
    assert set(config) == {"repo_url", "site_url", "publishers", "memory_contexts", "layer_colors"}


def test_vercel_builds_the_site_from_committed_records():
    import json as _json

    config = _json.loads((ROOT / "vercel.json").read_text())
    assert config["framework"] is None and config["outputDirectory"] == "site" and config["cleanUrls"] is True
    assert "build_site.py" in config["buildCommand"] and "GEMINI" not in json.dumps(config)
    assert not (ROOT / ".github/workflows/pages.yml").exists()
    assert "site/" in (ROOT / ".gitignore").read_text().split()


def test_site_builds_offline_from_committed_records(tmp_path, monkeypatch):
    import shutil
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("build_site.py tried to use the network")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for folder in ("data", "config", "results/explainer", "diagrams/generated"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    for folder in ("data/generated-diagrams",):
        if (ROOT / folder).exists() and not (tmp_path / folder).exists():
            shutil.copytree(ROOT / folder, tmp_path / folder)
    site = builder()
    monkeypatch.setattr(site, "ROOT", tmp_path)
    site.main()
    built = {p.relative_to(tmp_path / "site").as_posix() for p in (tmp_path / "site").rglob("*.html")}
    assert built == {p.relative_to(SITE).as_posix() for p in SITE.rglob("*.html")}


def test_one_colour_per_layer_type_on_every_page_and_in_the_diagrams():
    from archwatch import diagram, palette

    pal = palette.colors()
    assert len(set(pal.values())) == len(pal)  # no two layer types share a colour
    names = {}
    for path, page in pages().items():
        for colour in re.findall(r'<i style="background:(#[0-9a-fA-F]{6})"', page):
            assert colour in pal.values(), path
        for colour, label in re.findall(
            r'background:(#[0-9a-fA-F]{6})" aria-hidden="true"></span>(?:<b>\d+</b> )?([A-Za-z][A-Za-z ()-]*)', page
        ):
            names.setdefault(label.strip(), set()).add(colour)
    assert names and all(len(c) == 1 for c in names.values()), names
    css = diagram.kind_css()
    for kind, (_, cls) in diagram.KIND.items():
        assert f".{cls}{{fill:{pal[kind]}}}" in css


def test_public_diagrams_carry_no_extractor_notes():
    for path, page in pages().items():
        assert "Config keys not consumed" not in page and "not consumed by the extractor" not in page, path


def test_capped_cache_is_never_shown_as_zero():
    home = (SITE / "index.html").read_text()
    model = (SITE / "models" / "mistralai__Mistral-7B-v0.1.html").read_text()
    card = home.split('id="h-mistralai__Mistral-7B-v0.1"')[1].split("</article>")[0]
    assert "capped at 4,096 tokens" in card and "512 MiB" in card
    row = re.search(r"Mistral-7B-v0.1</a>.*?</tr>", home, re.S).group(0)
    assert "capped at 4,096 tokens" in row
    assert "cache, capped at 4,096 tokens" in model  # tile
    assert "when the window is full" in model and "Cache is capped at 4,096 tokens" in model  # facts + planner
    for page in pages().values():
        assert ">0 B<" not in page and not re.search(r"[^\d,]0 B per token", page)


def test_compare_derives_implied_values_instead_of_not_stated():
    d = json.loads((SITE / "data/models.json").read_text())
    k3 = next(m for m in d["models"] if m["stem"] == "moonshotai__Kimi-K3")
    k31 = next(m for m in d["models"] if m["stem"] == "moonshotai__Kimi-K3.1-Pro")
    assert d["implied"]["moonshotai__Kimi-K3"]["dsa_index_topk"] == "all earlier tokens"
    assert k31["fields"]["dsa_index_topk"][0] == "2,048" and "dsa_index_topk" not in k3["fields"]
    site = builder()
    dense = {"layer_types": ["full_attention"]}
    assert site.implied("dsa_index_topk", dense) == "all earlier tokens"
    assert site.implied("sliding_window", dense) == "no sliding window"
    assert site.implied("params_active", dense) == "all (dense)"
    assert site.implied("num_mtp_layers", dense) is None  # nothing implied: stays 'not stated'
