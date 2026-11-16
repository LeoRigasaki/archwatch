import importlib.util
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reference_agreement", ROOT / "tools/reference_agreement.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
CONFIG = ref.reference_config(repo="owner/reference", path="sheet.yml", commit="0" * 40)


def record(stem):
    return json.loads((ROOT / "data/models" / f"{stem}.json").read_text())


def test_reference_is_named_at_run_time_and_never_stored(monkeypatch):
    import pytest

    for key in ("REFERENCE_REPO", "REFERENCE_PATH", "REFERENCE_COMMIT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit):
        ref.reference_config()
    monkeypatch.setenv("REFERENCE_REPO", "owner/reference")
    monkeypatch.setenv("REFERENCE_PATH", "sheet.yml")
    monkeypatch.setenv("REFERENCE_COMMIT", "a" * 40)
    config = ref.reference_config()
    assert config["repo"] == "owner/reference" and config["commit"] == "a" * 40 and "{commit}" in config["raw_url"]
    assert re.fullmatch(r"a{40}", config["commit"])
    assert not list((ROOT / "config").glob("reference*"))  # nothing about the reference is committed


def test_fact_sheet_parsers():
    assert ref.parse_scale("300B total, 12B active (4% active)") == {
        "params_total": (300e9, 0.5e9),
        "params_active": (12e9, 0.5e9),
    }
    total = ref.parse_scale("100B core, 5B active; plus 40B n-gram embeddings and 2B MTP")["params_total"]
    assert total[0] == 142e9
    assert ref.parse_layers("30 sliding-window + 10 global") == (40, 0)
    assert ref.parse_layers("6 input + 10 shared × 3 passes + 6 output (42 applications)") == (22, 0)
    assert ref.parse_layers("12 encoder + 12 decoder layers; 2 SWA-only + 22 CSA2/SWA") == (24, 0)
    assert ref.parse_bytes("240 KiB") == (240 * 1024, 512)
    assert ref.parse_count("512,000") == (512000.0, 0)
    assert ref.decoder_kind("Sparse hybrid") == ref.decoder_kind("MoE") == "sparse"


def test_compare_reports_agreement_disagreement_evidence_and_unprocessed():
    kimi = record("moonshotai__Kimi-K3.1-Pro")
    gallery = {
        "Kimi entry": dict(
            config=dict(repo="moonshotai/Kimi-K3.1-Pro"),
            scale="2.8T total, 104B active",
            context_tokens="1,048,576",
            layer_mix="69 KDA + 25 Gated DSA",  # deliberately wrong: 94
            decoder_type="Sparse hybrid",
        ),
        "Other": dict(config=dict(repo="someone/New-Model"), date="2026-11-01", company="X"),
        "No repo": dict(date="2026-10-01", company="Y"),
    }
    out = ref.compare({kimi["id"]: kimi}, gallery)
    by_field = {r["field"]: r for r in out["rows"]}
    assert by_field["params_total"]["agree"] and by_field["context_length"]["agree"]
    layers = by_field["num_layers"]
    assert not layers["agree"] and layers["ours"] == 93 and "config:" in layers["our_evidence"]
    assert [m["gallery_name"] for m in out["not_processed"]] == ["Other", "No repo"]


def test_images_are_dropped_and_cache_is_not_vendored(tmp_path):
    cached = tmp_path / "reference" / "source" / CONFIG["commit"] / "sheet.yml"
    cached.parent.mkdir(parents=True)
    cached.write_text(yaml.safe_dump({"M": {"image": "https://example.invalid/x.webp", "scale": "1B parameters"}}))
    assert ref.fetch(CONFIG, tmp_path) == {"M": {"scale": "1B parameters"}}
    assert "cache/" in (ROOT / ".gitignore").read_text().splitlines()


def test_comparison_never_feeds_records_gold_or_diagrams():
    source = (ROOT / "tools/reference_agreement.py").read_text()
    writes = re.findall(r'\(REPORTS / "([^"]+)"\)\.write_text', source)
    assert set(writes) == {"agreement.json", "agreement.md"}
    assert 'REPORTS = ROOT / "cache" / "reference"' in source  # reports with reference values stay out of git
    for path in list((ROOT / "src").rglob("*.py")) + [ROOT / "run.py", ROOT / "watch.py", ROOT / "build_site.py"]:
        text = path.read_text()
        assert "reference_agreement" not in text, path
    assert not list((ROOT / "evals").rglob("*gallery*")) and not list((ROOT / "data").rglob("*gallery*"))


def test_every_disagreement_is_closed_by_recomputation_or_marked_ours():
    import pytest

    path = ref.REPORTS / "agreement.json"
    if not path.exists():
        pytest.skip("no local reference report (cache/reference/ is built on demand and never committed)")
    saved = json.loads(path.read_text())
    bad = [r for r in saved["rows"] if not r["agree"]]
    assert bad and all(r["cause"] in ref.CAUSES for r in bad)
    for r in bad:
        if r["cause"] == "definition":
            assert "Ours recomputed:" in r["cause_why"] and "accepted" in r["cause_why"]
    mistral = [
        r for r in saved["rows"] if r["model"] == "mistralai/Mistral-Large-4-1T-A52B" and r["field"] == "params_active"
    ]
    # Card "52B active" counts the input embedding; the fixed default leaves it out (a row lookup).
    assert mistral and (mistral[0]["agree"] or mistral[0]["cause"] == "definition")


def test_site_labels_the_growing_cache_definition():
    assert "sliding-window and constant states excluded" in (ROOT / "build_site.py").read_text()
