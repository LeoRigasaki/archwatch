import importlib.util
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reference_agreement", ROOT / "tools/reference_agreement.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
CONFIG = yaml.safe_load((ROOT / "config/reference-gallery.yaml").read_text())


def record(stem):
    return json.loads((ROOT / "data/models" / f"{stem}.json").read_text())


def test_reference_is_pinned_to_an_exact_commit_and_license():
    assert re.fullmatch(r"[0-9a-f]{40}", CONFIG["commit"]) and CONFIG["license"] == "Apache-2.0"
    assert "{commit}" in CONFIG["raw_url"] and CONFIG["path"] == "models.yml"


def test_fact_sheet_parsers():
    assert ref.parse_scale("400B total, 13B active (3.3% active)") == {
        "params_total": (400e9, 0.5e9),
        "params_active": (13e9, 0.5e9),
    }
    total = ref.parse_scale("125B core, 6B active; plus 51B n-gram embeddings and 4B MTP")["params_total"]
    assert total[0] == 180e9
    assert ref.parse_layers("45 sliding-window + 15 global") == (60, 0)
    assert ref.parse_layers("8 input + 20 shared × 2 passes + 8 output (56 applications)") == (36, 0)
    assert ref.parse_layers("20 encoder + 20 decoder layers; 2 SWA-only + 38 CSA2/SWA") == (40, 0)
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
    cached = tmp_path / "reference-gallery" / CONFIG["commit"] / "models.yml"
    cached.parent.mkdir(parents=True)
    cached.write_text(yaml.safe_dump({"M": {"image": "https://example.invalid/x.webp", "scale": "1B parameters"}}))
    assert ref.fetch(CONFIG, tmp_path) == {"M": {"scale": "1B parameters"}}
    assert "cache/" in (ROOT / ".gitignore").read_text().splitlines()


def test_comparison_never_feeds_records_gold_or_diagrams():
    source = (ROOT / "tools/reference_agreement.py").read_text()
    writes = re.findall(r'\(out / "([^"]+)"\)\.write_text', source)
    assert set(writes) == {"reference-agreement.json", "reference-agreement.md"}
    assert 'out = ROOT / "results"' in source
    for path in list((ROOT / "src").rglob("*.py")) + [ROOT / "run.py", ROOT / "watch.py", ROOT / "build_site.py"]:
        text = path.read_text()
        assert "reference_agreement" not in text and "llm-architecture-gallery" not in text, path
    assert not list((ROOT / "evals").rglob("*gallery*")) and not list((ROOT / "data").rglob("*gallery*"))


def test_every_recorded_disagreement_has_a_checked_cause():
    saved = json.loads((ROOT / "results/reference-agreement.json").read_text())
    causes = yaml.safe_load((ROOT / "config/reference-causes.yaml").read_text())
    bad = [r for r in saved["rows"] if not r["agree"]]
    assert bad and all(r["cause"] in ref.CAUSES for r in bad)
    for r in bad:
        entry = causes[r["model"]][r["field"]]
        assert entry["cause"] == r["cause"] and len(entry["why"]) > 40
    # The Mistral speculative-decoding-head error is fixed; its active count now agrees with the card.
    assert not [r for r in bad if r["cause"] == "error-ours"]
    mistral = [
        r for r in saved["rows"] if r["model"] == "mistralai/Mistral-Large-4-1T-A52B" and r["field"] == "params_active"
    ]
    assert mistral and mistral[0]["agree"]


def test_site_labels_the_growing_cache_definition():
    assert "sliding-window and constant states excluded" in (ROOT / "build_site.py").read_text()
