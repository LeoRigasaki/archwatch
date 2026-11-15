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
    assert "{commit}" in CONFIG["raw_url"] and CONFIG["path"] == "reference.yml"


def test_fact_sheet_parsers():
    assert ref.parse_scale("300B total, 12B active (4% active)") == {
        "params_total": (400e9, 0.5e9),
        "params_active": (13e9, 0.5e9),
    }
    total = ref.parse_scale("100B core, 5B active; plus 40B n-gram embeddings and 2B MTP")["params_total"]
    assert total[0] == 180e9
    assert ref.parse_layers("30 sliding-window + 10 global") == (60, 0)
    assert ref.parse_layers("6 input + 10 shared × 3 passes + 6 output (42 applications)") == (36, 0)
    assert ref.parse_layers("12 encoder + 12 decoder layers; 2 SWA-only + 22 CSA2/SWA") == (40, 0)
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
    cached = tmp_path / "reference-gallery" / CONFIG["commit"] / "reference.yml"
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
        assert "reference_agreement" not in text and "reference gallery" not in text, path
    assert not list((ROOT / "evals").rglob("*gallery*")) and not list((ROOT / "data").rglob("*gallery*"))
