import json
import shutil
from pathlib import Path

from archwatch import generated

ROOT = Path(__file__).resolve().parents[1]


def record(stem):
    return json.loads((ROOT / "data/models" / f"{stem}.json").read_text())


def copy_root(tmp_path, stem):
    for rel in ("config/providers.yaml", f"diagrams/generated/{stem}.svg", f"diagrams/generated/{stem}.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_published_generated_diagrams_are_selected():
    for stem in ("moonshotai__Kimi-K3.1-Pro", "nvidia__Nemotron-H-8B-Base-8K", "meta-llama__Llama-3.1-8B"):
        chosen = generated.select(record(stem))
        assert chosen and chosen[0].startswith("<svg") and len(chosen[1]["checks_passed"]) >= 24


def test_falls_back_when_svg_edited_spec_changed_or_missing(tmp_path):
    stem = "meta-llama__Llama-3.1-8B"
    root = copy_root(tmp_path, stem)
    rec = record(stem)
    assert generated.select(rec, root)
    changed = json.loads(json.dumps(rec))
    changed["published"]["context_length"] = 65536  # architecture spec changed since the drawing
    assert generated.select(changed, root) is None
    svg = root / f"diagrams/generated/{stem}.svg"
    svg.write_text(svg.read_text().replace("128K", "256K"))  # edited after review
    assert generated.select(rec, root) is None
    assert generated.select(record("microsoft__phi-2"), root) is None  # no diagram files in this root


def test_publish_refuses_failed_or_stale_diagrams(tmp_path):
    import pytest

    src = tmp_path / "results/run/meta-llama__Llama-3.1-8B"
    src.mkdir(parents=True)
    shutil.copy(ROOT / "results/diagram-gen/meta-llama__Llama-3.1-8B/v1.svg", src / "v1.svg")
    shutil.copy(ROOT / "results/diagram-gen/meta-llama__Llama-3.1-8B/v1.review.json", src / "v1.review.json")
    shutil.copy(ROOT / "results/diagram-gen/meta-llama__Llama-3.1-8B/spec.yaml", src / "spec.yaml")
    with pytest.raises(ValueError, match="passed every"):
        generated.publish(tmp_path, record("meta-llama__Llama-3.1-8B"), src / "v1.svg")  # v1 failed the brace rule


def site_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("site_for_diagram_labels", ROOT / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_site_labels_which_diagram_is_shown():
    # Never a fixed model: which models have an approved diagram changes with every diagram run.
    site = site_builder()
    published, _, _ = site.inventory()
    for stem, rec, _ in published.values():
        page = (ROOT / "site/models" / f"{stem}.html").read_text()
        if generated.select(rec):
            assert "Drawn by Gemini from verified fields," in page and "checks passed" in page, stem
            assert f'src="../diagrams/{stem}.svg"' in page, stem
        else:
            assert "Drawn by code from verified fields; no generated diagram has passed every check yet" in page, stem


def test_code_drawn_label_when_no_generated_diagram_passes(monkeypatch):
    # Fixture for the other branch, so it is covered even when every published model has a Gemini diagram.
    site = site_builder()
    published, _, _ = site.inventory()
    stem, rec, pub = next(iter(published.values()))
    monkeypatch.setattr(site.generated, "select", lambda record, root=None: None)
    page = site.model_page(stem, rec, pub, published)
    assert "Drawn by code from verified fields; no generated diagram has passed every check yet" in page
    assert "<svg" in page and "../diagrams/" not in page


def test_dense_models_show_all_active_parameters():
    page = (ROOT / "site/models/microsoft__phi-2.html").read_text()
    assert "active per token (dense)" in page and "not reported" not in page
