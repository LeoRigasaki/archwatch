"""Source-driven gallery and extraction metadata, independent of named demo models."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from archwatch import card, diagram, extract, verify, watcher

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location("gallery", ROOT / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gallery_discovers_unseen_record_and_watcher_sources(tmp_path, monkeypatch):
    module = builder()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    rec = json.loads((ROOT / "data/models/LiquidAI__LFM3-1.7B.json").read_text())
    rec.update(id="new-publisher/UnseenArchitecture", source_files=["config.json", "README.md", "weights.safetensors"])
    rec.pop("predecessor", None)
    folder = tmp_path / "data/models"
    folder.mkdir(parents=True)
    (folder / "arbitrary-filename.json").write_text(json.dumps(rec))
    (tmp_path / "data/seen.json").write_text(
        json.dumps({"models": {"other/FutureModel": {"status": "pending", "reason": "<unverified>"}}})
    )
    module.main()
    page = (tmp_path / "site/index.html").read_text()
    assert "Search 1 models" in page and "UnseenArchitecture" in page and "FutureModel" in page
    assert "<unverified>" not in page and "&lt;unverified&gt;" not in page  # held: plain reason only
    model = (tmp_path / "site/models/arbitrary-filename.html").read_text()
    assert f"huggingface.co/new-publisher/UnseenArchitecture/tree/{rec['source_commit']}" in model
    before = (page, (tmp_path / "site/index.html").stat().st_mtime_ns)
    module.main()
    assert ((tmp_path / "site/index.html").read_text(), (tmp_path / "site/index.html").stat().st_mtime_ns) == before
    assert "Kimi-K3" not in (ROOT / "build_site.py").read_text()
    assert "LFM3-1.7B" not in (ROOT / "build_site.py").read_text()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Mistral-7B-v0.1", "mistral"),
        ("gemma-3-1b-pt", "gemma"),
        ("Meta-Llama-3-8B", "llama"),
        ("Phi-3-mini", "phi"),
        ("GLM-4-9B", "glm"),
        ("NVIDIA-Nemotron-3-Nano", "nemotron"),
    ],
)
def test_family_rules(name, expected):
    pol = watcher.load_policy(ROOT / "config/watcher.yaml")
    assert watcher.family("org/" + name, pol) == expected


@pytest.mark.parametrize("kind", ["finetune", "adapter", "merge", "quantized"])
def test_tagged_derivatives_do_not_reach_config_or_processing(tmp_path, kind):
    pol = watcher.load_policy(ROOT / "config/watcher.yaml")

    class Source:
        def metadata(self, repo):
            return dict(
                sha="a" * 40, files=["config.json", "model.safetensors"], tags=[f"base_model:{kind}:lab/Primary"]
            )

        def config(self, *args):
            raise AssertionError("derivative consumed an admission/processing slot")

        def analyze(self, *args):
            raise AssertionError("derivative analyzed")

    report = watcher.run(tmp_path, pol, source=Source(), repos=["lab/Derived"], process=True)
    assert report["processed"] == 0
    assert report["outcomes"]["lab/Derived"] == "rejected"


def test_every_unread_nested_key_is_reported_without_mutating_config():
    cfg = dict(
        model_type="toy",
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=1024,
        hidden_act="silu",
        future={"mechanism": True, "layers": [{"new_flag": 7}]},
        training_only=123,
    )
    original = copy.deepcopy(cfg)
    facts, usage = extract.extract_with_usage(cfg)
    assert facts["hidden_size"][0] == 8
    assert {"future.mechanism", "future.layers[0].new_flag", "training_only"} <= set(usage["unconsumed"])
    assert {"hidden_size", "num_hidden_layers", "hidden_act"} <= set(usage["consumed"])
    assert cfg == original


@pytest.mark.parametrize(
    "text",
    [
        "| Field | Value |\n| --- | --- |\n| Number of Layers | 24 |\n| Activation Function | SiLU |",
        "- **Number of Layers:** 24\n- **Activation Function**: SiLU",
        "<table><tr><td>Number of Layers</td><td>24</td></tr><tr><td>Activation Function</td><td>SiLU</td></tr></table>",
    ],
)
def test_card_formats(text):
    parsed = card.parse_card(text)
    assert parsed["num_layers"]["value"] == 24
    assert parsed["activation"]["raw"] == "SiLU"


def test_fenced_examples_and_benchmark_columns_are_not_architecture_claims():
    assert not card.parse_card("```md\n- Number of Layers: 999\n```\n| Parameters | 123 | 456 |")


def test_activation_card_name_flows_through_verifier_into_svg():
    rec = json.loads((ROOT / "data/models/moonshotai__Kimi-K3.1-Pro.json").read_text())
    claims = card.parse_card("- Activation Function: SiTU alternate card name")
    facts = {k: (v["value"], v["evidence"]) for k, v in rec["facts"].items()}
    decisions = verify.verify(facts, rec["params"]["counted_from_weights"], {}, None, {"facts": {}}, claims)
    name = next(c for c in decisions if c["field"] == "activation_card_name")
    assert name["published"] and name["value"] == claims["activation"]["raw"]
    rec["published"]["activation_card_name"] = name["value"]
    rec["checks"] = [c for c in rec["checks"] if c["field"] != name["field"]] + [name]
    assert "SiTU alternate card name" in diagram.render(rec)
    assert "SiTU-GLU" not in (ROOT / "src/archwatch/diagram.py").read_text()
