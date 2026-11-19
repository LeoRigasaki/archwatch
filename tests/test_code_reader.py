"""The LLM code reader: claims are verified in code, every call is logged once, responses are cached."""

import copy
import json
import types
from pathlib import Path

import pytest
import yaml

from archwatch import code_reader as cr

ROOT = Path(__file__).resolve().parents[1]
CODE = "\n".join(
    [
        "class Attention(nn.Module):",
        "    def __init__(self, config):",
        "        self.num_heads = config.num_attention_heads",
        "        self.head_dim = config.head_dim",
        "        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)",
        "        self.act = nn.SiLU()",
    ]
)
FILES = {"modeling_toy.py": CODE}
CONFIG = {"num_attention_heads": 4, "head_dim": 8, "hidden_size": 16, "num_hidden_layers": 2}
TENSORS = {
    "model.layers.0.self_attn.q_proj.weight": {"shape": [32, 16], "dtype": "BF16"},
    "model.layers.1.self_attn.q_proj.weight": {"shape": [32, 16], "dtype": "BF16"},
    "model.layers.0.mlp.up_proj.weight": {"shape": [64, 16], "dtype": "BF16"},
    "model.layers.1.mamba.in_proj.weight": {"shape": [64, 16], "dtype": "BF16"},
}


def claim(**overrides):
    base = dict(
        field="num_attention_heads",
        value="4",
        file="modeling_toy.py",
        line_start=3,
        line_end=3,
        quote="self.num_heads = config.num_attention_heads",
        config_keys=["num_attention_heads"],
        tensor_patterns=["model.layers.N.self_attn.q_proj.weight"],
    )
    return cr.Claim(**(base | overrides))


def test_lines_are_numbered_by_us():
    assert cr.number("a\nb").splitlines() == ["    1| a", "    2| b"]


def test_a_correct_claim_passes_every_check():
    assert cr.check_claim(claim(), FILES, CONFIG, TENSORS) == []


@pytest.mark.parametrize(
    "overrides, problem",
    [
        (dict(quote="self.num_heads = 4"), "quote does not equal the cited lines"),
        (dict(line_start=9, line_end=9), "outside modeling_toy.py"),
        (dict(file="modeling_other.py"), "not among the fetched files"),
        (dict(config_keys=["n_heads"]), "config key n_heads does not exist"),
        (dict(value="5"), "config num_attention_heads = 4, claim says 5"),
        (dict(tensor_patterns=["model.layers.N.attn.wq.weight"]), "matches no tensor"),
    ],
)
def test_each_failure_is_named(overrides, problem):
    assert any(problem in p for p in cr.check_claim(claim(**overrides), FILES, CONFIG, TENSORS))


def test_layer_count_and_projection_widths_come_from_the_headers():
    layers = claim(field="num_layers", value="3", config_keys=["num_hidden_layers"], quote=CODE.splitlines()[2])
    assert any("config num_hidden_layers = 2" in p for p in cr.check_claim(layers, FILES, CONFIG, TENSORS))
    ok = claim(field="num_layers", value="2", config_keys=["num_hidden_layers"])
    assert cr.check_claim(ok, FILES, CONFIG, TENSORS) == []
    assert cr.cross_checks({"num_attention_heads": 4, "head_dim": 8}, TENSORS) == {}
    wrong = cr.cross_checks({"num_attention_heads": 4, "head_dim": 16}, TENSORS)
    assert "projection rows [32]" in wrong["num_attention_heads"][0]


def test_layer_pattern_must_match_per_layer_tensor_groups():
    assert cr._pattern_problems(["attention", "mamba"], TENSORS) == []
    assert cr._pattern_problems(["attention", "attention"], TENSORS) == [
        "layers typed 'attention' hold different tensors (layer 1)"
    ]
    assert "2 layers" in cr._pattern_problems(["attention"] * 3, TENSORS)[0]


def test_conflicting_or_mismatched_claims_go_to_review():
    code = dict(files=FILES, version_matches=True)
    rows = cr.verdicts([claim(), claim(value="4")], code, CONFIG, TENSORS, {"num_attention_heads": 4})
    assert [r["verdict"] for r in rows] == ["verified", "verified"]
    rows = cr.verdicts([claim()], code, CONFIG, TENSORS, {"num_attention_heads": 8})
    assert rows[0]["verdict"] == "review" and "rule-extracted value is 8" in rows[0]["conflicts"][0]
    rows = cr.verdicts([claim()], dict(code, version_matches=False), CONFIG, TENSORS, {})
    assert rows[0]["verdict"] == "review"  # code read at another transformers version is never verified


def test_dev_versions_are_flagged_and_a_later_release_is_marked_as_a_mismatch(monkeypatch):
    available = {"v4.54.0"}
    monkeypatch.setattr(cr, "_transformers_file", lambda tag, mt, name: "code" if tag in available else None)
    monkeypatch.setattr(cr, "_release_tags", lambda: [(4, 47, 0), (4, 48, 0), (4, 53, 0), (4, 54, 0)])
    info = {"sha": "a" * 40, "files": ["config.json"]}
    code = cr.fetch_code("org/m", info, {"model_type": "xlstm", "transformers_version": "4.48.0.dev0"})
    assert code["source"] == "transformers v4.54.0" and code["version_matches"] is False
    assert "development build" in code["flags"][0] and "absent at v4.48.0" in code["flags"][1]
    available.add("v4.48.0")
    exact = cr.fetch_code("org/m", info, {"model_type": "xlstm", "transformers_version": "4.48.0"})
    assert exact["source"] == "transformers v4.48.0" and exact["version_matches"] is True and exact["flags"] == []


class FakeInteractions:
    def __init__(self, text):
        self.text, self.requests = text, []

    def create(self, **request):
        self.requests.append(request)
        return types.SimpleNamespace(output_text=self.text)


def reader(tmp_path, monkeypatch, client):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cr, "CACHE", tmp_path / "cache")
    config = copy.deepcopy(yaml.safe_load((ROOT / "config/providers.yaml").read_text()))
    config["code_reader"]["ledger"] = str(tmp_path / "ledger.json")
    return cr.CodeReader(
        config, role="review", run_id="r1", client=types.SimpleNamespace(interactions=client), wait=lambda s: None
    )


def test_one_logged_call_per_prompt_then_the_cache_answers(tmp_path, monkeypatch):
    answer = json.dumps({"claims": [claim().model_dump()]})
    fake = FakeInteractions(answer)
    r = reader(tmp_path, monkeypatch, fake)
    claims, meta = r.read("org/m", "c" * 40, "prompt")
    assert claims.claims[0].field == "num_attention_heads" and meta["cached"] is False
    request = fake.requests[0]
    assert request["model"] == "gemini-3.5-flash-lite" and request["store"] is False and "tools" not in request
    assert request["response_format"]["schema"] == cr.Claims.model_json_schema()
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger) == 1 and ledger[0]["repo"] == "org/m" and ledger[0]["run_id"] == "r1"
    assert ledger[0]["experiment"] == "code-reader-v1" and len(ledger[0]["prompt_sha256"]) == 64
    again, meta = r.read("org/m", "c" * 40, "prompt")
    assert (
        meta["cached"] is True
        and len(fake.requests) == 1
        and len(json.loads((tmp_path / "ledger.json").read_text())) == 1
    )
    r.read("org/m", "d" * 40, "prompt")  # another commit is another cache key
    assert len(fake.requests) == 2


def test_the_reader_models_and_caps_come_from_config():
    section = yaml.safe_load((ROOT / "config/providers.yaml").read_text())["code_reader"]
    models = yaml.safe_load((ROOT / "config/providers.yaml").read_text())["models"]
    assert models[section["first"]]["id"] == "gemini-3.5-flash-lite"
    assert models[section["second_opinion"]]["id"] == "gemini-3.8-flash"
    assert section["max_calls_per_run"] <= 6 and section["ledger"] == "data/gemini-requests.json"
    source = (ROOT / "src/archwatch/code_reader.py").read_text()
    assert "url_context" not in source and "gemini-3" not in source  # no hard-coded model, no URL tool


def test_a_name_is_not_refuted_by_a_numeric_config_key():
    norm = claim(field="normalization_type", value='"RMSNorm"', config_keys=["head_dim"], tensor_patterns=[])
    assert cr.check_claim(norm, FILES, CONFIG, TENSORS) == []
    key_name = claim(value='"num_attention_heads"')  # xLSTM's first reading answered key names instead of values
    problems = cr.check_claim(key_name, FILES, CONFIG, TENSORS)
    assert "num_attention_heads needs a number, got 'num_attention_heads'" in problems
    assert "value 'num_attention_heads' is a config key name, not its value" in problems


def test_claims_must_cite_modelling_code_and_shape_fields_must_name_tensors():
    files = dict(FILES, **{"configuration_toy.py": "num_attention_heads=4"})
    from_defaults = claim(file="configuration_toy.py", line_start=1, line_end=1, quote="num_attention_heads=4")
    assert any("configuration class" in p for p in cr.check_claim(from_defaults, files, CONFIG, TENSORS))
    no_tensor = claim(tensor_patterns=[])
    assert "num_attention_heads is a weight shape, but the claim names no tensor to check it against" in cr.check_claim(
        no_tensor, FILES, CONFIG, TENSORS
    )
    activation = claim(
        field="activation",
        value='"silu"',
        line_start=6,
        line_end=6,
        quote="self.act = nn.SiLU()",
        config_keys=[],
        tensor_patterns=[],
    )
    assert cr.check_claim(activation, FILES, CONFIG, TENSORS) == []  # no weight to check: code and quote decide
