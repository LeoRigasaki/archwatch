"""The LLM code reader: claims are verified in code, cited modules are mapped to tensors by code, every call is
logged once with its prompt version, and responses are cached by (commit, model, prompt version, prompt hash)."""

import copy
import hashlib
import json
import types
from pathlib import Path

import pytest
import yaml

from archwatch import code_reader as cr

ROOT = Path(__file__).resolve().parents[1]
CODE = "\n".join(
    [
        "class ToyAttention(nn.Module):",  # 1
        "    def __init__(self, config):",  # 2
        "        self.num_heads = config.num_attention_heads",  # 3
        "        self.head_dim = config.head_dim",  # 4
        "        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)",  # 5
        "        self.act = nn.SiLU()",  # 6
        "class ToyMamba(nn.Module):",  # 7
        "    def __init__(self, config):",  # 8
        "        self.in_proj = nn.Linear(config.hidden_size, 64)",  # 9
        "MIXERS = {'attention': ToyAttention, 'mamba': ToyMamba}",  # 10
        "class ToyBlock(nn.Module):",  # 11
        "    def __init__(self, config, kind):",  # 12
        "        self.mixer = MIXERS[kind](config)",  # 13
        "class ToyModel(nn.Module):",  # 14
        "    def __init__(self, config):",  # 15
        "        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)",  # 16
        "        self.layers = nn.ModuleList([ToyBlock(config, k) for k in config.kinds])",  # 17
        "class ToyForCausalLM(nn.Module):",  # 18
        "    def __init__(self, config):",  # 19
        "        self.model = ToyModel(config)",  # 20
    ]
)
CODE_INFO = dict(files={"modeling_toy.py": CODE}, version_matches=True)
CONFIG = {
    "architectures": ["ToyForCausalLM"],
    "num_attention_heads": 4,
    "head_dim": 8,
    "hidden_size": 16,
    "vocab_size": 100,
    "num_hidden_layers": 2,
}
TENSORS = {
    "model.embed_tokens.weight": {"shape": [100, 16], "dtype": "BF16"},
    "model.layers.0.mixer.q_proj.weight": {"shape": [32, 16], "dtype": "BF16"},
    "model.layers.1.mixer.in_proj.weight": {"shape": [64, 16], "dtype": "BF16"},
}
PATHS = cr.state_paths(cr.module_tree(CODE_INFO["files"]), "ToyForCausalLM")
LINES = CODE.splitlines()


def claim(field="num_attention_heads", value="4", line=5, keys=("num_attention_heads",), file="modeling_toy.py"):
    return cr.Claim(
        field=field,
        value=value,
        file=file,
        line_start=line,
        line_end=line,
        quote=LINES[line - 1],
        config_keys=list(keys),
    )


def assess(c):
    return cr.assess(c, CODE_INFO, CONFIG, TENSORS, PATHS)


def test_the_prompt_is_frozen_and_versioned():
    prompt = cr.load_prompt(ROOT / "prompts/code_reader_v1.md")
    assert prompt["version"] == "code_reader_v1" and "num_layers" in prompt["fields"]
    assert "Never cite configuration-class defaults" in prompt["system"]
    assert hashlib.sha256((ROOT / "prompts/code_reader_v1.md").read_bytes()).hexdigest() == cr.PROMPTS["code_reader_v1"]
    text = cr.prompt_text(prompt, CODE_INFO["files"], CONFIG)
    assert "   16| " in text and '"vocab_size": 100' in text and "{code}" not in text
    assert "tensor" not in prompt["task"].lower()  # the model is not asked to name tensors


def test_an_edited_prompt_is_refused(tmp_path):
    edited = tmp_path / "code_reader_v1.md"
    edited.write_text((ROOT / "prompts/code_reader_v1.md").read_text() + "\nextra rule\n")
    with pytest.raises(ValueError, match="new version"):
        cr.load_prompt(edited)


def test_cited_lines_map_to_state_dict_paths_through_the_module_tree():
    assert PATHS[("modeling_toy.py", 16)] == {"model.embed_tokens"}
    assert PATHS[("modeling_toy.py", 17)] == {"model.layers.N"}
    assert PATHS[("modeling_toy.py", 5)] == {"model.layers.N.mixer.q_proj"}  # through MIXERS[kind](config)
    assert PATHS[("modeling_toy.py", 9)] == {"model.layers.N.mixer.in_proj"}
    assert cr.tensors_under({"model.layers.N.mixer.q_proj"}, TENSORS) == ["model.layers.0.mixer.q_proj.weight"]
    prefixed = {"language_model." + k: v for k, v in TENSORS.items()}
    assert cr.tensors_under({"model.embed_tokens"}, prefixed) == ["language_model.model.embed_tokens.weight"]


def test_a_correct_size_claim_passes_its_weight_check():
    a = assess(claim(field="vocab_size", value="100", line=16, keys=["vocab_size"]))
    assert a["citation"] == a["agreement"] == a["weights"] == [] and a["weight_status"] == "passed"
    layers = assess(claim(field="num_layers", value="2", line=17, keys=["num_hidden_layers"]))
    assert layers["weight_status"] == "passed" and layers["tensors"] == 2


@pytest.mark.parametrize(
    "c, part, problem",
    [
        (claim(field="vocab_size", value="99", line=16, keys=[]), "weights", "has a dimension 99"),
        (claim(field="num_layers", value="3", line=17, keys=[]), "weights", "holds 2 layers, claim says 3"),
        (claim(field="vocab_size", value="100", line=3, keys=["vocab_size"]), "weights", "build no module"),
        (claim(value="5"), "agreement", "config num_attention_heads = 4, claim says 5"),
        (claim(keys=["n_heads"]), "agreement", "config key n_heads does not exist"),
        (claim(value='"num_attention_heads"'), "agreement", "is a config key name, not its value"),
        (claim(file="modeling_other.py"), "citation", "not among the fetched modelling files"),
        (claim(file="configuration_toy.py"), "citation", "a configuration class"),
    ],
)
def test_each_failure_is_named(c, part, problem):
    assert any(problem in p for p in assess(c)[part]), assess(c)


def test_a_misquoted_line_fails_the_citation():
    bad = cr.Claim(**(claim().model_dump() | {"quote": "self.q_proj = nn.Linear(16, 32)"}))
    assert assess(bad)["citation"] == ["quote does not equal the cited lines"]


def test_fields_without_weights_are_not_weight_checked():
    a = assess(claim(field="activation", value='"silu"', line=6, keys=[]))
    assert a["weight_status"] == "n/a" and a["citation"] == a["agreement"] == []


def test_heads_times_head_dim_must_match_the_projection_rows():
    pair = [claim(), claim(field="head_dim", value="8", line=5, keys=["head_dim"])]
    assert [r["verdict"] for r in cr.verdicts(pair, CODE_INFO, CONFIG, TENSORS, {})] == ["verified", "verified"]
    wrong = [claim(), claim(field="head_dim", value="16", line=5, keys=["head_dim"])]
    rows = cr.verdicts(wrong, CODE_INFO, dict(CONFIG, head_dim=16), TENSORS, {})
    assert all(r["verdict"] == "review" and r["weight_status"] == "failed" for r in rows)


def test_code_from_another_revision_is_capped_at_review():
    rows = cr.verdicts([claim()], dict(CODE_INFO, version_matches=False), CONFIG, TENSORS, {})
    assert rows[0]["verdict"] == "review" and "capped at review" in rows[0]["problems"][-1]


def test_dev_versions_are_flagged_and_a_later_release_is_marked_as_a_mismatch(monkeypatch):
    available = {"v4.54.0"}
    monkeypatch.setattr(cr, "_transformers_file", lambda tag, mt, name: "code" if tag in available else None)
    monkeypatch.setattr(cr, "_release_tags", lambda: [(4, 47, 0), (4, 48, 0), (4, 53, 0), (4, 54, 0)])
    info = {"sha": "a" * 40, "files": ["config.json"]}
    code = cr.fetch_code("org/m", info, {"model_type": "xlstm", "transformers_version": "4.48.0.dev0"})
    assert code["source"] == "transformers v4.54.0" and code["version_matches"] is False
    assert "development build" in code["flags"][0] and "absent at v4.48.0" in code["flags"][1]


def test_repo_code_never_includes_configuration_classes(monkeypatch):
    monkeypatch.setattr(cr.hub, "get_text", lambda repo, f, revision=None: f"# {f}")
    info = {"sha": "a" * 40, "files": ["configuration_m.py", "modeling_m.py", "config.json"]}
    assert list(cr.fetch_code("org/m", info, {})["files"]) == ["modeling_m.py"]


def test_authors_code_is_pinned_to_the_last_commit_before_the_release(monkeypatch):
    seen = {}

    def commit_before(repo, until):
        seen["until"] = (repo, until)
        return "f" * 40, until

    def cached_fetch(key, url):
        seen[url] = True
        return "class M: pass"

    monkeypatch.setattr(cr, "_commit_before", commit_before)
    monkeypatch.setattr(cr, "_cached_fetch", cached_fetch)
    info = {"sha": "a" * 40, "files": ["config.json"], "created_at": "2024-12-11T00:00:00+00:00"}
    sources = {"org/m": {"github": "authors/code", "paths": ["pkg/model.py"]}}
    code = cr.fetch_code("org/m", info, {"model_type": "x", "transformers_version": "4.48.0"}, sources)
    assert seen["until"] == ("authors/code", "2024-12-11T00:00:00+00:00")
    assert list(code["files"]) == ["model.py"] and code["version_matches"] is False  # always capped at review
    assert "https://raw.githubusercontent.com/authors/code/" + "f" * 40 + "/pkg/model.py" in seen
    assert "last commit before the Hub release" in code["flags"][0]


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
    prompt = cr.load_prompt(ROOT / "prompts/code_reader_v1.md")
    fake = FakeInteractions(json.dumps({"claims": [claim().model_dump()]}))
    r = reader(tmp_path, monkeypatch, fake)
    claims, meta = r.read("org/m", "c" * 40, prompt, "text", purpose="eval")
    assert claims.claims[0].field == "num_attention_heads" and meta["prompt_version"] == "code_reader_v1"
    request = fake.requests[0]
    assert request["model"] == "gemini-3.5-flash-lite" and request["store"] is False and "tools" not in request
    assert request["system_instruction"] == prompt["system"]
    assert request["response_format"]["schema"] == cr.Claims.model_json_schema()
    assert "tensor_patterns" not in json.dumps(request["response_format"]["schema"])
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger) == 1 and ledger[0]["repo"] == "org/m" and ledger[0]["prompt_version"] == "code_reader_v1"
    assert (
        ledger[0]["purpose"] == "eval" and ledger[0]["run_id"] == "r1" and ledger[0]["experiment"] == "code-reader-v1"
    )
    _, meta = r.read("org/m", "c" * 40, prompt, "text")
    assert meta["cached"] is True and len(fake.requests) == 1 and r.calls == 1
    cached = list((tmp_path / "cache" / "responses").iterdir())
    assert len(cached) == 1 and "-code_reader_v1-" in cached[0].name


def test_a_failed_call_still_counts(tmp_path, monkeypatch):
    class Boom:
        def create(self, **request):
            raise RuntimeError("500 from provider")

    r = reader(tmp_path, monkeypatch, Boom())
    with pytest.raises(cr.ProviderError):
        r.read("org/m", "c" * 40, cr.load_prompt(ROOT / "prompts/code_reader_v1.md"), "text")
    assert r.calls == 1 and len(json.loads((tmp_path / "ledger.json").read_text())) == 1


def test_the_reader_models_caps_and_prompt_come_from_config():
    config = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
    section = config["code_reader"]
    assert config["models"][section["first"]]["id"] == "gemini-3.5-flash-lite"
    assert config["models"][section["second_opinion"]]["id"] == "gemini-3.8-flash"
    assert section["prompt"] == "prompts/code_reader_v1.md" and section["max_calls_per_run"] <= 6
    assert section["eval"]["max_calls_per_run"] <= config["models"]["fallback"]["rpd"]
    source = (ROOT / "src/archwatch/code_reader.py").read_text()
    assert "url_context" not in source and "gemini-3" not in source  # no hard-coded model, no URL tool


def test_the_sdk_client_never_retries_on_its_own(tmp_path, monkeypatch):
    """Every SDK retry is a real request against the free daily limit, invisible to the ledger: turn them off."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config = copy.deepcopy(yaml.safe_load((ROOT / "config/providers.yaml").read_text()))
    config["code_reader"]["ledger"] = str(tmp_path / "ledger.json")
    client = cr.CodeReader(config, role="review", run_id="r1")._client()
    options = client._api_client._http_options
    assert options.retry_options.attempts == 1
    assert options.timeout == config["request_timeout_seconds"] * 1000


def test_booleans_must_be_booleans():
    """flash-lite answered tied_embeddings = "True" (a string) on LFM3 and it passed every check: no longer."""
    c = claim(field="tied_embeddings", value='"True"', line=20, keys=[])
    assert "tied_embeddings needs true or false, got 'True'" in assess(c)["agreement"]


def test_head_dim_from_a_plain_line_is_checked_through_the_projection_rows():
    plain = cr.Claim(
        field="head_dim",
        value="8",
        file="modeling_toy.py",
        line_start=4,
        line_end=4,
        quote=LINES[3],
        config_keys=["head_dim"],
    )
    assert PATHS.get(("modeling_toy.py", 4)) is None  # self.head_dim = config.head_dim builds nothing
    rows = cr.verdicts([claim(), plain], CODE_INFO, CONFIG, TENSORS, {})
    assert rows[1]["weight_status"] == "passed" and rows[1]["verdict"] == "verified"
    alone = cr.verdicts([plain], CODE_INFO, CONFIG, TENSORS, {})[0]
    assert alone["weight_status"] == "failed" and "no heads x head_dim check" in alone["problems"][-1]


def test_getattr_and_builtins_do_not_build_modules():
    code = {
        "modeling_g.py": "class A(nn.Module):\n    def __init__(self, c):\n        self.head_dim = getattr(c, 'head_dim', 8)\n        self.proj = nn.Linear(8, 8)\n"
    }
    paths = cr.state_paths(cr.module_tree(code), "A")
    assert ("modeling_g.py", 3) not in paths and paths[("modeling_g.py", 4)] == {"proj"}
