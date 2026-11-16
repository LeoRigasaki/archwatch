"""The compare tool: data written from published values only, shareable URL state, chart numbers = calculator."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def data():
    return json.loads((SITE / "data/models.json").read_text())


def builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("site_builder_compare", ROOT / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_holds_published_values_only():
    site = builder()
    published, held, _ = site.inventory()
    d = data()
    assert {m["id"] for m in d["models"]} == set(published)
    for repo in held:
        assert repo not in json.dumps(d)  # held records are not in the data at all
    for m in d["models"]:
        _, rec, pub = published[m["id"]]
        in_review = {r["field"] for r in rec["review_queue"]}
        for field, (shown, raw) in m["fields"].items():
            if field == "cache":
                continue
            assert field in pub and field not in in_review, (m["id"], field)
            assert shown == site.display(field, pub[field])
        assert m["layers"] == pub["layer_types"] and m["counts"] == pub["layer_type_counts"]
        assert m["params"]["total"] == pub["params_total"] and m["context"] == pub["context_length"]
        for field in in_review:
            assert field not in m["fields"]


def test_chart_numbers_match_the_calculator():
    site = builder()
    published, _, _ = site.inventory()
    for m in data()["models"]:
        cache = site.render_cache(published[m["id"]][2])
        c = m["cache"]
        assert c["grow"] + 0 == cache["grows_per_token_bytes"]
        assert c["fixed"] + c["window_full"] == cache["constant_bytes"]
        assert m["series"][-1][0] == m["context"]
        for n, value in m["series"]:
            assert n <= m["context"]
            if n >= c["window"]:  # window full (or none): exactly the calculator's grows x n + constant
                assert value == cache["grows_per_token_bytes"] * n + cache["constant_bytes"], (m["id"], n)
            else:  # windowed layers have only n of their window tokens so far
                assert value == c["grow"] * n + c["window_full"] * n // c["window"] + c["fixed"]
    mistral = next(m for m in data()["models"] if m["id"] == "mistralai/Mistral-7B-v0.1")
    assert mistral["cache"]["grow"] == 0 and mistral["cache"]["window"] == 4096  # capped, never shown as 0 B


def node(script):
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def test_url_state_round_trips_and_ignores_unknown_or_repeated_models():
    js = json.dumps(str(SITE / "compare.js"))
    result = node(
        f"""const AW=require({js});const stems=['x','y','z','w'];
        console.log(JSON.stringify([
          AW.parseState('?a=x&b=y&c=z',stems,null),
          AW.parseState('?a=x&b=x&c=nope',stems,['w','z']),
          AW.parseState('',stems,['y','x']),
          AW.stateQuery(['x','y']), AW.stateQuery(['x','y','z']),
          AW.parseState(AW.stateQuery(['z','w','x']),stems,null)]))"""
    )
    assert result[0] == ["x", "y", "z"]
    assert result[1] == ["w", "z"]  # one valid model is not a comparison: the page preset is used
    assert result[2] == ["y", "x"]
    assert result[3] == "?a=x&b=y" and result[4] == "?a=x&b=y&c=z"
    assert result[5] == ["z", "w", "x"]


def test_browser_cache_formula_matches_python():
    site = builder()
    js = json.dumps(str(SITE / "compare.js"))
    models = data()["models"]
    points = [512, 4096, 8192, 131072]
    got = node(
        f"""const AW=require({js});const d=require({json.dumps(str(SITE / "data/models.json"))});
        console.log(JSON.stringify(d.models.map(m=>{json.dumps(points)}.map(n=>AW.cacheBytes(m.cache,n)))))"""
    )
    for m, values in zip(models, got):
        assert values == [site.cache_bytes(m["cache"], n) for n in points], m["id"]


def test_pair_pages_are_prefilled_and_the_tool_works_without_js_as_a_list():
    d = data()
    preds = [s for s in d["suggested"] if s["why"] == "predecessor"]
    assert preds  # every predecessor pair is suggested first
    for s in preds:
        page = (SITE / "compare" / f"{s['b']}.html").read_text()
        assert f'data-a="{s["a"]}" data-b="{s["b"]}"' in page
    index = (SITE / "compare/index.html").read_text()
    assert "<noscript>" in index and f'href="../compare/{preds[0]["b"]}.html"' in index
    assert 'src="../compare.js"' in index
    js = (SITE / "compare.js").read_text()
    assert "http" not in js.replace("http://www.w3.org/2000/svg", "")  # no external requests from the tool
    whys = {s["why"] for s in d["suggested"]}
    assert any(w.startswith("closest total parameters") for w in whys)


def test_key_rows_first_merged_fields_once_and_rest_collapsed():
    site = builder()
    d = data()
    assert d["order"] == [f for f in site.KEY_ROWS if f in d["labels"]]
    assert not set(d["order"]) & set(d["rest"]) and not site.MERGED & set(d["labels"])
    assert d["labels"]["ffn"] == "FFN hidden size" and "dense_ffn_intermediate" not in d["rest"]


def test_values_the_architecture_implies_are_derived_not_not_stated():
    site = builder()
    published, _, _ = site.inventory()
    models = {m["id"]: m for m in data()["models"]}
    llama, pub = models["meta-llama/Llama-3.1-8B"], published["meta-llama/Llama-3.1-8B"][2]
    comp = pub["parameter_components"]
    assert (
        llama["derived"]["params_active_non_embedding"][1] == pub["params_total"] - comp["embedding"] - comp["lm_head"]
    )
    assert llama["derived"]["v_head_dim"] == [f"{pub['head_dim']:,} (same as head size)", pub["head_dim"]]
    assert llama["derived"]["ffn"][1] == pub["ffn_intermediate"] and llama["derived"]["experts"][0] == "none (dense)"
    lfm = models["LiquidAI/LFM3-1.7B-Base"]  # tied embeddings: the shared matrix is subtracted once
    lpub = published["LiquidAI/LFM3-1.7B-Base"][2]
    assert (
        lfm["derived"]["params_active_non_embedding"][1]
        == lpub["params_total"] - lpub["parameter_components"]["embedding"]
    )
    ds = models["deepseek-ai/DeepSeek-V2-Lite"]
    assert "v_head_dim" not in ds["derived"]  # MLA has its own value head size; nothing is assumed
    assert ds["derived"]["ffn"][0].endswith("(dense layers)")
