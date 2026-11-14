import copy
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from archwatch import weights
from archwatch.diagram_checks import extract_svg, review, static_checks
from archwatch.diagram_spec import SpecError, build_spec, compress, dump
from archwatch.gemini import Gemini, ProviderError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
SETTINGS = CONFIG["diagram_gen"]


def record(stem):
    return json.loads((ROOT / "data/models" / f"{stem}.json").read_text())


def numbers(text):
    text = "\n".join("" if line.lstrip().startswith("#") else line.split(" #")[0] for line in text.splitlines())
    return {x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


# ---------------------------------------------------------------- spec builder


@pytest.mark.parametrize(
    "stem,kit,deviations",
    [
        # The kit mixes units: 33 KB is 33,792 / 1024 but 5.7 MB is 5,713,920 / 10^6.
        # The builder uses 1024 for both, giving 5.4 MB.
        ("moonshotai__Kimi-K3.1-Pro", "kimi-k3.1-pro", ({"5.7"}, {"5.4"})),
        ("LiquidAI__LFM3-1.7B", "lfm3-1.7b", (set(), set())),
    ],
)
def test_generated_spec_matches_hand_written_kit_numbers(stem, kit, deviations):
    rec = record(stem)
    body = dump(build_spec(rec), rec).split("\n\n", 1)[1]
    expected = numbers((ROOT / "docs/diagram-kit/specs" / f"{kit}.yaml").read_text())
    got = numbers(body)
    assert expected - got == deviations[0]
    assert got - expected == deviations[1]


def test_kimi_spec_structure_matches_kit():
    spec = build_spec(record("moonshotai__Kimi-K3.1-Pro"))
    view = spec["layer_pattern"]["compressed_view"]
    assert [(g["repeat"], len(g["layers"])) for g in view] == [(1, 4), (22, 4), (1, 1)]
    assert view[0]["layers"][0] == "KDA + dense FFN"
    assert set(spec["zoom_panels"]) == {"kda", "gated_dsa", "latent_moe", "dense_ffn"}
    assert "the word RoPE" in spec["do_not_include"]
    assert not any("rope" in p.lower().replace("no", "") for p in spec["zoom_panels"]["gated_dsa"]["parts"][2:3])
    assert spec["callouts"][-1].startswith("KV cache: about 33 KB")


@pytest.mark.parametrize(
    "stem",
    ["moonshotai__Kimi-K3.1-Pro", "LiquidAI__LFM3-1.7B", "nvidia__Nemotron-H-8B-Base-8K", "meta-llama__Llama-3.1-8B"],
)
def test_every_spec_layer_sum_and_panel_order(stem):
    spec = build_spec(record(stem))
    view = spec["layer_pattern"]["compressed_view"]
    assert sum(g["repeat"] * len(g["layers"]) for g in view) == spec["model"]["layers"]
    assert 1 <= len(spec["callouts"]) <= 5
    for panel in spec["zoom_panels"].values():
        assert panel["flow"] == "bottom to top, in the order listed" and panel["parts"]


def test_spec_requires_published_block_shapes():
    rec = record("LiquidAI__LFM3-1.7B")
    rec["published"].pop("block_shapes")
    with pytest.raises(SpecError, match="block_shapes"):
        build_spec(rec)


def test_spec_rejects_tensor_shapes_that_disagree_with_published_facts():
    rec = record("LiquidAI__LFM3-1.7B")
    rec["published"]["block_shapes"]["globals"]["embed_tokens.weight"] = [128000, 4096]
    with pytest.raises(SpecError, match="Embedding"):
        build_spec(rec)


@pytest.mark.parametrize(
    "seq",
    [
        list("KKKD" * 23 + "D"),
        list("CCGCSCGCSCGCSCGCSCGCSCGC"),
        list("ABCABDABCABDX"),
        list("A" * 7),
        list("ABCDEFG"),
        [],
    ],
)
def test_compress_always_covers_every_layer_once(seq):
    groups = compress(seq)
    flat = [x for g in groups for _ in range(g["repeat"]) for x in g["layers"]]
    assert flat == seq


def test_compress_splits_a_variant_repeat_into_its_own_group():
    seq = list("KKKD" * 23 + "D")
    labels = ["Kd"] + seq[1:]
    groups = compress(seq, labels)
    assert [(g["repeat"], g["layers"]) for g in groups] == [
        (1, ["Kd", "K", "K", "D"]),
        (22, ["K", "K", "K", "D"]),
        (1, ["D"]),
    ]


# ---------------------------------------------------------------- tensor shapes


def tensor(shape, dtype="BF16"):
    return {"dtype": dtype, "shape": shape}


def test_block_shapes_groups_by_config_type_and_ffn_signature():
    t = {
        "model.embed_tokens.weight": tensor([100, 8]),
        "model.layers.0.self_attn.q_proj.weight": tensor([8, 8]),
        "model.layers.0.mlp.gate_proj.weight": tensor([32, 8]),
        "model.layers.1.self_attn.q_proj.weight": tensor([8, 8]),
        "model.layers.1.mlp.experts.0.w1.weight": tensor([4, 8]),
        "model.layers.1.mlp.gate.weight": tensor([2, 8]),
        "model.layers.1.mlp.gate.weight_scale": tensor([2], "F32"),
        "model.layers.2.conv.in_proj.weight_packed": tensor([24, 4], "U8"),
        "model.layers.2.mlp.gate.weight": tensor([2, 8]),
        "model.vision_tower.x.weight": tensor([3, 3]),
    }
    out = weights.block_shapes(t, ["attn", "attn", "conv"], num_layers=3, quant_bits=4)
    assert out["mixers"]["attn"] == {"layers": 2, "tensors": {"self_attn.q_proj.weight": [8, 8]}}
    assert out["mixers"]["conv"]["tensors"] == {"conv.in_proj.weight": [24, 8]}  # 4-bit packed columns unpacked
    assert [g["layer_ranges"] for g in out["ffns"]] == [[[0, 0]], [[1, 2]]]  # experts excluded, scales skipped
    assert out["globals"] == {"embed_tokens.weight": [100, 8]}


def test_block_shapes_withholds_inconsistent_layers_of_one_type():
    t = {
        "model.layers.0.self_attn.q_proj.weight": tensor([8, 8]),
        "model.layers.1.self_attn.q_proj.weight": tensor([16, 8]),
    }
    assert weights.block_shapes(t, ["attn", "attn"], num_layers=2) is None
    assert weights.block_shapes(t, ["attn"], num_layers=2) is None  # pattern/header length mismatch


# ---------------------------------------------------------------- SVG checks


def good_svg(spec, *, height=1200, extra=""):
    """A deliberately plain SVG that satisfies the structural contract for `spec`."""
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 {height}">',
        '<rect width="1600" height="100%" fill="#fff"/>',
    ]
    y = 40

    def text(s, x=40, size=14):
        nonlocal y
        y += 22
        return f'<text x="{x}" y="{y}" font-size="{size}" font-family="Arial">{s}</text>'

    out.append(text(spec["model"]["title_label"], size=28))
    for i, g in enumerate(spec["layer_pattern"]["compressed_view"], 1):
        out.append(f'<g data-group="{i}" data-repeat="{g["repeat"]}">')
        for _ in g["layers"]:
            y += 6
            out.append(f'<rect data-layer="x" x="40" y="{y}" width="100" height="4" fill="#ccc"/>')
        out.append(text(f"{g['repeat']} x") + "</g>")
    for key, panel in spec["zoom_panels"].items():
        out.append(f'<g data-zoom="{key}">' + text(panel["title"], x=800) + "</g>")
    for i, c in enumerate(spec["callouts"], 1):
        out.append(f'<g data-callout="{i}">' + text(c, x=800) + "</g>")
    return "".join(out) + extra + "</svg>"


@pytest.fixture
def lfm_spec():
    return build_spec(record("LiquidAI__LFM3-1.7B"))


def test_static_checks_pass_on_a_contract_compliant_svg(lfm_spec):
    checks = static_checks(good_svg(lfm_spec), lfm_spec, SETTINGS)
    assert all(c["passed"] for c in checks.values()), checks


@pytest.mark.parametrize(
    "mutate,check",
    [
        (lambda s: s.replace("</svg>", '<text x="50" y="900">4096 dims</text></svg>'), "numbers_from_spec"),
        (lambda s: s.replace("</svg>", '<text x="50" y="900">RoPE MoE</text></svg>'), "labels_from_spec"),
        (lambda s: s.replace('data-repeat="5"', 'data-repeat="6"'), "layer_sum"),
        (lambda s: s.replace('data-zoom="gqa"', 'data-zoom="attention"'), "zoom_panels_present"),
        (lambda s: s.replace('data-callout="2"', 'data-callout="9"'), "callouts_present"),
        (lambda s: s.replace('viewBox="0 0 1600 1200"', 'viewBox="0 0 1200 900"'), "canvas_size"),
    ],
)
def test_static_checks_catch_each_failure(lfm_spec, mutate, check):
    checks = static_checks(mutate(good_svg(lfm_spec)), lfm_spec, SETTINGS)
    assert not checks[check]["passed"]


def test_do_not_include_catches_rope_for_nope_model():
    spec = build_spec(record("moonshotai__Kimi-K3.1-Pro"))
    svg = good_svg(spec).replace("</svg>", '<text x="50" y="900">RoPE</text></svg>')
    assert not static_checks(svg, spec, SETTINGS)["do_not_include"]["passed"]


def test_unsafe_svg_is_rejected_before_any_browser(lfm_spec):
    svg = good_svg(lfm_spec).replace("</svg>", '<image href="http://x/y.png"/></svg>')
    result = review(svg, lfm_spec, SETTINGS, browser=False)
    assert not result["passed"] and "Disallowed" in result["failures"][0]


def test_extract_svg_strips_markdown_fences():
    assert extract_svg("```svg\n<svg a='1'></svg>\n```") == "<svg a='1'></svg>"


def chrome_available():
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            p.chromium.launch(channel=SETTINGS["browser_channel"], headless=True).close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not chrome_available(), reason="installed Chrome is required for geometry checks")
def test_browser_checks_find_layout_problems(lfm_spec):
    clean = review(good_svg(lfm_spec), lfm_spec, SETTINGS)
    assert clean["passed"], clean["failures"]
    bad = good_svg(
        lfm_spec,
        extra=(
            '<text x="60" y="62" font-size="28">LFM3-1.7B</text>'  # overlaps the title
            '<text x="40" y="1195" font-size="14">2048</text>'  # inside the bottom margin
            '<text x="40" y="1000" font-size="14">7680</text><rect x="30" y="985" width="80" height="20" fill="#000"/>'
            '<line x1="0" y1="995" x2="200" y2="995" stroke="#000" stroke-dasharray="4 4"/>'
        ),
    )
    checks = review(bad, lfm_spec, SETTINGS)["checks"]
    for name in ("no_text_overlap", "inside_canvas_with_margin", "no_covered_text", "no_leader_lines"):
        assert not checks[name]["passed"], name
    assert not checks["no_lines_through_text"]["passed"]


# ---------------------------------------------------------------- provider adapter


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self, n):
        return self.body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_adapter_sends_system_instruction_and_enforces_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sent = []

    def opener(request, timeout):
        sent.append(json.loads(request.data))
        return FakeResponse({"candidates": [{"content": {"parts": [{"text": "<svg/>"}]}, "finishReason": "STOP"}]})

    config = copy.deepcopy(CONFIG)
    config["diagram_gen"]["max_generation_requests"] = 2
    clock = iter(range(0, 10_000, 100))
    client = Gemini(config, tmp_path / "ledger.json", opener=opener, clock=lambda: next(clock), wait=lambda s: None)
    assert client.generate("SYSTEM", "hello")[0] == "<svg/>"
    client.generate("SYSTEM", "again")
    with pytest.raises(ProviderError, match="budget"):
        client.generate("SYSTEM", "third")
    assert sent[0]["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    assert sent[0]["generationConfig"]["maxOutputTokens"] == config["diagram_gen"]["max_output_tokens"]
    assert len(json.loads((tmp_path / "ledger.json").read_text())) == 2
    assert "test-key" not in (tmp_path / "ledger.json").read_text()


def test_adapter_refuses_without_key_or_free_model(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="key"):
        Gemini(CONFIG, tmp_path / "l.json")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    config = copy.deepcopy(CONFIG)
    config["models"]["diagram"]["free"] = False
    with pytest.raises(ProviderError, match="free"):
        Gemini(config, tmp_path / "l.json")


def test_no_model_names_in_python_sources():
    ids = [m["id"] for m in CONFIG["models"].values()]
    for path in (ROOT / "src/archwatch").glob("*.py"):
        text = path.read_text()
        assert not any(i in text for i in ids), path


# ---------------------------------------------------------------- repair loop


def test_repair_loop_sends_failures_and_stops_on_pass(tmp_path, monkeypatch):
    import importlib.util

    spec_path = ROOT / "tools/diagram_gen.py"
    module_spec = importlib.util.spec_from_file_location("diagram_gen", spec_path)
    gen = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(gen)
    monkeypatch.setattr(gen, "OUT", tmp_path)
    spec = build_spec(record("LiquidAI__LFM3-1.7B"))
    good = good_svg(spec)
    bad = good.replace("</svg>", '<text x="50" y="900">4096</text></svg>')

    class Client:
        prompts = []

        def generate(self, system, prompt):
            self.prompts.append(prompt)
            return (bad if len(self.prompts) == 1 else "```svg\n" + good + "\n```"), {"usage": {}}

    monkeypatch.setattr(gen, "review", lambda svg, s, st, screenshot=None: review(svg, s, st, browser=False))
    client = Client()
    row = gen.run_model("LiquidAI/LFM3-1.7B", CONFIG, client)
    assert row["passed"] and row["final_version"] == 2 and len(client.prompts) == 2
    assert "number 4096" in client.prompts[1] and "PREVIOUS SVG" in client.prompts[1]
    folder = tmp_path / "LiquidAI__LFM3-1.7B"
    assert {p.name for p in folder.iterdir()} >= {"spec.yaml", "v1.svg", "v1.review.json", "v2.svg", "v2.review.json"}
    # Resume: saved SVGs are reused, no new calls.
    gen.run_model("LiquidAI/LFM3-1.7B", CONFIG, client)
    assert len(client.prompts) == 2
    shutil.rmtree(folder)
