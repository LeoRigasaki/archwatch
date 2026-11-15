import copy
import json
import re
import shutil
import textwrap
from html import escape
from pathlib import Path

import pytest
import yaml

from archwatch import weights
from archwatch.diagram_checks import extract_svg, review, static_checks
from archwatch.diagram_spec import MAX_WORDS, SpecError, _words, build_spec, compress, dump
from archwatch.gemini import Gemini, ProviderError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
SETTINGS = CONFIG["diagram_gen"]
SUBJECTS = [
    "moonshotai__Kimi-K3.1-Pro",
    "LiquidAI__LFM3-1.7B",
    "nvidia__Nemotron-H-8B-Base-8K",
    "meta-llama__Llama-3.1-8B",
]


def record(stem):
    return json.loads((ROOT / "data/models" / f"{stem}.json").read_text())


def numbers(text):
    text = "\n".join("" if line.lstrip().startswith("#") else line.split(" #")[0] for line in text.splitlines())
    return {x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def edges(panel):
    labels = {n["id"]: n["label"] for n in panel["nodes"]}
    return {(labels[a], labels[b]) for a, b in (e.split(" -> ") for e in panel["edges"])}


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
    spec = build_spec(rec)
    spec.pop("layout")  # page geometry, not architecture facts
    body = dump(spec, rec).split("\n\n", 1)[1]
    expected = numbers((ROOT / "docs/diagram-kit/specs" / f"{kit}.yaml").read_text())
    got = numbers(body)
    assert expected - got == deviations[0]
    assert got - expected == deviations[1]


def test_kimi_panels_encode_the_real_wiring():
    spec = build_spec(record("moonshotai__Kimi-K3.1-Pro"))
    view = spec["layer_pattern"]["compressed_view"]
    assert [(g["repeat"], len(g["layers"])) for g in view] == [(1, 4), (22, 4), (1, 1)]
    assert set(spec["zoom_panels"]) == {"kda", "gated_dsa", "latent_moe", "dense_ffn"}
    kda = edges(spec["zoom_panels"]["kda"])
    delta = "gated delta rule"
    # q, k, v -> conv -> delta rule; beta and decay are parallel inputs into the delta rule.
    assert ("q, k, v: Linear 7168 -> 12288", "Conv1d, kernel 4") in kda
    assert ("Conv1d, kernel 4", delta) in kda
    assert ("beta: Linear 7168 -> 96", delta) in kda and ("decay: Linear 7168 -> 128 -> 12288", delta) in kda
    assert ("beta: Linear 7168 -> 96", "decay: Linear 7168 -> 128 -> 12288") not in kda
    # output norm -> gate multiplied in -> output projection
    assert {(delta, "output norm (128)"), ("output norm (128)", "multiply")} <= kda
    assert ("output gate: Linear 7168 -> 12288", "multiply") in kda
    assert ("multiply", "output projection: 12288 -> 7168") in kda
    dsa = edges(spec["zoom_panels"]["gated_dsa"])
    attn = "sparse attention, NoPE"
    assert ("Linear 1536 -> 96 x 192", attn) in dsa and ("Linear 512 -> 96 x (128 + 128)", attn) in dsa
    assert ("select top 2048 tokens", attn) in dsa and ("indexer: 32 heads x 128", "select top 2048 tokens") in dsa
    moe = edges(spec["zoom_panels"]["latent_moe"])
    assert ("router: 896 experts, top 16", "16 experts: 3584 -> 3072 -> 3584") in moe
    assert ("2 shared experts (6144)", "sum") in moe and ("Linear 3584 -> 7168", "sum") in moe
    assert "the word RoPE" in spec["do_not_include"]
    assert not any("RoPE" in s for s in json.dumps(spec["zoom_panels"]).split('"') if "NoPE" not in s)


def test_parallel_parts_are_not_drawn_in_sequence():
    lfm = build_spec(record("LiquidAI__LFM3-1.7B"))["zoom_panels"]
    gqa = edges(lfm["gqa"])
    q, k, v = "q: Linear 2048 -> 2048", "k: Linear 2048 -> 512", "v: Linear 2048 -> 512"
    assert {("input (2048)", q), ("input (2048)", k), ("input (2048)", v)} <= gqa
    assert not ({(q, k), (k, v)} & gqa)
    assert ("input (2048)", "head gate: Linear 2048 -> 16") in gqa  # gate reads the layer input (modeling_lfm3.py)
    conv = edges(lfm["short_conv"])
    assert ("in_proj: Linear 2048 -> 6144", "multiply with C") in conv  # C skips the convolution
    ffn = edges(lfm["ffn"])
    assert ("w3: Linear 2048 -> 7680", "multiply") in ffn and ("SiLU", "multiply") in ffn
    mamba = edges(build_spec(record("nvidia__Nemotron-H-8B-Base-8K"))["zoom_panels"]["mamba2"])
    assert ("in_proj: Linear 4096 -> 18560", "norm (8192) gated by z") in mamba  # z branch


@pytest.mark.parametrize("stem", SUBJECTS)
def test_every_spec_is_well_formed(stem):
    spec = build_spec(record(stem))
    view = spec["layer_pattern"]["compressed_view"]
    assert sum(g["repeat"] * len(g["layers"]) for g in view) == spec["model"]["layers"]
    assert 1 <= len(spec["callouts"]) <= 5
    bars = {"bar:" + label for g in view for label in g["layers"]}
    assert all(c["beside"] in {"stack", "embedding"} | bars for c in spec["callouts"])
    assert set(spec["zoom_panels"]) == {k for col in spec["layout"]["zoom_columns"] for k in col["panels"]}
    assert len(spec["layout"]["zoom_columns"]) == (2 if len(spec["zoom_panels"]) > 3 else 1)
    for panel in list(spec["zoom_panels"].values()) + [spec["each_layer"]]:
        ids = {n["id"] for n in panel["nodes"]}
        assert all(_words(n["label"]) <= MAX_WORDS for n in panel["nodes"])
        assert all(set(e.split(" -> ")) <= ids for e in panel["edges"])
        touched = {x for e in panel["edges"] for x in e.split(" -> ")}
        assert touched == ids  # no orphan nodes
        assert not any("does not grow" in n["label"] for n in panel["nodes"])  # notes are captions


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
    assert [x for g in groups for _ in range(g["repeat"]) for x in g["layers"]] == seq


def test_compress_splits_a_variant_repeat_into_its_own_group():
    seq = list("KKKD" * 23 + "D")
    groups = compress(seq, ["Kd"] + seq[1:])
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
    assert out["mixers"]["conv"]["tensors"] == {"conv.in_proj.weight": [24, 8]}
    assert [g["layer_ranges"] for g in out["ffns"]] == [[[0, 0]], [[1, 2]]]
    assert out["globals"] == {"embed_tokens.weight": [100, 8]}


def test_block_shapes_withholds_inconsistent_layers_of_one_type():
    t = {
        "model.layers.0.self_attn.q_proj.weight": tensor([8, 8]),
        "model.layers.1.self_attn.q_proj.weight": tensor([16, 8]),
    }
    assert weights.block_shapes(t, ["attn", "attn"], num_layers=2) is None
    assert weights.block_shapes(t, ["attn"], num_layers=2) is None


# ---------------------------------------------------------------- a contract-compliant reference SVG


def good_svg(spec, *, extra="", drop_edge=None, add_edge=None):
    """A plain SVG that follows the fixed layout and draws every node and edge of `spec`."""
    layout = spec["layout"]
    W = layout["canvas_width"]
    t = lambda x, yy, s, a="start", size=14: (  # noqa: E731
        f'<text x="{x}" y="{yy}" font-size="{size}" font-family="Arial" text-anchor="{a}">{escape(str(s))}</text>'
    )

    def graph(attr, key, title, panel, x0, y0, out):
        out.append(f'<g {attr}="{key}">' + t(x0, y0 + 16, title))
        rows = {n["id"]: y0 + 30 + 30 * i for i, n in enumerate(panel["nodes"])}
        for n in panel["nodes"]:
            r = rows[n["id"]]
            out.append(
                f'<g data-node="{n["id"]}"><rect x="{x0}" y="{r}" width="360" height="22" fill="#fff" stroke="#333"/>'
            )
            out.append(t(x0 + 6, r + 16, n["label"]) + "</g>")
        lane = x0 + 380
        es = [e for e in panel["edges"] if e != drop_edge]
        if add_edge and set(add_edge.split(" -> ")) <= set(rows):
            es.append(add_edge)
        for e in es:
            a, b = e.split(" -> ")
            ya, yb = rows[a] + 11, rows[b] + 11
            out.append(
                f'<path data-edge="{a}->{b}" d="M{x0 + 360} {ya}H{lane}V{yb}H{x0 + 362}" fill="none" stroke="#333"/>'
            )
            lane += 6
        yy = y0 + 30 + 30 * len(panel["nodes"]) + 10
        for line in [w for c in panel["caption"] for w in textwrap.wrap(c, 64)]:
            out.append(t(x0, yy + 14, line))
            yy += 20
        out.append("</g>")
        return yy + 20

    # zoom columns first: the left column is then scaled to match their height
    zooms, bottom = ['<g data-section="zooms">'], 130
    for col in layout["zoom_columns"]:
        zy = 130
        for key in col["panels"]:
            zy = graph(
                "data-zoom", key, spec["zoom_panels"][key]["title"], spec["zoom_panels"][key], col["x"][0], zy, zooms
            )
        bottom = max(bottom, zy)
    zooms.append("</g>")
    each = []
    each_height = graph("data-each", "x", "Each layer", spec["each_layer"], 40, 0, each)
    n_bars = sum(len(g["layers"]) for g in spec["layer_pattern"]["compressed_view"])
    n_groups = len(spec["layer_pattern"]["compressed_view"])
    n_io = len(spec["io"]["top"]) + len(spec["io"]["bottom"])
    budget = (bottom - 130) - each_height - 40 - 34 * n_io - 26 * n_groups
    bar = max(8, min(34, budget // max(n_bars, 1) - 4))
    group_gap = 8 + max(0, (budget - n_bars * (bar + 4)) // max(n_groups, 1))

    out, y, marks, bars = [], 130, {}, {}
    out.append('<g data-section="title">' + t(W // 2, 50, spec["model"]["title_label"], "middle", 32))
    out.append(t(W // 2, 80, spec["model"]["summary_line"], "middle", 16) + "</g>")

    def io(item):
        nonlocal y
        marks[item["id"]] = y
        out.append(
            f'<g data-io="{item["id"]}"><rect x="40" y="{y}" width="400" height="24" fill="#fff" stroke="#333"/>'
        )
        out.append(t(48, y + 17, item["label"]) + "</g>")
        y += 34

    out.append('<g data-section="stack">')
    stack_top = y
    for item in reversed(spec["io"]["top"]):
        io(item)
    for i in range(n_groups, 0, -1):
        g = spec["layer_pattern"]["compressed_view"][i - 1]
        out.append(f'<g data-group="{i}" data-repeat="{g["repeat"]}">')
        top = y
        for label in reversed(g["layers"]):
            bars.setdefault(label, y + bar // 2)
            out.append(f'<rect data-layer="{escape(label)}" x="40" y="{y}" width="260" height="{bar}" fill="#ccc"/>')
            y += bar + 4
        out.append(t(320, max(y - 2, top + 14), f"{g['repeat']} x") + "</g>")
        y = max(y, top + 18) + group_gap
    groups_bottom = y
    if n_groups > 1:
        out.append(t(420, (stack_top + y) // 2, f"{spec['model']['layers']} layers"))
    for item in reversed(spec["io"]["bottom"]):
        io(item)
    wanted = []
    for i, c in enumerate(spec["callouts"], 1):
        if c["beside"] == "stack":
            cy = (stack_top + groups_bottom) // 2
        elif c["beside"] == "embedding":
            cy = marks["embedding"] + 12
        else:
            cy = bars[c["beside"][4:]]
        wanted.append((cy, i, textwrap.wrap(c["text"], 24)))
    floor = 0
    for cy, i, lines in sorted(wanted):
        h = 18 * len(lines)
        top = max(cy - h // 2, floor)
        floor = top + h + 6
        x = layout["callout_margin_x"][0]
        spans = "".join(
            f'<tspan x="{x}" dy="{0 if k == 0 else 18}">{escape(line)}</tspan>' for k, line in enumerate(lines)
        )
        out.append(
            f'<g data-callout="{i}"><text x="{x}" y="{top + 13}" font-size="14" font-family="Arial">{spans}</text></g>'
        )
    out.append("</g>")
    each_y = y + 20
    shifted = "".join(each).replace('<g data-each="x">', f'<g data-each="x" transform="translate(0 {each_y})">')
    out.append('<g data-section="each-layer">' + shifted + "</g>")
    height = max(each_y + each_height, bottom, 1000) + 40
    head = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}"><rect width="{W}" height="{height}" fill="#fff"/>'
    return head + "".join(out) + "".join(zooms) + extra + "</svg>"


@pytest.fixture
def lfm_spec():
    return build_spec(record("LiquidAI__LFM3-1.7B"))


def test_static_checks_pass_on_a_contract_compliant_svg(lfm_spec):
    checks = static_checks(good_svg(lfm_spec), lfm_spec, SETTINGS)
    assert all(c["passed"] for c in checks.values()), [c["problems"] for c in checks.values() if not c["passed"]]


@pytest.mark.parametrize(
    "mutate,check",
    [
        (lambda s: s.replace("</svg>", '<text x="50" y="900">4096 dims</text></svg>'), "numbers_from_spec"),
        (lambda s: s.replace("</svg>", '<text x="50" y="900">RoPE MoE</text></svg>'), "labels_from_spec"),
        (lambda s: s.replace('data-repeat="5"', 'data-repeat="6"'), "layer_sum"),
        (lambda s: s.replace('data-zoom="gqa"', 'data-zoom="attention"'), "zoom_panels_present"),
        (lambda s: s.replace('data-callout="2"', 'data-callout="9"'), "callouts_present"),
        (lambda s: re.sub(r'viewBox="0 0 (\d+) (\d+)"', r'viewBox="0 0 1200 \2"', s), "canvas_size"),
        (lambda s: s.replace(">24 layers<", ">24 x<"), "repeat_counts_once"),
        (lambda s: s.replace('data-section="each-layer"', 'data-section="layer"'), "fixed_layout_sections"),
        (lambda s: s.replace(">multiply with C<", ">multiply C<"), "graph_nodes_match_spec"),
    ],
)
def test_static_checks_catch_each_failure(lfm_spec, mutate, check):
    checks = static_checks(mutate(good_svg(lfm_spec)), lfm_spec, SETTINGS)
    assert not checks[check]["passed"]


def test_edge_checks_catch_missing_and_invented_edges(lfm_spec):
    missing = static_checks(good_svg(lfm_spec, drop_edge="in_proj -> mul_c"), lfm_spec, SETTINGS)
    assert any("in_proj->mul_c is missing" in p for p in missing["graph_edges_match_spec"]["problems"])
    # A sequential q -> k wire is exactly the wrong wiring a flat list produced.
    extra = static_checks(good_svg(lfm_spec, add_edge="q -> k"), lfm_spec, SETTINGS)
    assert any("q->k is drawn but not in the spec" in p for p in extra["graph_edges_match_spec"]["problems"])


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


CHROME = pytest.mark.skipif(not chrome_available(), reason="installed Chrome is required for geometry checks")


@CHROME
@pytest.mark.parametrize("stem", SUBJECTS)
def test_reference_svg_passes_every_check(stem):
    spec = build_spec(record(stem))
    result = review(good_svg(spec), spec, SETTINGS)
    assert result["passed"], result["failures"]


@CHROME
def test_browser_checks_find_layout_problems(lfm_spec):
    bad = good_svg(
        lfm_spec,
        extra=(
            '<text x="{X}" y="50" font-size="28" text-anchor="middle">LFM3-1.7B</text>'
            '<text x="40" y="{H}" font-size="14">2048</text>'
            '<text x="1000" y="2000" font-size="14">7680</text><rect x="990" y="1985" width="80" height="20" fill="#000"/>'
            '<line x1="960" y1="1995" x2="1200" y2="1995" stroke="#000" stroke-dasharray="4 4"/>'
        ),
    )
    width, height = map(int, re.search(r'viewBox="0 0 (\d+) (\d+)"', bad).groups())
    bad = (
        bad.replace("{X}", str(width // 2))
        .replace("{H}", str(height - 4))
        .replace(f'0 0 {width} {height}"', f'0 0 {width} 2100"')
    )
    checks = review(bad, lfm_spec, SETTINGS)["checks"]
    for name in (
        "no_text_overlap",
        "inside_canvas_with_margin",
        "no_covered_text",
        "no_leader_lines",
        "no_lines_through_text",
    ):
        assert not checks[name]["passed"], name


@CHROME
def test_layer_bars_must_match_spec_bottom_to_top(lfm_spec):
    svg = good_svg(lfm_spec)
    swapped = (
        svg.replace('data-layer="Global GQA + FFN"', 'data-layer="TMP"')
        .replace('data-layer="Sliding-window GQA + FFN"', 'data-layer="Global GQA + FFN"')
        .replace('data-layer="TMP"', 'data-layer="Sliding-window GQA + FFN"')
    )
    check = review(swapped, lfm_spec, SETTINGS)["checks"]["layer_bars_match_spec"]
    assert not check["passed"] and "group 2" in check["problems"][0]


@CHROME
def test_mislabelled_arrow_and_misplaced_parts_are_caught(lfm_spec):
    svg = good_svg(lfm_spec)
    # Relabel one drawn arrow so its attribute claims a different source: geometry no longer matches.
    wrong = svg.replace('data-edge="in->q"', 'data-edge="TMP"').replace('data-edge="in->v"', 'data-edge="in->q"')
    wrong = wrong.replace('data-edge="TMP"', 'data-edge="in->v"')
    checks = review(wrong, lfm_spec, SETTINGS)["checks"]
    assert not checks["edges_connect_their_nodes"]["passed"]
    x = lfm_spec["layout"]["callout_margin_x"][0]
    moved = re.sub(r'(<g data-callout="3">.*?</g>)', lambda m: m.group(1).replace(f'x="{x}"', 'x="1300"'), svg)
    assert not review(moved, lfm_spec, SETTINGS)["checks"]["callouts_beside_their_part"]["passed"]
    swapped = svg.replace('<g data-section="zooms">', '<g data-section="zooms" transform="translate(-880 0)">')
    assert not review(swapped, lfm_spec, SETTINGS)["checks"]["fixed_layout_positions"]["passed"]


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


def test_adapter_counts_every_call_against_the_one_daily_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sent = []

    def opener(request, timeout):
        sent.append(json.loads(request.data))
        return FakeResponse({"candidates": [{"content": {"parts": [{"text": "<svg/>"}]}, "finishReason": "STOP"}]})

    config = copy.deepcopy(CONFIG)
    config["models"]["diagram"]["rpd"] = 3
    ledger = tmp_path / "ledger.json"
    # A call made by another tool earlier the same day already counts.
    ledger.write_text(
        json.dumps([dict(time=0, day="1969-12-31", model=config["models"]["diagram"]["id"], experiment="other")])
    )
    clock = iter(range(100, 10_000, 100))
    client = Gemini(config, ledger, opener=opener, clock=lambda: next(clock), wait=lambda s: None)
    assert client.generate("SYSTEM", "hello")[0] == "<svg/>"
    client.generate("SYSTEM", "again")
    with pytest.raises(ProviderError, match="daily"):
        client.generate("SYSTEM", "third")
    assert sent[0]["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    assert sent[0]["generationConfig"]["maxOutputTokens"] == config["diagram_gen"]["max_output_tokens"]
    assert len(json.loads(ledger.read_text())) == 3
    assert "test-key" not in ledger.read_text()


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
    for path in list((ROOT / "src/archwatch").glob("*.py")) + list((ROOT / "tools").glob("*.py")):
        assert not any(i in path.read_text() for i in ids), path


# ---------------------------------------------------------------- repair loop


def test_repair_loop_sends_failures_and_stops_on_pass(tmp_path, monkeypatch):
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("diagram_gen", ROOT / "tools/diagram_gen.py")
    gen = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(gen)
    monkeypatch.setattr(gen, "OUT", tmp_path)
    spec = build_spec(record("LiquidAI__LFM3-1.7B"))
    good = good_svg(spec)
    bad = good.replace("</svg>", '<text x="50" y="900">4096</text></svg>')

    class Client:
        prompts = []

        images = []

        def generate(self, system, prompt, images=()):
            self.prompts.append(prompt)
            self.images.append(list(images))
            return (bad if len(self.prompts) == 1 else "```svg\n" + good + "\n```"), {"usage": {}}

    def fake_review(svg, s, st, screenshot=None):
        Path(screenshot).write_bytes(b"PNG-of-" + str(Path(screenshot).name).encode())
        return review(svg, s, st, browser=False)

    monkeypatch.setattr(gen, "review", fake_review)
    client = Client()
    row = gen.run_model("LiquidAI/LFM3-1.7B", CONFIG, client)
    assert row["passed"] and row["final_version"] == 2 and len(client.prompts) == 2
    assert "number 4096" in client.prompts[1] and "PREVIOUS SVG" in client.prompts[1]
    reference_png = (ROOT / "docs/diagram-kit/reference.png").read_bytes()
    assert client.images[0] == [reference_png] and "HOUSE STYLE" in client.prompts[0]
    assert client.images[1] == [b"PNG-of-v1.png", reference_png]  # render of the failed attempt + house style
    folder = tmp_path / "LiquidAI__LFM3-1.7B"
    assert {p.name for p in folder.iterdir()} >= {"spec.yaml", "v1.svg", "v1.review.json", "v2.svg", "v2.review.json"}
    gen.run_model("LiquidAI/LFM3-1.7B", CONFIG, client)  # resume: saved SVGs are reused, no new calls
    assert len(client.prompts) == 2
    shutil.rmtree(folder)


@CHROME
def test_line_through_an_unconnected_box_is_caught():
    spec = build_spec(record("moonshotai__Kimi-K3.1-Pro"))
    svg = good_svg(spec)
    assert review(svg, spec, SETTINGS)["checks"]["lines_avoid_other_boxes"]["passed"]
    x0 = spec["layout"]["zoom_columns"][0]["x"][0]
    # A straight line across a row of the first zoom panel, like the Kimi DSA lines that hugged boxes.
    crossing = svg.replace("</svg>", f'<line x1="{x0 - 60}" y1="171" x2="{x0 + 520}" y2="171" stroke="#333"/></svg>')
    check = review(crossing, spec, SETTINGS)["checks"]["lines_avoid_other_boxes"]
    assert not check["passed"] and "crosses or runs along" in check["problems"][0]


@CHROME
def test_unbalanced_columns_are_caught(lfm_spec):
    svg = re.sub(
        r'transform="translate\(0 (\d+)\)"',
        lambda m: f'transform="translate(0 {int(m.group(1)) + 900})"',
        good_svg(lfm_spec),
    )
    svg = re.sub(r'viewBox="0 0 (\d+) (\d+)"', lambda m: f'viewBox="0 0 {m.group(1)} 2400"', svg)
    check = review(svg, lfm_spec, SETTINGS)["checks"]["columns_balanced"]
    assert not check["passed"] and "under 25%" in check["problems"][0]


def test_repairs_stop_after_three_calls(tmp_path, monkeypatch):
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("diagram_gen", ROOT / "tools/diagram_gen.py")
    gen = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(gen)
    monkeypatch.setattr(gen, "OUT", tmp_path)
    bad = good_svg(build_spec(record("LiquidAI__LFM3-1.7B"))).replace(
        "</svg>", '<text x="50" y="900">4096</text></svg>'
    )

    class Client:
        calls = 0

        def generate(self, system, prompt, images=()):
            self.calls += 1
            return bad, {"usage": {}}

    def fake_review(svg, s, st, screenshot=None):
        Path(screenshot).write_bytes(b"png")
        return review(svg, s, st, browser=False)

    monkeypatch.setattr(gen, "review", fake_review)
    client = Client()
    row = gen.run_model("LiquidAI/LFM3-1.7B", CONFIG, client)
    assert client.calls == 3 and not row["passed"] and row["final_version"] == 3


# ---------------------------------------------------------------- house-style reference


def test_reference_must_be_archwatch_output_with_matching_hashes(tmp_path):
    from archwatch import house_style

    svg_text, png = house_style.load(ROOT)  # the committed reference is valid
    assert svg_text.startswith("<svg") and png[:8] == b"\x89PNG\r\n\x1a\n"
    meta = json.loads((ROOT / "docs/diagram-kit/reference.json").read_text())
    assert meta["origin"] == "archwatch-generated" and meta["source"].startswith("results/diagram-gen")

    (tmp_path / "docs/diagram-kit").mkdir(parents=True)
    for name in house_style.NAMES:
        shutil.copy(ROOT / "docs/diagram-kit" / name, tmp_path / "docs/diagram-kit" / name)
    (tmp_path / "docs/diagram-kit/reference.svg").write_text(svg_text + "<!-- edited -->")
    with pytest.raises(house_style.ReferenceError, match="hashes"):
        house_style.load(tmp_path)
    for outside in ("downloads/gallery.svg", "research-reference/x.svg", "docs/other.png"):
        with pytest.raises(house_style.ReferenceError, match="ArchWatch"):
            house_style.approve(tmp_path, outside)


def test_adapter_sends_images_as_inline_png(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = []

    def opener(request, timeout):
        sent.append(json.loads(request.data))
        return FakeResponse({"candidates": [{"content": {"parts": [{"text": "<svg/>"}]}}]})

    Gemini(CONFIG, tmp_path / "l.json", opener=opener, wait=lambda s: None).generate("S", "P", [b"\x89PNG-a", b"b"])
    parts = sent[0]["contents"][0]["parts"]
    assert parts[0] == {"text": "P"} and [p["inline_data"]["mime_type"] for p in parts[1:]] == ["image/png"] * 2


def test_single_group_shows_its_count_once():
    spec = build_spec(record("meta-llama__Llama-3.1-8B"))
    assert len(spec["layer_pattern"]["compressed_view"]) == 1
    svg = good_svg(spec)
    assert static_checks(svg, spec, SETTINGS)["repeat_counts_once"]["passed"]
    doubled = svg.replace("</svg>", '<text x="420" y="300" font-size="14">32 layers</text></svg>')
    check = static_checks(doubled, spec, SETTINGS)["repeat_counts_once"]
    assert not check["passed"] and "drop the brace" in check["problems"][0]


def test_rules_for_direct_q_mla_parallel_blocks_and_fc_mlps():
    ds = build_spec(record("deepseek-ai__DeepSeek-V2-Lite"))
    mla = edges(ds["zoom_panels"]["mla"])
    assert ("input (2048)", "q: Linear 2048 -> 16 x 192") in mla  # no query latent in this model
    assert any("yarn scaling, factor 40" in c for c in ds["zoom_panels"]["mla"]["caption"])
    assert {"mla", "dense_ffn", "moe"} == set(ds["zoom_panels"])
    phi = build_spec(record("microsoft__phi-2"))
    each = edges(phi["each_layer"])
    assert ("norm (shared)", "FFN") in each and ("norm (shared)", "attention: MHA") in each  # parallel block
    assert ("layer input", "add input and both branches") in each
    assert ("output projection: 2560 -> 2560", "output (2560)") in edges(phi["zoom_panels"]["mha"])  # 'dense'
    ffn = edges(phi["zoom_panels"]["ffn"])
    assert ("fc1: Linear 2560 -> 10240", "GELU") in ffn and phi["zoom_panels"]["ffn"]["title"] == "FFN"
    gemma = build_spec(record("google__gemma-3-1b-pt"))
    assert gemma["io"]["top"][-1]["label"] == "LM head (tied to the token embedding)"
    assert gemma["zoom_panels"]["ffn"]["title"] == "GeGLU FFN"
