import copy
import json
import shutil
from pathlib import Path

import pytest

from archwatch.diagram_routes import browser_geometry, inspect_svg, passed, render_layout, specification

ROOT = Path(__file__).resolve().parents[1]


def spec():
    return dict(
        labels={
            "count": dict(text="layers: 2", role="layer_count", value=2),
            "width": dict(text="hidden_size: 8", role="fact", value=8),
        },
        nodes=[dict(id="main", labels=["count", "width"])],
        edges=[],
        groups=[],
        layer_count=2,
        bounds=dict(max_width=1200, max_height=1800),
    )


def layout():
    return dict(
        canvas=dict(width=400, height=200),
        nodes=[dict(id="main", x=30, y=20, width=300, height=100)],
        groups=[],
        labels=[dict(id="count", node="main"), dict(id="width", node="main")],
        edges=[],
    )


def test_constrained_layout_copies_all_labels_without_accepting_facts_from_model():
    s = spec()
    drawing = render_layout(layout(), s)
    result = inspect_svg(drawing, s)
    assert result["labels_and_numbers_from_published"] and result["label_coverage"]
    assert result["layer_sum_matches"] and result["labels_attached_to_nodes"]
    formatted = inspect_svg(
        drawing.replace("hidden_size: 8", "hidden_size:8").replace("url(#arrow)", "url('#arrow')"), s
    )
    assert formatted["labels_and_numbers_from_published"]
    bad = layout()
    bad["nodes"][0]["label"] = "invented width: 999"
    with pytest.raises(ValueError, match="Unexpected"):
        render_layout(bad, s)


def test_full_svg_cannot_hallucinate_numbers_or_duplicate_layer_counts():
    s = spec()
    drawing = render_layout(layout(), s)
    wrong = inspect_svg(drawing.replace("layers: 2", "layers: 3"), s)
    assert not wrong["labels_and_numbers_from_published"] and not wrong["layer_sum_matches"]
    duplicated = drawing.replace("</g>", '<text data-label-id="count">layers: 2</text></g>')
    result = inspect_svg(duplicated, s)
    assert not result["label_coverage"] and result["layer_sum"] == 4


@pytest.mark.parametrize(
    "content",
    [
        '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>',
        '<svg xmlns="http://www.w3.org/2000/svg"><style>@import "https://example.invalid/font";</style></svg>',
        '<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///private/file">]><svg/>',
    ],
)
def test_active_or_external_svg_never_reaches_browser(content):
    with pytest.raises(ValueError, match="Unsafe"):
        inspect_svg(content, spec())


@pytest.mark.parametrize("change", ["nan", "huge_canvas", "missing_edge", "duplicate_node"])
def test_layout_rejects_unbounded_or_incomplete_geometry(change):
    s = spec()
    value = layout()
    if change == "nan":
        value["nodes"][0]["x"] = float("nan")
    elif change == "huge_canvas":
        value["canvas"]["width"] = 1000000
    elif change == "missing_edge":
        s["edges"] = [dict(id="e", source="main", target="main")]
    else:
        value["nodes"].append(copy.deepcopy(value["nodes"][0]))
    with pytest.raises(ValueError):
        render_layout(value, s)


def test_spec_refuses_published_values_that_differ_from_verifier_decision():
    record = json.loads((ROOT / "data/models/meta-llama__Llama-3.1-8B.json").read_text())
    specification(record)
    record["published"]["hidden_size"] = 99999
    with pytest.raises(ValueError, match="differs"):
        specification(record)


BROWSER_AVAILABLE = bool(
    shutil.which("google-chrome") or Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").exists()
)


@pytest.mark.skipif(not BROWSER_AVAILABLE, reason="Installed Chrome needed for browser geometry fixtures")
@pytest.mark.parametrize("change", ["valid", "overlap", "translated_outside", "hidden"])
def test_browser_geometry_measures_rendered_transforms_and_visibility(change):
    s = spec()
    drawing = render_layout(layout(), s)
    if change == "overlap":
        drawing = drawing.replace('y="70"', 'y="48"')
    elif change == "translated_outside":
        drawing = drawing.replace("<g data-node-id", '<g transform="translate(500,0)" data-node-id')
    elif change == "hidden":
        drawing = drawing.replace("<g data-node-id", '<g opacity="0" data-node-id')
    checks = inspect_svg(drawing, s) | browser_geometry(drawing, s)
    if change == "valid":
        assert passed(checks)
    elif change == "overlap":
        assert not checks["no_text_overlap"]
    elif change == "translated_outside":
        assert not checks["everything_inside_canvas"]
    else:
        assert not checks["all_labels_visible"]
