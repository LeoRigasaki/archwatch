"""Fact-locked inputs, deterministic layout rendering and route validators."""

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from html import escape

from .diagram import KIND
from .publication import render_values


def specification(record):
    v = render_values(record)
    decisions = {c["field"]: c for c in record["checks"]}
    for key, value in v.items():
        if json.dumps(value, sort_keys=True) != json.dumps(decisions[key].get("value"), sort_keys=True):
            raise ValueError("Published value differs from its verifier decision")
    labels, nodes = {}, []

    def label(key, text, path, value, role="fact"):
        labels[key] = dict(text=text, source_path=path, value=value, role=role)
        return key

    def field(key):
        value = v[key]
        return label("f_" + key, f"{key}: {str(value).lower() if isinstance(value, bool) else value}", "/" + key, value)

    def node(key, fields, title=None):
        ids = []
        if title:
            ids.append(title)
        ids += [field(f) for f in fields if f in v]
        nodes.append(dict(id=key, labels=ids))

    node("embedding", ["vocab_size", "hidden_size"])
    mixer_ids = []
    for kind, count in v["layer_type_counts"].items():
        key = "mix_" + kind
        mixer_ids.append(key)
        title = label("kind_" + kind, KIND.get(kind, (kind,))[0], "/layer_type_counts/" + kind, count, "kind")
        count_id = label("count_" + kind, f"layers: {count}", "/layer_type_counts/" + kind, count, "layer_count")
        fields = {
            "kda_linear": ["linear_attn_num_heads", "linear_attn_head_dim", "linear_attn_conv_kernel"],
            "gated_dsa_mla": [
                "mla_kv_lora_rank",
                "mla_qk_rope_head_dim",
                "dsa_index_num_heads",
                "dsa_index_head_dim",
                "dsa_index_topk",
            ],
            "short_conv": ["short_conv_kernel"],
            "mamba2": [
                "mamba_num_heads",
                "mamba_head_dim",
                "mamba_state_size",
                "mamba_groups",
                "mamba_conv_kernel",
                "mamba_conv_channels",
            ],
            "mlp": ["ffn_intermediate", "activation", "ffn_gated"],
            "sliding_attention": ["sliding_window"],
        }.get(kind, [])
        if kind in ("full_attention", "sliding_attention"):
            fields = fields + ["num_attention_heads", "num_kv_heads", "head_dim"]
        # Shared facts appear once in a facts node, not duplicated visible labels.
        fresh = [f for f in fields if "f_" + f not in labels]
        node(key, fresh, title)
        nodes[-1]["labels"].insert(1, count_id)
    standalone = v.get("block_layout") == "standalone"
    if not standalone:
        node(
            "ffn",
            [
                "moe_num_experts",
                "moe_active_experts",
                "moe_shared_experts",
                "moe_expert_intermediate",
                "moe_latent_dim",
                "dense_layers",
                "dense_ffn_intermediate",
                "ffn_intermediate",
                "activation",
                "ffn_gated",
            ],
        )
    node("summary", ["num_layers", "context_length", "params_total"])
    for item in nodes:
        item["min_height"] = 32 + 22 * len(item["labels"])
        item["min_width"] = max(240, 32 + 8 * max(len(labels[k]["text"]) for k in item["labels"]))
    edges = []
    for key in mixer_ids:
        edges.append(dict(id="embedding_to_" + key, source="embedding", target=key))
        edges.append(dict(id=key + "_out", source=key, target="summary" if standalone else "ffn"))
    if not standalone:
        edges.append(dict(id="ffn_out", source="ffn", target="summary"))
    result = dict(
        schema_version=1,
        source_id=record["id"],
        source_commit=record["source_commit"],
        labels=labels,
        nodes=nodes,
        groups=[],
        edges=edges,
        layer_count=v["num_layers"],
        bounds=dict(max_width=1200, max_height=1800),
        scope="Layer-kind architecture overview; semantic nodes, labels and edges fixed by published data. Only layout varies.",
    )
    result["spec_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def prompt(spec, route):
    common = """Arrange this fixed architecture overview with top-to-bottom flow (embedding above mixers, summary below). Do not add or infer any architecture fact, node, label or edge.
Use every supplied node exactly once and every label exactly once, attached to its supplied node.
Copy label text verbatim. Do not add visible titles, legends or annotations. Numbers in visible labels are facts; geometric coordinates are not.
Keep text readable at 13px or larger, with generous padding; no overlapping text or nodes and nothing outside the canvas.
Keep every supplied edge, with arrows, but do not add edges. You may choose positions, sizes, canvas and colors.
Make a professional, restrained technical diagram with clear hierarchy. Model names belong to external captions, not visible SVG text.
"""
    if route == "A":
        instructions = """Return ONLY a complete standalone SVG (no Markdown). Use an explicit viewBox='0 0 WIDTH HEIGHT'.
Use only svg, g, rect, text, tspan, path, line, polyline, polygon, defs, marker, style and title elements.
Put each node in a <g data-node-id='NODE_ID'> containing one main <rect> and its text labels.
Each <text data-label-id='LABEL_ID'> must contain the exact corresponding label text.
Put data-edge-id='EDGE_ID' on each edge path/line. Use system fonts; no external resources, scripts, images, foreignObject or HTML.
Make dimensions fit the max_width/max_height limits. Do not hide labels or use transforms to fake validation.
"""
    elif route == "B":
        instructions = """Return ONLY a JSON object (no Markdown) with exactly:
{"canvas":{"width":960,"height":1100},"nodes":[{"id":"supplied node ID","x":0,"y":0,"width":280,"height":180}],"groups":[],"labels":[{"id":"supplied label ID","node":"its supplied node ID"}],"edges":["supplied edge ID"]}.
Every supplied node/edge ID must occur exactly once. groups must be empty for this experiment.
Return every supplied label ID exactly once, on its supplied node. You may order labels within a node.
Do not return label text or facts: our renderer inserts the published text for each ID.
Each node needs height >= 32 + 22 * number_of_labels and width enough for its longest label at Arial 13px.
The renderer draws centered labels with a 22px line spacing and routes arrows between node centers. Plan clear spacing.
"""
    else:
        raise ValueError("Unknown route")
    return common + instructions + "\nFIXED SPECIFICATION:\n" + json.dumps(spec, ensure_ascii=False)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Layout coordinates must be finite numbers")
    return value


def render_layout(layout, spec):
    if set(layout) != {"canvas", "nodes", "groups", "labels", "edges"} or layout["groups"] != []:
        raise ValueError("Unexpected layout fields/groups")
    canvas = layout["canvas"]
    if set(canvas) != {"width", "height"}:
        raise ValueError("Invalid canvas")
    w, h = _number(canvas["width"]), _number(canvas["height"])
    if not (0 < w <= spec["bounds"]["max_width"] and 0 < h <= spec["bounds"]["max_height"]):
        raise ValueError("Canvas exceeds experiment bounds")
    expected = {n["id"]: n for n in spec["nodes"]}
    expected_labels = Counter((label, node["id"]) for node in spec["nodes"] for label in node["labels"])
    if (
        any(set(label) != {"id", "node"} for label in layout["labels"])
        or Counter((label["id"], label["node"]) for label in layout["labels"]) != expected_labels
    ):
        raise ValueError("Layout labels differ from the fixed published label references")
    positions = {}
    for item in layout["nodes"]:
        if set(item) != {"id", "x", "y", "width", "height"} or item["id"] not in expected or item["id"] in positions:
            raise ValueError("Unexpected or duplicated node")
        positions[item["id"]] = {k: _number(item[k]) for k in ("x", "y", "width", "height")}
        if item["width"] <= 0 or item["height"] < 32 + 22 * len(expected[item["id"]]["labels"]):
            raise ValueError("Node cannot contain the required label lines")
    if set(positions) != set(expected) or Counter(layout["edges"]) != Counter(e["id"] for e in spec["edges"]):
        raise ValueError("Missing or duplicated nodes/edges")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#687686"/></marker></defs>',
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
    ]
    for edge in spec["edges"]:
        a, b = positions[edge["source"]], positions[edge["target"]]
        x1, y1 = a["x"] + a["width"] / 2, a["y"] + a["height"]
        x2, y2 = b["x"] + b["width"] / 2, b["y"]
        mid = (y1 + y2) / 2
        out.append(
            f'<path data-edge-id="{edge["id"]}" d="M{x1} {y1}V{mid}H{x2}V{y2}" fill="none" stroke="#687686" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )
    for node in spec["nodes"]:
        p = positions[node["id"]]
        out.append(
            f'<g data-node-id="{node["id"]}"><rect x="{p["x"]}" y="{p["y"]}" width="{p["width"]}" height="{p["height"]}" rx="10" fill="#f3f7fb" stroke="#8095aa"/>'
        )
        node_labels = [label["id"] for label in layout["labels"] if label["node"] == node["id"]]
        for index, label in enumerate(node_labels):
            text = escape(spec["labels"][label]["text"])
            out.append(
                f'<text data-label-id="{label}" x="{p["x"] + p["width"] / 2}" y="{p["y"] + 28 + 22 * index}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#182c40">{text}</text>'
            )
        out.append("</g>")
    return "".join(out) + "</svg>"


def inspect_svg(svg, spec):
    def external_url(value):
        return any(not url.strip().strip("\"'").startswith("#") for url in re.findall(r"url\(([^)]*)\)", value, re.I))

    def canonical(text):
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        return re.sub(r"\s*:\s*", ":", " ".join(text.split()))

    if len(svg.encode()) > 1024 * 1024 or re.search(r"<!DOCTYPE|<!ENTITY", svg, re.I):
        raise ValueError("Unsafe or oversized SVG")
    root = ET.fromstring(svg)
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("Output is not SVG")
    allowed = {
        "svg",
        "g",
        "rect",
        "text",
        "tspan",
        "path",
        "line",
        "polyline",
        "polygon",
        "defs",
        "marker",
        "style",
        "title",
    }
    texts, nodes, edges = [], [], []
    parents = {child: parent for parent in root.iter() for child in parent}
    expected_owner = {label: node["id"] for node in spec["nodes"] for label in node["labels"]}
    misplaced = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in allowed:
            raise ValueError("Unsafe SVG element")
        for key, value in element.attrib.items():
            key = key.rsplit("}", 1)[-1].lower()
            if key.startswith("on") or key in ("href", "src") or external_url(value):
                raise ValueError("Unsafe SVG resource/event")
        if tag == "style" and (
            re.search(r"@import|https?:", element.text or "", re.I) or external_url(element.text or "")
        ):
            raise ValueError("Unsafe SVG stylesheet")
        if element.get("data-node-id"):
            nodes.append(element.get("data-node-id"))
        if element.get("data-edge-id"):
            edges.append(element.get("data-edge-id"))
        if tag == "text":
            texts.append((element.get("data-label-id"), " ".join("".join(element.itertext()).split())))
            ancestor = parents.get(element)
            while ancestor is not None and not ancestor.get("data-node-id"):
                ancestor = parents.get(ancestor)
            if ancestor is None or ancestor.get("data-node-id") != expected_owner.get(element.get("data-label-id")):
                misplaced.append(element.get("data-label-id"))
    unknown = [
        dict(id=k, text=t)
        for k, t in texts
        if k not in spec["labels"] or canonical(t) != canonical(spec["labels"][k]["text"])
    ]
    counts = Counter(k for k, _ in texts)
    coverage = counts == Counter({k: 1 for k in spec["labels"]})
    layer_sum = 0
    for key, text in texts:
        if spec["labels"].get(key, {}).get("role") == "layer_count":
            numbers = re.findall(r"\d+", text)
            layer_sum += int(numbers[0]) if len(numbers) == 1 else -1000000
    return dict(
        labels_and_numbers_from_published=not unknown,
        label_coverage=coverage,
        labels_attached_to_nodes=not misplaced,
        invalid_labels=unknown,
        layer_sum=layer_sum,
        expected_layers=spec["layer_count"],
        layer_sum_matches=layer_sum == spec["layer_count"],
        nodes_and_edges_match=Counter(nodes) == Counter(n["id"] for n in spec["nodes"])
        and Counter(edges) == Counter(e["id"] for e in spec["edges"]),
    )


GEOMETRY_JS = r"""() => {
 const svg=document.querySelector('svg'),v=svg.viewBox.baseVal;
 const rect=e=>{const b=e.getBBox(),m=svg.getScreenCTM().inverse().multiply(e.getScreenCTM());const points=[[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(m));const xs=points.map(p=>p.x),ys=points.map(p=>p.y);return {x:Math.min(...xs),y:Math.min(...ys),width:Math.max(...xs)-Math.min(...xs),height:Math.max(...ys)-Math.min(...ys)}};
 const visible=e=>{for(let n=e;n&&n!==document;n=n.parentElement){const s=getComputedStyle(n);if(s.display==='none'||s.visibility==='hidden'||Number(s.opacity)<=0)return false}return true};
 const texts=[...svg.querySelectorAll('text')].map(e=>({id:e.dataset.labelId,box:rect(e),visible:visible(e)}));
 const textRuns=[...svg.querySelectorAll('text,tspan')].filter(e=>[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim()));
 const readable=e=>{const m=e.getScreenCTM();return Number.parseFloat(getComputedStyle(e).fontSize)*Math.min(Math.hypot(m.a,m.b),Math.hypot(m.c,m.d))>=12.99};
 const overlaps=[];
 for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
  const a=texts[i].box,b=texts[j].box;
  if(Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x)>0.5 && Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y)>0.5)overlaps.push([texts[i].id,texts[j].id]);
 }
 const outside=[...svg.querySelectorAll('text,rect,path,line,polyline,polygon')].filter(e=>!e.closest('defs')).filter(e=>{const b=rect(e);return b.x<v.x-.5||b.y<v.y-.5||b.x+b.width>v.x+v.width+.5||b.y+b.height>v.y+v.height+.5}).map(e=>e.dataset.labelId||e.dataset.edgeId||e.tagName);
 const invisible=texts.filter(t=>!t.visible||t.box.width<=0||t.box.height<=0).map(t=>t.id);
 const nodeBoxes=[...svg.querySelectorAll('[data-node-id]')].map(e=>({id:e.dataset.nodeId,box:e.querySelector('rect')?rect(e.querySelector('rect')):null}));
 const nodeOverlaps=[];
 for(let i=0;i<nodeBoxes.length;i++)for(let j=i+1;j<nodeBoxes.length;j++){const a=nodeBoxes[i].box,b=nodeBoxes[j].box;if(a&&b&&Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x)>.5&&Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y)>.5)nodeOverlaps.push([nodeBoxes[i].id,nodeBoxes[j].id]);}
 const overflow=[];
 for(const n of [...svg.querySelectorAll('[data-node-id]')]){const shape=n.querySelector('rect');if(!shape){overflow.push(n.dataset.nodeId);continue;}const a=rect(shape);for(const t of n.querySelectorAll('text')){const b=rect(t);if(b.x<a.x-.5||b.y<a.y-.5||b.x+b.width>a.x+a.width+.5||b.y+b.height>a.y+a.height+.5)overflow.push(t.dataset.labelId);}}
 return {canvas:{width:v.width,height:v.height},overlaps,outside,invisible,nodeOverlaps,text_node_overflow:overflow,no_text_overlap:overlaps.length===0,everything_inside_canvas:outside.length===0,all_labels_visible:invisible.length===0,readable_font_sizes:textRuns.length>0&&textRuns.every(readable),no_node_overlap:nodeBoxes.every(n=>n.box)&&nodeOverlaps.length===0,text_fits_nodes:overflow.length===0};
}"""


def browser_geometry(svg, spec, *, channel="chrome", screenshot=None):
    # Reject active content BEFORE loading model output into a browser.
    inspect_svg(svg, spec)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        context = browser.new_context(
            viewport={"width": 1400, "height": 1900}, device_scale_factor=1, service_workers="block"
        )
        context.route("**/*", lambda route: route.abort())
        page = context.new_page()
        page.set_content("<!doctype html><style>body{margin:0;background:white}</style>" + svg)
        page.evaluate(
            "() => { const s=document.querySelector('svg'),v=s.viewBox.baseVal;s.style.width=v.width+'px';s.style.height=v.height+'px'; }"
        )
        page.evaluate("document.fonts.ready")
        result = page.evaluate(GEOMETRY_JS)
        result["canvas_within_experiment_bounds"] = (
            0 < result["canvas"]["width"] <= spec["bounds"]["max_width"]
            and 0 < result["canvas"]["height"] <= spec["bounds"]["max_height"]
        )
        if screenshot:
            page.locator("svg").screenshot(path=str(screenshot))
        browser.close()
    return result


def passed(review):
    fields = (
        "labels_and_numbers_from_published",
        "label_coverage",
        "layer_sum_matches",
        "nodes_and_edges_match",
        "no_text_overlap",
        "everything_inside_canvas",
        "all_labels_visible",
        "canvas_within_experiment_bounds",
        "readable_font_sizes",
        "no_node_overlap",
        "labels_attached_to_nodes",
        "text_fits_nodes",
    )
    return all(review.get(field) is True for field in fields)
