"""Code checks for Gemini-written architecture SVGs against their spec (no LLM grading)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

ALLOWED = {
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
BENCHMARK_WORDS = {"mmlu", "gsm8k", "humaneval", "benchmark", "score", "accuracy"}
POSITION_WORDS = {"rope", "rotary", "positional"}
SECTIONS = ("title", "stack", "each-layer", "zooms")
REPEAT_LABEL = re.compile(r"^\s*(\d+)\s*[x×]\s*$")


# ---------------------------------------------------------------- tokens


def _numbers(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return re.findall(r"\d+(?:\.\d+)?", text)


def _word(w: str) -> str:
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _words(text: str) -> list[str]:
    return [_word(w) for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", text)]


def _tokens(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return [_word(t) if t[0].isalpha() else t for t in re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?", text)]


def _strings(value, skip=("id", "beside", "edges", "layout")):
    # Layout geometry and wiring ids are drawing instructions, never visible facts.
    if isinstance(value, dict):
        for k, v in value.items():
            if k not in skip:
                yield from _strings(v, skip)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v, skip)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def vocabulary(spec: dict) -> tuple[set[str], set[str]]:
    body = "\n".join(_strings(spec))
    return set(_numbers(body)), set(_words(body))


def graphs(spec: dict) -> dict:
    """Every graph the diagram must draw, keyed by where it lives in the SVG."""
    out = {("data-section", "each-layer"): spec["each_layer"]}
    out.update({("data-zoom", key): panel for key, panel in spec["zoom_panels"].items()})
    return out


def _edge(text: str) -> str:
    return "->".join(part.strip() for part in text.split("->"))


# ---------------------------------------------------------------- parsing


def extract_svg(response: str) -> str:
    start, end = response.find("<svg"), response.rfind("</svg>")
    if start < 0 or end < 0:
        raise ValueError("Response contains no complete <svg> element")
    return response[start : end + len("</svg>")]


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _text_of(element) -> str:
    parts = [element.text or ""]
    for child in element:
        parts.append(" " + _text_of(child) + " ")
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


def parse(svg: str):
    if len(svg.encode()) > 2 * 1024 * 1024 or re.search(r"<!DOCTYPE|<!ENTITY", svg, re.I):
        raise ValueError("Unsafe or oversized SVG")
    root = ET.fromstring(svg)
    if _local(root.tag) != "svg":
        raise ValueError("Output is not SVG")
    for element in root.iter():
        if _local(element.tag) not in ALLOWED:
            raise ValueError(f"Disallowed SVG element <{_local(element.tag)}>")
        for key, value in element.attrib.items():
            key = _local(key).lower()
            if key.startswith("on") or key in ("href", "src"):
                raise ValueError("SVG contains an event handler or external reference")
            for url in re.findall(r"url\(([^)]*)\)", value, re.I):
                if not url.strip().strip("\"'").startswith("#"):
                    raise ValueError("SVG references an external resource")
        if _local(element.tag) == "style" and re.search(
            r"@import|https?:|url\((?!\s*['\"]?#)", element.text or "", re.I
        ):
            raise ValueError("SVG stylesheet loads an external resource")
    return root


def _visible_texts(root):
    hidden = {
        id(e) for parent in root.iter() if _local(parent.tag) in ("defs", "title", "style") for e in parent.iter()
    }
    return [e for e in root.iter() if _local(e.tag) == "text" and id(e) not in hidden]


def _find_all(root, attr, value=None):
    return [e for e in root.iter() if e.get(attr) is not None and (value is None or e.get(attr) == value)]


def _group_text(group) -> str:
    return " ".join(_text_of(t) for t in group.iter() if _local(t.tag) == "text")


def _subsequence(needle, haystack):
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def _within(root, element):
    parents = {child: parent for parent in root.iter() for child in parent}
    found = []
    node = parents.get(element)
    while node is not None:
        found.append(node)
        node = parents.get(node)
    return found


# ---------------------------------------------------------------- static checks


def static_checks(svg: str, spec: dict, settings: dict) -> dict:
    out = {}
    try:
        root = parse(svg)
    except (ValueError, ET.ParseError) as error:
        return {"svg_valid_and_safe": dict(passed=False, problems=[str(error)])}
    out["svg_valid_and_safe"] = dict(passed=True, problems=[])

    def result(problems, **extra):
        return dict(passed=not problems, problems=problems, **extra)

    view = [float(x) for x in (root.get("viewBox") or "").replace(",", " ").split()] or [0, 0, 0, 0]
    low, high = settings["canvas_height_range"]
    ok = len(view) == 4 and view[:3] == [0, 0, spec["layout"]["canvas_width"]] and low <= view[3] <= high
    out["canvas_size"] = result(
        [] if ok else [f"viewBox must be 0 0 {spec['layout']['canvas_width']} H, {low} <= H <= {high}"]
    )

    numbers, words = vocabulary(spec)
    texts = [_text_of(t) for t in _visible_texts(root)]
    bad = sorted({(n, t) for t in texts for n in _numbers(t) if n not in numbers})
    out["numbers_from_spec"] = result([f'number {n} in "{t}" is not in the spec' for n, t in bad])
    bad = sorted({(w, t) for t in texts for w in _words(t) if w not in words})
    out["labels_from_spec"] = result([f'word "{w}" in "{t}" is not in the spec' for w, t in bad])

    forbidden, dont = set(), spec.get("do_not_include", [])
    for item in dont:
        match = re.match(r"the word (\S+)", item)
        if match:
            forbidden.add(_word(match.group(1)))
        if "position-encoding" in item:
            forbidden |= POSITION_WORDS
        if "benchmark" in item:
            forbidden |= BENCHMARK_WORDS
    hits = sorted({(w, t) for t in texts for w in _words(t) if w in forbidden})
    hits += sorted({("%", t) for t in texts if "%" in t and any("benchmark" in d for d in dont)})
    out["do_not_include"] = result([f'"{w}" appears in "{t}" (do_not_include)' for w, t in hits])

    # ---- layer groups, sum and repeat labels (each repeat count appears exactly once)
    expected = spec["layer_pattern"]["compressed_view"]
    found = {g.get("data-group"): g for g in _find_all(root, "data-group")}
    problems, total = [], 0
    if sorted(found) != [str(i + 1) for i in range(len(expected))]:
        problems.append(f"expected data-group 1..{len(expected)}, found {sorted(found)}")
    for i, entry in enumerate(expected, 1):
        group = found.get(str(i))
        if group is None:
            continue
        bars = _find_all(group, "data-layer")
        repeat = group.get("data-repeat")
        if repeat != str(entry["repeat"]):
            problems.append(f"group {i}: data-repeat is {repeat}, spec says {entry['repeat']}")
        if len(bars) != len(entry["layers"]):
            problems.append(f"group {i}: {len(bars)} layer bars, spec lists {len(entry['layers'])}")
        total += int(repeat) * len(bars) if repeat and repeat.isdigit() else 0
    layers = spec["model"]["layers"]
    if total != layers:
        problems.append(f"drawn groups add up to {total} layers, spec says {layers}")
    out["layer_sum"] = result(problems, drawn_total=total, expected=layers)

    problems = []
    group_ids = {id(e): g.get("data-group") for g in found.values() for e in g.iter()}
    labels = [(t, REPEAT_LABEL.match(_text_of(t))) for t in _visible_texts(root)]
    for i, entry in enumerate(expected, 1):
        inside = [m.group(1) for t, m in labels if m and group_ids.get(id(t)) == str(i)]
        if inside != [str(entry["repeat"])]:
            problems.append(f'group {i} must show its repeat label "{entry["repeat"]} x" exactly once, found {inside}')
    stray = [_text_of(t) for t, m in labels if m and id(t) not in group_ids]
    problems += [f'repeat-style label "{s}" outside a layer group; label the total as "{layers} layers"' for s in stray]
    if len(expected) == 1:
        totals = [_text_of(t) for t in _visible_texts(root) if re.fullmatch(rf"\s*{layers}\s+layers?\s*", _text_of(t))]
        problems += [
            f'single layer group: drop the brace and its "{t}" label; "{layers} x" already gives the count'
            for t in totals
        ]
    out["repeat_counts_once"] = result(problems)

    # ---- fixed layout sections and title
    problems = [f'missing <g data-section="{s}">' for s in SECTIONS if not _find_all(root, "data-section", s)]
    title = _find_all(root, "data-section", "title")
    if title:
        tt = _tokens(_group_text(title[0]))
        for key in ("title_label", "summary_line"):
            if not _subsequence(_tokens(spec["model"][key]), tt):
                problems.append(f'title section must show model.{key} "{spec["model"][key]}"')
    stack = _find_all(root, "data-section", "stack")
    if stack:
        inside = {id(e) for e in stack[0].iter()}
        problems += [f"layer group {g} is outside the stack section" for g, e in found.items() if id(e) not in inside]
        for item in spec["io"]["bottom"] + spec["io"]["top"]:
            boxes = [e for e in _find_all(root, "data-io", item["id"]) if id(e) in inside]
            if not boxes:
                problems.append(f'io box "{item["id"]}" ({item["label"]}) missing from the stack section')
            elif _tokens(_group_text(boxes[0])) != _tokens(item["label"]):
                problems.append(f'io box "{item["id"]}" must read "{item["label"]}"')
        problems += [
            f"callout {c.get('data-callout')} must sit inside the left panel (stack section)"
            for c in _find_all(root, "data-callout")
            if id(c) not in inside
        ]
    zooms = _find_all(root, "data-section", "zooms")
    if zooms:
        inside = {id(e) for e in zooms[0].iter()}
        problems += [
            f'zoom panel "{z.get("data-zoom")}" is outside the zooms section'
            for z in _find_all(root, "data-zoom")
            if id(z) not in inside
        ]
    out["fixed_layout_sections"] = result(problems)

    # ---- zoom panels and graphs: every node and edge, nothing extra
    zoom_found = {z.get("data-zoom"): z for z in _find_all(root, "data-zoom")}
    problems = [
        f'zoom panel "{k}" ({p["title"]}) is missing' for k, p in spec["zoom_panels"].items() if k not in zoom_found
    ]
    problems += [f'unexpected zoom panel "{k}"' for k in zoom_found if k not in spec["zoom_panels"]]
    for key, panel in spec["zoom_panels"].items():
        if key in zoom_found and not _subsequence(_tokens(panel["title"]), _tokens(_group_text(zoom_found[key]))):
            problems.append(f'zoom panel "{key}" does not show its title "{panel["title"]}"')
    out["zoom_panels_present"] = result(problems)

    node_problems, edge_problems = [], []
    for (attr, key), graph in graphs(spec).items():
        holders = _find_all(root, attr, key)
        where = f"{'zoom panel' if attr == 'data-zoom' else 'section'} {key}"
        if not holders:
            continue  # reported by the section/panel checks
        holder = holders[0]
        drawn = {}
        for n in _find_all(holder, "data-node"):
            drawn.setdefault(n.get("data-node"), []).append(n)
        for node in graph["nodes"]:
            got = drawn.get(node["id"], [])
            if len(got) != 1:
                node_problems.append(f'{where}: node "{node["id"]}" ({node["label"]}) drawn {len(got)} times')
            elif _tokens(_group_text(got[0])) != _tokens(node["label"]):
                node_problems.append(f'{where}: node "{node["id"]}" must read "{node["label"]}"')
        extra = set(drawn) - {n["id"] for n in graph["nodes"]}
        node_problems += [f'{where}: node "{n}" is not in the spec' for n in sorted(extra)]
        want = sorted(_edge(e) for e in graph["edges"])
        got = sorted(_edge(e.get("data-edge")) for e in _find_all(holder, "data-edge"))
        for e in sorted(set(want) - set(got)):
            edge_problems.append(f"{where}: edge {e} is missing")
        for e in sorted(set(got) - set(want)):
            edge_problems.append(f"{where}: edge {e} is drawn but not in the spec")
        for e in sorted({e for e in got if got.count(e) > 1}):
            edge_problems.append(f"{where}: edge {e} is drawn more than once")
    out["graph_nodes_match_spec"] = result(node_problems)
    out["graph_edges_match_spec"] = result(edge_problems)

    calls = {g.get("data-callout"): g for g in _find_all(root, "data-callout")}
    problems = []
    for i, callout in enumerate(spec["callouts"], 1):
        group = calls.get(str(i))
        if group is None:
            problems.append(f'callout {i} "{callout["text"]}" is missing')
        elif _tokens(_group_text(group)) != _tokens(callout["text"]):
            problems.append(f'callout {i} reads "{_group_text(group)}", spec says "{callout["text"]}"')
    out["callouts_present"] = result(problems)
    return out


# ---------------------------------------------------------------- browser checks

GEOMETRY_JS = r"""(settings) => {
 const svg=document.querySelector('svg'), v=svg.viewBox.baseVal, m=settings.margin_px;
 const box=e=>{const b=e.getBBox(),t=svg.getScreenCTM().inverse().multiply(e.getScreenCTM());
   const p=[[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(t));
   const xs=p.map(q=>q.x),ys=p.map(q=>q.y);return {x:Math.min(...xs),y:Math.min(...ys),w:Math.max(...xs)-Math.min(...xs),h:Math.max(...ys)-Math.min(...ys)}};
 const hidden=e=>e.closest('defs,marker,title,style');
 const label=e=>(e.textContent||'').replace(/\s+/g,' ').trim().slice(0,60);
 const inter=(a,b)=>Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));
 const all=[...svg.querySelectorAll('*')].filter(e=>!hidden(e));
 const order=new Map(all.map((e,i)=>[e,i]));
 const texts=all.filter(e=>e.tagName==='text'&&label(e)).map(e=>({e,t:label(e),b:box(e)}));
 const overlaps=[];
 for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){const a=texts[i].b,b=texts[j].b;
   if(inter(a,b)>Math.min(a.w*a.h,b.w*b.h)*0.05&&inter(a,b)>4)overlaps.push([texts[i].t,texts[j].t]);}
 const painted=e=>{const s=getComputedStyle(e);return s.fill!=='none'&&Number(s.fillOpacity)>0.05&&Number(s.opacity)>0.05};
 const shapes=all.filter(e=>['rect','path','polygon'].includes(e.tagName)&&painted(e));
 const covered=[];
 for(const t of texts){for(const s of shapes){if(order.get(s)<=order.get(t.e))continue;const a=inter(t.b,box(s));
   if(a>t.b.w*t.b.h*0.2){covered.push(t.t);break;}}}
 const strokes=all.filter(e=>['line','polyline','path'].includes(e.tagName)).filter(e=>{const s=getComputedStyle(e);return s.stroke!=='none'&&(e.tagName!=='path'||s.fill==='none')});
 const leaders=strokes.filter(e=>{const d=getComputedStyle(e).strokeDasharray;const closed=e.tagName==='path'&&/z\s*$/i.test(e.getAttribute('d')||'');return d&&d!=='none'&&!closed}).length;
 const pts=e=>{const len=e.getTotalLength?e.getTotalLength():0;const t=svg.getScreenCTM().inverse().multiply(e.getScreenCTM());const o=[];if(!len)return o;for(let d=0;d<=len;d+=2)o.push(e.getPointAtLength(d).matrixTransform(t));o.push(e.getPointAtLength(len).matrixTransform(t));return o};
 const through=[];
 for(const s of strokes){const hit=new Set();for(const p of pts(s)){for(const x of texts){const b=x.b;if(p.x>b.x+2&&p.x<b.x+b.w-2&&p.y>b.y+2&&p.y<b.y+b.h-2)hit.add(x.t);}}for(const h of hit)through.push(h);}
 const bg=e=>e.tagName==='rect'&&(()=>{const b=box(e);return b.w>=v.width*0.99&&b.h>=v.height*0.99})();
 const outside=all.filter(e=>['text','rect','path','line','polyline','polygon'].includes(e.tagName)&&!bg(e)).filter(e=>{const b=box(e);
   return b.w+b.h>0&&(b.x<v.x+m-0.5||b.y<v.y+m-0.5||b.x+b.w>v.x+v.width-m+0.5||b.y+b.h>v.y+v.height-m+0.5)}).map(e=>e.tagName==='text'?label(e):e.tagName+(e.closest('[data-zoom]')?' in zoom '+e.closest('[data-zoom]').dataset.zoom:''));
 const runs=[...svg.querySelectorAll('text,tspan')].filter(e=>!hidden(e)&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim()));
 const small=runs.filter(e=>{const c=e.getScreenCTM();return Number.parseFloat(getComputedStyle(e).fontSize)*Math.min(Math.hypot(c.a,c.b),Math.hypot(c.c,c.d))<settings.min_font_px}).map(label);
 const bars=[...svg.querySelectorAll('[data-group]')].map(g=>({group:g.dataset.group,bars:[...g.querySelectorAll('[data-layer]')].map(b=>{const r=box(b);return {label:b.dataset.layer,y:r.y+r.h/2}}).sort((a,b)=>b.y-a.y).map(b=>b.label)}));
 // edges: the arrow must start at its source box and end at its target box (within 14 px)
 const near=(p,b,d)=>p.x>=b.x-d&&p.x<=b.x+b.w+d&&p.y>=b.y-d&&p.y<=b.y+b.h+d;
 const nodeBox=(holder,id)=>{const n=holder.querySelector(`[data-node="${CSS.escape(id)}"]`);if(!n)return null;const r=n.querySelector('rect');return box(r||n)};
 const edges=[];
 for(const e of svg.querySelectorAll('[data-edge]')){const holder=e.closest('[data-zoom],[data-section]');if(!holder)continue;
   const [a,b]=e.dataset.edge.split('->').map(s=>s.trim());const A=nodeBox(holder,a),B=nodeBox(holder,b);const p=pts(e);
   if(!A||!B||!p.length){edges.push({edge:e.dataset.edge,where:holder.dataset.zoom||holder.dataset.section,ok:false,why:'unknown node or empty line'});continue;}
   const s=p[0],t=p[p.length-1];const fwd=near(s,A,14)&&near(t,B,14),rev=near(s,B,14)&&near(t,A,14);
   edges.push({edge:e.dataset.edge,where:holder.dataset.zoom||holder.dataset.section,ok:fwd,why:fwd?'':(rev?'arrow points the wrong way':'arrow does not connect its two boxes')});}
 const sec=n=>{const e=svg.querySelector(`[data-section="${n}"]`);return e?box(e):null};
 const union=bs=>{bs=bs.filter(Boolean);if(!bs.length)return null;const x=Math.min(...bs.map(b=>b.x)),y=Math.min(...bs.map(b=>b.y));return {x,y,w:Math.max(...bs.map(b=>b.x+b.w))-x,h:Math.max(...bs.map(b=>b.y+b.h))-y}};
 const targets=k=>{if(k==='stack'){const u=union([...svg.querySelectorAll('[data-group]')].map(box));return u?[u]:[]}
   if(k.startsWith('bar:'))return [...svg.querySelectorAll('[data-layer]')].filter(b=>b.dataset.layer===k.slice(4)).map(box);
   return [...svg.querySelectorAll(`[data-io="${CSS.escape(k)}"]`)].map(box)};
 const callouts=settings.callouts.map((c,i)=>{const e=svg.querySelector(`[data-callout="${i+1}"]`);const T=targets(c.beside);
   if(!e||!T.length)return {i:i+1,ok:false,why:'callout or its target missing'};
   const b=box(e),cy=b.y+b.h/2;let best=null;
   for(const t of T){const gap=b.x-(t.x+t.w);const dy=cy<t.y?t.y-cy:(cy>t.y+t.h?cy-(t.y+t.h):0);const ok=gap>=-4&&gap<=settings.callout_gap_px&&dy<=settings.callout_level_px;
     if(!best||ok||(!best.ok&&gap+dy<best.d))best={ok,d:gap+dy,why:`${Math.round(gap)} px right of and ${Math.round(dy)} px off level with ${c.beside}`};if(ok)break;}
   return {i:i+1,ok:best.ok,why:best.why}});
 // lines may touch only the boxes they connect: no crossing, and no running within line_clearance_px of another box for 12 px or more
 const boxes=[...svg.querySelectorAll('[data-node],[data-io],[data-layer]')].filter(e=>!hidden(e)).map(e=>{const r=e.tagName==='rect'?e:e.querySelector('rect');return r?{e,b:box(r),t:label(e)||e.dataset.layer||''}:null}).filter(Boolean);
 const crossings=[];
 for(const s of strokes){if(s.tagName==='path'&&/z\s*$/i.test(s.getAttribute('d')||''))continue;const p=pts(s);if(p.length<2)continue;
   let allowed;if(s.dataset.edge){const holder=s.closest('[data-zoom],[data-section]');const ids=s.dataset.edge.split('->').map(x=>x.trim());
     allowed=new Set(ids.map(id=>holder&&holder.querySelector(`[data-node="${CSS.escape(id)}"]`)).filter(Boolean));}
   else{allowed=new Set(boxes.filter(o=>near(p[0],o.b,14)||near(p[p.length-1],o.b,14)).map(o=>o.e));}
   for(const o of boxes){if(allowed.has(o.e)||[...allowed].some(a=>a.contains(o.e)||o.e.contains(a)))continue;
     const n=p.filter(q=>near(q,o.b,settings.line_clearance_px)).length;if(n>=6)crossings.push(`${s.dataset.edge?'edge '+s.dataset.edge:'a line'} crosses or runs along the box "${o.t}"`);}}
 const zoomBox=k=>{const z=svg.querySelector(`[data-zoom="${CSS.escape(k)}"]`);return z?box(z):null};
 const columns=settings.zoom_columns.map(c=>({x:c.x,panels:c.panels.map(k=>({k,b:zoomBox(k)}))}));
 return {canvas:{width:v.width,height:v.height},bars,edges,callouts,crossings:[...new Set(crossings)],columns,sections:{title:sec('title'),stack:sec('stack'),each:sec('each-layer'),zooms:sec('zooms')},
   overlaps,covered:[...new Set(covered)],through:[...new Set(through)],leaders,outside:[...new Set(outside)],small:[...new Set(small)]};
}"""


def _layout_problems(s, columns, layout):
    """Same sections in the same places for every model; geometry comes from the spec's layout."""
    if not all(s.values()):
        return ["layout sections missing; see fixed_layout_sections"]
    problems, width = [], layout["canvas_width"]
    title, stack, each, zooms = s["title"], s["stack"], s["each"], s["zooms"]
    lx0, lx1 = layout["left_panel"]["x"]
    if abs(title["x"] + title["w"] / 2 - width / 2) > width * 0.05:
        problems.append("title section is not centered")
    if title["y"] + title["h"] > min(stack["y"], zooms["y"]) + 1:
        problems.append("title section must sit above the left panel and the zoom panels")
    for name, b in (("left panel", stack), ("Each layer box", each)):
        if b["x"] < lx0 - 20 or b["x"] + b["w"] > lx1 + 20:
            problems.append(f"{name} must stay within x {lx0}-{lx1} (found {round(b['x'])}-{round(b['x'] + b['w'])})")
    if each["y"] < stack["y"] + stack["h"] - 1:
        problems.append("the Each layer box must sit under the left panel")
    for col in columns:
        x0, x1 = col["x"]
        for p in col["panels"]:
            b = p["b"]
            if b and (b["x"] < x0 - 20 or b["x"] + b["w"] > x1 + 20):
                problems.append(f'zoom panel "{p["k"]}" must sit in its column x {x0}-{x1}')
    return problems


def _balance_problems(s, columns):
    if not (s["stack"] and s["each"]):
        return ["layout sections missing; see fixed_layout_sections"]
    left = max(s["stack"]["y"] + s["stack"]["h"], s["each"]["y"] + s["each"]["h"]) - min(
        s["stack"]["y"], s["each"]["y"]
    )
    heights = []
    for col in columns:
        bs = [p["b"] for p in col["panels"] if p["b"]]
        if bs:
            heights.append(max(b["y"] + b["h"] for b in bs) - min(b["y"] for b in bs))
    if not heights:
        return ["zoom panels missing"]
    right = max(heights)
    diff = abs(left - right) / max(left, right)
    if diff >= 0.25:
        return [
            f"left column is {round(left)} px tall and the tallest zoom column {round(right)} px "
            f"({diff:.0%} apart; must be under 25%). Scale the layer stack or the Each layer box"
        ]
    return []


def browser_checks(svg: str, settings: dict, *, spec=None, screenshot=None) -> dict:
    parse(svg)  # reject active content before a browser ever loads model output
    from playwright.sync_api import sync_playwright

    layout = (spec or {}).get("layout", {})
    js_settings = dict(settings, callouts=(spec or {}).get("callouts", []), zoom_columns=layout.get("zoom_columns", []))
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=settings["browser_channel"], headless=True)
        context = browser.new_context(viewport={"width": 2300, "height": 2500}, device_scale_factor=1)
        context.route("**/*", lambda route: route.abort())
        page = context.new_page()
        page.set_content("<!doctype html><style>body{margin:0;background:white}</style>" + svg)
        page.evaluate(
            "() => { const s=document.querySelector('svg'),v=s.viewBox.baseVal;"
            "s.style.width=v.width+'px';s.style.height=v.height+'px'; }"
        )
        page.evaluate("document.fonts.ready")
        geo = page.evaluate(GEOMETRY_JS, js_settings)
        if screenshot:
            page.locator("svg").screenshot(path=str(screenshot))
        browser.close()

    def result(items, message="{}"):
        return dict(passed=not items, problems=[message.format(i) for i in items])

    order = []
    if spec:
        drawn = {g["group"]: g["bars"] for g in geo["bars"]}
        for i, entry in enumerate(spec["layer_pattern"]["compressed_view"], 1):
            got = drawn.get(str(i))
            if got is not None and got != entry["layers"]:
                order.append(
                    f"group {i}: bars from bottom to top are {got}, spec says {entry['layers']} "
                    "(data-layer must be each layer's exact label)"
                )
    checks = {"layer_bars_match_spec": result(order)}
    if spec:
        checks["edges_connect_their_nodes"] = result(
            [f"{e['where']}: edge {e['edge']}: {e['why']}" for e in geo["edges"] if not e["ok"]]
        )
        checks["fixed_layout_positions"] = result(_layout_problems(geo["sections"], geo["columns"], spec["layout"]))
        checks["columns_balanced"] = result(_balance_problems(geo["sections"], geo["columns"]))
        checks["lines_avoid_other_boxes"] = result(geo["crossings"])
        checks["callouts_beside_their_part"] = result(
            [
                f'callout {c["i"]} "{spec["callouts"][c["i"] - 1]["text"]}" is not beside its part inside the left panel ({c["why"]})'
                for c in geo["callouts"]
                if not c["ok"]
            ]
        )
    checks.update(
        {
            "no_text_overlap": result([f'"{a}" / "{b}"' for a, b in geo["overlaps"]], "text overlaps text: {}"),
            "no_covered_text": result(geo["covered"], 'a shape is drawn on top of the text "{}"'),
            "no_lines_through_text": result(geo["through"], 'a line or arrow passes through the text "{}"'),
            "no_leader_lines": dict(
                passed=geo["leaders"] == 0,
                problems=[f"{geo['leaders']} dashed/dotted connector line(s) found; nothing needs leader lines"]
                if geo["leaders"]
                else [],
            ),
            "inside_canvas_with_margin": result(
                geo["outside"], "{} is outside the canvas or within " + str(settings["margin_px"]) + " px of its edge"
            ),
            "readable_font_sizes": result(
                geo["small"], 'text "{}" is smaller than ' + str(settings["min_font_px"]) + " px"
            ),
        }
    )
    return checks


def review(svg: str, spec: dict, settings: dict, *, screenshot=None, browser=True) -> dict:
    checks = static_checks(svg, spec, settings)
    if checks["svg_valid_and_safe"]["passed"] and browser:
        checks.update(browser_checks(svg, settings, spec=spec, screenshot=screenshot))
    failures = [p for name, c in checks.items() if not c["passed"] for p in c["problems"][:12]]
    return dict(passed=all(c["passed"] for c in checks.values()), checks=checks, failures=failures)
