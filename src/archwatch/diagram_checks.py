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


def _strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def vocabulary(spec: dict) -> tuple[set[str], set[str]]:
    body = "\n".join(_strings(spec))
    return set(_numbers(body)), set(_words(body))


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


def _groups(root, attr):
    return [e for e in root.iter() if _local(e.tag) == "g" and e.get(attr) is not None]


def _group_text(group) -> str:
    return " ".join(_text_of(t) for t in group.iter() if _local(t.tag) == "text")


def _subsequence(needle, haystack):
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


# ---------------------------------------------------------------- static checks


def static_checks(svg: str, spec: dict, settings: dict) -> dict:
    out = {}
    try:
        root = parse(svg)
    except (ValueError, ET.ParseError) as error:
        return {"svg_valid_and_safe": dict(passed=False, problems=[str(error)])}
    out["svg_valid_and_safe"] = dict(passed=True, problems=[])

    view = [float(x) for x in (root.get("viewBox") or "").replace(",", " ").split()] or [0, 0, 0, 0]
    low, high = settings["canvas_height_range"]
    ok = len(view) == 4 and view[:3] == [0, 0, settings["canvas_width"]] and low <= view[3] <= high
    out["canvas_size"] = dict(
        passed=ok,
        problems=[] if ok else [f"viewBox must be 0 0 {settings['canvas_width']} H with {low} <= H <= {high}"],
    )

    numbers, words = vocabulary(spec)
    texts = [_text_of(t) for t in _visible_texts(root)]
    bad_numbers = sorted({(n, t) for t in texts for n in _numbers(t) if n not in numbers})
    out["numbers_from_spec"] = dict(
        passed=not bad_numbers, problems=[f'number {n} in "{t}" is not in the spec' for n, t in bad_numbers]
    )
    bad_words = sorted({(w, t) for t in texts for w in _words(t) if w not in words})
    out["labels_from_spec"] = dict(
        passed=not bad_words, problems=[f'word "{w}" in "{t}" is not in the spec' for w, t in bad_words]
    )

    forbidden = set()
    for item in spec.get("do_not_include", []):
        match = re.match(r"the word (\S+)", item)
        if match:
            forbidden.add(_word(match.group(1)))
        if "position-encoding" in item:
            forbidden |= POSITION_WORDS
        if "benchmark" in item:
            forbidden |= BENCHMARK_WORDS
    hits = sorted({(w, t) for t in texts for w in _words(t) if w in forbidden})
    hits += sorted({("%", t) for t in texts if "%" in t and "benchmark" in " ".join(spec.get("do_not_include", []))})
    out["do_not_include"] = dict(
        passed=not hits, problems=[f'"{w}" appears in "{t}" (do_not_include)' for w, t in hits]
    )

    expected = spec["layer_pattern"]["compressed_view"]
    found = {g.get("data-group"): g for g in _groups(root, "data-group")}
    problems, total = [], 0
    if sorted(found) != [str(i + 1) for i in range(len(expected))]:
        problems.append(f"expected data-group 1..{len(expected)}, found {sorted(found)}")
    for i, entry in enumerate(expected, 1):
        group = found.get(str(i))
        if group is None:
            continue
        bars = [e for e in group.iter() if e.get("data-layer") is not None]
        repeat = group.get("data-repeat")
        if repeat != str(entry["repeat"]):
            problems.append(f"group {i}: data-repeat is {repeat}, spec says {entry['repeat']}")
        if len(bars) != len(entry["layers"]):
            problems.append(f"group {i}: {len(bars)} layer bars, spec lists {len(entry['layers'])}")
        if not re.search(rf"(?<![\d.]){entry['repeat']}\s*[x×]", _group_text(group)):
            problems.append(f'group {i}: visible "{entry["repeat"]} x" label missing')
        total += int(repeat) * len(bars) if repeat and repeat.isdigit() else 0
    layers = spec["model"]["layers"]
    if total != layers:
        problems.append(f"drawn groups add up to {total} layers, spec says {layers}")
    out["layer_sum"] = dict(passed=not problems, problems=problems, drawn_total=total, expected=layers)

    zooms = {g.get("data-zoom"): g for g in _groups(root, "data-zoom")}
    problems = []
    for key, panel in spec["zoom_panels"].items():
        if key not in zooms:
            problems.append(f'zoom panel "{key}" ({panel["title"]}) is missing')
        elif not _subsequence(_tokens(panel["title"]), _tokens(_group_text(zooms[key]))):
            problems.append(f'zoom panel "{key}" does not show its title "{panel["title"]}"')
    problems += [f'unexpected zoom panel "{k}"' for k in zooms if k not in spec["zoom_panels"]]
    out["zoom_panels_present"] = dict(passed=not problems, problems=problems)

    calls = {g.get("data-callout"): g for g in _groups(root, "data-callout")}
    problems = []
    for i, text in enumerate(spec["callouts"], 1):
        group = calls.get(str(i))
        if group is None:
            problems.append(f'callout {i} "{text}" is missing')
        elif _tokens(_group_text(group)) != _tokens(text):
            problems.append(f'callout {i} reads "{_group_text(group)}", spec says "{text}"')
    out["callouts_present"] = dict(passed=not problems, problems=problems)
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
 const through=[];
 for(const s of strokes){const len=s.getTotalLength?s.getTotalLength():0;if(!len)continue;const t=svg.getScreenCTM().inverse().multiply(s.getScreenCTM());
   const hit=new Set();for(let d=0;d<=len;d+=2){const p=s.getPointAtLength(d).matrixTransform(t);
     for(const x of texts){const b=x.b;if(p.x>b.x+2&&p.x<b.x+b.w-2&&p.y>b.y+2&&p.y<b.y+b.h-2)hit.add(x.t);}}
   for(const h of hit)through.push(h);}
 const bg=e=>e.tagName==='rect'&&(()=>{const b=box(e);return b.w>=v.width*0.99&&b.h>=v.height*0.99})();
 const outside=all.filter(e=>['text','rect','path','line','polyline','polygon'].includes(e.tagName)&&!bg(e)).filter(e=>{const b=box(e);
   return b.w+b.h>0&&(b.x<v.x+m-0.5||b.y<v.y+m-0.5||b.x+b.w>v.x+v.width-m+0.5||b.y+b.h>v.y+v.height-m+0.5)}).map(e=>e.tagName==='text'?label(e):e.tagName+(e.closest('[data-zoom]')?' in zoom '+e.closest('[data-zoom]').dataset.zoom:''));
 const runs=[...svg.querySelectorAll('text,tspan')].filter(e=>!hidden(e)&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim()));
 const small=runs.filter(e=>{const c=e.getScreenCTM();return Number.parseFloat(getComputedStyle(e).fontSize)*Math.min(Math.hypot(c.a,c.b),Math.hypot(c.c,c.d))<settings.min_font_px}).map(label);
 return {canvas:{width:v.width,height:v.height},overlaps,covered:[...new Set(covered)],through:[...new Set(through)],leaders,outside:[...new Set(outside)],small:[...new Set(small)]};
}"""


def browser_checks(svg: str, settings: dict, *, screenshot=None) -> dict:
    parse(svg)  # reject active content before a browser ever loads model output
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=settings["browser_channel"], headless=True)
        context = browser.new_context(viewport={"width": 1700, "height": 2100}, device_scale_factor=1)
        context.route("**/*", lambda route: route.abort())
        page = context.new_page()
        page.set_content("<!doctype html><style>body{margin:0;background:white}</style>" + svg)
        page.evaluate(
            "() => { const s=document.querySelector('svg'),v=s.viewBox.baseVal;"
            "s.style.width=v.width+'px';s.style.height=v.height+'px'; }"
        )
        page.evaluate("document.fonts.ready")
        geo = page.evaluate(GEOMETRY_JS, settings)
        if screenshot:
            page.locator("svg").screenshot(path=str(screenshot))
        browser.close()

    def result(items, message):
        return dict(passed=not items, problems=[message.format(i) for i in items])

    return {
        "no_text_overlap": result([f'"{a}" / "{b}"' for a, b in geo["overlaps"]], "text overlaps text: {}"),
        "no_covered_text": result(geo["covered"], 'a shape is drawn on top of the text "{}"'),
        "no_lines_through_text": result(geo["through"], 'a line or arrow passes through the text "{}"'),
        "no_leader_lines": dict(
            passed=geo["leaders"] == 0,
            problems=[
                f"{geo['leaders']} dashed/dotted connector line(s) found; callouts and panels need no leader lines"
            ]
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


def review(svg: str, spec: dict, settings: dict, *, screenshot=None, browser=True) -> dict:
    checks = static_checks(svg, spec, settings)
    if checks["svg_valid_and_safe"]["passed"] and browser:
        checks.update(browser_checks(svg, settings, screenshot=screenshot))
    failures = [p for name, c in checks.items() if not c["passed"] for p in c["problems"][:12]]
    return dict(passed=all(c["passed"] for c in checks.values()), checks=checks, failures=failures)
