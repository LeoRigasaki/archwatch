"""Build the static gallery page from verified records: site/index.html (one self-contained file)."""

import html
import json
from pathlib import Path

from archwatch import diagram, pipeline
from archwatch.publication import render_cache, render_values

ROOT = Path(__file__).parent
MODELS = [("moonshotai__Kimi-K3.1-Pro", "kimi-k3-1-pro"), ("LiquidAI__LFM3-1.7B", "lfm3-1-7b")]
LABEL = {
    "kda_linear": "KDA linear attention",
    "gated_dsa_mla": "Gated DSA on MLA",
    "full_attention_mla": "MLA (full)",
    "short_conv": "Gated short conv",
    "sliding_attention": "Sliding-window attention",
    "full_attention": "Global attention",
}
COLOR = {
    "kda_linear": "#2a9d8f",
    "gated_dsa_mla": "#e76f51",
    "full_attention_mla": "#f4a261",
    "short_conv": "#5b7fbf",
    "sliding_attention": "#8ab17d",
    "full_attention": "#e76f51",
}
FIELD_NAMES = {
    "num_layers": "Layers",
    "hidden_size": "Hidden size",
    "vocab_size": "Vocabulary",
    "context_length": "Context length",
    "num_attention_heads": "Attention heads",
    "num_kv_heads": "KV heads",
    "head_dim": "Head dim (q/k)",
    "attention_type": "Attention type",
    "moe_num_experts": "Routed experts",
    "moe_active_experts": "Experts per token",
    "moe_shared_experts": "Shared experts",
    "moe_expert_intermediate": "Expert hidden size",
    "moe_latent_dim": "Latent MoE dim",
    "dense_layers": "Dense-FFN layers",
    "dsa_index_topk": "DSA top-k",
    "dsa_index_num_heads": "DSA indexer heads",
    "dsa_index_head_dim": "DSA indexer head dim",
    "layer_type_counts": "Layer mix",
    "layer_pattern_per_layer": "Per-layer pattern",
    "mla_cached_width_per_token": "MLA values cached per token",
    "dsa_index_key_width": "DSA key width",
    "activation": "Activation",
    "layer_index_base_in_config": "Config layer lists are",
    "attention_output_gate": "Attention output gate",
    "params_total": "Total parameters",
    "params_active": "Active parameters",
    "vision_params": "Vision encoder parameters",
    "ffn_intermediate": "FFN hidden size",
    "short_conv_kernel": "Conv kernel",
    "sliding_window": "Sliding window",
    "qk_norm": "QK-norm",
    "kv_cache_per_token_bytes": "Cache growth per token (bytes)",
    "norm": "Norm",
    "released_weights_format": "Released weights",
}


def e(x):
    return html.escape(str(x))


def human(n):
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= d:
            return f"{n / d:.2f} {unit}".replace(".00 ", " ")
    return f"{n:,}"


def show(field, v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if field == "layer_index_base_in_config":
        return "1-based" if v == 1 else "0-based"
    if isinstance(v, dict):
        return ", ".join(f"{LABEL.get(k, k)} {n}" for k, n in v.items())
    if isinstance(v, list):
        runs, prev, c = [], None, 0
        for x in v:
            if x == prev:
                c += 1
            else:
                if prev is not None:
                    runs.append(f"{prev}x{c}")
                prev, c = x, 1
        runs.append(f"{prev}x{c}")
        return f"{len(v)} layers: " + " ".join(runs[:8]) + (" ..." if len(runs) > 8 else "")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int) and field.startswith("params") or field == "vision_params":
        return f"{human(v)} ({v:,})"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def evidence(c):
    parts = []
    for x in c["candidates"]:
        ev = x["evidence"]
        src = ev.split(":", 1)[0]
        val = show(c["field"], x["value"]) if c["published"] and not isinstance(x["value"], list) else ""
        raw = f' (card says "{e(x["raw"])}")' if c["published"] and "raw" in x else ""
        parts.append(
            f'<span class="src src-{e(src)}">{e(src)}</span> <code>{e(ev.split(":", 1)[-1][:90])}</code>'
            + (f" = {e(val)}" if val and src != "card" else "")
            + raw
        )
    return "<br>".join(parts)


def model_section(name, anchor):
    rec = json.loads((ROOT / "data/models" / f"{name}.json").read_text())
    pub = render_values(rec)
    cache = render_cache(pub)
    components = pub["parameter_components"]
    inventory = pub["weight_header_summary"]
    repo = rec["id"]
    commit = rec["source_commit"]
    ctx = pub["context_length"]
    full = cache["grows_per_token_bytes"] * ctx + cache["constant_bytes"]
    tiles = [
        ("Total parameters", human(pub["params_total"]), "summed from weight shapes"),
        (
            "Active per token",
            human(pub["params_active"])
            if "params_active" in pub
            else ("withheld" if "MoE" in pub["decoder_type"] else "all (dense)"),
            "published counting convention" if "params_active" in pub else "see verifier evidence",
        ),
        ("Layers", f"{pub['num_layers']}", show("layer_type_counts", pub["layer_type_counts"])),
        ("Context", f"{int(ctx):,}", "tokens"),
        ("Cache growth", f"{cache['grows_per_token_bytes']:,} B", "per token, bf16"),
        ("Cache at full context", f"{full / 2**30:.1f} GiB", f"{cache['reduction_vs_naive_mha']}x less than plain MHA"),
    ]
    t_html = "".join(
        f'<div class="tile"><div class="tl">{e(a)}</div><div class="tv">{e(b)}</div><div class="ts">{e(c)}</div></div>'
        for a, b, c in tiles
    )

    svg = diagram.render(rec)

    # verification table
    order = {"verified": 0, "single source": 1, "conflict": 2}
    rows = sorted(rec["checks"], key=lambda c: order.get(c["status"], 3))
    v_rows = "".join(
        f"<tr><td>{e(FIELD_NAMES.get(c['field'], c['field']))}</td><td>{e(show(c['field'], pub[c['field']]) if c['field'] in pub else 'Withheld for review')}</td>"
        f'<td><span class="st st-{c["status"].replace(" ", "-")}">{e(c["status"])}</span></td><td class="ev">{evidence(c)}</td></tr>'
        for c in rows
    )
    n_ver = sum(1 for c in rows if c["status"] == "verified")
    n_single = sum(1 for c in rows if c["status"] == "single source")
    n_conf = sum(1 for c in rows if c["status"] == "conflict")

    # diff
    pred = rec.get("predecessor", {}).get("id")
    prec = json.loads((ROOT / "data/models" / f"{pred.replace('/', '__')}.json").read_text()) if pred else None
    field_diff = pipeline.diff(rec, prec) if prec else []
    d_rows = "".join(
        f"<tr><td>{e(FIELD_NAMES.get(r['field'], r['field']))}</td><td>{e(show(r['field'], r['before']) if r['before'] is not None else 'none')}</td>"
        f"<td>{e(show(r['field'], r['after']) if r['after'] is not None else 'none')}</td></tr>"
        for r in field_diff
    )
    td = rec.get("tensor_diff_vs_predecessor") or {}
    t_add = "".join(f"<li><code>{e(x['pattern'])}</code> +{x['params']:,}</li>" for x in td.get("added", []))
    t_rem = "".join(f"<li><code>{e(x['pattern'])}</code> &minus;{x['params']:,}</li>" for x in td.get("removed", []))
    tensor_html = (
        f'<div class="cols"><div><h4>Tensor groups added ({td.get("added_count", 0)}, +{td.get("added_total", 0):,} params)</h4><ul>{t_add or "<li>none</li>"}</ul></div>'
        f"<div><h4>Tensor groups removed ({td.get('removed_count', 0)}, &minus;{td.get('removed_total', 0):,} params)</h4><ul>{t_rem or '<li>none</li>'}</ul></div></div>"
        '<p class="small">Groups fold layer and expert numbers together. A matching add and remove of similar names (for example <code>out_proj</code> and <code>o_proj</code>) is usually a rename.</p>'
    )

    if pred:
        items = "".join(
            "<li><b>"
            + e(FIELD_NAMES.get(r["field"], r["field"]))
            + "</b>: "
            + "withheld from published comparisons pending review.</li>"
            for r in prec["review_queue"]
        )
        if items:
            tensor_html += f"<h4>Conflicts found in {e(pred.split('/')[-1])} while diffing</h4><ul>{items}</ul>"
    # params breakdown
    tot = pub["params_total"]
    p_rows = "".join(
        f'<tr><td>{e(k.replace("_", " "))}</td><td class="num">{v:,}</td><td class="num">{100 * v / tot:.2f}%</td></tr>'
        for k, v in components.items()
    )

    # cache breakdown table
    c_rows = "".join(
        f'<tr><td><span class="dot" style="background:{COLOR.get(r["layer_type"], "#999")}"></span>{e(LABEL.get(r["layer_type"], r["layer_type"]))}</td>'
        f'<td class="num">{r["layers"]}</td><td class="num">{(r["grows_per_token_bytes"] or 0):,}</td>'
        f'<td class="num">{(r["constant_bytes"] or 0) / 2**20:,.2f}</td><td class="small">{e(r["formula"])}</td></tr>'
        for r in cache["breakdown"]
    )
    assumptions = "".join(f"<li>{e(a)}</li>" for a in cache["assumptions"])
    notes = "".join(f"<li>{e(n)}</li>" for n in rec.get("unchecked_notes", [])) or "<li>Nothing left unchecked.</li>"
    calc_data = {
        "grow": cache["grows_per_token_bytes"],
        "constant": cache["constant_bytes"],
        "naive": cache["naive_mha_per_token_bytes"],
        "max": int(ctx),
        "rows": [
            {
                "t": LABEL.get(r["layer_type"], r["layer_type"]),
                "c": COLOR.get(r["layer_type"], "#999"),
                "g": r["grows_per_token_bytes"] or 0,
                "k": r["constant_bytes"] or 0,
            }
            for r in cache["breakdown"]
        ],
    }

    return f"""
<section id="{anchor}" class="model">
  <h2>{e(repo.split("/")[-1])}</h2>
  <p class="meta"><a href="https://huggingface.co/{e(repo)}">{e(repo)}</a> · commit <code>{e(commit[:10])}</code> · license {e(rec.get("license") or "see repo")} · analyzed {e(rec["analyzed_at"][:16].replace("T", " "))} UTC</p>
  <div class="tiles">{t_html}</div>

  <h3>Architecture</h3>
  <div class="diagram">{svg}</div>

  <h3>Inference cache calculator</h3>
  <p>Drag to a context length. Layers whose cache grows with context are the only ones that matter at long context; the rest cost a fixed amount.</p>
  <div class="calc" data-model='{json.dumps(calc_data)}'>
    <input type="range" min="0" max="1000" value="1000" aria-label="Context length">
    <div class="calc-out"></div>
    <div class="bars"></div>
  </div>
  <table><thead><tr><th>Layer type</th><th class="num">Layers</th><th class="num">Bytes per token</th><th class="num">Fixed (MiB)</th><th>How it is computed</th></tr></thead><tbody>{c_rows}</tbody></table>
  <details><summary>Assumptions in the cache numbers</summary><ul>{assumptions}</ul></details>

  <h3>What changed since {e(pred.split("/")[-1]) if pred else "the predecessor"}</h3>
  <table><thead><tr><th>Field</th><th>{e(pred.split("/")[-1]) if pred else "Before"}</th><th>{e(repo.split("/")[-1])}</th></tr></thead><tbody>{d_rows}</tbody></table>
  {tensor_html}

  <h3>Verification: {n_ver} verified, {n_single} single source, {n_conf} conflicts</h3>
  <p>Each field is read from every source that states it. A field is published when sources agree or only one source states it; a disagreement goes to a human instead.</p>
  <div class="tablewrap"><table class="vt"><thead><tr><th>Field</th><th>Value</th><th>Status</th><th>Sources</th></tr></thead><tbody>{v_rows}</tbody></table></div>

  <h3>Where the parameters are</h3>
  <table><thead><tr><th>Component</th><th class="num">Parameters</th><th class="num">Share</th></tr></thead><tbody>{p_rows}</tbody></table>
  <p class="small">{inventory["tensor_count"]:,} tensors read from {e(repo)} weight headers; {inventory["skipped_scale_tensors"]:,} quantization-scale tensors skipped as metadata. 4-bit packed weights count two parameters per stored byte.</p>

  <h3>Not checked</h3>
  <ul>{notes}</ul>
</section>"""


def main():
    sections = "".join(model_section(n, a) for n, a in MODELS)
    nav = " · ".join(f'<a href="#{a}">{n.split("__")[1]}</a>' for n, a in MODELS)
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ArchWatch prototype: Kimi K3.1 Pro and LFM3-1.7B</title>
<style>
:root{{--bg:#fff;--fg:#1f2328;--mut:#59636e;--line:#d1d9e0;--soft:#f6f8fa;--acc:#0969da}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--line:#30363d;--soft:#161b22;--acc:#4493f8}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:28px;margin:0 0 6px}}h2{{font-size:24px;margin:56px 0 4px;padding-top:24px;border-top:1px solid var(--line)}}
h3{{font-size:17px;margin:32px 0 8px}}h4{{font-size:14px;margin:12px 0 6px}}
a{{color:var(--acc)}}code{{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--soft);padding:1px 4px;border-radius:4px}}
.lead{{font-size:16.5px;color:var(--fg);max-width:780px}}.meta,.small{{color:var(--mut);font-size:13px}}
.stages{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:20px 0}}
.stage{{border:1px solid var(--line);border-radius:8px;padding:10px;background:var(--soft)}}.stage b{{display:block;font-size:13.5px}}.stage span{{font-size:12.5px;color:var(--mut)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.tile{{border:1px solid var(--line);border-radius:10px;padding:12px}}.tl{{font-size:12.5px;color:var(--mut)}}.tv{{font-size:22px;font-weight:650;margin:2px 0}}.ts{{font-size:12px;color:var(--mut)}}
.diagram{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px;overflow-x:auto}}.diagram svg{{width:100%;height:auto;min-width:640px;display:block}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0}}th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-weight:600;color:var(--mut);font-size:12.5px}}.num{{text-align:right;font-variant-numeric:tabular-nums}}
.tablewrap{{overflow-x:auto}}.vt td.ev{{font-size:12.5px;color:var(--mut);min-width:340px}}
.st{{font-size:12px;padding:2px 8px;border-radius:99px;white-space:nowrap}}.st-verified{{background:#1a7f3722;color:#1a7f37}}
.st-single-source{{background:#9a670022;color:#9a6700}}.st-conflict{{background:#cf222e22;color:#cf222e}}
@media (prefers-color-scheme:dark){{.st-verified{{color:#3fb950}}.st-single-source{{color:#d29922}}.st-conflict{{color:#f85149}}}}
.src{{font-size:11px;padding:0 5px;border-radius:4px;border:1px solid var(--line);color:var(--fg)}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:700px){{.cols{{grid-template-columns:1fr}}}}
ul{{padding-left:20px}}li{{margin:3px 0}}
.calc{{border:1px solid var(--line);border-radius:10px;padding:14px;background:var(--soft)}}.calc input{{width:100%}}
.calc-out{{font-size:15px;margin:8px 0}}.calc-out b{{font-size:20px}}.bars{{display:flex;height:22px;border-radius:5px;overflow:hidden;background:var(--line)}}
.bars div{{height:100%}}details{{margin:8px 0;color:var(--mut);font-size:13.5px}}
</style></head><body><main>
<h1>ArchWatch prototype run</h1>
<p class="lead">Two models analyzed at the source commits shown below. Architecture views use published verifier-approved values and recompute their cache estimates from those values. Model-card statements are claims to cross-check against config, code and tensor headers.</p>
<p class="meta">Models: {nav} · predecessors diffed: Kimi-K3, LFM2.5-1.2B-Instruct · no weights downloaded: shapes come from safetensors headers via HTTP range requests</p>
<div class="stages">
<div class="stage"><b>1 Input</b><span>Fixed watchlist, commit pinned</span></div>
<div class="stage"><b>2 Config</b><span>config.json to schema, no LLM</span></div>
<div class="stage"><b>3 Weights</b><span>tensor shapes, exact counts</span></div>
<div class="stage"><b>4 Code</b><span>AST scan, line citations</span></div>
<div class="stage"><b>5 Card</b><span>claims, with rounding tolerance</span></div>
<div class="stage"><b>6 Calculator</b><span>cache per layer type</span></div>
<div class="stage"><b>7 Verify</b><span>agree, single, or conflict</span></div>
<div class="stage"><b>8 Diff + publish</b><span>vs predecessor, diagram, page</span></div>
</div>
{sections}
<h2>Sources</h2>
<ul class="small">
<li><a href="https://huggingface.co/moonshotai/Kimi-K3.1-Pro">moonshotai/Kimi-K3.1-Pro</a> and <a href="https://huggingface.co/moonshotai/Kimi-K3">moonshotai/Kimi-K3</a> on Hugging Face: config.json, configuration_kimi_k3.py, README.md, 96 safetensors headers each</li>
<li><a href="https://huggingface.co/LiquidAI/LFM3-1.7B">LiquidAI/LFM3-1.7B</a> and <a href="https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct">LiquidAI/LFM2.5-1.2B-Instruct</a>: config.json, modeling_lfm3.py, configuration_lfm3.py, README.md, safetensors headers</li>
<li>Inspired by Sebastian Raschka's hand-curated <a href="https://sebastianraschka.com/llm-architecture-gallery/">LLM Architecture Gallery</a>; nothing here is taken from it.</li>
</ul>
</main>
<script>
function fmtB(b){{const u=['B','KiB','MiB','GiB','TiB'];let i=0;while(b>=1024&&i<u.length-1){{b/=1024;i++}}return b.toFixed(b<10&&i?2:1)+' '+u[i]}}
document.querySelectorAll('.calc').forEach(el=>{{
  const m=JSON.parse(el.dataset.model),r=el.querySelector('input'),out=el.querySelector('.calc-out'),bars=el.querySelector('.bars');
  const lo=Math.log(1024),hi=Math.log(m.max);
  function upd(){{
    const n=Math.round(Math.exp(lo+(hi-lo)*r.value/1000));
    const tot=m.grow*n+m.constant, naive=m.naive*n;
    out.innerHTML='At <b>'+n.toLocaleString()+'</b> tokens the cache is <b>'+fmtB(tot)+'</b> ('+fmtB(m.grow*n)+' growing + '+fmtB(m.constant)+' fixed). The same model with plain multi-head attention would need '+fmtB(naive)+'.';
    bars.innerHTML=m.rows.map(x=>{{const v=x.g*n+x.k;return v?'<div title="'+x.t+': '+fmtB(v)+'" style="width:'+(100*v/tot)+'%;background:'+x.c+'"></div>':''}}).join('');
  }}
  r.addEventListener('input',upd);upd();
}});
</script></body></html>"""
    (ROOT / "site").mkdir(exist_ok=True)
    (ROOT / "site" / "index.html").write_text(page)
    print("site/index.html", len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
