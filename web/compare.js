/* ArchWatch compare tool. Plain JS; reads ../data/models.json (published values only). */
var AW = (function () {
  function parseState(search, stems, preset) {
    /* ?a=&b=&c= -> up to three distinct known stems; falls back to the page preset. */
    var q = new URLSearchParams(search || ''), out = [];
    ['a', 'b', 'c'].forEach(function (k) {
      var v = q.get(k);
      if (v && stems.indexOf(v) >= 0 && out.indexOf(v) < 0) out.push(v);
    });
    if (out.length < 2 && preset && preset.length === 2) return preset.slice();
    return out;
  }
  function stateQuery(sel) {
    var keys = ['a', 'b', 'c'], parts = [];
    sel.forEach(function (s, i) { if (s) parts.push(keys[i] + '=' + encodeURIComponent(s)); });
    return parts.length ? '?' + parts.join('&') : '';
  }
  function cacheBytes(c, n) {
    var windowed = c.window ? Math.floor(c.window_full * Math.min(n, c.window) / c.window) : 0;
    return c.grow * n + windowed + c.fixed;
  }
  function csv(rows) {
    return rows.map(function(row){return row.map(function(value){var s=String(value==null?'':value);if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"'}).join(',')}).join('\r\n')+'\r\n';
  }
  return { csv:csv, parseState: parseState, stateQuery: stateQuery, cacheBytes: cacheBytes };
})();
if (typeof module !== 'undefined') module.exports = AW;

(function () {
  if (typeof document === 'undefined') return;
  var tool = document.getElementById('tool');
  if (!tool) return;
  var D, byStem = {}, byName = {}, sel = [];
  var ctxPick, precPick = '16', editingThird = false;

  function el(tag, attrs, text) {
    var x = document.createElement(tag);
    for (var k in (attrs || {})) x.setAttribute(k, attrs[k]);
    if (text !== undefined) x.textContent = text;
    return x;
  }
  function sig(x) { var t = Number(x).toPrecision(3); return String(parseFloat(t)); }
  function params(n) {
    if (n === null || n === undefined) return 'n/a';
    if (n >= 1e12) return sig(n / 1e12) + 'T';
    if (n >= 1e9) return sig(n / 1e9) + 'B';
    return sig(n / 1e6) + 'M';
  }
  function bytes(b) {
    if (b >= Math.pow(2, 30)) return (b / Math.pow(2, 30)).toFixed(2) + ' GiB';
    if (b >= Math.pow(2, 20)) return sig(b / Math.pow(2, 20)) + ' MiB';
    if (b >= 1024) return sig(b / 1024) + ' KiB';
    return b + ' B';
  }
  function tokens(n) {
    if (n >= 1048576 && n % 1048576 === 0) return (n / 1048576) + 'M';
    if (n % 1024 === 0) return (n / 1024) + 'K';
    return n.toLocaleString('en-US');
  }
  function models() { return sel.map(function (s) { return byStem[s]; }).filter(Boolean); }

  function setUrl() { history.replaceState(null, '', location.pathname + AW.stateQuery(sel)); }

  function fillPickers() {
    ['a', 'b', 'c'].forEach(function (k, i) {
      var box = tool.querySelector('[data-slot=' + k + ']'), m = byStem[sel[i]];
      box.querySelector('input').value = m ? m.name : '';
      box.querySelector('.slotmeta').textContent = m ? m.publisher + ' · ' + m.date + ' · ' + m.design : '';
      box.style.setProperty('--slot', D.slots[i]);
      if(k==='c')box.hidden=!m&&!editingThird;
      var preview=box.querySelector('.slot-preview');preview.textContent='';
      if(m){var strip=el('div',{'class':'cells',role:'img','aria-label':m.name+' layer order, layer 1 first'});m.layers.forEach(function(kind){strip.appendChild(el('i',{style:'background:'+D.colors[kind]}))});preview.appendChild(strip);
      var legend=el('ul',{'class':'legend'});Object.keys(m.counts).forEach(function(kind){var li=el('li');li.appendChild(el('span',{'class':'sw',style:'background:'+D.colors[kind],'aria-hidden':'true'}));li.appendChild(document.createTextNode(D.kinds[kind]+' '+m.counts[kind]));legend.appendChild(li)});preview.appendChild(legend);
      var link=el('a',{'class':'slot-evidence',href:'../models/'+m.stem+'.html#facts'},'Inspect '+m.layers.length+' layers and source evidence →');preview.appendChild(link)}
    });
    document.getElementById('add-third').hidden = !!sel[2] || editingThird;
  }

  function value(m, f) {
    if (m.fields[f]) return m.fields[f];
    if (m.derived[f]) return m.derived[f];
    var imp = (D.implied[m.stem] || {})[f];
    return [imp || 'not stated', null];
  }
  function rowValues(ms,f){var vals=ms.map(function(m){return value(m,f).slice()});if(f.indexOf('params')!==0)return vals;
    function collision(){return vals.some(function(a,i){return vals.some(function(b,j){return i!==j&&a[1]!==b[1]&&typeof a[1]==='number'&&typeof b[1]==='number'&&a[0]===b[0]})})}
    for(var digits=4;digits<=12&&collision();digits++)vals.forEach(function(v){if(typeof v[1]!=='number')return;var n=v[1],unit=n>=1e12?'T':n>=1e9?'B':'M',div=unit==='T'?1e12:unit==='B'?1e9:1e6;v[0]=String(Number((n/div).toPrecision(digits)))+unit});return vals;
  }
  function spread(nums) {
    var xs = nums.filter(function (x) { return x; });
    if (xs.length < 2) return 0;
    return Math.log(Math.max.apply(null, xs) / Math.min.apply(null, xs));
  }
  function list(ms, fn) { return ms.map(function (m) { return m.name + ' ' + fn(m); }).join(', '); }
  function renderKeyDiff(ms) {
    /* one sentence per field, covering every selected model; ranked by the size of the difference */
    var box = document.getElementById('keydiff'); box.textContent = '';
    var items = [], n = Math.min.apply(null, [131072].concat(ms.map(function (m) { return m.context; })));
    function same(fn) { var v = ms.map(fn); return v.every(function (x) { return x === v[0]; }); }
    if (!same(function (m) { return m.design + m.hybrid; }))
      items.push([2.5, 'Design: ' + list(ms, function (m) {
        return m.design === 'MoE' ? 'MoE (' + params(m.params.active) + ' of ' + params(m.params.total) + ' active per token)' : m.design + ' (every parameter active)';
      }) + '.']);
    var tot = spread(ms.map(function (m) { return m.params.total; }));
    if (tot > Math.log(1.1)) items.push([tot, 'Total parameters: ' + list(ms, function (m) { return params(m.params.total); }) + '.']);
    if (ms.every(function (m) { return m.params.active; })) {
      var act = spread(ms.map(function (m) { return m.params.active; }));
      if (act > Math.log(1.1)) items.push([act, 'Active parameters per token: ' + list(ms, function (m) { return params(m.params.active); }) + '.']);
    }
    if (!same(function (m) { return Object.keys(m.counts).sort().join(); }))
      items.push([2, 'Layer types: ' + ms.map(function (m) {
        return m.name + ' ' + Object.keys(m.counts).map(function (k) { return m.counts[k] + ' ' + D.kinds[k]; }).join(' + ');
      }).join('; ') + '.']);
    var lay = spread(ms.map(function (m) { return m.layers.length; }));
    if (lay > Math.log(1.1)) items.push([lay, 'Layers: ' + list(ms, function (m) { return m.layers.length; }) + '.']);
    var ctx = spread(ms.map(function (m) { return m.context; }));
    if (ctx > 0) items.push([ctx, 'Context: ' + list(ms, function (m) { return tokens(m.context); }) + '.']);
    var cs = ms.map(function (m) { return AW.cacheBytes(m.cache, n); }), cache = spread(cs);
    if (cache > Math.log(1.1)) items.push([cache, 'Cache at ' + tokens(n) + ' tokens: ' + ms.map(function (m, i) {
      return m.name + ' ' + bytes(cs[i]) + (!m.cache.grow && m.cache.window ? ' (stops growing at ' + m.cache.window.toLocaleString('en-US') + ' tokens)' : '');
    }).join(', ') + '.']);
    if (!same(function (m) { return value(m, 'attention')[0]; }))
      items.push([1, 'Attention: ' + list(ms, function (m) { return value(m, 'attention')[0]; }) + '.']);
    ['hidden_size', 'vocab_size'].forEach(function (f) {
      var r = spread(ms.map(function (m) { return value(m, f)[1]; }));
      if (r > Math.log(1.2)) items.push([r * 0.6, D.labels[f] + ': ' + list(ms, function (m) { return value(m, f)[0]; }) + '.']);
    });
    items.sort(function (x, y) { return y[0] - x[0] || x[1].localeCompare(y[1]); });
    var overview=el('p',{'class':'comparison-overview plain-only'}),sentences=[];
    if(ms.length===2){var a=ms[0],b=ms[1];if(a.layers.length===b.layers.length)sentences.push('Both use '+a.layers.length+' decoder layers.');
    if(JSON.stringify(a.counts)!==JSON.stringify(b.counts))sentences.push('The layer mix changes from '+Object.keys(a.counts).map(function(k){return a.counts[k]+' '+D.kinds[k]}).join(' + ')+' to '+Object.keys(b.counts).map(function(k){return b.counts[k]+' '+D.kinds[k]}).join(' + ')+'.');
    if(a.cache.grow!==b.cache.grow)sentences.push('Growing cache changes from '+bytes(a.cache.grow)+' to '+bytes(b.cache.grow)+' per token. Fixed state and windowed cache are included separately in the comparison below.');
    if(spread([a.params.total,b.params.total])>Math.log(1.1))sentences.push('Stored parameters change from '+params(a.params.total)+' to '+params(b.params.total)+'.');
    if(a.context!==b.context)sentences.push('The stated maximum context changes from '+tokens(a.context)+' to '+tokens(b.context)+' tokens.');
    if(!sentences.length)sentences=items.slice(0,3).map(function(x){return x[1]});
    }else sentences=items.slice(0,3).map(function(x){return x[1]});
    overview.textContent=sentences.join(' ')||'No architecture difference was found in the displayed fields.';box.appendChild(overview);
    var ul = el('ul', { 'class': 'changes technical-only' });
    items.forEach(function (it) { ul.appendChild(el('li', {}, it[1])); });
    if (!items.length) ul.appendChild(el('li', {}, 'No difference of 10% or more in design, size, layers, context or cache.'));
    box.appendChild(ul);
    var pred = ms.length === 2 && (ms[1].predecessor === ms[0].stem ? ms[1] : ms[0].predecessor === ms[1].stem ? ms[0] : null);
    if (pred && pred.note) {
      box.appendChild(el('p', { 'class': 'prose' }, pred.note));
      box.appendChild(el('p', { 'class': 'byline' }, 'Written by Gemini from the verified diff; every number checked by code.'));
    }
    box.appendChild(el('p', { 'class': 'byline' }, 'One sentence per field, written by code from published values; largest differences first.'));
  }

  function renderCacheBars(ms){var box=document.getElementById('cache-summarybars');box.textContent='';var common=Math.min.apply(null,ms.map(function(m){return m.context})),points=[Math.min(131072,common),common].filter(function(n,i,a){return a.indexOf(n)===i}),rows=[];
  points.forEach(function(n){ms.forEach(function(m,i){rows.push({label:m.name+' at '+tokens(n),bytes:AW.cacheBytes(m.cache,n),color:D.slots[i]})})});var maximum=Math.max.apply(null,rows.map(function(r){return r.bytes}))||1;var list=el('ul',{'class':'cache-comparison-bars'});rows.forEach(function(r){var li=el('li');li.appendChild(el('span',{},r.label));var track=el('span',{'class':'track'});track.appendChild(el('span',{'class':'fill',style:'width:'+Math.min(100,100*r.bytes/maximum)+'%;background:'+r.color}));li.appendChild(track);li.appendChild(el('span',{'class':'mono'},bytes(r.bytes)));list.appendChild(li)});box.appendChild(list)}

  function renderStrips(ms) {
    var box = document.getElementById('strips'); box.textContent = '';
    ms.forEach(function (m, i) {
      var card = el('div', { 'class': 'panel stripcard' });
      var head = el('div', { 'class': 'stackhead' });
      var h = el('h3'); var a = el('a', { href: '../models/' + m.stem + '.html' }, m.name); h.appendChild(a); head.appendChild(h);
      head.appendChild(el('span', { 'class': 'mono' }, m.layers.length + ' layers'));
      card.appendChild(head);
      var strip = el('div', { 'class': 'cells', role: 'img', 'aria-label': m.name + ' layer order, layer 1 first' });
      m.layers.forEach(function (k) { strip.appendChild(el('i', { style: 'background:' + D.colors[k] })); });
      card.appendChild(strip);
      var lg = el('ul', { 'class': 'legend' });
      Object.keys(m.counts).forEach(function (k) {
        var li = el('li'); li.appendChild(el('span', { 'class': 'sw', style: 'background:' + D.colors[k], 'aria-hidden': 'true' }));
        li.appendChild(document.createTextNode(D.kinds[k] + ' ' + m.counts[k])); lg.appendChild(li);
      });
      card.appendChild(lg); box.appendChild(card);
    });
  }

  function renderChart(ms) {
    var box = document.getElementById('chart'), dataBox = document.getElementById('chartdata');
    box.textContent = ''; dataBox.textContent = '';
    var maxX = Math.max.apply(null, ms.map(function (m) { return m.context; }));
    var maxY = 0;
    ms.forEach(function (m) { m.series.forEach(function (p) { if (p[0] <= maxX && p[1] > maxY) maxY = p[1]; }); });
    maxY = maxY || 1;
    var W = Math.max(300, Math.min(900, box.clientWidth || 760)), H = W < 500 ? 260 : 300, L = 64, R = 14, T = 16, B = 40, NS = 'http://www.w3.org/2000/svg';
    var lo = Math.log(1024), hi = Math.log(maxX);
    function X(n) { return L + (W - L - R) * (Math.log(n) - lo) / (hi - lo || 1); }
    function Y(v) { return T + (H - T - B) * (1 - v / maxY); }
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H); svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Cache memory against context length for ' + ms.map(function (m) { return m.name; }).join(', '));
    function line(x1, y1, x2, y2, cls) { var l = document.createElementNS(NS, 'line'); l.setAttribute('x1', x1); l.setAttribute('y1', y1); l.setAttribute('x2', x2); l.setAttribute('y2', y2); l.setAttribute('class', cls); svg.appendChild(l); }
    function text(x, y, s, anchor) { var t = document.createElementNS(NS, 'text'); t.setAttribute('x', x); t.setAttribute('y', y); t.setAttribute('text-anchor', anchor || 'middle'); t.setAttribute('class', 'axis'); t.textContent = s; svg.appendChild(t); }
    for (var g = 0; g <= 4; g++) { var v = maxY * g / 4; line(L, Y(v), W - R, Y(v), 'grid'); text(L - 8, Y(v) + 4, bytes(v), 'end'); }
    [1024, 8192, 32768, 131072, 1048576].filter(function (n) { return n <= maxX; }).forEach(function (n) { line(X(n), T, X(n), H - B, 'grid'); text(X(n), H - B + 18, tokens(n)); });
    text((L + W - R) / 2, H - 4, 'context length (tokens, log scale)');
    ms.forEach(function (m, i) {
      var pts = m.series.filter(function (p) { return p[0] <= maxX; }).map(function (p) { return X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1); });
      var pl = document.createElementNS(NS, 'polyline'); pl.setAttribute('points', pts.join(' ')); pl.setAttribute('class', 'series');
      pl.setAttribute('stroke', D.slots[i]); svg.appendChild(pl);
      var last = m.series[m.series.length - 1], dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('cx', X(last[0])); dot.setAttribute('cy', Y(last[1])); dot.setAttribute('r', 4); dot.setAttribute('fill', D.slots[i]); svg.appendChild(dot);
    });
    box.appendChild(svg);
    var lg = el('ul', { 'class': 'legend' });
    ms.forEach(function (m, i) {
      var li = el('li'); li.appendChild(el('span', { 'class': 'sw', style: 'background:' + D.slots[i], 'aria-hidden': 'true' }));
      li.appendChild(document.createTextNode(m.name + ' (max ' + tokens(m.context) + ', ' + m.cache_label + ')')); lg.appendChild(li);
    });
    box.appendChild(lg);
    var t = el('table', { 'class': 'facts' }), hd = el('tr'); hd.appendChild(el('th', { scope: 'col' }, 'Context'));
    ms.forEach(function (m) { hd.appendChild(el('th', { scope: 'col' }, m.name)); });
    var th = el('thead'); th.appendChild(hd); t.appendChild(th); var tb = el('tbody');
    var xs = []; ms.forEach(function (m) { m.series.forEach(function (p) { if (p[0] <= maxX && xs.indexOf(p[0]) < 0) xs.push(p[0]); }); });
    xs.sort(function (a, b) { return a - b; }).forEach(function (n) {
      var tr = el('tr'); tr.appendChild(el('th', { scope: 'row' }, tokens(n)));
      ms.forEach(function (m) { var p = m.series.filter(function (q) { return q[0] === n; })[0]; tr.appendChild(el('td', { 'class': 'mono' }, p ? bytes(p[1]) : (n > m.context ? 'beyond max' : bytes(AW.cacheBytes(m.cache, n))))); });
      tb.appendChild(tr);
    });
    t.appendChild(tb); dataBox.appendChild(t);
  }

  function renderFit(ms) {
    var maxX = Math.max.apply(null, ms.map(function (m) { return m.context; }));
    var ctxs = D.contexts.filter(function (n) { return n <= maxX; });
    if (ctxs.indexOf(ctxPick) < 0) ctxPick = ctxs.indexOf(32768) >= 0 ? 32768 : ctxs[ctxs.length - 1];
    function seg(id, items, current, onpick) {
      var g = document.getElementById(id); g.textContent = '';
      items.forEach(function (it) {
        var b = el('button', { type: 'button', 'aria-pressed': String(it[0] === current) }, it[1]);
        b.addEventListener('click', function () { onpick(it[0]); renderFit(models()); }); g.appendChild(b);
      });
    }
    seg('ctxseg', ctxs.map(function (n) { return [n, tokens(n)]; }), ctxPick, function (v) { ctxPick = v; });
    seg('precseg', D.precision.map(function (p) { return [p.key, p.label]; }), precPick, function (v) { precPick = v; });
    var per = D.precision.filter(function (p) { return p.key === precPick; })[0].bytes;
    var box = document.getElementById('fit-compare'); box.textContent = '';
    ms.forEach(function (m, i) {
      var row = el('div', { 'class': 'fitrow' }), head = el('div', { 'class': 'fithead' });
      head.appendChild(el('b', { style: 'color:' + D.slots[i] }, m.name));
      if (ctxPick > m.context) {
        head.appendChild(el('span', { 'class': 'sub' }, 'beyond its maximum context (' + tokens(m.context) + ')'));
        row.appendChild(head); box.appendChild(row); return;
      }
      var w = m.params.total * per, c = AW.cacheBytes(m.cache, ctxPick), total = w + c;
      head.appendChild(el('span', { 'class': 'mono' }, bytes(total) + '  (weights ' + bytes(w) + ', cache ' + bytes(c) + ')'));
      row.appendChild(head);
      var ul = el('ul', { 'class': 'devices' });
      D.devices.forEach(function (d) {
        var word = total <= 0.75 * d.bytes ? 'fits' : total <= d.bytes ? 'tight' : 'too big';
        var li = el('li'); li.appendChild(el('span', {}, d.label));
        var tr = el('span', { 'class': 'track' }); tr.appendChild(el('span', { 'class': 'fill', style: 'width:' + Math.min(100, 100 * total / d.bytes).toFixed(1) + '%;background:' + D.slots[i] }));
        li.appendChild(tr); li.appendChild(el('span', { 'class': 'verdict v-' + word.replace(' ', '-') }, word)); ul.appendChild(li);
      });
      row.appendChild(ul); box.appendChild(row);
    });
  }

  function chip(a, b) {
    if (a === b) return ['same', 'same'];
    return ['differs', 'up'];
  }
  function chipNum(ra, rb, da, db) {
    if (ra && rb && ra === rb) return ['same', 'same'];
    if (!(ra && rb) && da === db) return ['same', 'same'];
    if (ra && rb) {
      if (rb === 2 * ra) return ['doubled', 'up'];
      if (2 * rb === ra) return ['halved', 'warn'];
      var pct = 100 * (rb - ra) / ra;
      if (Math.abs(pct) < 0.1) return [(pct > 0 ? '<+0.1%' : '<-0.1%'), 'same'];
      return [(pct > 0 ? '+' : '') + pct.toFixed(Math.abs(pct) >= 1 ? 0 : 1) + '%', rb > ra ? 'up' : 'warn'];
    }
    return ['differs', 'up'];
  }
  function table(ms, fieldsList) {
    var t = el('table', { 'class': 'facts cmp' }), hd = el('tr');
    hd.appendChild(el('th', { scope: 'col' }, 'Field'));
    ms.forEach(function (m) { hd.appendChild(el('th', { scope: 'col' }, m.name)); });
    hd.appendChild(el('th', { scope: 'col' }, 'Change'));
    var th = el('thead'); th.appendChild(hd); t.appendChild(th);
    var tb = el('tbody');
    fieldsList.forEach(function (f) {
      if (!ms.some(function (m) { return m.fields[f] || m.derived[f]; })) return;
      var vals = rowValues(ms,f);
      var shown = vals.map(function (v) { return v[0]; }), raws = vals.map(function (v) { return v[1]; });
      var c = ms.length === 2 ? chipNum(raws[0], raws[1], shown[0], shown[1]) : chip(shown.every(function (x) { return x === shown[0]; }) ? 1 : 0, 1);
      var tr = el('tr', { 'data-diff': c[0] === 'same' ? '0' : '1' });
      var label=el('th',{scope:'row'},D.labels[f]);label.appendChild(el('span',{'class':'key technical-only'},f));tr.appendChild(label);
      shown.forEach(function (x) { tr.appendChild(el('td', { 'class': 'mono' }, x)); });
      var td = el('td'); td.appendChild(el('span', { 'class': 'chip chip-' + c[1] }, c[0])); tr.appendChild(td);
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    return t;
  }
  function renderFields(ms) {
    var box = document.getElementById('fields'); box.textContent = '';
    box.appendChild(table(ms, D.order));
    var more = el('details', { 'class': 'allfields' }), sum = el('summary', {}, 'All fields');
    more.appendChild(sum); more.appendChild(table(ms, D.rest)); box.appendChild(more);
    applyOnlyDiff();
  }
  function applyOnlyDiff() {
    var on = document.getElementById('onlydiff').checked;
    document.querySelectorAll('#fields tr[data-diff="0"]').forEach(function (r) { r.hidden = on; });
  }

  function render() {
    var ms = models();
    fillPickers(); setUrl();
    if (ms.length < 2) return;
    renderKeyDiff(ms); renderStrips(ms); renderChart(ms); renderFit(ms); renderFields(ms); renderCacheBars(ms);
  }

  function start(data) {
    D = data;
    D.models.forEach(function (m) { byStem[m.stem] = m; byName[m.name.toLowerCase()] = m.stem; });
    var stems = D.models.map(function (m) { return m.stem; });
    var preset = tool.dataset.a ? [tool.dataset.a, tool.dataset.b] : null;
    if (!preset && D.suggested.length) preset = [D.suggested[0].a, D.suggested[0].b];
    sel = AW.parseState(location.search, stems, preset);
    tool.hidden = false;
    ['a', 'b', 'c'].forEach(function (k, i) {
      var input = tool.querySelector('#pick-' + k);
      input.addEventListener('change', function () {
        var s = byName[input.value.trim().toLowerCase()];
        var message=document.getElementById('selection-error');message.textContent='';
        if(s&&(sel.indexOf(s)<0||sel[i]===s)){sel[i]=s}
        else if(!input.value.trim()&&i===2){sel.splice(2,1);editingThird=false}
        else{render();message.textContent='Selection unchanged. Choose a different published model from the suggestions.';return}
        render();
      });
    });
    document.getElementById('swap').addEventListener('click', function () { var t = sel[0]; sel[0] = sel[1]; sel[1] = t; render(); });
    document.getElementById('clear-c').addEventListener('click', function () { sel = sel.slice(0, 2); editingThird=false; render(); });
    document.getElementById('add-third').addEventListener('click',function(){editingThird=true;fillPickers();document.getElementById('pick-c').focus()});
    tool.querySelectorAll('[data-edit]').forEach(function(button){button.addEventListener('click',function(){var input=document.getElementById('pick-'+button.dataset.edit);input.focus();input.select()})});
    document.querySelectorAll('[data-reading]').forEach(function(button){if(button.tagName!=='BUTTON')return;button.addEventListener('click',function(){var mode=button.dataset.reading;tool.dataset.reading=mode;document.querySelectorAll('button[data-reading]').forEach(function(b){b.setAttribute('aria-pressed',String(b===button))})})});
    document.getElementById('pickers').addEventListener('submit',function(ev){ev.preventDefault()});
    document.getElementById('onlydiff').addEventListener('change', applyOnlyDiff);
    document.querySelector('.compare-chart-details').addEventListener('toggle',function(ev){if(ev.currentTarget.open&&models().length>=2)renderChart(models())});
    document.getElementById('export-csv').addEventListener('click',function(){
      var ms=models(),rows=[['Field'].concat(ms.map(function(m){return m.name}))];
      rows.push(['Repository'].concat(ms.map(function(m){return m.id})));
      rows.push(['Source commit'].concat(ms.map(function(m){return m.source_commit})));
      rows.push(['Pinned primary files'].concat(ms.map(function(m){return m.source_url})));
      D.order.concat(D.rest).forEach(function(f){if(ms.some(function(m){return m.fields[f]||m.derived[f]}))rows.push([D.labels[f]].concat(rowValues(ms,f).map(function(v){return v[0]})))});
      var url=URL.createObjectURL(new Blob([AW.csv(rows)],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='archwatch-comparison.csv';a.click();setTimeout(function(){URL.revokeObjectURL(url)},1000);
    });
    document.getElementById('copylink').addEventListener('click', function () {
      var url = location.origin + location.pathname + AW.stateQuery(sel);
      var done = function () { document.getElementById('copied').textContent = 'Link copied'; };
      var failed = function () { document.getElementById('copied').textContent = 'Could not copy. Copy the URL from your address bar.'; };
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(done, failed); else failed();
    });
    tool.querySelectorAll('.chipbtn').forEach(function (a) {
      a.addEventListener('click', function (ev) { ev.preventDefault(); sel = [a.dataset.a, a.dataset.b]; render(); });
    });
    render();
  }
  fetch('../data/models.json?v='+tool.dataset.version).then(function (r) { return r.json(); }).then(start).catch(function () {
    tool.hidden = true; document.getElementById('compare-error').hidden=false;
  });
})();
