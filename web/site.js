(function(){var root=document.documentElement,btn=document.querySelector('.themebtn');
  function cur(){return root.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}
  if(btn){btn.setAttribute('aria-pressed',cur()==='dark');btn.addEventListener('click',function(){
    var t=cur()==='dark'?'light':'dark';root.dataset.theme=t;btn.setAttribute('aria-pressed',t==='dark');
    try{localStorage.setItem('aw-theme',t)}catch(e){}});}
})();
(function(){var p=document.getElementById('fit');if(!p||!p.dataset.combos)return;var combos=JSON.parse(p.dataset.combos);
  function sel(attr){return p.querySelector('['+attr+'][aria-pressed=true]').getAttribute(attr)}
  function bytes(n){if(n===0)return '0 B';var units=['B','KiB','MiB','GiB','TiB'],i=Math.min(4,Math.floor(Math.log(n)/Math.log(1024)));return +(n/Math.pow(1024,i)).toPrecision(3)+' '+units[i]}
  function draw(){var c=combos[sel('data-ctx')+'-'+sel('data-prec')],seqInput=document.getElementById('sequences'),ramInput=document.getElementById('device-memory'),headInput=document.getElementById('headroom');
    var status=document.getElementById('budget-result');
    if(![seqInput,ramInput,headInput].every(function(x){return x.checkValidity()&&x.value!==''})){status.textContent='Enter valid memory, sequence and headroom values.';status.className='budget-result over';return}
    var sequences=+seqInput.value,w=c.weights_bytes,k=c.cache_bytes*sequences,total=w+k;
    p.querySelector('[data-out=total]').textContent=bytes(total);p.querySelector('[data-out=weights]').textContent=bytes(w);p.querySelector('[data-out=cache]').textContent=bytes(k);
    var memory=+ramInput.value*1e9,available=memory*(1-(+headInput.value)/100),fits=total<=available;
    status.textContent=(fits?'Within':'Exceeds')+' your planning budget: '+bytes(total)+' required / '+bytes(available)+' available after headroom. '+(fits?'Confirm with your runtime before deployment.':'Reduce precision, context or concurrent sequences.');status.className='budget-result'+(fits?'':' over');
    [8e9,16e9,24e9].forEach(function(cap,i){var li=p.querySelectorAll('.devices li')[i],word=total<=cap*.75?'fits':total<=cap?'tight':'too big';li.querySelector('.fill').style.width=Math.min(100,100*total/cap)+'%';var v=li.querySelector('.verdict');v.textContent=word;v.className='verdict v-'+word.replace(' ','-')});}
  p.querySelectorAll('.seg').forEach(function(g){g.addEventListener('click',function(ev){var b=ev.target.closest('button');if(!b)return;g.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});draw()})});
  p.querySelectorAll('input').forEach(function(x){x.addEventListener('input',draw)});draw();
})();
(function(){var t=document.getElementById('onlydiff');if(!t)return;t.addEventListener('change',function(){
  document.querySelectorAll('tr[data-diff="0"]').forEach(function(r){r.hidden=t.checked})});})();
(function(){var f=document.getElementById('filters');if(!f)return;
  var q=document.getElementById('q'),sort=document.getElementById('sort'),list=document.querySelector('.cards'),
      cards=[].slice.call(list.children),rows=[].slice.call(document.querySelectorAll('.tableview tbody tr')),
      shown=document.querySelector('.shown'),empty=document.querySelector('.empty'),now=new Date();
  function vals(n){return [].slice.call(f.querySelectorAll('input[name='+n+']:checked')).map(function(x){return x.value})}
  function ok(c){var d=c.dataset,des=vals('design'),lay=vals('layers'),siz=vals('size'),rel=f.querySelector('input[name=released]:checked').value;
    var dm=(des.indexOf(d.design==='moe'?'moe':'dense')>=0)||(d.hybrid==='1'&&des.indexOf('hybrid')>=0);
    var lm=!lay.length||lay.some(function(l){return d.layers.split(' ').indexOf(l)>=0});
    var p=+d.params,sm=!siz.length||siz.some(function(s){return s==='s'?p<3e9:s==='m'?p>=3e9&&p<30e9:p>=30e9});
    var age=(now-new Date(d.date))/864e5,rm=rel==='any'||(rel==='6m'?age<=183:d.date.slice(0,4)==String(now.getFullYear()));
    var publisher=f.querySelector('[name=publisher]').value,pm=!publisher||publisher===d.publisher;var t=q.value.trim().toLowerCase();return pm&&dm&&lm&&sm&&rm&&(!t||d.search.indexOf(t)>=0)}
  function apply(){var key=sort.value,n=0;cards.sort(function(a,b){var x=a.dataset,y=b.dataset;
      return (key==='big'?y.params-x.params:key==='small'?x.params-y.params:key==='name'?x.name.localeCompare(y.name):y.date.localeCompare(x.date))||x.name.localeCompare(y.name)});
    cards.forEach(function(c){list.appendChild(c);var m=ok(c);c.hidden=!m;if(m)n++});
    var body=document.querySelector('.tableview tbody'),byHref={};
    rows.forEach(function(r){byHref[r.querySelector('a').getAttribute('href')]=r});
    cards.forEach(function(c){var r=byHref[c.querySelector('h3 a').getAttribute('href')];if(r){r.hidden=c.hidden;body.appendChild(r)}});
    shown.textContent=n;empty.hidden=n>0}
  f.addEventListener('change',apply);f.addEventListener('reset',function(){setTimeout(apply)});q.addEventListener('input',apply);sort.addEventListener('change',apply);
  document.querySelectorAll('[data-view]').forEach(function(b){b.addEventListener('click',function(){var t=b.dataset.view==='table';
    document.querySelectorAll('[data-view]').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});
    list.hidden=t;document.querySelector('.tableview').hidden=!t;})});
})();

(function(){var btn=document.querySelector('.menubtn'),nav=document.getElementById('main-nav');if(!btn||!nav)return;function close(){nav.classList.remove('open');btn.setAttribute('aria-expanded','false')}btn.addEventListener('click',function(){var open=btn.getAttribute('aria-expanded')!=='true';nav.classList.toggle('open',open);btn.setAttribute('aria-expanded',String(open))});document.addEventListener('keydown',function(ev){if(ev.key==='Escape'){var was=nav.classList.contains('open');close();if(was)btn.focus()}});document.addEventListener('click',function(ev){if(!nav.contains(ev.target)&&!btn.contains(ev.target))close()});})();
(function(){var panel=document.getElementById('filter-panel');if(!panel)return;var media=matchMedia('(max-width:860px)');function adapt(){panel.open=!media.matches}adapt();media.addEventListener('change',adapt);})();
(function(){var tray=document.getElementById('compare-tray');if(!tray)return;var chosen=[],buttons=[].slice.call(document.querySelectorAll('.compare-pick')),status=document.getElementById('compare-selection'),link=document.getElementById('compare-selection-link');
function draw(){buttons.forEach(function(b){var on=chosen.indexOf(b.dataset.stem)>=0;b.setAttribute('aria-pressed',String(on));b.textContent=on?'✓ Selected':'+ Compare';b.setAttribute('aria-label',(on?'Remove ':'Add ')+b.dataset.name+(on?' from comparison':' to comparison'))});tray.hidden=!chosen.length;status.textContent=chosen.map(function(stem){return buttons.find(function(b){return b.dataset.stem===stem}).dataset.name}).join(' · ');link.setAttribute('aria-disabled',String(chosen.length<2));link.textContent=chosen.length<2?'Select at least 2':'Compare '+chosen.length+' models →';link.href='compare/index.html?'+chosen.map(function(stem,i){return ['a','b','c'][i]+'='+encodeURIComponent(stem)}).join('&')}
buttons.forEach(function(b){b.addEventListener('click',function(){var i=chosen.indexOf(b.dataset.stem);if(i>=0)chosen.splice(i,1);else if(chosen.length<3)chosen.push(b.dataset.stem);else{status.textContent='Choose up to 3 models. Remove one to add another.';return}draw()})});document.getElementById('clear-selection').addEventListener('click',function(){chosen=[];draw()});link.addEventListener('click',function(ev){if(chosen.length<2)ev.preventDefault()});})();
(function(){var expand=document.getElementById('expand-diagram'),dialog=document.getElementById('diagram-dialog');if(!expand||!dialog)return;var figure=document.querySelector('#diagram figure'),download=document.getElementById('download-diagram');expand.addEventListener('click',function(){dialog.querySelector('.dialog-art').replaceChildren(figure.firstElementChild.cloneNode(true));dialog.showModal()});document.getElementById('close-diagram').addEventListener('click',function(){dialog.close()});dialog.addEventListener('click',function(ev){if(ev.target===dialog){var r=dialog.getBoundingClientRect();if(ev.clientX<r.left||ev.clientX>r.right||ev.clientY<r.top||ev.clientY>r.bottom)dialog.close()}});download.addEventListener('click',async function(){var art=figure.firstElementChild;try{var text=art.tagName.toLowerCase()==='img'?await fetch(art.src).then(function(r){if(!r.ok)throw Error();return r.text()}):new XMLSerializer().serializeToString(art);var url=URL.createObjectURL(new Blob([text],{type:'image/svg+xml'})),a=document.createElement('a');a.href=url;a.download=download.dataset.filename;a.click();setTimeout(function(){URL.revokeObjectURL(url)},1000)}catch(err){download.textContent='Download failed. Try again.'}});})();
