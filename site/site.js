(function(){var root=document.documentElement,btn=document.querySelector('.themebtn');
  function cur(){return root.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}
  if(btn){btn.setAttribute('aria-pressed',cur()==='dark');btn.addEventListener('click',function(){
    var t=cur()==='dark'?'light':'dark';root.dataset.theme=t;btn.setAttribute('aria-pressed',t==='dark');
    try{localStorage.setItem('aw-theme',t)}catch(e){}});}
})();
(function(){var p=document.getElementById('fit');if(!p)return;var combos=JSON.parse(p.dataset.combos),ctx,prec;
  function sel(attr){var b=p.querySelector('['+attr+'][aria-pressed=true]');return b.getAttribute(attr)}
  function draw(){var c=combos[sel('data-ctx')+'-'+sel('data-prec')];['total','weights','cache'].forEach(function(k){
      p.querySelector('[data-out='+k+']').textContent=c[k]});
    p.querySelectorAll('.devices li').forEach(function(li,i){var d=c.devices[i];li.querySelector('.fill').style.width=d[0]+'%';
      var v=li.querySelector('.verdict');v.textContent=d[1];v.className='verdict v-'+d[1].replace(' ','-')});}
  p.querySelectorAll('.seg').forEach(function(g){g.addEventListener('click',function(ev){var b=ev.target.closest('button');if(!b)return;
    g.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});draw();});});
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
    var t=q.value.trim().toLowerCase();return dm&&lm&&sm&&rm&&(!t||d.search.indexOf(t)>=0)}
  function apply(){var key=sort.value,n=0;cards.sort(function(a,b){var x=a.dataset,y=b.dataset;
      return key==='big'?y.params-x.params:key==='small'?x.params-y.params:key==='name'?x.name.localeCompare(y.name):y.date.localeCompare(x.date)});
    cards.forEach(function(c,i){list.appendChild(c);var m=ok(c);c.hidden=!m;if(rows[i]){}if(m)n++});
    rows.forEach(function(r){var name=r.querySelector('a').getAttribute('href');var c=cards.filter(function(c){return c.querySelector('h3 a').getAttribute('href')===name})[0];r.hidden=c.hidden});
    shown.textContent=n;empty.hidden=n>0}
  f.addEventListener('change',apply);f.addEventListener('reset',function(){setTimeout(apply)});q.addEventListener('input',apply);sort.addEventListener('change',apply);
  document.querySelectorAll('[data-view]').forEach(function(b){b.addEventListener('click',function(){var t=b.dataset.view==='table';
    document.querySelectorAll('[data-view]').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});
    list.hidden=t;document.querySelector('.tableview').hidden=!t;})});
})();
