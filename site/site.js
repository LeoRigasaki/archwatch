document.querySelectorAll('.panel .pills').forEach(function(group){
  var panel=group.closest('.panel'),out=panel.querySelector('.membytes'),tok=panel.querySelector('.memtokens');
  group.addEventListener('click',function(ev){var b=ev.target.closest('button');if(!b)return;
    group.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});
    out.textContent=b.dataset.bytes;tok.textContent=b.dataset.tokens;});
});
(function(){var box=document.querySelector('.controls');if(!box)return;box.hidden=false;
  var q=document.getElementById('q'),pills=box.querySelectorAll('.pill'),cards=document.querySelectorAll('.card'),
      empty=document.querySelector('.empty'),filter='All';
  function apply(){var t=q.value.trim().toLowerCase(),n=0;cards.forEach(function(c){
      var ok=(filter==='All'||c.dataset.tags.split('|').indexOf(filter)>=0)&&(!t||c.dataset.search.indexOf(t)>=0);
      c.hidden=!ok;if(ok)n++;});empty.hidden=n>0;}
  q.addEventListener('input',apply);
  pills.forEach(function(p){p.addEventListener('click',function(){filter=p.dataset.filter;
    pills.forEach(function(x){x.setAttribute('aria-pressed',x===p?'true':'false')});apply();});});
})();
