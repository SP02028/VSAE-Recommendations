// app.js — lightweight loader for self_contained_app.js
(function(){
  try{
    var s=document.createElement('script');
    s.src='self_contained_app.js?v=2026-10-11.1'; // bump when the script changes so browsers don't use a cached copy
    s.defer=true;
    s.onload=function(){console.log('self_contained_app.js loaded');};
    s.onerror=function(){console.error('Failed to load self_contained_app.js');};
    document.head.appendChild(s);
  }catch(e){console.error(e);}  
})();
