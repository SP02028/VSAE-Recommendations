// ════════════════════════════════════════════════════════════════
// UTILITIES
// ════════════════════════════════════════════════════════════════
var PITCH_CLASS={C:0,'C#':1,Db:1,D:2,'D#':3,Eb:3,E:4,F:5,'F#':6,Gb:6,G:7,'G#':8,Ab:8,A:9,'A#':10,Bb:10,B:11};

function noteToMidi(s){
  if(typeof s!=='string') return null;
  s=s.trim().split(/[(),]/)[0].trim();
  if(!s||s.toUpperCase()==='N/A') return null;
  var m=s.match(/([A-Ga-g][b#]?)(\d)/);
  if(!m) return null;
  var p=m[1].length>1?m[1][0].toUpperCase()+m[1][1].toLowerCase():m[1].toUpperCase();
  if(!(p in PITCH_CLASS)) return null;
  return(parseInt(m[2],10)+1)*12+PITCH_CLASS[p];
}

function parseNoteInput(s){
  if(!s||typeof s!=='string'||!s.trim()) return null;
  return noteToMidi(s.trim());
}

function runtimeToSeconds(s){
  if(typeof s!=='string') return null;
  s=s.trim();
  if(!s||s.toUpperCase()==='N/A') return null;
  var m=s.match(/^(\d+):(\d{2}(?:\.\d+)?)$/);
  if(!m) return null;
  return Math.round(parseInt(m[1],10)*60+parseFloat(m[2]));
}

function median(arr){
  if(!arr.length) return null;
  var s=arr.slice().sort(function(a,b){return a-b;});
  var mid=Math.floor(s.length/2);
  return s.length%2?s[mid]:(s[mid-1]+s[mid])/2;
}

function esc(v){
  return String(v==null?'':v)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function uniq(arr){
  var s={},o=[];
  arr.forEach(function(v){var t=String(v||'').trim();if(t&&!s[t]){s[t]=1;o.push(t);}});
  return o.sort();
}

// ════════════════════════════════════════════════════════════════
// CSV PARSING & FEATURE ENGINEERING
// ════════════════════════════════════════════════════════════════
function parseCSV(text){
  var rows=[],row=[],field='',inQ=false;
  for(var i=0;i<text.length;i++){
    var ch=text[i];
    if(ch==='"'){if(inQ&&text[i+1]==='"'){field+='"';i++;}else{inQ=!inQ;}}
    else if(ch===','&&!inQ){row.push(field);field='';}
    else if((ch==='\n'||ch==='\r')&&!inQ){
      if(ch==='\r'&&text[i+1]==='\n') i++;
      row.push(field);rows.push(row);row=[];field='';
    } else field+=ch;
  }
  if(field||row.length){row.push(field);rows.push(row);}
  return rows;
}

function loadVSAEData(text){
  var rows=parseCSV(text||'');
  if(!rows.length) return [];
  var headers=rows[0].map(function(h,i){
    var t=String(h||'').trim();
    return i===0?t.replace(/^\uFEFF/,''):t;
  });
  var tIdx=headers.findIndex(function(h){return h.toLowerCase()==='title';});
  var songs=[];
  for(var r=1;r<rows.length;r++){
    var vals=(rows[r]||[]).map(function(v){return String(v||'').trim();});
    if(!vals.length) continue;
    var title=tIdx>=0?vals[tIdx]||'':'';
    if(!title) continue;
    var last=(vals[vals.length-1]||'').toLowerCase();
    if(last.includes('should probably be removed')) continue;
    var obj={};
    headers.forEach(function(h,i){obj[h]=vals[i]||'';});
    songs.push(obj);
  }
  return songs;
}

var ERA_KEYS=['Renaissance','Baroque','Classical','Romantic','Modern'];
var VR_DEFAULTS={
  Soprano:{high:79,low:65},'Mezzo Soprano':{high:77,low:62},
  Alto:{high:77,low:60},Tenor:{high:76,low:60},
  Baritone:{high:64,low:48},Bass:{high:62,low:43},'Vocal All':{high:77,low:60}
};

function inferEra(tp){
  if(!tp||typeof tp!=='string') return 'Unknown';
  var low=tp.toLowerCase();
  for(var i=0;i<ERA_KEYS.length;i++) if(low.includes(ERA_KEYS[i].toLowerCase())) return ERA_KEYS[i];
  var m=tp.match(/(\d{3,4})/);
  if(!m) return 'Unknown';
  var yr=parseInt(m[1],10);
  if(yr<1600) return 'Renaissance';
  if(yr<1750) return 'Baroque';
  if(yr<1820) return 'Classical';
  if(yr<1910) return 'Romantic';
  return 'Modern';
}

function engineerFeatures(songs){
  var rts=[];
  songs.forEach(function(s){
    s.Era=inferEra(s['Time Period']);
    var d=VR_DEFAULTS[s.VocalRange]||null;
    var hi=noteToMidi(s['Highest Note']); if(hi==null&&d) hi=d.high; s.HighestNote_MIDI=hi;
    var lo=noteToMidi(s['Lowest Note']);  if(lo==null&&d) lo=d.low;  s.LowestNote_MIDI=lo;
    s.RangeSpan=(hi!=null&&lo!=null)?Math.max(0,hi-lo):0;
    var rt=runtimeToSeconds(s['Runtime of Song']); s.RuntimeSeconds=rt; if(rt!=null) rts.push(rt);
    var ck={A:1,B:2,C:3}; s.ClassOrdinal=ck[String(s.Class||'').trim().toUpperCase()]||2;
    var ek={Renaissance:1,Baroque:2,Classical:3,Romantic:4,Modern:5,Unknown:3};
    s.EraOrdinal=ek[s.Era]||3;
    s._pprScore=null;
  });
  var med=median(rts);
  if(med!=null) songs.forEach(function(s){if(s.RuntimeSeconds==null) s.RuntimeSeconds=Math.round(med);});
  return songs;
}

// ════════════════════════════════════════════════════════════════
// MISSING DATA CHECK
// ════════════════════════════════════════════════════════════════
function isMissingData(s){
  var hi=String(s['Highest Note']||'').trim().toUpperCase();
  var lo=String(s['Lowest Note']||'').trim().toUpperCase();
  var rt=String(s['Runtime of Song']||'').trim().toUpperCase();
  var missingNote=hi==='N/A'||hi===''||lo==='N/A'||lo==='';
  var missingRuntime=rt==='N/A'||rt==='';
  return missingNote||missingRuntime;
}

// ════════════════════════════════════════════════════════════════
// FEATURE MATRIX
// ════════════════════════════════════════════════════════════════
function normalizeMinMax(vals){
  var mn=Infinity,mx=-Infinity;
  vals.forEach(function(v){var n=Number(v);if(isFinite(n)){if(n<mn)mn=n;if(n>mx)mx=n;}});
  if(!isFinite(mn)||mn===mx) return vals.map(function(){return 0;});
  var span=mx-mn;
  return vals.map(function(v){var n=Number(v);return isFinite(n)?(n-mn)/span:0;});
}

function buildFeatureMatrix(songs,features){
  if(!features||!features.length) throw new Error('No features selected');
  var matrix=songs.map(function(){return[];});
  function has(f){return features.indexOf(f)>=0;}

  if(has('VocalRange')){
    var vrCats=uniq(songs.map(function(s){return s.VocalRange||'Vocal All';}));
    songs.forEach(function(s,i){vrCats.forEach(function(c){matrix[i].push(((s.VocalRange||'Vocal All')===c?1:0)*3);});});
  }
  if(has('Class')){
    songs.forEach(function(s,i){matrix[i].push((s.ClassOrdinal/3)*2.5);});
  }
  if(has('Language')){
    var langCats=uniq(songs.map(function(s){return s.Language||'English';}));
    songs.forEach(function(s,i){langCats.forEach(function(c){matrix[i].push(((s.Language||'English')===c?1:0)*1.5);});});
  }
  if(has('Genre')){
    var genreCats=uniq(songs.map(function(s){return(s.Genre||'unknown').toLowerCase();}));
    songs.forEach(function(s,i){var g=(s.Genre||'unknown').toLowerCase();genreCats.forEach(function(c){matrix[i].push((g===c?1:0)*1.5);});});
  }
  if(has('Era')){
    songs.forEach(function(s,i){matrix[i].push((s.EraOrdinal/5)*1);});
  }
  if(has('RangeSpan')){
    var rsNorm=normalizeMinMax(songs.map(function(s){return s.RangeSpan||0;}));
    songs.forEach(function(s,i){matrix[i].push(rsNorm[i]);});
  }
  if(has('Runtime')){
    var rtNorm=normalizeMinMax(songs.map(function(s){return s.RuntimeSeconds||0;}));
    songs.forEach(function(s,i){matrix[i].push(rtNorm[i]*0.5);});
  }
  return matrix;
}

// ════════════════════════════════════════════════════════════════
// PERSONALIZED PAGERANK
// ════════════════════════════════════════════════════════════════
function cosine(a,b){
  var dot=0,na=0,nb=0,len=Math.min(a.length,b.length);
  for(var i=0;i<len;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
  if(!na||!nb) return 0;
  return Math.min(1,Math.max(0,dot/(Math.sqrt(na)*Math.sqrt(nb))));
}

function computePPR(matrix,queryIdx){
  var n=matrix.length;
  var sim=[];
  for(var i=0;i<n;i++){sim.push(new Float64Array(n));}
  for(var i=0;i<n;i++) for(var j=0;j<n;j++) sim[i][j]=i===j?0:cosine(matrix[i],matrix[j]);
  var k=Math.min(12,Math.max(3,Math.floor(n/10)));
  var sparse=[];
  for(var i=0;i<n;i++){sparse.push(new Float64Array(n));}
  for(var i=0;i<n;i++){
    var row=Array.from(sim[i]).map(function(v,j){return{j:j,v:v};}).sort(function(a,b){return b.v-a.v;});
    for(var t=0;t<Math.min(k,row.length);t++) sparse[i][row[t].j]=Math.max(0,row[t].v);
  }
  var T=[];
  for(var i=0;i<n;i++){
    T.push(new Float64Array(n));
    var sum=0; for(var j=0;j<n;j++) sum+=sparse[i][j];
    if(sum>0) for(var j=0;j<n;j++) T[i][j]=sparse[i][j]/sum;
  }
  var alpha=0.85;
  var teleport=new Float64Array(n); teleport[queryIdx]=1;
  var rank=new Float64Array(teleport);
  for(var iter=0;iter<100;iter++){
    var next=new Float64Array(n);
    for(var j=0;j<n;j++){var s=0;for(var i=0;i<n;i++) s+=T[i][j]*rank[i];next[j]=alpha*s+(1-alpha)*teleport[j];}
    var l1=0;for(var i=0;i<n;i++) l1+=Math.abs(next[i]-rank[i]);
    rank=next;
    if(l1<1e-10) break;
  }
  return rank;
}

function baseTitle(t){
  return(t||'').replace(/\s*\((?:High|Low|Medium|Vocal All|Bass|Baritone|Soprano|Alto|Tenor|Mezzo Soprano)[^)]*\)\s*$/i,'').trim();
}

// ════════════════════════════════════════════════════════════════
// FILTERING
// ════════════════════════════════════════════════════════════════
var RANGE_BOUNDS={
  Soprano:{low:65,high:79},'Mezzo Soprano':{low:62,high:77},
  Alto:{low:60,high:77},Tenor:{low:60,high:76},
  Baritone:{low:48,high:64},Bass:{low:43,high:62},'Vocal All':{low:60,high:77}
};
var MALE_VR=['tenor','baritone','bass'];

function shouldTransposeForMen(selectedRanges){
  return selectedRanges.length===1&&MALE_VR.indexOf(selectedRanges[0].toLowerCase())>=0;
}

function applyVocalRangeFilter(songs,selectedRanges){
  if(!selectedRanges.length) return songs;
  var set={};selectedRanges.forEach(function(v){set[v]=1;});
  var filtered=songs.filter(function(s){return set[s.VocalRange];});
  if(selectedRanges.length===1){
    var bounds=RANGE_BOUNDS[selectedRanges[0]];
    if(bounds){
      var isMale=shouldTransposeForMen(selectedRanges);
      filtered=filtered.filter(function(s){
        var hi=s.HighestNote_MIDI,lo=s.LowestNote_MIDI;
        if(hi==null||lo==null) return false;
        if(isMale&&s.VocalRange==='Vocal All'){lo-=12;hi-=12;}
        return lo<=bounds.high&&hi>=bounds.low;
      });
    }
  }
  return filtered;
}

// ════════════════════════════════════════════════════════════════
// RANGE MATCH SCORING
// ════════════════════════════════════════════════════════════════
function scoreRangeMatch(songs,userLow,userHigh,isMale){
  var lo=Math.min(userLow,userHigh),hi=Math.max(userLow,userHigh);
  var span=Math.max(1,hi-lo);
  songs.forEach(function(s){
    var sLo=s.LowestNote_MIDI,sHi=s.HighestNote_MIDI;
    if(sLo==null||sHi==null){s._rangeScore=0;return;}
    if(isMale&&s.VocalRange==='Vocal All'){sLo-=12;sHi-=12;}
    var overlap=Math.max(0,Math.min(sHi,hi)-Math.max(sLo,lo));
    var score=(overlap/span)*100;
    if(sLo>=lo&&sHi<=hi) score=100;
    s._rangeScore=Math.round(Math.max(0,Math.min(100,score)));
  });
}

// ════════════════════════════════════════════════════════════════
// NOTE RANGE LABEL
// ════════════════════════════════════════════════════════════════
function transposeNoteLabel(noteStr,shift){
  if(!noteStr||shift===0) return String(noteStr||'').trim();
  var m=noteStr.trim().match(/^([A-Ga-g][b#]?)(\d+)$/);
  if(!m) return noteStr.trim();
  var pitch=m[1].length>1?m[1][0].toUpperCase()+m[1][1].toLowerCase():m[1].toUpperCase();
  return pitch+(parseInt(m[2],10)+shift);
}

function noteRangeLabel(song,transposeVocalAll){
  var lo=String(song['Lowest Note']||'').trim();
  var hi=String(song['Highest Note']||'').trim();
  if(!lo||!hi) return 'Missing';
  var bad=['N/A','NAN',''];
  if(bad.indexOf(lo.toUpperCase())>=0||bad.indexOf(hi.toUpperCase())>=0) return 'Missing';
  var isVA=String(song.VocalRange||'').trim()==='Vocal All';
  var shift=transposeVocalAll&&isVA?-1:0;
  return transposeNoteLabel(lo,shift)+' - '+transposeNoteLabel(hi,shift);
}

// ════════════════════════════════════════════════════════════════
// TABLE RENDERING
// ════════════════════════════════════════════════════════════════
var DISPLAY_COLS=['Title','Composer','VocalRange','Class','Language','Genre','Era','Note Range','Runtime'];

function renderTable(songs,containerId,transposeForMen,topK){
  var el=document.getElementById(containerId);
  if(!el) return;
  if(!songs||!songs.length){el.innerHTML='<div class="empty-row">No songs match the current filters.</div>';return;}
  var rows=songs.slice(0,topK||songs.length);
  var html='<table><thead><tr><th>#</th>';
  DISPLAY_COLS.forEach(function(c){html+='<th>'+esc(c)+'</th>';});
  html+='</tr></thead><tbody>';
  rows.forEach(function(s,i){
    var rt=String(s['Runtime of Song']||'').trim();
    if(!rt||rt.toUpperCase()==='N/A'||rt.toUpperCase()==='NAN') rt='Missing';
    var noteRange=noteRangeLabel(s,transposeForMen);
    html+='<tr><td style="color:var(--muted);">'+(i+1)+'</td>';
    ['Title','Composer','VocalRange','Class','Language','Genre','Era'].forEach(function(c){
      html+='<td>'+esc(s[c]||'')+'</td>';
    });
    html+='<td>'+esc(noteRange)+'</td><td>'+esc(rt)+'</td></tr>';
  });
  html+='</tbody></table>';
  el.innerHTML=html;
}

// ════════════════════════════════════════════════════════════════
// CHIP HELPERS
// ════════════════════════════════════════════════════════════════
function buildChips(containerId,values){
  var el=document.getElementById(containerId); if(!el) return;
  el.innerHTML='';
  values.forEach(function(v){
    var chip=document.createElement('span');
    chip.className='chip';chip.textContent=v;chip.dataset.value=v;
    chip.addEventListener('click',function(){
      chip.classList.toggle('off');
      runPipeline();
    });
    el.appendChild(chip);
  });
}

function getActiveChips(containerId){
  var el=document.getElementById(containerId); if(!el) return [];
  var out=[];
  el.querySelectorAll('.chip:not(.off)').forEach(function(c){out.push(c.dataset.value);});
  return out;
}

// ════════════════════════════════════════════════════════════════
// DATASET OVERVIEW CHARTS
// ════════════════════════════════════════════════════════════════
var _charts={};

function countBy(songs,field){
  var counts={};
  songs.forEach(function(s){var v=String(s[field]||'Unknown').trim();counts[v]=(counts[v]||0)+1;});
  return counts;
}

function makeChart(canvasId,counts,color){
  var entries=Object.entries(counts).sort(function(a,b){return b[1]-a[1];});
  var labels=entries.map(function(e){return e[0];});
  var data=entries.map(function(e){return e[1];});

  if(_charts[canvasId]) _charts[canvasId].destroy();
  var ctx=document.getElementById(canvasId);
  if(!ctx) return;
  _charts[canvasId]=new Chart(ctx,{
    type:'bar',
    data:{
      labels:labels,
      datasets:[{data:data,backgroundColor:color||'rgba(255,106,95,.7)',borderColor:'rgba(255,106,95,1)',borderWidth:1,borderRadius:4}]
    },
    options:{
      responsive:true,maintainAspectRatio:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:'#bac3d8',font:{size:11}},grid:{color:'rgba(152,168,198,.1)'}},
        y:{ticks:{color:'#bac3d8',font:{size:11},stepSize:1},grid:{color:'rgba(152,168,198,.1)'}}
      }
    }
  });
}

function renderOverview(songs){
  document.getElementById('m-total').textContent=songs.length;
  document.getElementById('m-ranges').textContent=uniq(songs.map(function(s){return s.VocalRange;})).length;
  document.getElementById('m-langs').textContent=uniq(songs.map(function(s){return s.Language;})).length;
  document.getElementById('m-composers').textContent=uniq(songs.map(function(s){return s.Composer;})).length;

  makeChart('chart-vr',countBy(songs,'VocalRange'));
  makeChart('chart-class',countBy(songs,'Class'),'rgba(100,180,255,.7)');
  makeChart('chart-lang',countBy(songs,'Language'),'rgba(100,220,150,.7)');
  makeChart('chart-era',countBy(songs,'Era'),'rgba(220,160,255,.7)');

  var cols=['Title','Composer','VocalRange','Class','Language','Genre','Era','RangeSpan','RuntimeSeconds'];
  var html='<table><thead><tr><th>#</th>'+cols.map(function(c){return'<th>'+esc(c)+'</th>';}).join('')+'</tr></thead><tbody>';
  songs.forEach(function(s,i){
    html+='<tr><td style="color:var(--muted);">'+(i+1)+'</td>'+
      cols.map(function(c){return'<td>'+esc(s[c]!=null?s[c]:'')+'</td>';}).join('')+'</tr>';
  });
  document.getElementById('tbl-full').innerHTML=html+'</tbody></table>';
}

// ════════════════════════════════════════════════════════════════
// PIPELINE CACHE
// ════════════════════════════════════════════════════════════════
var _pprKey=null;

function getPprScores(querySongIdx,features,excludeTranspositions){
  var key=querySongIdx+'|'+features.join(',')+'|'+excludeTranspositions;
  if(key===_pprKey) return;

  _pprKey=key;
  var matrix=buildFeatureMatrix(allSongs,features);
  var scores=computePPR(matrix,querySongIdx);

  allSongs.forEach(function(s){s._pprScore=null;});

  var querySong=allSongs[querySongIdx];
  var queryBase=baseTitle(querySong.Title).toLowerCase();

  allSongs.forEach(function(s,i){
    if(i===querySongIdx){s._pprScore=null;return;}
    if(excludeTranspositions&&queryBase&&(s.Title||'').toLowerCase().startsWith(queryBase)){
      s._pprScore=null;return;
    }
    s._pprScore=scores[i];
  });
}

// ════════════════════════════════════════════════════════════════
// MAIN PIPELINE
// ════════════════════════════════════════════════════════════════
var allSongs=null;

function runPipeline(){
  if(!allSongs) return;

  var topK=parseInt(document.getElementById('slider-topk').value,10)||10;

  var selVR   =getActiveChips('chips-vr');
  var selClass=getActiveChips('chips-class');
  var selLang =getActiveChips('chips-lang');
  var transposeForMen=shouldTransposeForMen(selVR);

  var filtered=allSongs.slice();

  filtered=applyVocalRangeFilter(filtered,selVR);

  if(selClass.length){var cs={};selClass.forEach(function(v){cs[v]=1;});filtered=filtered.filter(function(s){return cs[s.Class];});}

  if(selLang.length){var ls={};selLang.forEach(function(v){ls[v.toLowerCase()]=1;});filtered=filtered.filter(function(s){return ls[(s.Language||'').toLowerCase()];});}

  var simEnabled=false;
  var chkSim=document.getElementById('chk-similarity');
  if(chkSim&&chkSim.checked){
    var refVal=document.getElementById('sel-reference').value;
    var simMin=parseFloat(document.getElementById('slider-sim-min').value)||0;
    var feats=[];
    var featMap=[['feat-vocalrange','VocalRange'],['feat-class','Class'],['feat-language','Language'],
      ['feat-genre','Genre'],['feat-era','Era'],['feat-rangespan','RangeSpan'],['feat-runtime','Runtime']];
    featMap.forEach(function(p){var el=document.getElementById(p[0]);if(el&&el.checked) feats.push(p[1]);});
    var excl=document.getElementById('chk-exclude-transpositions').checked;

    if(refVal!==''&&feats.length){
      var qIdx=parseInt(refVal,10);
      if(isFinite(qIdx)&&qIdx>=0&&qIdx<allSongs.length){
        try{
          getPprScores(qIdx,feats,excl);
          filtered=filtered.filter(function(s){return s._pprScore!==null&&s._pprScore>=simMin;});
          if(filtered.length) simEnabled=true;
        }catch(e){console.warn('PPR error:',e);}
      }
    }
  } else {
    allSongs.forEach(function(s){s._pprScore=null;});
    _pprKey=null;
  }

  var basedFiltered=filtered.slice();

  var displaySongs=basedFiltered.filter(function(s){return!isMissingData(s);});
  var missingSongs=basedFiltered.filter(function(s){return isMissingData(s);});

  var chkRange=document.getElementById('chk-range-match');
  var rangeEnabled=chkRange&&chkRange.checked;
  var lowMidi=null,highMidi=null;
  var hintLow=document.getElementById('hint-low');
  var hintHigh=document.getElementById('hint-high');
  var hintBoth=document.getElementById('hint-both');

  if(rangeEnabled){
    var lowStr=(document.getElementById('input-low').value||'').trim();
    var highStr=(document.getElementById('input-high').value||'').trim();
    lowMidi=parseNoteInput(lowStr);
    highMidi=parseNoteInput(highStr);

    if(hintLow) hintLow.innerHTML=lowStr?(lowMidi!=null?'<span class="parse-ok">MIDI '+lowMidi+'</span>':'<span class="parse-err">Could not parse \''+esc(lowStr)+'\'.</span>'):'';
    if(hintHigh) hintHigh.innerHTML=highStr?(highMidi!=null?'<span class="parse-ok">MIDI '+highMidi+'</span>':'<span class="parse-err">Could not parse \''+esc(highStr)+'\'.</span>'):'';
    if(hintBoth){
      if(lowMidi!=null&&highMidi!=null) hintBoth.textContent='';
      else hintBoth.textContent='Enter both notes to score range matches.';
    }

    if(lowMidi!=null&&highMidi!=null){
      scoreRangeMatch(displaySongs,lowMidi,highMidi,transposeForMen);
      displaySongs.sort(function(a,b){
        var rd=(b._rangeScore||0)-(a._rangeScore||0);
        if(rd!==0) return rd;
        return(b._pprScore||0)-(a._pprScore||0);
      });
    }
  } else {
    if(hintLow) hintLow.textContent='';
    if(hintHigh) hintHigh.textContent='';
    if(hintBoth) hintBoth.textContent='';
    if(simEnabled){
      displaySongs.sort(function(a,b){return(b._pprScore||0)-(a._pprScore||0);});
    }
  }

  if(simEnabled){
    missingSongs.sort(function(a,b){return(b._pprScore||0)-(a._pprScore||0);});
  }

  renderTable(displaySongs,'tbl-main',transposeForMen,topK);
  renderTable(missingSongs,'tbl-missing',transposeForMen,topK);

  // ── NEW: Populate song-selection dropdown ────────────────────
  var selChosen=document.getElementById('sel-chosen-song');
  if(selChosen){
    var prev=selChosen.value;
    selChosen.innerHTML='';

    var placeholder=document.createElement('option');
    placeholder.value='';
    placeholder.textContent='\u2014 Select a song \u2014';
    selChosen.appendChild(placeholder);

    var recommended=displaySongs.slice(0,topK).concat(missingSongs.slice(0,topK));
    for(var ri=0;ri<recommended.length;ri++){
      var song=recommended[ri];
      var opt=document.createElement('option');
      opt.value=String(song.Song_Code||'');
      opt.textContent=song.Title+' | '+song.VocalRange+' | Class '+(song.Class||'')+' | '+(song.Language||'');
      selChosen.appendChild(opt);
    }

    if(prev){
      var stillExists=false;
      for(var oi=0;oi<selChosen.options.length;oi++){
        if(selChosen.options[oi].value===prev){selChosen.value=prev;stillExists=true;break;}
      }
      if(!stillExists) selChosen.value='';
    }

    var submitBtn=document.getElementById('btn-submit');
    if(submitBtn) submitBtn.disabled=!selChosen.value;
  }
}

// ════════════════════════════════════════════════════════════════
// WIRE EVENTS
// ════════════════════════════════════════════════════════════════
function wire(){
  document.querySelectorAll('.tab-btn').forEach(function(btn){
    btn.addEventListener('click',function(){
      document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
      document.querySelectorAll('.tab-pane').forEach(function(p){p.classList.remove('active');});
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });

  var chkSim=document.getElementById('chk-similarity');
  var simBlock=document.getElementById('sim-block');
  chkSim.addEventListener('change',function(){
    simBlock.classList.toggle('open',chkSim.checked);
    if(!chkSim.checked){allSongs.forEach(function(s){s._pprScore=null;});_pprKey=null;}
    runPipeline();
  });

  var chkRange=document.getElementById('chk-range-match');
  var rangeBlock=document.getElementById('range-block');
  chkRange.addEventListener('change',function(){
    rangeBlock.classList.toggle('open',chkRange.checked);
    runPipeline();
  });

  var sliderSim=document.getElementById('slider-sim-min');
  sliderSim.addEventListener('input',function(){
    document.getElementById('sim-min-label').textContent=parseFloat(sliderSim.value).toFixed(2);
    runPipeline();
  });

  var sliderK=document.getElementById('slider-topk');
  sliderK.addEventListener('input',function(){
    document.getElementById('topk-label').textContent=sliderK.value;
    runPipeline();
  });

  ['sel-reference','feat-vocalrange','feat-class','feat-language','feat-genre',
   'feat-era','feat-rangespan','feat-runtime','chk-exclude-transpositions'].forEach(function(id){
    var el=document.getElementById(id); if(!el) return;
    el.addEventListener('change',function(){_pprKey=null;runPipeline();});
  });

  ['input-low','input-high'].forEach(function(id){
    var el=document.getElementById(id); if(!el) return;
    el.addEventListener('input',runPipeline);
  });

  // ── NEW: Song selection dropdown + submit button ─────────────
  var selChosen=document.getElementById('sel-chosen-song');
  var btnSubmit=document.getElementById('btn-submit');

  if(selChosen){
    selChosen.addEventListener('change',function(){
      if(btnSubmit) btnSubmit.disabled=!selChosen.value;
    });
  }

  if(btnSubmit){
    btnSubmit.addEventListener('click',function(){
      var songCode=(selChosen&&selChosen.value)||'';
      if(!songCode){
        alert('Please select a song from the dropdown first.');
        return;
      }
      var params=new URLSearchParams(window.location.search||'');
      var studentId=params.get('StudentID')||'';
      window.location.href='https://aveschoir.org/Vocal-Solo-Event?StudentID='+encodeURIComponent(studentId)+'&song-code='+encodeURIComponent(songCode);
    });
  }
}

// ════════════════════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded',function(){
  wire();

  fetch('VSAE_Data_Final.csv')
    .then(function(r){
      if(!r.ok) throw new Error('CSV returned status '+r.status+'. Make sure VSAE_Data_Final.csv is in the same folder as index.html and GitHub Pages is enabled.');
      return r.text();
    })
    .then(function(text){
      var raw=loadVSAEData(text);
      if(!raw.length) throw new Error('CSV parsed to 0 songs \u2014 check the file format.');
      allSongs=engineerFeatures(raw);

      document.getElementById('loading').style.display='none';
      document.getElementById('app-shell').style.display='block';

      buildChips('chips-vr',uniq(allSongs.map(function(s){return s.VocalRange;})));
      buildChips('chips-class',uniq(allSongs.map(function(s){return s.Class;})));
      buildChips('chips-lang',['English','French','German','Spanish','Latin','Italian']);

      var ref=document.getElementById('sel-reference');
      ref.innerHTML='';
      var labeled=allSongs.map(function(s,i){
        return{i:i,label:[s.Title,s.VocalRange,'Class '+s.Class,s.Language].join(' | ')};
      }).sort(function(a,b){return a.label.localeCompare(b.label);});
      labeled.forEach(function(item){
        var o=document.createElement('option');
        o.value=String(item.i);o.textContent=item.label;
        ref.appendChild(o);
      });

      renderOverview(allSongs);
      runPipeline();
    })
    .catch(function(err){
      document.getElementById('loading').style.display='none';
      var box=document.getElementById('error-msg');
      box.style.display='block';
      box.textContent=err.message+'\n\nTo deploy on GitHub Pages:\n1. Commit index.html and VSAE_Data_Final.csv to the same folder in your repo\n2. Go to repo Settings \u2192 Pages \u2192 Branch: main \u2192 Save\n3. Visit https://sp02028.github.io/VSAE-Recommendations/';
    });
});