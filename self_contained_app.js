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

// Student input must be exactly a note name and one octave digit
// ("B3", "f#5", "Bb 2"). noteToMidi alone is lenient: it read "C10" as C1.
function parseNoteInput(s){
  if(!s||typeof s!=='string') return null;
  var t=s.replace(/\s+/g,'').replace('♯','#').replace('♭','b');
  if(!/^[A-Ga-g][#b]?\d$/.test(t)) return null;
  return noteToMidi(t);
}

var NOTE_NAMES=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
function midiToNote(m){return NOTE_NAMES[m%12]+(Math.floor(m/12)-1);}

// Limits for the singer's range. Songs in the list span D#2–A5 and a
// median of 15 semitones (90% need at least an octave).
var VOICE_MIN=33;   // A1
var VOICE_MAX=96;   // C7
var NARROW_SPAN=12; // under an octave: warn, few songs fit
var WIDE_SPAN=36;   // over three octaves: warn, likely an octave typo

// Returns {low, high, lowErr, highErr, error, warning}. low/high are only
// set when the pair is usable for range matching.
function validateVocalRange(lowStr,highStr){
  var r={low:null,high:null,lowErr:'',highErr:'',error:'',warning:''};
  var lo=parseNoteInput(lowStr),hi=parseNoteInput(highStr);
  function check(str,m){
    if(!str) return '';
    if(m==null) return 'Use a note name and octave, like B3 or F#5.';
    if(m<VOICE_MIN||m>VOICE_MAX) return midiToNote(m)+' is outside the range of the human voice (A1–C7). Check the octave number: middle C is C4.';
    return '';
  }
  r.lowErr=check(lowStr,lo);
  r.highErr=check(highStr,hi);
  if(r.lowErr||r.highErr||lo==null||hi==null) return r;
  if(lo>hi){r.error='Your lowest note ('+midiToNote(lo)+') is above your highest note ('+midiToNote(hi)+'). Swap them?';return r;}
  if(lo===hi){r.error='Your lowest and highest notes are the same. Enter the full range you can sing.';return r;}
  var span=hi-lo;
  if(span<NARROW_SPAN) r.warning='That range is under an octave ('+span+' semitones). Most songs need at least an octave, so few will fit.';
  else if(span>WIDE_SPAN) r.warning='That range is over three octaves. Double-check the octave numbers (middle C is C4).';
  r.low=lo;r.high=hi;
  return r;
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

// Language values in the sheet vary in case and sometimes carry a '?'
// (e.g. 'English?', 'italian'); compare on a cleaned form.
function normLang(v){
  return String(v||'').replace(/\?/g,'').trim().toLowerCase()||'english';
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
    s.Language=titleCase(normLang(s.Language));  // 'English?', 'italian' → 'English', 'Italian'
    s.Era=inferEra(s['Time Period']);
    // Missing notes stay null (no per-voice guess): those songs are listed
    // under "Also worth a look" and are never scored on range.
    var hi=noteToMidi(s['Highest Note']); s.HighestNote_MIDI=hi;
    var lo=noteToMidi(s['Lowest Note']);  s.LowestNote_MIDI=lo;
    s.RangeSpan=(hi!=null&&lo!=null)?Math.max(0,hi-lo):0;
    var rt=runtimeToSeconds(s['Runtime of Song']); s.RuntimeSeconds=rt; if(rt!=null) rts.push(rt);
    var ck={A:1,B:2,C:3}; s.ClassOrdinal=ck[String(s.Class||'').trim().toUpperCase()]||2;
    var ek={Renaissance:1,Baroque:2,Classical:3,Romantic:4,Modern:5,Unknown:3};
    s.EraOrdinal=ek[s.Era]||3;
    s._similarityScore=null;
    s._pprRaw=null;
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
// SONG–ATTRIBUTE GRAPH + PERSONALIZED PAGERANK
// ════════════════════════════════════════════════════════════════
// Nodes are songs plus one node per attribute value (composer:scarlatti,
// lang:italian, class:A, era:Baroque, voice:Soprano, ...). Each song links
// to its attribute values; neighbouring ordinal values link to each other
// (Baroque–Classical, Soprano–Mezzo) so close values earn partial
// credit through 2-hop paths.
//
// A walk from the reference song that passes through an attribute node
// splits its mass across every song sharing that value, so a rare shared
// value (a composer with 6 songs) counts far more than a common one
// (English, 248 songs) — an IDF-like weighting that falls out of the walk.
//
// Score = lift = PPR(song) / PageRank(song). Dividing by global PageRank
// removes the advantage of songs whose attributes are simply common.

// Relation weights and walk settings come from similarity_config.json, shared
// with vsae_contentbased.py so the two versions can't drift. The "Compare
// songs by" checkboxes switch relations on/off. See that file for what each
// setting does (rarity_damping is the β that tames rare composers).
var RELATIONS=null,PPR_RESTART=null,NEIGHBOR_SHARE=null,RARITY_DAMPING=null;
var SIMILARITY_CONFIG_URL='similarity_config.json';

// Throws with a readable message if the file is malformed.
function applySimilarityConfig(cfg){
  function num(v,name,lo,hi){
    if(typeof v!=='number'||!isFinite(v)||v<lo||v>hi) throw new Error(name+' must be a number from '+lo+' to '+hi+'.');
    return v;
  }
  if(!cfg||typeof cfg.relations!=='object') throw new Error('missing "relations".');
  var rel={};
  ['Composer','Language','Genre','Era','VocalRange','RangeSpan','Runtime'].forEach(function(k){
    var r=cfg.relations[k];
    if(!r) throw new Error('relations.'+k+' is missing.');
    rel[k]={weight:num(r.weight,'relations.'+k+'.weight',0,100),label:String(r.label||k.toLowerCase())};
  });
  RELATIONS=rel;
  PPR_RESTART=num(cfg.ppr_restart,'ppr_restart',0.01,0.99);
  NEIGHBOR_SHARE=num(cfg.neighbor_share,'neighbor_share',0,10);
  RARITY_DAMPING=num(cfg.rarity_damping,'rarity_damping',0,1);
}

// Without a usable config, similarity is switched off but the rest of the
// page (filters, range matching, song choice) keeps working.
function disableSimilarity(reason){
  var chk=document.getElementById('chk-similarity');
  chk.checked=false;chk.disabled=true;
  document.getElementById('sim-block').classList.remove('open');
  var hint=document.getElementById('sim-unavailable');
  if(hint){hint.textContent='Similarity ranking is unavailable right now ('+String(reason).replace(/\.$/,'')+').';hint.style.display='block';}
  console.error('similarity_config.json: '+reason);
}

var VOICE_LADDER=['Soprano','Mezzo Soprano','Alto','Tenor','Baritone','Bass'];
var SPAN_BINS=[{max:9,label:'narrow range'},{max:12,label:'moderate range'},{max:15,label:'wide range'},{max:99,label:'very wide range'}];
var RUNTIME_BINS=[{max:119,label:'under 2 min'},{max:179,label:'2–3 min'},{max:239,label:'3–4 min'},{max:99999,label:'over 4 min'}];

// The Genre column is free text ("Classical / Folk", "spirtual", "Mélodie/Classic").
// Map it to a few tags; a song can carry more than one. Era words used as
// genres ("Baroque", "Romantic") are left to the Era relation.
var GENRE_TAGS=[
  ['spiritual',/spiritual|spirtual|gospel/],
  ['sacred',/sacred|hymn/],
  ['opera / aria',/opera|aria|operetta/],
  ['art song',/art|m.lodie|romance|berceuse|lied/],
  ['folk',/folk|shanty|ballad|tonadilla/],
  ['madrigal',/madrigal|mardrigal/],
  ['classical',/classic/]
];
function genreTags(g){
  var low=String(g||'').toLowerCase(),out=[];
  GENRE_TAGS.forEach(function(t){if(t[1].test(low)) out.push(t[0]);});
  return out;
}

function binLabel(bins,v){
  for(var i=0;i<bins.length;i++) if(v<=bins[i].max) return i;
  return bins.length-1;
}

function titleCase(s){return s.replace(/\b\w/g,function(c){return c.toUpperCase();});}

function buildAttributeGraph(songs,relations){
  var n=songs.length,ids={},nodes=[],adj=[];
  function node(key,rel,label){
    if(!(key in ids)){ids[key]=nodes.length;nodes.push({key:key,rel:rel,label:label});adj.push({});}
    return ids[key];
  }
  function link(a,b,w){adj[a][b]=(adj[a][b]||0)+w;adj[b][a]=(adj[b][a]||0)+w;}
  function on(r){return relations.indexOf(r)>=0;}
  songs.forEach(function(s,i){node('song:'+i,'song',s.Title);});

  songs.forEach(function(s,i){
    var u=i;
    if(on('Composer')&&String(s.Composer||'').trim()){
      var c=s.Composer.trim();
      link(u,node('composer:'+c.toLowerCase(),'Composer',c),RELATIONS.Composer.weight);
    }
    if(on('Language')){
      var l=normLang(s.Language);
      link(u,node('lang:'+l,'Language',titleCase(l)),RELATIONS.Language.weight);
    }
    if(on('Genre')){
      var tags=genreTags(s.Genre);
      tags.forEach(function(t){link(u,node('genre:'+t,'Genre',titleCase(t)),RELATIONS.Genre.weight/tags.length);});
    }
    if(on('Era')&&s.Era&&s.Era!=='Unknown'){
      link(u,node('era:'+s.Era,'Era',s.Era),RELATIONS.Era.weight);
    }
    if(on('VocalRange')&&s.VocalRange){
      if(s.VocalRange==='Vocal All'){
        // Vocal All songs can be sung by any voice: spread the link evenly.
        VOICE_LADDER.forEach(function(v){link(u,node('voice:'+v,'VocalRange',v),RELATIONS.VocalRange.weight/VOICE_LADDER.length);});
      }else if(VOICE_LADDER.indexOf(s.VocalRange)>=0){
        link(u,node('voice:'+s.VocalRange,'VocalRange',s.VocalRange),RELATIONS.VocalRange.weight);
      }
    }
    if(on('RangeSpan')&&noteToMidi(s['Highest Note'])!=null&&noteToMidi(s['Lowest Note'])!=null){
      var b=binLabel(SPAN_BINS,s.RangeSpan);
      link(u,node('span:'+b,'RangeSpan',SPAN_BINS[b].label),RELATIONS.RangeSpan.weight);
    }
    if(on('Runtime')&&runtimeToSeconds(s['Runtime of Song'])!=null){
      var rb=binLabel(RUNTIME_BINS,s.RuntimeSeconds);
      link(u,node('runtime:'+rb,'Runtime',RUNTIME_BINS[rb].label),RELATIONS.Runtime.weight);
    }
  });

  // Rarity damping (see RARITY_DAMPING).
  for(var a0=n;a0<nodes.length;a0++){
    var members=Object.keys(adj[a0]).filter(function(k){return +k<n;});
    var scale=Math.pow(members.length,RARITY_DAMPING);
    members.forEach(function(k){adj[a0][k]*=scale;adj[k][a0]*=scale;});
  }

  // Ordinal neighbours, weighted relative to how many songs each value has
  // so a walker at "Class A" steps to "Class B" ~20% of the time.
  function chain(prefix,values){
    for(var k=0;k+1<values.length;k++){
      var a=ids[prefix+values[k]],b=ids[prefix+values[k+1]];
      if(a==null||b==null) continue;
      var da=0,db=0,x;
      for(x in adj[a]) da+=adj[a][x];
      for(x in adj[b]) db+=adj[b][x];
      link(a,b,NEIGHBOR_SHARE*Math.min(da,db));
    }
  }
  chain('era:',ERA_KEYS);
  chain('voice:',VOICE_LADDER);
  chain('span:',SPAN_BINS.map(function(_,i){return i;}));
  chain('runtime:',RUNTIME_BINS.map(function(_,i){return i;}));

  // Flatten to arrays for the power iteration.
  var N=nodes.length,nbr=[],wts=[],deg=new Float64Array(N);
  for(var v=0;v<N;v++){
    var ks=Object.keys(adj[v]);
    nbr.push(new Int32Array(ks.map(Number)));
    wts.push(new Float64Array(ks.map(function(k){return adj[v][k];})));
    for(var t=0;t<ks.length;t++) deg[v]+=wts[v][t];
  }
  var g={N:N,nSongs:n,nodes:nodes,nbr:nbr,wts:wts,deg:deg};
  // Global PageRank with restarts spread over all songs = how reachable a
  // song is from anywhere; the baseline for lift.
  var teleport=new Float64Array(N);
  for(var i=0;i<n;i++) teleport[i]=1/n;
  g.baseline=randomWalk(g,teleport);
  return g;
}

function randomWalk(g,teleport){
  var N=g.N,rank=new Float64Array(teleport);
  for(var iter=0;iter<200;iter++){
    var next=new Float64Array(N);
    for(var u=0;u<N;u++){
      if(!rank[u]||!g.deg[u]) continue;
      var f=(1-PPR_RESTART)*rank[u]/g.deg[u],nb=g.nbr[u],w=g.wts[u];
      for(var t=0;t<nb.length;t++) next[nb[t]]+=f*w[t];
    }
    var l1=0;
    for(var v=0;v<N;v++){next[v]+=PPR_RESTART*teleport[v];l1+=Math.abs(next[v]-rank[v]);}
    rank=next;
    if(l1<1e-12) break;
  }
  return rank;
}

function personalizedPageRank(g,songIdx){
  var teleport=new Float64Array(g.N);teleport[songIdx]=1;
  return randomWalk(g,teleport);
}

// "Why" for one candidate: the attribute nodes that sent it the most mass
// in the final step of the walk. Values the reference doesn't share but is
// next to (Class B for a Class A reference) are reported as "similar …".
function explainScore(g,ppr,songIdx,queryIdx){
  var qLinks={},nb=g.nbr[queryIdx];
  for(var t=0;t<nb.length;t++) qLinks[nb[t]]=1;
  var flows=[],total=0,cn=g.nbr[songIdx],cw=g.wts[songIdx];
  for(var t2=0;t2<cn.length;t2++){
    var a=cn[t2];
    if(a<g.nSongs) continue;
    var f=ppr[a]*cw[t2]/g.deg[a];
    total+=f;flows.push({node:g.nodes[a],shared:!!qLinks[a],flow:f});
  }
  flows.sort(function(x,y){return y.flow-x.flow;});
  var seen={},out=[];
  flows.forEach(function(x){
    if(out.length>=3||x.flow<0.12*total) return;
    // Only ordinal values have neighbours; any other unshared value got its
    // mass via longer paths and isn't a meaningful reason to show.
    var ordinal=['Era','VocalRange','RangeSpan','Runtime'].indexOf(x.node.rel)>=0;
    if(!x.shared&&!ordinal) return;
    var label=x.shared?x.node.label:'similar '+RELATIONS[x.node.rel].label;
    if(seen[label]) return;
    seen[label]=1;out.push(label);
  });
  return out;
}

function baseTitle(t){
  return(t||'').replace(/\s*\((?:High|Low|Medium|Vocal All|Bass|Baritone|Soprano|Alto|Tenor|Mezzo(?: Soprano)?)[^)]*\)\s*$/i,'').trim();
}
function songGroupKey(s){
  return baseTitle(s.Title).toLowerCase()+'|'+String(s.Composer||'').trim().toLowerCase();
}

// "Also in" lists other versions by key 
function versionLabel(s){
  var k=String(s.Key||'').trim();
  if(k) return k;
  var m=String(s.Title||'').match(/\(([^)]*)\)\s*$/);
  return m?m[1]:(s.VocalRange||'other version');
}


function bestFitVersion(group){
  var best=null;
  group.forEach(function(s){
    if(s._rangeOut==null) return;
    if(!best||s._rangeOut<best._rangeOut||
       (s._rangeOut===best._rangeOut&&s._rangeCenterDist<best._rangeCenterDist)) best=s;
  });
  return best;
}

// `ranked` lists are already sorted best-first. Returns them with later
// versions of an already-shown song removed and each kept song's
// remaining versions (from `pool`) attached as _otherKeys.
function collapseVersions(rankedLists,pool,useRange){
  var groups={};
  pool.forEach(function(s){var k=songGroupKey(s);(groups[k]=groups[k]||[]).push(s);});
  var rep={};
  if(useRange) Object.keys(groups).forEach(function(k){var b=bestFitVersion(groups[k]);if(b) rep[k]=b;});
  var seen={};
  return rankedLists.map(function(list){
    return list.filter(function(s){
      var k=songGroupKey(s);
      if(rep[k]&&rep[k]!==s) return false;
      if(seen[k]) return false;
      seen[k]=1;
      s._otherKeys=groups[k].filter(function(o){return o!==s;});
      return true;
    });
  });
}

// ════════════════════════════════════════════════════════════════
// FILTERING
// ════════════════════════════════════════════════════════════════
var MALE_VR=['tenor','baritone','bass'];

// Vocal All songs are written at treble pitch. Men sing them an octave
// lower, so transpose only when every selected voice is male. Any female
// voice in the selection (e.g. Alto + Tenor for a contralto) keeps the
// written pitch.
function shouldTransposeForMen(selectedRanges){
  return selectedRanges.length>0&&selectedRanges.every(function(v){return MALE_VR.indexOf(v.toLowerCase())>=0;});
}

// Songs come in by their voice tag. Vocal All isn't a voice chip: those
// songs are included whenever the student narrows the voices. Whether a song
// suits the singer is decided by their own notes (Range fit), not by
// per-voice note limits.
function applyVocalRangeFilter(songs,selectedRanges){
  if(!selectedRanges.length||selectedRanges.length>=VOICE_LADDER.length) return songs;
  var set={'Vocal All':1};
  selectedRanges.forEach(function(v){set[v]=1;});
  return songs.filter(function(s){return set[s.VocalRange];});
}

// ════════════════════════════════════════════════════════════════
// RANGE MATCH SCORING
// ════════════════════════════════════════════════════════════════
// A song fits when every note is inside the singer's range. _rangeOut is
// how many semitones the song goes past the range (below + above), so songs
// that fit always sort first. (The old score was overlap / singer's span,
// which gave a song wider than the range at both ends a perfect 100.)
function scoreRangeMatch(songs,userLow,userHigh,isMale){
  var lo=Math.min(userLow,userHigh),hi=Math.max(userLow,userHigh);
  songs.forEach(function(s){
    var sLo=s.LowestNote_MIDI,sHi=s.HighestNote_MIDI;
    if(sLo==null||sHi==null){s._rangeOut=null;return;}
    if(isMale&&s.VocalRange==='Vocal All'){sLo-=12;sHi-=12;}
    s._rangeBelow=Math.max(0,lo-sLo);
    s._rangeAbove=Math.max(0,sHi-hi);
    s._rangeOut=s._rangeBelow+s._rangeAbove;
    s._rangeCenterDist=Math.abs((sLo+sHi)/2-(lo+hi)/2);
  });
}

function rangeFitLabel(s){
  if(s._rangeOut==null) return '';
  if(s._rangeOut===0) return 'Fits';
  var parts=[];
  if(s._rangeBelow) parts.push(s._rangeBelow+' too low');
  if(s._rangeAbove) parts.push(s._rangeAbove+' too high');
  return parts.join(', ')+' (semitones)';
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
var DISPLAY_COLS=['Title','Composer','VocalRange','Class','Language','Genre','Era','Key','Note Range','Runtime'];
var COL_LABELS={VocalRange:'Vocal range','Note Range':'Note range',RangeSpan:'Range span',RuntimeSeconds:'Runtime (s)'};
function colLabel(c){return COL_LABELS[c]||c;}

function renderTable(songs,containerId,transposeForMen,topK){
  var el=document.getElementById(containerId);
  if(!el) return;
  if(!songs||!songs.length){el.innerHTML='<div class="empty-row">No songs match the current filters.</div>';return;}
  var rows=songs.slice(0,topK||songs.length);
  var showSimilarity=rows.some(function(s){return s._similarityScore!==null&&s._similarityScore!==undefined;});
  var showFit=rows.some(function(s){return s._rangeOut!=null;});
  var cols=DISPLAY_COLS.slice();
  if(showFit) cols.push('Range fit');
  if(showSimilarity) cols.push('Similarity','Why');
  var html='<table><thead><tr><th>#</th>';
  cols.forEach(function(c){html+='<th>'+esc(colLabel(c))+'</th>';});
  html+='</tr></thead><tbody>';
  rows.forEach(function(s,i){
    var rt=String(s['Runtime of Song']||'').trim();
    if(!rt||rt.toUpperCase()==='N/A'||rt.toUpperCase()==='NAN') rt='Missing';
    var noteRange=noteRangeLabel(s,transposeForMen);
    html+='<tr><td class="num">'+(i+1)+'</td>';
    var others=(s._otherKeys||[]).map(versionLabel);
    html+='<td>'+esc(s.Title||'')+(others.length?'<div class="also-in">Also in keys: '+esc(others.join(' \u00b7 '))+'</div>':'')+'</td>';
    ['Composer','VocalRange','Class','Language','Genre','Era','Key'].forEach(function(c){
      var v=s[c]||'';
      if(c==='VocalRange'&&transposeForMen&&v==='Vocal All') v='Vocal All (octave lower)';
      html+='<td>'+esc(v)+'</td>';
    });
    html+='<td>'+esc(noteRange)+'</td><td>'+esc(rt)+'</td>';
    if(showFit){
      html+='<td class="'+(s._rangeOut===0?'fit-ok':'fit-out')+'">'+esc(rangeFitLabel(s))+'</td>';
    }
    if(showSimilarity){
      html+='<td>'+esc(s._similarityScore==null?'':s._similarityScore.toFixed(2))+'</td>';
      html+='<td class="why">'+esc((s._why||[]).join(' · '))+'</td>';
    }
    html+='</tr>';
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
    var chip=document.createElement('button');
    chip.type='button';chip.className='chip';chip.textContent=v;chip.dataset.value=v;
    chip.setAttribute('aria-pressed','true');
    chip.addEventListener('click',function(){
      setChip(chip,chip.classList.contains('off'));
      runPipeline();
    });
    el.appendChild(chip);
  });
}

function setChip(chip,on){
  chip.classList.toggle('off',!on);
  chip.setAttribute('aria-pressed',on?'true':'false');
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
      datasets:[{data:data,backgroundColor:color||'#03553E',borderRadius:4}]
    },
    options:{
      responsive:true,maintainAspectRatio:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:'#444',font:{family:'Roboto',size:12}},grid:{display:false}},
        y:{ticks:{color:'#444',font:{family:'Roboto',size:12},precision:0},grid:{color:'#E3E5E8'}}
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
  makeChart('chart-class',countBy(songs,'Class'));
  makeChart('chart-lang',countBy(songs,'Language'));
  makeChart('chart-era',countBy(songs,'Era'));

  var cols=['Title','Composer','VocalRange','Class','Language','Genre','Era','RangeSpan','RuntimeSeconds'];
  var html='<table><thead><tr><th>#</th>'+cols.map(function(c){return'<th>'+esc(colLabel(c))+'</th>';}).join('')+'</tr></thead><tbody>';
  songs.forEach(function(s,i){
    html+='<tr><td class="num">'+(i+1)+'</td>'+
      cols.map(function(c){return'<td>'+esc(s[c]!=null?s[c]:'')+'</td>';}).join('')+'</tr>';
  });
  document.getElementById('tbl-full').innerHTML=html+'</tbody></table>';
}

// ════════════════════════════════════════════════════════════════
// PIPELINE CACHE
// ════════════════════════════════════════════════════════════════
var _similarityKey=null;
var _graphKey=null,_graph=null;

function getSimilarityScores(querySongIdx,features,excludeTranspositions){
  var key=querySongIdx+'|'+features.join(',')+'|'+excludeTranspositions;
  if(key===_similarityKey) return;

  _similarityKey=key;
  var fKey=features.join(',');
  if(fKey!==_graphKey){
    _graph=buildAttributeGraph(allSongs,features);
    _graphKey=fKey;
  }
  var ppr=personalizedPageRank(_graph,querySongIdx);

  allSongs.forEach(function(s){s._pprRaw=null;s._why=null;});

  var queryBase=baseTitle(allSongs[querySongIdx].Title).toLowerCase();
  allSongs.forEach(function(s,i){
    if(i===querySongIdx) return;
    if(excludeTranspositions&&queryBase&&(s.Title||'').toLowerCase().startsWith(queryBase)) return;
    s._pprRaw=_graph.baseline[i]>0?ppr[i]/_graph.baseline[i]:0;
    s._why=explainScore(_graph,ppr,i,querySongIdx);
  });
}

// Lift spans orders of magnitude, so scores are log-scaled between the
// weakest and strongest lift among the songs the student can currently see:
// 1.00 = best visible match, 0.00 = weakest.
function scaleSimilarity(songs){
  allSongs.forEach(function(s){s._similarityScore=null;});
  var max=0,min=Infinity;
  songs.forEach(function(s){if(s._pprRaw>max) max=s._pprRaw;if(s._pprRaw<min) min=s._pprRaw;});
  min=Math.max(min,max*1e-4);
  var span=Math.log(max/min);
  songs.forEach(function(s){
    var v=span>0?Math.log(Math.max(s._pprRaw,min)/min)/span:1;
    s._similarityScore=Math.max(0,Math.min(1,v));
  });
}

// Rank candidates by the similarity level selected on the slider:
// 0.00 favors the least similar songs, 1.00 favors the most similar songs,
// and intermediate values favor scores closest to that level.
function compareSimilarityPreference(a,b,level){
  var aDistance=Math.abs((a._similarityScore||0)-level);
  var bDistance=Math.abs((b._similarityScore||0)-level);
  if(aDistance!==bDistance) return aDistance-bDistance;
  return level<0.5?
    (a._similarityScore||0)-(b._similarityScore||0):
    (b._similarityScore||0)-(a._similarityScore||0);
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

  if(selLang.length){var ls={};selLang.forEach(function(v){ls[v.toLowerCase()]=1;});filtered=filtered.filter(function(s){return ls[normLang(s.Language)];});}

  var simEnabled=false;
  var chkSim=document.getElementById('chk-similarity');
  if(chkSim&&chkSim.checked){
    var refVal=document.getElementById('sel-reference').value;
    var simLevel=parseFloat(document.getElementById('slider-sim-min').value)||0;
    var feats=[];
    var featMap=[['feat-composer','Composer'],['feat-vocalrange','VocalRange'],['feat-language','Language'],
      ['feat-genre','Genre'],['feat-era','Era'],['feat-rangespan','RangeSpan'],['feat-runtime','Runtime']];
    featMap.forEach(function(p){var el=document.getElementById(p[0]);if(el&&el.checked) feats.push(p[1]);});
    var excl=document.getElementById('chk-exclude-transpositions').checked;

    if(refVal!==''&&feats.length){
      var qIdx=parseInt(refVal,10);
      if(isFinite(qIdx)&&qIdx>=0&&qIdx<allSongs.length){
        try{
          getSimilarityScores(qIdx,feats,excl);
          filtered=filtered.filter(function(s){return s._pprRaw!=null;});
          scaleSimilarity(filtered);
          filtered.sort(function(a,b){return compareSimilarityPreference(a,b,simLevel);});
          if(filtered.length){
            simEnabled=true;
            var simValues=filtered.map(function(s){return s._similarityScore;});
            var simHint=document.getElementById('sim-score-hint');
            if(simHint){
              var direction=simLevel===0?'the least similar songs':
                (simLevel===1?'the most similar songs':'songs with similarity near '+simLevel.toFixed(2));
              simHint.textContent='Showing '+direction+' first. Scores range from '+
                Math.min.apply(null,simValues).toFixed(2)+'–'+
                Math.max.apply(null,simValues).toFixed(2)+
                ' (1.00 = most similar).';
            }
          }else{
            var simHint=document.getElementById('sim-score-hint');
            if(simHint) simHint.textContent='';
          }
        }catch(e){console.warn('Similarity scoring error:',e);}
      }
    }
  } else {
    allSongs.forEach(function(s){s._similarityScore=null;});
    _similarityKey=null;
    var simHint=document.getElementById('sim-score-hint');
    if(simHint) simHint.textContent='';
  }

  allSongs.forEach(function(s){s._rangeOut=null;});
  var basedFiltered=filtered.slice();

  var displaySongs=basedFiltered.filter(function(s){return!isMissingData(s);});
  var missingSongs=basedFiltered.filter(function(s){return isMissingData(s);});

  // Range matching applies whenever both notes are entered.
  var rangeEnabled=true;
  var lowMidi=null,highMidi=null;
  var hintLow=document.getElementById('hint-low');
  var hintHigh=document.getElementById('hint-high');
  var hintBoth=document.getElementById('hint-both');

  if(rangeEnabled){
    var lowStr=(document.getElementById('input-low').value||'').trim();
    var highStr=(document.getElementById('input-high').value||'').trim();
    var vr=validateVocalRange(lowStr,highStr);
    lowMidi=vr.low;highMidi=vr.high;

    function fieldHint(el,input,str,err){
      if(!el) return;
      el.innerHTML=err?'<span class="parse-err">'+esc(err)+'</span>':
        (str&&parseNoteInput(str)!=null?'<span class="parse-ok">'+esc(midiToNote(parseNoteInput(str)))+'</span>':'');
      input.setAttribute('aria-invalid',err?'true':'false');
    }
    fieldHint(hintLow,document.getElementById('input-low'),lowStr,vr.lowErr);
    fieldHint(hintHigh,document.getElementById('input-high'),highStr,vr.highErr);
    if(hintBoth){
      if(vr.error) hintBoth.innerHTML='<span class="parse-err">'+esc(vr.error)+'</span>';
      else if(vr.warning) hintBoth.innerHTML='<span class="parse-warn">'+esc(vr.warning)+'</span>';
      else if(!vr.lowErr&&!vr.highErr&&(lowStr||highStr)&&(lowMidi==null)) hintBoth.textContent='Enter both notes to sort by range fit.';
      else hintBoth.textContent='';
    }

    if(lowMidi!=null&&highMidi!=null){
      scoreRangeMatch(displaySongs,lowMidi,highMidi,transposeForMen);
      displaySongs.sort(function(a,b){
        var rd=(a._rangeOut==null?999:a._rangeOut)-(b._rangeOut==null?999:b._rangeOut);
        if(rd!==0) return rd;
        if(simEnabled){
          var sd=compareSimilarityPreference(a,b,simLevel);
          if(sd!==0) return sd;
        }
        return(a._rangeCenterDist||0)-(b._rangeCenterDist||0);
      });
    }else if(simEnabled){
      // No usable range yet: rank by the selected similarity preference.
      displaySongs.sort(function(a,b){return compareSimilarityPreference(a,b,simLevel);});
    }
  }

  if(simEnabled){
    missingSongs.sort(function(a,b){return compareSimilarityPreference(a,b,simLevel);});
  }

  var collapsed=collapseVersions([displaySongs,missingSongs],basedFiltered,lowMidi!=null&&highMidi!=null&&rangeEnabled);
  displaySongs=collapsed[0];missingSongs=collapsed[1];

  var octaveNote=document.getElementById('octave-note');
  if(octaveNote) octaveNote.style.display=transposeForMen?'block':'none';

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

    // Every version of each recommended song is selectable, so students
    // can still register the key that suits them.
    var recommended=displaySongs.slice(0,topK).concat(missingSongs.slice(0,topK));
    recommended.forEach(function(song){
      [song].concat(song._otherKeys||[]).forEach(function(v,vi){
        var opt=document.createElement('option');
        opt.value=String(v.Song_Code||'');
        opt.textContent=(vi?'\u00a0\u00a0\u21b3 ':'')+v.Title+' | '+v.VocalRange+' | Class '+(v.Class||'')+(v.Key?' | Key '+v.Key:'')+' | '+(v.Language||'');
        selChosen.appendChild(opt);
      });
    });

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
    if(!chkSim.checked){allSongs.forEach(function(s){s._similarityScore=null;});_similarityKey=null;}
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

  ['sel-reference','feat-composer','feat-vocalrange','feat-language','feat-genre',
   'feat-era','feat-rangespan','feat-runtime','chk-exclude-transpositions'].forEach(function(id){
    var el=document.getElementById(id); if(!el) return;
    el.addEventListener('change',function(){_similarityKey=null;runPipeline();});
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
      window.location.href=REGISTRATION_URL+'?StudentID='+encodeURIComponent(getParam('StudentID'))+'&song-code='+encodeURIComponent(songCode);
    });
  }

  var btnCancel=document.getElementById('btn-cancel');
  if(btnCancel){
    btnCancel.addEventListener('click',function(){
      window.location.href=REGISTRATION_URL+'?StudentID='+encodeURIComponent(getParam('StudentID'));
    });
  }
}

// ════════════════════════════════════════════════════════════════
// URL PARAMETERS (from the aveschoir.org registration form)
// ════════════════════════════════════════════════════════════════
var REGISTRATION_URL='https://aveschoir.org/Vocal-Solo-Event';

function getParam(name){
  var v=new URLSearchParams(window.location.search||'').get(name);
  v=(v||'').trim();
  return(!v||v.toUpperCase()==='NA')?'':v;
}

// PreviousVocalRange → default Vocal Range filter (students can change it).
function applyPreviousVocalRange(){
  var prev=getParam('PreviousVocalRange').toLowerCase();
  if(!prev) return;
  var chips=document.querySelectorAll('#chips-vr .chip');
  var match=null;
  chips.forEach(function(c){if(c.dataset.value.toLowerCase()===prev) match=c;});
  if(!match) return;
  chips.forEach(function(c){setChip(c,c===match);});
}

// PreviousSongCode → default reference song for similarity (optional).
function applyPreviousSongCode(){
  var code=getParam('PreviousSongCode');
  if(!code) return;
  var idx=allSongs.findIndex(function(s){return String(s.Song_Code||'').trim()===code;});
  if(idx<0) return;
  document.getElementById('sel-reference').value=String(idx);
  var chk=document.getElementById('chk-similarity');
  chk.checked=true;
  document.getElementById('sim-block').classList.add('open');
  var note=document.getElementById('prev-song-note');
  if(note){
    note.textContent='We started with the song you sang last time, '+allSongs[idx].Title+
      '. Pick a different reference song, or turn this off to skip it.';
    note.style.display='block';
  }
}

// ════════════════════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════════════════════
function initializeSongSelector(){
  wire();

  // no-cache: revalidate so a retuned config takes effect without a version bump
  var configLoaded=fetch(SIMILARITY_CONFIG_URL,{cache:'no-cache'})
    .then(function(r){if(!r.ok) throw new Error('status '+r.status);return r.json();})
    .then(function(cfg){applySimilarityConfig(cfg);return true;})
    .catch(function(err){disableSimilarity(err.message);return false;});

  Promise.all([
    fetch('VSAE_Data_Final.csv').then(function(r){
      if(!r.ok) throw new Error('CSV returned status '+r.status+'. Make sure VSAE_Data_Final.csv is in the same folder as index.html.');
      return r.text();
    }),
    configLoaded
  ])
    .then(function(results){
      var text=results[0];
      var raw=loadVSAEData(text);
      if(!raw.length) throw new Error('CSV parsed to 0 songs — check the file format.');
      allSongs=engineerFeatures(raw);

      document.getElementById('loading').style.display='none';
      document.getElementById('app-shell').style.display='block';

      buildChips('chips-vr',VOICE_LADDER.filter(function(v){return allSongs.some(function(s){return s.VocalRange===v;});}));
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

      applyPreviousVocalRange();
      if(results[1]) applyPreviousSongCode();

      renderOverview(allSongs);
      runPipeline();
    })
    .catch(function(err){
      document.getElementById('loading').style.display='none';
      var box=document.getElementById('error-msg');
      box.style.display='block';
      box.textContent=err.message;
    });
}

// app.js injects this script after the page has loaded, so
// DOMContentLoaded may already have fired (fix from Mark, Oct 2026).
if(document.readyState==='loading'){
  document.addEventListener('DOMContentLoaded',initializeSongSelector);
}else{
  initializeSongSelector();
}
