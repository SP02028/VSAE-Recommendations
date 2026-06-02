function loadVSAEData(csvText) {
  function parseCSV(text) {
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;

    for (let i = 0; i < text.length; i += 1) {
      const ch = text[i];

      if (ch === '"') {
        if (inQuotes && text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === ',' && !inQuotes) {
        row.push(field);
        field = "";
      } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
        if (ch === '\r' && text[i + 1] === '\n') {
          i += 1;
        }

        row.push(field);
        rows.push(row);
        row = [];
        field = "";
      } else {
        field += ch;
      }
    }

    if (field.length > 0 || row.length > 0) {
      row.push(field);
      rows.push(row);
    }

    return rows;
  }

  const rows = parseCSV(csvText || "");
  if (rows.length === 0) {
    return [];
  }

  const headers = rows[0].map((h, idx) => {
    const trimmed = String(h == null ? "" : h).trim();
    if (idx === 0) {
      return trimmed.replace(/^\uFEFF/, "");
    }
    return trimmed || `__extra_col_${idx + 1}`;
  });

  const titleIndex = headers.findIndex((h) => h.toLowerCase() === "title");
  const songs = [];

  for (let r = 1; r < rows.length; r += 1) {
    const raw = rows[r];
    if (!raw || raw.length === 0) {
      continue;
    }

    const values = raw.map((v) => String(v == null ? "" : v).trim());

    const title = titleIndex >= 0 ? (values[titleIndex] || "").trim() : "";
    if (!title) {
      continue;
    }

    const lastVal = (values[values.length - 1] || "").toLowerCase();
    if (lastVal.includes("should probably be removed")) {
      continue;
    }

    const obj = {};
    for (let i = 0; i < headers.length; i += 1) {
      obj[headers[i]] = values[i] == null ? "" : values[i];
    }

    songs.push(obj);
  }

  return songs;
}

const PITCH_CLASS = {
  C: 0,
  "C#": 1,
  Db: 1,
  D: 2,
  "D#": 3,
  Eb: 3,
  E: 4,
  F: 5,
  "F#": 6,
  Gb: 6,
  G: 7,
  "G#": 8,
  Ab: 8,
  A: 9,
  "A#": 10,
  Bb: 10,
  B: 11,
};

function noteToMidi(noteStr) {
  if (typeof noteStr !== "string") {
    return null;
  }

  let cleaned = noteStr.trim();
  cleaned = cleaned.split(/[(),]/)[0].trim();
  if (cleaned === "N/A" || cleaned === "" || cleaned === "n/a") {
    return null;
  }

  const match = cleaned.match(/([A-Ga-g][b#]?)(\d)/);
  if (!match) {
    return null;
  }

  const rawPitch = match[1];
  const octave = Number.parseInt(match[2], 10);
  const pitch =
    rawPitch.length > 1
      ? rawPitch[0].toUpperCase() + rawPitch.slice(1).toLowerCase()
      : rawPitch.toUpperCase();

  if (!Object.prototype.hasOwnProperty.call(PITCH_CLASS, pitch)) {
    return null;
  }

  return (octave + 1) * 12 + PITCH_CLASS[pitch];
}

function runtimeToSeconds(rtStr) {
  if (typeof rtStr !== "string") {
    return null;
  }

  const cleaned = rtStr.trim();
  if (!cleaned || cleaned.toUpperCase() === "N/A") {
    return null;
  }

  const match = cleaned.match(/^(\d+):(\d{2}(?:\.\d+)?)$/);
  if (!match) {
    return null;
  }

  const minutes = Number.parseInt(match[1], 10);
  const secondsPart = Number.parseFloat(match[2]);
  if (!Number.isFinite(minutes) || !Number.isFinite(secondsPart)) {
    return null;
  }

  return Math.round(minutes * 60 + secondsPart);
}

const ERA_KEYWORDS = ["Renaissance", "Baroque", "Classical", "Romantic", "Modern"];

const VOCAL_RANGE_MIDI_DEFAULTS = {
  Soprano: { high: 79, low: 65 },
  "Mezzo Soprano": { high: 77, low: 62 },
  Alto: { high: 77, low: 60 },
  Tenor: { high: 76, low: 60 },
  Baritone: { high: 64, low: 48 },
  Bass: { high: 62, low: 43 },
  "Vocal All": { high: 77, low: 60 },
};

const RANGE_MIDI_BOUNDS = {
  Soprano: { low: 65, high: 79 },
  "Mezzo Soprano": { low: 62, high: 77 },
  Alto: { low: 60, high: 77 },
  Tenor: { low: 60, high: 76 },
  Baritone: { low: 48, high: 64 },
  Bass: { low: 43, high: 62 },
  "Vocal All": { low: 60, high: 77 },
};

const MALE_RANGES = ["tenor", "baritone", "bass"];

const VOICE_PART_OPTIONS = ["Soprano", "Mezzo Soprano", "Alto", "Tenor", "Baritone", "Bass"];

const LANGUAGE_OPTIONS = ["English", "Italian", "French", "German", "Spanish", "Latin"];

const CLASS_ORDINAL_MAP = { A: 1, B: 2, C: 3 };

const ERA_ORDINAL_MAP = {
  Renaissance: 1,
  Baroque: 2,
  Classical: 3,
  Romantic: 4,
  Modern: 5,
  Unknown: 3,
};

let allSongs = [];
let currentStudentId = null;
let currentPrevSongCode = null;
let selectedSongCode = null;

function inferEra(timePeriod) {
  const value = typeof timePeriod === "string" ? timePeriod.trim() : "";
  if (!value) {
    return "Unknown";
  }

  const lower = value.toLowerCase();
  for (let i = 0; i < ERA_KEYWORDS.length; i += 1) {
    const era = ERA_KEYWORDS[i];
    if (lower.includes(era.toLowerCase())) {
      return era;
    }
  }

  const yearMatch = value.match(/(\d{3,4})/);
  if (!yearMatch) {
    return "Unknown";
  }

  const year = Number.parseInt(yearMatch[1], 10);
  if (!Number.isFinite(year)) {
    return "Unknown";
  }
  if (year < 1600) {
    return "Renaissance";
  }
  if (year < 1750) {
    return "Baroque";
  }
  if (year < 1820) {
    return "Classical";
  }
  if (year < 1910) {
    return "Romantic";
  }
  return "Modern";
}

function median(numbers) {
  if (!numbers || numbers.length === 0) {
    return null;
  }

  const sorted = numbers.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[mid];
  }
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function engineerFeatures(songs) {
  if (!Array.isArray(songs)) {
    return songs;
  }

  const runtimeValues = [];

  for (let i = 0; i < songs.length; i += 1) {
    const song = songs[i];
    if (!song || typeof song !== "object") {
      continue;
    }

    const era = inferEra(song["Time Period"]);
    song.Era = era;

    const vocalRange = typeof song.VocalRange === "string" ? song.VocalRange.trim() : "";
    const defaults = VOCAL_RANGE_MIDI_DEFAULTS[vocalRange] || null;

    let highestMidi = noteToMidi(song["Highest Note"]);
    let lowestMidi = noteToMidi(song["Lowest Note"]);

    if (highestMidi == null && defaults) {
      highestMidi = defaults.high;
    }
    if (lowestMidi == null && defaults) {
      lowestMidi = defaults.low;
    }

    song.HighestNote_MIDI = highestMidi;
    song.LowestNote_MIDI = lowestMidi;

    if (highestMidi == null || lowestMidi == null) {
      song.RangeSpan = 0;
    } else {
      song.RangeSpan = Math.max(0, highestMidi - lowestMidi);
    }

    const runtimeSec = runtimeToSeconds(song["Runtime of Song"]);
    song.RuntimeSeconds = runtimeSec;
    if (runtimeSec != null) {
      runtimeValues.push(runtimeSec);
    }

    const classKey = typeof song.Class === "string" ? song.Class.trim().toUpperCase() : "";
    song.ClassOrdinal = CLASS_ORDINAL_MAP[classKey] || 2;
    song.EraOrdinal = ERA_ORDINAL_MAP[era] || 3;
  }

  const runtimeMedian = median(runtimeValues);
  const filledRuntime = runtimeMedian == null ? null : Math.round(runtimeMedian);
  if (filledRuntime != null) {
    for (let i = 0; i < songs.length; i += 1) {
      const song = songs[i];
      if (!song || typeof song !== "object") {
        continue;
      }
      if (song.RuntimeSeconds == null) {
        song.RuntimeSeconds = filledRuntime;
      }
    }
  }

  return songs;
}

function normalizeMinMax(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return [];
  }

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const value = Number(values[i]);
    if (!Number.isFinite(value)) {
      continue;
    }
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) {
    return values.map(() => 0.0);
  }

  const span = max - min;
  return values.map((v) => {
    const value = Number(v);
    if (!Number.isFinite(value)) {
      return 0.0;
    }
    return (value - min) / span;
  });
}

function buildFeatureMatrix(songs, selectedFeatures) {
  if (!Array.isArray(selectedFeatures) || selectedFeatures.length === 0) {
    throw new Error("selectedFeatures must not be empty");
  }

  const rows = Array.isArray(songs) ? songs : [];
  const matrix = rows.map(() => []);

  const hasFeature = (name) => selectedFeatures.includes(name);

  if (hasFeature("VocalRange")) {
    const categories = Array.from(
      new Set(
        rows.map((song) => {
          const vr = song && typeof song.VocalRange === "string" ? song.VocalRange.trim() : "";
          return vr || "Vocal All";
        })
      )
    );

    for (let i = 0; i < rows.length; i += 1) {
      const song = rows[i];
      const vr = song && typeof song.VocalRange === "string" ? song.VocalRange.trim() : "";
      const value = vr || "Vocal All";
      for (let j = 0; j < categories.length; j += 1) {
        matrix[i].push((value === categories[j] ? 1.0 : 0.0) * 3.0);
      }
    }
  }

  if (hasFeature("Class")) {
    for (let i = 0; i < rows.length; i += 1) {
      const song = rows[i] || {};
      const classOrdinalRaw = Number(song.ClassOrdinal);
      const classOrdinal = Number.isFinite(classOrdinalRaw) ? classOrdinalRaw : 2;
      matrix[i].push((classOrdinal / 3.0) * 2.5);
    }
  }

  if (hasFeature("Language")) {
    const categories = Array.from(
      new Set(
        rows.map((song) => {
          const lang = song && typeof song.Language === "string" ? song.Language.trim() : "";
          return lang || "English";
        })
      )
    );

    for (let i = 0; i < rows.length; i += 1) {
      const song = rows[i];
      const lang = song && typeof song.Language === "string" ? song.Language.trim() : "";
      const value = lang || "English";
      for (let j = 0; j < categories.length; j += 1) {
        matrix[i].push((value === categories[j] ? 1.0 : 0.0) * 1.5);
      }
    }
  }

  if (hasFeature("Genre")) {
    const categories = Array.from(
      new Set(
        rows.map((song) => {
          const genre = song && typeof song.Genre === "string" ? song.Genre.trim().toLowerCase() : "";
          return genre || "unknown";
        })
      )
    );

    for (let i = 0; i < rows.length; i += 1) {
      const song = rows[i];
      const genre = song && typeof song.Genre === "string" ? song.Genre.trim().toLowerCase() : "";
      const value = genre || "unknown";
      for (let j = 0; j < categories.length; j += 1) {
        matrix[i].push((value === categories[j] ? 1.0 : 0.0) * 1.5);
      }
    }
  }

  if (hasFeature("Era")) {
    for (let i = 0; i < rows.length; i += 1) {
      const song = rows[i] || {};
      const eraOrdinalRaw = Number(song.EraOrdinal);
      const eraOrdinal = Number.isFinite(eraOrdinalRaw) ? eraOrdinalRaw : 3;
      matrix[i].push((eraOrdinal / 5.0) * 1.0);
    }
  }

  if (hasFeature("RangeSpan")) {
    const normalized = normalizeMinMax(
      rows.map((song) => {
        const value = Number(song && song.RangeSpan);
        return Number.isFinite(value) ? value : 0;
      })
    );
    for (let i = 0; i < normalized.length; i += 1) {
      matrix[i].push(normalized[i] * 1.0);
    }
  }

  if (hasFeature("Runtime")) {
    const normalized = normalizeMinMax(
      rows.map((song) => {
        const value = Number(song && song.RuntimeSeconds);
        return Number.isFinite(value) ? value : 0;
      })
    );
    for (let i = 0; i < normalized.length; i += 1) {
      matrix[i].push(normalized[i] * 0.5);
    }
  }

  for (let i = 0; i < matrix.length; i += 1) {
    for (let j = 0; j < matrix[i].length; j += 1) {
      matrix[i][j] = Number(matrix[i][j]);
    }
  }

  return matrix;
}

function cosineSimilarity(vecA, vecB) {
  const len = Math.min(vecA.length, vecB.length);
  let dot = 0.0;
  let normA = 0.0;
  let normB = 0.0;

  for (let i = 0; i < len; i += 1) {
    const a = Number(vecA[i]);
    const b = Number(vecB[i]);
    const aa = Number.isFinite(a) ? a : 0.0;
    const bb = Number.isFinite(b) ? b : 0.0;
    dot += aa * bb;
    normA += aa * aa;
    normB += bb * bb;
  }

  if (normA === 0 || normB === 0) {
    return 0.0;
  }

  const sim = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  if (sim < 0) {
    return 0.0;
  }
  if (sim > 1) {
    return 1.0;
  }
  return sim;
}

function baseTitle(title) {
  const raw = typeof title === "string" ? title.trim() : "";
  return raw
    .replace(
      /\s*\((?:High Voice|Low Voice|Soprano|Alto|Tenor|Bass|Baritone|Mezzo Soprano|Vocal All)\)\s*$/i,
      ""
    )
    .trim();
}

function personalizedPageRank(featureMatrix, queryIndex, songs, excludeSameBase) {
  const n = Array.isArray(featureMatrix) ? featureMatrix.length : 0;
  if (!Array.isArray(songs)) {
    return songs;
  }
  if (n === 0 || songs.length === 0) {
    return songs;
  }

  const similarity = Array.from({ length: n }, () => Array(n).fill(0.0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      if (i === j) {
        similarity[i][j] = 0.0;
      } else {
        similarity[i][j] = cosineSimilarity(featureMatrix[i] || [], featureMatrix[j] || []);
      }
    }
  }

  const k = Math.min(12, Math.max(3, Math.floor(n / 10)));
  const sparse = Array.from({ length: n }, () => Array(n).fill(0.0));
  for (let i = 0; i < n; i += 1) {
    const pairs = [];
    for (let j = 0; j < n; j += 1) {
      pairs.push({ idx: j, value: similarity[i][j] });
    }
    pairs.sort((a, b) => b.value - a.value);
    const keep = Math.min(k, pairs.length);
    for (let t = 0; t < keep; t += 1) {
      const p = pairs[t];
      sparse[i][p.idx] = p.value;
    }
  }

  const transition = Array.from({ length: n }, () => Array(n).fill(0.0));
  for (let i = 0; i < n; i += 1) {
    let rowSum = 0.0;
    for (let j = 0; j < n; j += 1) {
      rowSum += sparse[i][j];
    }
    if (rowSum !== 0) {
      for (let j = 0; j < n; j += 1) {
        transition[i][j] = sparse[i][j] / rowSum;
      }
    }
  }

  const alpha = 0.85;
  const teleport = Array(n).fill(0.0);
  if (queryIndex >= 0 && queryIndex < n) {
    teleport[queryIndex] = 1.0;
  }
  let rank = teleport.slice();

  for (let iter = 0; iter < 100; iter += 1) {
    const nextRank = Array(n).fill(0.0);
    for (let j = 0; j < n; j += 1) {
      let sum = 0.0;
      for (let i = 0; i < n; i += 1) {
        sum += transition[i][j] * rank[i];
      }
      nextRank[j] = alpha * sum + (1 - alpha) * teleport[j];
    }

    let l1 = 0.0;
    for (let i = 0; i < n; i += 1) {
      l1 += Math.abs(nextRank[i] - rank[i]);
    }
    rank = nextRank;
    if (l1 < 1e-10) {
      break;
    }
  }

  if (queryIndex >= 0 && queryIndex < n) {
    rank[queryIndex] = -Infinity;
  }

  if (excludeSameBase && queryIndex >= 0 && queryIndex < songs.length) {
    const queryTitle = songs[queryIndex] && songs[queryIndex].Title;
    const queryBase = baseTitle(queryTitle).toLowerCase();
    if (queryBase) {
      for (let i = 0; i < Math.min(n, songs.length); i += 1) {
        if (i === queryIndex) {
          continue;
        }
        const title = songs[i] && typeof songs[i].Title === "string" ? songs[i].Title : "";
        if (title.trim().toLowerCase().startsWith(queryBase)) {
          rank[i] = 0.0;
        }
      }
    }
  }

  for (let i = 0; i < songs.length; i += 1) {
    songs[i].pprScore = i < rank.length ? rank[i] : 0.0;
  }

  songs.sort((a, b) => b.pprScore - a.pprScore);
  return songs;
}

function filterSongs(songs, filters) {
  if (!Array.isArray(songs)) {
    return [];
  }

  const cfg = filters && typeof filters === "object" ? filters : {};
  const vocalRanges = Array.isArray(cfg.vocalRanges) ? cfg.vocalRanges : [];
  const classes = Array.isArray(cfg.classes) ? cfg.classes : [];
  const languages = Array.isArray(cfg.languages) ? cfg.languages : [];

  let filtered = songs.slice();

  if (vocalRanges.length > 0) {
    const vocalSet = new Set(vocalRanges);
    filtered = filtered.filter((song) => {
      const songRange = song && typeof song.VocalRange === "string" ? song.VocalRange : "";
      return vocalSet.has(songRange) || songRange === "Vocal All";
    });
  }

  if (vocalRanges.length === 1) {
    const selectedRange = vocalRanges[0];
    const bounds = RANGE_MIDI_BOUNDS[selectedRange];
    if (bounds) {
      const transposeForMen = vocalRanges.some((range) => MALE_RANGES.includes(String(range).trim().toLowerCase()));

      filtered = filtered.filter((song) => {
        const lowestRaw = Number(song && song.LowestNote_MIDI);
        const highestRaw = Number(song && song.HighestNote_MIDI);
        if (!Number.isFinite(lowestRaw) || !Number.isFinite(highestRaw)) {
          return false;
        }

        let lowest = lowestRaw;
        let highest = highestRaw;
        const songRange = song && typeof song.VocalRange === "string" ? song.VocalRange.trim() : "";
        if (transposeForMen && songRange === "Vocal All") {
          lowest -= 12;
          highest -= 12;
        }

        return lowest <= bounds.high && highest >= bounds.low;
      });
    }
  }

  if (classes.length > 0) {
    const classSet = new Set(classes);
    filtered = filtered.filter((song) => classSet.has(song && song.Class));
  }

  if (languages.length > 0) {
    const languageSet = new Set(languages.map((lang) => normalizeLanguageValue(lang)));
    filtered = filtered.filter((song) => {
      const songLang = normalizeLanguageValue(song && song.Language != null ? song.Language : "");
      return languageSet.has(songLang);
    });
  }

  return filtered;
}

function parseNoteInput(str) {
  if (str == null) {
    return null;
  }
  if (typeof str !== "string") {
    return null;
  }
  if (!str.trim()) {
    return null;
  }

  const midi = noteToMidi(str);
  return midi == null ? null : midi;
}

function scoreRangeMatch(songs, userLowMidi, userHighMidi, isMaleRange) {
  if (!Array.isArray(songs)) {
    return songs;
  }

  const low = Number(userLowMidi);
  const high = Number(userHighMidi);
  if (!Number.isFinite(low) || !Number.isFinite(high)) {
    for (let i = 0; i < songs.length; i += 1) {
      if (songs[i] && typeof songs[i] === "object") {
        songs[i].rangeMatchScore = 0;
      }
    }
    return songs;
  }

  const userLow = Math.min(low, high);
  const userHigh = Math.max(low, high);
  const userSpan = Math.max(1, userHigh - userLow);

  for (let i = 0; i < songs.length; i += 1) {
    const song = songs[i];
    if (!song || typeof song !== "object") {
      continue;
    }

    let songLow = Number(song.LowestNote_MIDI);
    let songHigh = Number(song.HighestNote_MIDI);
    if (!Number.isFinite(songLow) || !Number.isFinite(songHigh)) {
      song.rangeMatchScore = 0;
      continue;
    }

    if (isMaleRange && song.VocalRange === "Vocal All") {
      songLow -= 12;
      songHigh -= 12;
    }

    const overlap = Math.max(0, Math.min(songHigh, userHigh) - Math.max(songLow, userLow));
    let score = Math.round((overlap / userSpan) * 100);

    if (songLow >= userLow && songHigh <= userHigh) {
      score = 100;
    }

    if (score < 0) {
      score = 0;
    } else if (score > 100) {
      score = 100;
    }

    song.rangeMatchScore = score;
  }

  return songs;
}

function hasMissingData(song) {
  if (!song || typeof song !== "object") {
    return true;
  }

  const lowestRaw = String(song["Lowest Note"] == null ? "" : song["Lowest Note"]).trim();
  const highestRaw = String(song["Highest Note"] == null ? "" : song["Highest Note"]).trim();
  const runtimeRaw = String(song["Runtime of Song"] == null ? "" : song["Runtime of Song"]).trim();

  const missingNote = !lowestRaw || !highestRaw || lowestRaw.toUpperCase() === "N/A" || highestRaw.toUpperCase() === "N/A";
  const missingRuntime = !runtimeRaw || runtimeRaw.toUpperCase() === "N/A";

  if (missingNote || missingRuntime) {
    return true;
  }

  const musicNotes = String(song.Music_Notes == null ? "" : song.Music_Notes).trim().toLowerCase();
  if (musicNotes && musicNotes !== "n/a") {
    return true;
  }

  return false;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function shiftDisplayNoteDownOctave(noteStr) {
  if (typeof noteStr !== "string") {
    return "";
  }
  const cleaned = noteStr.trim();
  const match = cleaned.match(/^([A-Ga-g][b#]?)(\d+)$/);
  if (!match) {
    return cleaned;
  }
  const pitch = match[1];
  const octave = Number.parseInt(match[2], 10);
  if (!Number.isFinite(octave)) {
    return cleaned;
  }
  const normalizedPitch =
    pitch.length > 1 ? pitch[0].toUpperCase() + pitch.slice(1).toLowerCase() : pitch.toUpperCase();
  return `${normalizedPitch}${octave - 1}`;
}

function renderSongTable(songs, containerId, options) {
  const container = typeof document !== "undefined" ? document.getElementById(containerId) : null;
  if (!container) {
    return;
  }

  const cfg = options && typeof options === "object" ? options : {};
  const topKRaw = Number(cfg.topK);
  const topK = Number.isFinite(topKRaw) ? Math.max(0, Math.floor(topKRaw)) : (Array.isArray(songs) ? songs.length : 0);
  const showSimilarity = Boolean(cfg.showSimilarity);
  const showRangeMatch = Boolean(cfg.showRangeMatch);
  const isMaleRange = Boolean(cfg.isMaleRange);
  const selectedCode = cfg.selectedSongCode == null ? null : String(cfg.selectedSongCode);

  const source = Array.isArray(songs) ? songs : [];
  const rows = source.slice(0, topK);

  const headers = [
    "",
    "#",
    "Title",
    "Composer",
    "VocalRange",
    "Class",
    "Language",
    "Genre",
    "Era",
    "Note Range",
    "Runtime",
  ];
  if (showSimilarity) {
    headers.push("PPR Score");
  }
  if (showRangeMatch) {
    headers.push("Range Match");
  }

  let html = "";
  html += '<table style="width:100%;border-collapse:collapse;border:1px solid var(--accent, #ff6a5f);background:var(--bg-input, #2b3242);color:var(--text-main, #eef3ff);font-size:14px;">';
  html += "<thead><tr>";
  for (let i = 0; i < headers.length; i += 1) {
    html += `<th style="text-align:left;padding:8px 10px;border:1px solid var(--accent, #ff6a5f);">${escapeHtml(headers[i])}</th>`;
  }
  html += "</tr></thead><tbody>";

  for (let i = 0; i < rows.length; i += 1) {
    const song = rows[i] && typeof rows[i] === "object" ? rows[i] : {};

    const lowestRaw = String(song["Lowest Note"] == null ? "" : song["Lowest Note"]).trim();
    const highestRaw = String(song["Highest Note"] == null ? "" : song["Highest Note"]).trim();
    const lowestMissing = !lowestRaw || lowestRaw.toUpperCase() === "N/A";
    const highestMissing = !highestRaw || highestRaw.toUpperCase() === "N/A";

    let noteRange = "Missing";
    if (!lowestMissing && !highestMissing) {
      const transpose = isMaleRange && String(song.VocalRange || "").trim() === "Vocal All";
      const lowestLabel = transpose ? shiftDisplayNoteDownOctave(lowestRaw) : lowestRaw;
      const highestLabel = transpose ? shiftDisplayNoteDownOctave(highestRaw) : highestRaw;
      noteRange = `${lowestLabel} - ${highestLabel}`;
    }

    const runtimeRaw = String(song["Runtime of Song"] == null ? "" : song["Runtime of Song"]).trim();
    const runtime = !runtimeRaw || runtimeRaw.toUpperCase() === "N/A" ? "Missing" : runtimeRaw;

    const songCode = song.Song_Code == null ? "" : String(song.Song_Code);
    const isSelected = selectedCode != null && songCode === selectedCode;

    const radioHtml = isSelected
      ? '<span style="display:inline-block;width:18px;height:18px;border-radius:50%;border:2px solid var(--accent, #ff6a5f);background:var(--accent, #ff6a5f);vertical-align:middle;"><span style="display:block;width:8px;height:8px;border-radius:50%;background:#fff;margin:3px auto;"></span></span>'
      : '<span style="display:inline-block;width:18px;height:18px;border-radius:50%;border:2px solid rgba(255,255,255,0.4);background:transparent;vertical-align:middle;"></span>';

    const cells = [
      radioHtml,
      String(i + 1),
      String(song.Title == null ? "" : song.Title),
      String(song.Composer == null ? "" : song.Composer),
      String(song.VocalRange == null ? "" : song.VocalRange),
      String(song.Class == null ? "" : song.Class),
      String(song.Language == null ? "" : song.Language),
      String(song.Genre == null ? "" : song.Genre),
      String(song.Era == null ? "" : song.Era),
      noteRange,
      runtime,
    ];

    if (showSimilarity) {
      const ppr = Number(song.pprScore);
      cells.push(Number.isFinite(ppr) ? ppr.toFixed(4) : String(song.pprScore == null ? "" : song.pprScore));
    }
    if (showRangeMatch) {
      const rangeScore = Number(song.rangeMatchScore);
      const displayScore = Number.isFinite(rangeScore) ? Math.round(rangeScore) : 0;
      cells.push(`${displayScore}%`);
    }

    const rowBg = i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.04)";
    const selectedBg = isSelected ? "rgba(255, 106, 95, 0.35)" : rowBg;
    const borderStyle = isSelected ? "2px solid var(--accent, #ff6a5f)" : "1px solid rgba(255,255,255,0.2)";
    html += `<tr data-row-index="${i}" data-song-code="${escapeHtml(songCode)}" style="background:${selectedBg};cursor:pointer;border:${borderStyle};transition:all 0.2s;">`;
    for (let j = 0; j < cells.length; j += 1) {
      if (j === 0) {
        html += `<td style="padding:8px 10px;border:1px solid rgba(255,255,255,0.2);text-align:center;width:36px;">${cells[j]}</td>`;
      } else {
        html += `<td style="padding:8px 10px;border:1px solid rgba(255,255,255,0.2);">${escapeHtml(cells[j])}</td>`;
      }
    }
    html += "</tr>";
  }

  html += "</tbody></table>";
  container.innerHTML = html;
}

function buildSongDisplayLabel(song) {
  if (!song || typeof song !== "object") {
    return "";
  }
  const title = String(song.Title || "");
  const vocalRange = String(song.VocalRange || "");
  const cls = String(song.Class || "");
  const language = normalizeLanguageValue(song.Language);
  return `${title} | ${vocalRange} | Class ${cls} | ${language}`;
}

function resolveReferenceSong(referenceText, referenceItems, songCodeToIndex) {
  const raw = String(referenceText == null ? "" : referenceText).trim();
  if (!raw) {
    return null;
  }

  if (songCodeToIndex && songCodeToIndex.has(raw)) {
    return { code: raw, index: songCodeToIndex.get(raw) };
  }

  const lower = raw.toLowerCase();
  const exact = referenceItems.find((item) => item.label.toLowerCase() === lower);
  if (exact) {
    return exact;
  }

  const startsWith = referenceItems.find((item) => item.label.toLowerCase().startsWith(lower));
  if (startsWith) {
    return startsWith;
  }

  const includes = referenceItems.find((item) => item.label.toLowerCase().includes(lower));
  if (includes) {
    return includes;
  }

  return null;
}

function filterReferenceItems(query, referenceItems) {
  const needle = String(query == null ? "" : query).trim().toLowerCase();
  if (!needle) {
    return referenceItems;
  }

  return referenceItems.filter((item) => item.label.toLowerCase().includes(needle));
}

function normalizeLanguageValue(value) {
  const raw = String(value == null ? "" : value).trim();
  if (!raw) {
    return "";
  }

  const cleaned = raw.replace(/[?]+$/g, "").trim();
  const lower = cleaned.toLowerCase();
  for (let i = 0; i < LANGUAGE_OPTIONS.length; i += 1) {
    if (lower === LANGUAGE_OPTIONS[i].toLowerCase()) {
      return LANGUAGE_OPTIONS[i];
    }
  }

  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1).toLowerCase();
}

function readQueryParams() {
  if (typeof window === "undefined" || !window.location) {
    return { studentId: null, prevSongCode: null };
  }

  const params = new URLSearchParams(window.location.search || "");
  const studentIdRaw = params.get("StudentID");
  const songCodeRaw = params.get("song-code");

  return {
    studentId: studentIdRaw == null || studentIdRaw === "" ? null : studentIdRaw,
    prevSongCode: songCodeRaw == null || songCodeRaw === "" ? null : songCodeRaw,
  };
}

function redirectToAvesChoir(studentId, songCode) {
  const sid = encodeURIComponent(studentId == null ? "" : String(studentId));
  const sc = encodeURIComponent(songCode == null ? "" : String(songCode));
  window.location.href = `https://aveschoir.org/Vocal-Solo-Event?StudentID=${sid}&song-code=${sc}`;
}

function getSelectedValues(selectEl) {
  if (!selectEl || !selectEl.options) {
    return [];
  }
  const values = [];
  for (let i = 0; i < selectEl.options.length; i += 1) {
    const opt = selectEl.options[i];
    if (opt.selected) {
      values.push(opt.value);
    }
  }
  return values;
}

function setSelectOptions(selectEl, values) {
  if (!selectEl) {
    return;
  }
  selectEl.innerHTML = "";
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = true;
    selectEl.appendChild(option);
  }
}

function setSingleSelectOptions(selectEl, values, defaultIndex = 0) {
  if (!selectEl) {
    return;
  }

  selectEl.innerHTML = "";
  for (let i = 0; i < values.length; i += 1) {
    const option = document.createElement("option");
    option.value = values[i];
    option.textContent = values[i];
    option.selected = i === defaultIndex;
    selectEl.appendChild(option);
  }
  if (values.length > 0) {
    selectEl.value = values[Math.min(defaultIndex, values.length - 1)];
  }
}

function setTab(tabId) {
  const panes = document.querySelectorAll(".tab-pane");
  const buttons = document.querySelectorAll(".tab-btn");
  for (let i = 0; i < panes.length; i += 1) {
    panes[i].classList.toggle("active", panes[i].id === tabId);
  }
  for (let i = 0; i < buttons.length; i += 1) {
    buttons[i].classList.toggle("active", buttons[i].dataset.tab === tabId);
  }
}

function renderSimpleBarList(containerId, counts) {
  const container = document.getElementById(containerId);
  if (!container) {
    return;
  }

  const entries = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  const maxCount = entries.length ? Math.max(...entries.map(([, count]) => count)) : 1;
  container.innerHTML = entries
    .map(([label, count]) => {
      const width = Math.max(6, Math.round((count / maxCount) * 100));
      return `
        <div style="display:flex;align-items:center;gap:10px;margin:6px 0;">
          <div style="min-width:140px;color:var(--text-main);font-size:.82rem;">${escapeHtml(label)}</div>
          <div style="flex:1;height:14px;background:rgba(255,255,255,.08);border-radius:999px;overflow:hidden;">
            <div style="height:100%;width:${width}%;background:var(--accent);"></div>
          </div>
          <div style="min-width:32px;text-align:right;font-family:monospace;color:var(--text-main);">${count}</div>
        </div>`;
    })
    .join("");
}

async function init() {
  if (typeof document === "undefined") {
    return;
  }

  const { studentId, prevSongCode } = readQueryParams();
  currentStudentId = studentId;
  currentPrevSongCode = prevSongCode;

  const vocalSelect = document.getElementById("filter-vocal-range");
  const classSelect = document.getElementById("filter-class");
  const langSelect = document.getElementById("filter-language");
  const lowNoteInput = document.getElementById("input-low-note");
  const highNoteInput = document.getElementById("input-high-note");
  const toggleSimilarity = document.getElementById("toggle-similarity");
  const referenceInput = document.getElementById("select-reference");
  const referenceSuggestions = document.getElementById("reference-suggestions");
  const excludeTranspositions = document.getElementById("exclude-transpositions");
  const topKSlider = document.getElementById("slider-topk");
  const resultsContainer = document.getElementById("results-table");
  const missingResultsContainer = document.getElementById("missing-results-table");
  const submitButton = document.getElementById("btn-submit");

  const featureCheckboxes = [
    { id: "feature-vocalrange", name: "VocalRange" },
    { id: "feature-class", name: "Class" },
    { id: "feature-language", name: "Language" },
    { id: "feature-genre", name: "Genre" },
    { id: "feature-era", name: "Era" },
    { id: "feature-rangespan", name: "RangeSpan" },
    { id: "feature-runtime", name: "Runtime" },
  ];

  const songCodeToIndex = new Map();
  const referenceItems = [];
  let suggestionTimeout = null;

  /* ── NEW: submit-button state helper ── */
  function updateSubmitButton() {
    if (!submitButton) {
      return;
    }
    if (selectedSongCode) {
      submitButton.disabled = false;
      submitButton.style.opacity = "1";
      submitButton.style.cursor = "pointer";
      const match = allSongs.find(
        (s) => s && String(s.Song_Code) === selectedSongCode
      );
      submitButton.textContent = match
        ? `Submit: ${String(match.Title || "").substring(0, 50)}`
        : "Submit Selected Song";
    } else {
      submitButton.disabled = true;
      submitButton.style.opacity = "0.5";
      submitButton.style.cursor = "not-allowed";
      submitButton.textContent = "Click a song row to select";
    }
  }
  /* ── END NEW ── */

  try {
    const resp = await fetch("VSAE_Data_Final.csv");
    if (!resp.ok) {
      throw new Error(`Failed to fetch CSV (${resp.status})`);
    }
    const csvText = await resp.text();
    allSongs = engineerFeatures(loadVSAEData(csvText));
    songCodeToIndex.clear();
    referenceItems.length = 0;
    for (let i = 0; i < allSongs.length; i += 1) {
      const code = allSongs[i] && allSongs[i].Song_Code != null ? String(allSongs[i].Song_Code) : String(i + 1);
      songCodeToIndex.set(code, i);
      referenceItems.push({
        code,
        index: i,
        label: buildSongDisplayLabel(allSongs[i]),
      });
    }
  } catch (err) {
    const loadingEl = document.getElementById("loading");
    const appEl = document.getElementById("app");
    const errorBox = document.getElementById("error-box");
    if (loadingEl) {
      loadingEl.style.display = "none";
    }
    if (appEl) {
      appEl.style.display = "none";
    }
    if (errorBox) {
      errorBox.style.display = "block";
      errorBox.textContent = `Could not load data: ${err.message}`;
    }
    if (resultsContainer) {
      resultsContainer.textContent = `Could not load data: ${err.message}`;
    }
    return;
  }

  const loadingEl = document.getElementById("loading");
  const appEl = document.getElementById("app");
  if (loadingEl) {
    loadingEl.style.display = "none";
  }
  if (appEl) {
    appEl.style.display = "block";
  }

  const totalStat = document.getElementById("stat-total");
  const rangesStat = document.getElementById("stat-ranges");
  const langsStat = document.getElementById("stat-langs");
  const composersStat = document.getElementById("stat-composers");
  if (totalStat) {
    totalStat.textContent = String(allSongs.length);
  }
  if (rangesStat) {
    rangesStat.textContent = String(new Set(allSongs.map((song) => song.VocalRange)).size);
  }
  if (langsStat) {
    langsStat.textContent = String(
      new Set(allSongs.map((song) => normalizeLanguageValue(song.Language)).filter(Boolean)).size
    );
  }
  if (composersStat) {
    composersStat.textContent = String(new Set(allSongs.map((song) => song.Composer)).size);
  }

  renderSimpleBarList(
    "overview-vocalrange",
    new Map(Object.entries(
      allSongs.reduce((acc, song) => {
        const key = String(song.VocalRange || "Unknown");
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }, {})
    ))
  );
  renderSimpleBarList(
    "overview-class",
    new Map(Object.entries(
      allSongs.reduce((acc, song) => {
        const key = String(song.Class || "Unknown");
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }, {})
    ))
  );
  renderSimpleBarList(
    "overview-language",
    new Map(Object.entries(
      allSongs.reduce((acc, song) => {
        const key = normalizeLanguageValue(song.Language) || "Unknown";
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }, {})
    ))
  );
  renderSimpleBarList(
    "overview-era",
    new Map(Object.entries(
      allSongs.reduce((acc, song) => {
        const key = String(song.Era || "Unknown");
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }, {})
    ))
  );

  referenceItems.sort((a, b) => a.label.localeCompare(b.label));

  const uniqueSorted = (arr) => Array.from(new Set(arr.filter((v) => String(v || "").trim() !== ""))).sort();
  setSingleSelectOptions(vocalSelect, VOICE_PART_OPTIONS, 0);
  setSelectOptions(classSelect, uniqueSorted(allSongs.map((s) => s.Class)));
  setSelectOptions(langSelect, LANGUAGE_OPTIONS.filter((lang) => allSongs.some((song) => normalizeLanguageValue(song.Language) === lang)));

  if (referenceInput) {
    if (currentPrevSongCode != null && songCodeToIndex.has(String(currentPrevSongCode))) {
      const refIdx = songCodeToIndex.get(String(currentPrevSongCode));
      referenceInput.value = buildSongDisplayLabel(allSongs[refIdx]);
    } else if (referenceItems.length > 0) {
      referenceInput.value = referenceItems[0].label;
    }
  }

  const tabButtons = document.querySelectorAll(".tab-btn");
  for (let i = 0; i < tabButtons.length; i += 1) {
    tabButtons[i].addEventListener("click", () => setTab(tabButtons[i].dataset.tab));
  }

  function hideReferenceSuggestions() {
    if (referenceSuggestions) {
      referenceSuggestions.classList.remove("on");
      referenceSuggestions.innerHTML = "";
    }
  }

  function showReferenceSuggestions() {
    if (!referenceSuggestions || !referenceInput) {
      return;
    }

    const matches = filterReferenceItems(referenceInput.value, referenceItems).slice(0, 30);
    referenceSuggestions.innerHTML = "";
    if (!matches.length) {
      hideReferenceSuggestions();
      return;
    }

    for (let i = 0; i < matches.length; i += 1) {
      const item = matches[i];
      const button = document.createElement("button");
      button.type = "button";
      button.className = "suggestion-item";
      button.textContent = item.label;
      button.dataset.code = item.code;
      button.dataset.label = item.label;
      button.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        referenceInput.value = item.label;
        hideReferenceSuggestions();
        rerunPipeline();
      });
      referenceSuggestions.appendChild(button);
    }

    referenceSuggestions.classList.add("on");
  }

  function rerunPipeline() {
    const vocalRanges = getSelectedValues(vocalSelect);
    const classes = getSelectedValues(classSelect);
    const languages = getSelectedValues(langSelect).map((value) => normalizeLanguageValue(value));

    const isMaleRange = vocalRanges.some((range) => MALE_RANGES.includes(String(range).trim().toLowerCase()));

    let working = filterSongs(allSongs, { vocalRanges, classes, languages });

    let similarityActive = false;
    const similarityEnabled = Boolean(toggleSimilarity && toggleSimilarity.checked);
    const referenceMatch = resolveReferenceSong(referenceInput ? referenceInput.value : "", referenceItems, songCodeToIndex);
    if (similarityEnabled && referenceMatch) {
      const queryAllIndex = referenceMatch.index;
      const checkedFeatures = [];
      for (let i = 0; i < featureCheckboxes.length; i += 1) {
        const box = document.getElementById(featureCheckboxes[i].id);
        if (box && box.checked) {
          checkedFeatures.push(featureCheckboxes[i].name);
        }
      }

      if (checkedFeatures.length > 0 && Number.isFinite(queryAllIndex) && queryAllIndex >= 0) {
        const featureMatrix = buildFeatureMatrix(allSongs, checkedFeatures);
        const rankedAllSongs = personalizedPageRank(
          featureMatrix,
          queryAllIndex,
          allSongs.slice(),
          Boolean(excludeTranspositions && excludeTranspositions.checked)
        );
        const pprByCode = new Map();
        for (let i = 0; i < rankedAllSongs.length; i += 1) {
          const song = rankedAllSongs[i];
          const code = song && song.Song_Code != null ? String(song.Song_Code) : String(i + 1);
          pprByCode.set(code, Number(song.pprScore) || 0);
        }
        working = working.map((song) => {
          const code = song && song.Song_Code != null ? String(song.Song_Code) : "";
          song.pprScore = pprByCode.has(code) ? pprByCode.get(code) : 0;
          return song;
        });
        similarityActive = true;
      }
    }

    const lowMidi = parseNoteInput(lowNoteInput ? lowNoteInput.value : null);
    const highMidi = parseNoteInput(highNoteInput ? highNoteInput.value : null);
    const rangeActive = lowMidi != null && highMidi != null;
    if (rangeActive) {
      scoreRangeMatch(working, lowMidi, highMidi, isMaleRange);
    }

    const completeSongs = [];
    const missingSongs = [];
    for (let i = 0; i < working.length; i += 1) {
      const song = working[i];
      if (hasMissingData(song)) {
        missingSongs.push(song);
      } else {
        completeSongs.push(song);
      }
    }

    if (rangeActive) {
      completeSongs.sort((a, b) => Number(b.rangeMatchScore || 0) - Number(a.rangeMatchScore || 0));
      missingSongs.sort((a, b) => Number(b.rangeMatchScore || 0) - Number(a.rangeMatchScore || 0));
    } else if (similarityActive) {
      completeSongs.sort((a, b) => Number(b.pprScore || 0) - Number(a.pprScore || 0));
      missingSongs.sort((a, b) => Number(b.pprScore || 0) - Number(a.pprScore || 0));
    }

    const topKValue = Number.parseInt(topKSlider && topKSlider.value ? topKSlider.value : "10", 10);
    renderSongTable(completeSongs, "results-table", {
      topK: Number.isFinite(topKValue) ? topKValue : 10,
      showSimilarity: similarityActive,
      showRangeMatch: rangeActive,
      isMaleRange,
      selectedSongCode,
    });

    renderSongTable(missingSongs, "missing-results-table", {
      topK: Number.isFinite(topKValue) ? topKValue : 10,
      showSimilarity: similarityActive,
      showRangeMatch: rangeActive,
      isMaleRange,
      selectedSongCode,
    });

    if (resultsContainer) {
      const completeCountEl = document.getElementById("complete-count");
      if (completeCountEl) {
        completeCountEl.textContent = `${completeSongs.length} songs`;
      }
    }
    if (missingResultsContainer) {
      const missingCountEl = document.getElementById("missing-count");
      if (missingCountEl) {
        missingCountEl.textContent = `${missingSongs.length} songs`;
      }
    }

    const wireSelection = (container) => {
      if (!container) {
        return;
      }
      const clickableRows = container.querySelectorAll("tbody tr");
      for (let i = 0; i < clickableRows.length; i += 1) {
        clickableRows[i].addEventListener("click", () => {
          const code = clickableRows[i].getAttribute("data-song-code") || "";
          selectedSongCode = code || null;
          rerunPipeline();
        });
      }
    };

    wireSelection(resultsContainer);
    wireSelection(missingResultsContainer);

    /* ── NEW: keep submit button in sync after every render ── */
    updateSubmitButton();
  }

  const watchedIds = [
    "filter-vocal-range",
    "filter-class",
    "filter-language",
    "input-low-note",
    "input-high-note",
    "toggle-similarity",
    "select-reference",
    "exclude-transpositions",
    "feature-vocalrange",
    "feature-class",
    "feature-language",
    "feature-genre",
    "feature-era",
    "feature-rangespan",
    "feature-runtime",
    "slider-topk",
  ];

  for (let i = 0; i < watchedIds.length; i += 1) {
    const el = document.getElementById(watchedIds[i]);
    if (!el) {
      continue;
    }
    const eventName = el.tagName === "INPUT" && String(el.type).toLowerCase() === "text" ? "input" : "change";
    if (el === referenceInput) {
      el.addEventListener("input", () => {
        showReferenceSuggestions();
        rerunPipeline();
      });
      el.addEventListener("focus", showReferenceSuggestions);
      el.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          hideReferenceSuggestions();
        }
      });
      el.addEventListener("blur", () => {
        suggestionTimeout = window.setTimeout(() => {
          hideReferenceSuggestions();
        }, 150);
      });
      continue;
    }
    el.addEventListener(eventName, rerunPipeline);
  }

  if (referenceSuggestions) {
    referenceSuggestions.addEventListener("pointerdown", () => {
      if (suggestionTimeout != null) {
        window.clearTimeout(suggestionTimeout);
        suggestionTimeout = null;
      }
    });
  }

  /* ── CHANGED: submit button with user feedback ── */
  if (submitButton) {
    submitButton.addEventListener("click", () => {
      if (!selectedSongCode) {
        alert("Please select a song first by clicking on a row in the table.");
        return;
      }
      redirectToAvesChoir(currentStudentId, selectedSongCode);
    });
    updateSubmitButton();
  }

  if (referenceInput && currentPrevSongCode) {
    referenceInput.dataset.prevSongCode = currentPrevSongCode;
  }

  if (referenceInput) {
    showReferenceSuggestions();
  }

  rerunPipeline();
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    loadVSAEData,
    noteToMidi,
    runtimeToSeconds,
    engineerFeatures,
    buildFeatureMatrix,
    personalizedPageRank,
    filterSongs,
    parseNoteInput,
    scoreRangeMatch,
    hasMissingData,
    renderSongTable,
    readQueryParams,
    redirectToAvesChoir,
    init,
  };
}
