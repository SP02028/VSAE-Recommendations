"""
app.py  —  VSAE Song Selector (Streamlit)
-----------------------------------------
Python version of the song selector deployed on aveschoir.org
(index.html + self_contained_app.js). The UI, copy and recommendation
behaviour mirror the JS app so the two can be compared side by side:

  - Aves Choir look (theme in .streamlit/config.toml, fonts and header below)
  - StudentID / PreviousSongCode / PreviousVocalRange URL parameters
  - Personalized PageRank similarity with a Why column (vsae_contentbased.py)
  - Vocal range fit, one row per song, best-fitting key
  - Submit / Cancel back to the registration form
"""

import os
import re
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

from vsae_contentbased import (
    TRANSPOSITION_RE,
    VOICE_LADDER,
    get_recommendation_scores,
    scale_to_best,
)
from vsae_data import RANGE_MIDI_DEFAULTS, load_and_engineer, note_to_midi

REGISTRATION_URL = "https://aveschoir.org/Vocal-Solo-Event"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VSAE Song Selector | Aves Choir", layout="wide")

# Colours live in .streamlit/config.toml; fonts and the header band follow the
# Aves Choir Website Style Guide (Poppins headings, Roboto body, green + gold).
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp, .stMarkdown, button, input, textarea, select { font-family: Roboto, Arial, sans-serif; }
    h1, h2, h3, h4 { font-family: Poppins, Arial, sans-serif !important; color: #111111; }
    .block-container { max-width: 1140px; padding-top: 1.5rem; }
    .site-band { background: #03553E; border-bottom: 4px solid #FFD200; color: #FFFFFF;
                 padding: 32px 24px; border-radius: 6px; margin-bottom: 24px; }
    .site-band h1 { color: #FFFFFF !important; font-size: 44px; font-weight: 700; line-height: 1.15; margin: 0; padding: 0; }
    .site-band p { color: rgba(255,255,255,.92); margin: 8px 0 0; max-width: 70ch; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #03553E; font-weight: 700; }
    div[data-baseweb="tab-highlight"] { background-color: #FFD200; height: 3px; }
    @media (max-width: 700px) { .site-band h1 { font-size: 32px; } .site-band { padding: 24px 16px; } }
    </style>
    <div class="site-band">
      <h1>Find your VSAE song</h1>
      <p>Filter by voice type, difficulty and language. You can also find songs like one you already know,
      or match songs to your vocal range.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("VSAE_DATA_PATH", "VSAE_Data_Final.csv")


@st.cache_data
def load_data(path):
    return load_and_engineer(path)


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Data file not found at `{DATA_PATH}`. Set the `VSAE_DATA_PATH` environment variable "
        "or place `VSAE_Data_Final.csv` in the same directory as app.py."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers (mirror self_contained_app.js)
# ══════════════════════════════════════════════════════════════════════════════
def get_param(name: str) -> str:
    value = str(st.query_params.get(name, "") or "").strip()
    return "" if not value or value.upper() == "NA" else value


MALE_RANGES = {"tenor", "baritone", "bass"}


def should_transpose_for_men(selected_ranges: list[str]) -> bool:
    """Vocal All songs drop an octave only when every selected voice is male;
    any female voice (e.g. Alto + Tenor for a contralto) keeps written pitch."""
    return bool(selected_ranges) and all(r.casefold() in MALE_RANGES for r in selected_ranges)


def vocal_range_bounds(vocal_range: str) -> tuple[int, int] | None:
    bounds = RANGE_MIDI_DEFAULTS.get(vocal_range)
    return (bounds['low'], bounds['high']) if bounds else None


def transpose_note_label(note_str: str, octave_shift: int) -> str:
    cleaned = str(note_str).strip()
    if octave_shift == 0:
        return cleaned
    match = re.match(r'^([A-Ga-g][b#]?)(\d+)$', cleaned)
    if not match:
        return cleaned
    pitch, octave = match.group(1), int(match.group(2))
    pitch = pitch[0].upper() + pitch[1:].lower() if len(pitch) > 1 else pitch.upper()
    return f"{pitch}{octave + octave_shift}"


def note_range_label(row: pd.Series, transpose_vocal_all: bool = False) -> str:
    lowest, highest = row.get('Lowest Note'), row.get('Highest Note')
    if pd.isna(lowest) or pd.isna(highest):
        return "Missing"
    if str(lowest).strip().upper() in {"N/A", "", "NAN"} or str(highest).strip().upper() in {"N/A", "", "NAN"}:
        return "Missing"
    shift = -1 if transpose_vocal_all and str(row.get('VocalRange', '')).strip() == 'Vocal All' else 0
    return f"{transpose_note_label(lowest, shift)} - {transpose_note_label(highest, shift)}"


def missing_note_or_runtime(frame: pd.DataFrame) -> pd.Series:
    def blank(col):
        values = frame[col].fillna('').astype(str).str.strip().str.upper()
        return (values == 'N/A') | (values == '')
    return blank('Highest Note') | blank('Lowest Note') | blank('Runtime of Song')


def runtime_label(value) -> str:
    text = str(value if not pd.isna(value) else '').strip()
    return "Missing" if text.upper() in {"N/A", "NAN", ""} else text


# ── Vocal range input ─────────────────────────────────────────────────────────
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
VOICE_MIN, VOICE_MAX = 33, 96   # A1–C7
NARROW_SPAN, WIDE_SPAN = 12, 36


def parse_note_input(note_str: str) -> int | None:
    """Strict: a note name and one octave digit (note_to_midi alone read "C10" as C1)."""
    if not note_str or not isinstance(note_str, str):
        return None
    cleaned = re.sub(r'\s+', '', note_str).replace('♯', '#').replace('♭', 'b')
    if not re.fullmatch(r'[A-Ga-g][#b]?\d', cleaned):
        return None
    return note_to_midi(cleaned)


def midi_to_note(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def validate_vocal_range(low_str: str, high_str: str):
    """Return (low, high, errors, warning). low/high are None unless usable.
    Mirrors validateVocalRange in self_contained_app.js."""
    errors = []
    low, high = parse_note_input(low_str), parse_note_input(high_str)
    for label, text, midi in (("lowest", low_str, low), ("highest", high_str, high)):
        if not text:
            continue
        if midi is None:
            errors.append(f"Your {label} note: use a note name and octave, like B3 or F#5.")
        elif not VOICE_MIN <= midi <= VOICE_MAX:
            errors.append(f"{midi_to_note(midi)} is outside the range of the human voice (A1–C7). "
                          "Check the octave number: middle C is C4.")
    if errors or low is None or high is None:
        return None, None, errors, None
    if low > high:
        return None, None, [f"Your lowest note ({midi_to_note(low)}) is above your highest note "
                            f"({midi_to_note(high)}). Swap them?"], None
    if low == high:
        return None, None, ["Your lowest and highest notes are the same. Enter the full range you can sing."], None
    span = high - low
    warning = None
    if span < NARROW_SPAN:
        warning = (f"That range is under an octave ({span} semitones). "
                   "Most songs need at least an octave, so few will fit.")
    elif span > WIDE_SPAN:
        warning = "That range is over three octaves. Double-check the octave numbers (middle C is C4)."
    return low, high, [], warning


def score_range_fit(frame: pd.DataFrame, low: int, high: int, transpose: bool) -> pd.DataFrame:
    """Semitones each song goes outside the singer's range (0 = fits), plus how
    far its middle sits from the middle of the range. Songs that fit sort first."""
    song_low = frame['LowestNote_MIDI'].astype(float)
    song_high = frame['HighestNote_MIDI'].astype(float)
    if transpose:
        vocal_all = frame['VocalRange'] == 'Vocal All'
        song_low = song_low.where(~vocal_all, song_low - 12)
        song_high = song_high.where(~vocal_all, song_high - 12)
    out = frame.copy()
    out['RangeBelow'] = (low - song_low).clip(lower=0)
    out['RangeAbove'] = (song_high - high).clip(lower=0)
    out['RangeOut'] = out['RangeBelow'] + out['RangeAbove']
    out['CenterDist'] = ((song_low + song_high) / 2 - (low + high) / 2).abs()
    return out


def range_fit_label(row: pd.Series) -> str:
    if pd.isna(row.get('RangeOut')):
        return ""
    if row['RangeOut'] == 0:
        return "Fits"
    parts = []
    if row['RangeBelow']:
        parts.append(f"{int(row['RangeBelow'])} too low")
    if row['RangeAbove']:
        parts.append(f"{int(row['RangeAbove'])} too high")
    return ", ".join(parts) + " (semitones)"


# ── One row per song ──────────────────────────────────────────────────────────
def base_title(title) -> str:
    return TRANSPOSITION_RE.sub('', str(title or '')).strip()


def song_group_key(row: pd.Series) -> str:
    """Same base title and composer (Wayfaring Stranger by Walters vs. Niles
    are different arrangements)."""
    return base_title(row['Title']).lower() + '|' + str(row.get('Composer') or '').strip().lower()


def version_label(row: pd.Series) -> str:
    key = str(row.get('Key') or '').strip()
    if key and key.upper() != 'NAN':
        return key
    match = re.search(r'\(([^)]*)\)\s*$', str(row['Title']))
    return match.group(1) if match else str(row.get('VocalRange') or 'other version')


def collapse_versions(ranked: list[pd.DataFrame], pool: pd.DataFrame, use_range: bool) -> list[pd.DataFrame]:
    """Keep one version per song. With range matching, the version kept is the
    best fit (fewest semitones outside, then best centred); otherwise the
    best-ranked one. Other versions are attached as OtherVersions."""
    keys = pool.apply(song_group_key, axis=1)
    groups = {k: list(idx) for k, idx in keys.groupby(keys).groups.items()}
    rep = {}
    if use_range and 'RangeOut' in pool.columns:
        fit = pool.dropna(subset=['RangeOut']).assign(_key=keys)
        best = fit.sort_values(['RangeOut', 'CenterDist'], kind='stable').drop_duplicates('_key')
        rep = dict(zip(best['_key'], best.index))
    seen, out = set(), []
    for frame in ranked:
        keep = []
        for idx in frame.index:
            k = keys[idx]
            if (k in rep and rep[k] != idx) or k in seen:
                continue
            seen.add(k)
            keep.append(idx)
        kept = frame.loc[keep].copy()
        kept['OtherVersions'] = [[o for o in groups[keys[i]] if o != i] for i in keep]
        out.append(kept)
    return out


def build_table(frame: pd.DataFrame, pool: pd.DataFrame, transpose: bool, top_k: int,
                show_fit: bool, show_similarity: bool) -> pd.DataFrame:
    rows = frame.head(top_k)
    table = pd.DataFrame(index=range(1, len(rows) + 1))
    titles = []
    for _, row in rows.iterrows():
        others = [version_label(pool.loc[o]) for o in row['OtherVersions']]
        titles.append(row['Title'] + (f"  (also in keys: {' · '.join(others)})" if others else ""))
    table['Title'] = titles
    table['Composer'] = rows['Composer'].values
    vr = rows['VocalRange'].fillna('').values
    table['Vocal range'] = [f"{v} (octave lower)" if transpose and v == 'Vocal All' else v for v in vr]
    for col in ['Class', 'Language', 'Genre', 'Era', 'Key']:
        table[col] = rows[col].values
    table['Note range'] = [note_range_label(r, transpose) for _, r in rows.iterrows()]
    table['Runtime'] = [runtime_label(v) for v in rows['Runtime of Song']]
    if show_fit:
        table['Range fit'] = [range_fit_label(r) for _, r in rows.iterrows()]
    if show_similarity:
        table['Similarity'] = rows['SimilarityScore'].round(2).values
        table['Why'] = rows['Why'].values
    return table


# ══════════════════════════════════════════════════════════════════════════════
# URL parameters from the aveschoir.org registration form
# ══════════════════════════════════════════════════════════════════════════════
student_id = get_param("StudentID")
prev_range = next((v for v in VOICE_LADDER if v.casefold() == get_param("PreviousVocalRange").casefold()), None)
prev_code = get_param("PreviousSongCode")
prev_song_idx = None
if prev_code:
    matches = df.index[df['Song_Code'].astype(str).str.strip() == prev_code]
    prev_song_idx = matches[0] if len(matches) else None

tab_find, tab_about = st.tabs(["Find a song", "About the song list"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Find a song
# ══════════════════════════════════════════════════════════════════════════════
with tab_find:
    with st.container(border=True):
        st.subheader("Browse songs")
        st.caption("All options start on. Tap an option to turn it off.")
        col1, col2, col3 = st.columns(3)
        range_options = [v for v in VOICE_LADDER if v in set(df['VocalRange'].dropna())]
        with col1:
            selected_ranges = st.pills("Vocal range", range_options, selection_mode="multi",
                                       default=[prev_range] if prev_range else range_options)
        with col2:
            class_options = sorted(df['Class'].dropna().unique().tolist())
            selected_class = st.pills("Difficulty class", class_options, selection_mode="multi",
                                      default=class_options)
        with col3:
            lang_options = ["English", "French", "German", "Spanish", "Latin", "Italian"]
            selected_lang = st.pills("Language", lang_options, selection_mode="multi", default=lang_options)

    # ── Filters ──
    filtered = df.copy()
    transpose_for_men = should_transpose_for_men(selected_ranges)
    # Vocal All isn't a voice: those songs are included whenever voices are
    # narrowed, kept if they overlap the chosen voices (an octave lower for men).
    if selected_ranges and len(selected_ranges) < len(range_options):
        filtered = filtered[filtered['VocalRange'].isin(selected_ranges + ['Vocal All'])]
        all_bounds = [b for b in (vocal_range_bounds(r) for r in selected_ranges) if b]
        if all_bounds:
            low_bound, high_bound = min(b[0] for b in all_bounds), max(b[1] for b in all_bounds)
            low_midi, high_midi = filtered['LowestNote_MIDI'], filtered['HighestNote_MIDI']
            if transpose_for_men:
                vocal_all_mask = filtered['VocalRange'] == 'Vocal All'
                low_midi = low_midi.where(~vocal_all_mask, low_midi - 12)
                high_midi = high_midi.where(~vocal_all_mask, high_midi - 12)
            filtered = filtered[(low_midi <= high_bound) & (high_midi >= low_bound)]
    if selected_class:
        filtered = filtered[filtered['Class'].isin(selected_class)]
    if selected_lang:
        filtered = filtered[filtered['Language'].isin(selected_lang)]

    # ── Similarity ──
    st.divider()
    st.subheader("Find songs like one you know (optional)")
    st.caption(
        "Pick a reference song and we'll rank the others by how strongly they connect to it through shared "
        "composers, languages, genres and more. Rare connections (the same composer) count more than common ones "
        "(both in English). Difficulty isn't compared, since you choose your class above. The closest match "
        "scores 1.00, and the Why column shows the strongest connections."
    )
    similarity_enabled = False
    use_similarity = st.checkbox("Rank songs by similarity to a reference song", value=prev_song_idx is not None)
    if use_similarity:
        if prev_song_idx is not None:
            st.info(f"We started with the song you sang last time, {df.loc[prev_song_idx, 'Title']}. "
                    "Pick a different reference song, or turn this off to skip it.")
        query_options = sorted(
            ((f"{r.Title} | {r.VocalRange} | Class {r.Class} | {r.Language}", idx)
             for idx, r in df.iterrows() if str(r.Title).strip()),
            key=lambda o: o[0],
        )
        labels = [o[0] for o in query_options]
        default_pos = next((i for i, o in enumerate(query_options) if o[1] == prev_song_idx), 0)
        selected_label = st.selectbox("Reference song", labels, index=default_pos)
        query_idx = query_options[labels.index(selected_label)][1]
        similarity_min = st.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.01)

        st.markdown("#### Compare songs by")
        fc1, fc2 = st.columns(2)
        feature_boxes = [
            ('Composer', "Composer", True), ('VocalRange', "Vocal range", True),
            ('Language', "Language", True), ('Genre', "Genre", True),
            ('Era', "Era / time period", False), ('RangeSpan', "Note range span", False),
            ('Runtime', "Runtime", False),
        ]
        selected_features = []
        for i, (feature, label, default) in enumerate(feature_boxes):
            with (fc1 if i % 2 == 0 else fc2):
                if st.checkbox(label, value=default, key=f"feat_{feature}"):
                    selected_features.append(feature)
        exclude_transpositions = st.checkbox("Leave out other keys of the reference song", value=True)

        if not selected_features:
            st.warning("Please select at least one feature.")
        else:
            try:
                scores, reasons = get_recommendation_scores(
                    df=df, selected_features=selected_features, song_title=df.loc[query_idx, 'Title'],
                    query_index=query_idx, exclude_same_base=exclude_transpositions, with_reasons=True,
                )
                scored = filtered.copy()
                scored['SimilarityScore'] = scores.reindex(scored.index)
                scored = scored[scored['SimilarityScore'].notna()]
                scored['Why'] = reasons.reindex(scored.index).apply(' · '.join)
                # Scale so the best match among the songs the student can see is 1.00
                scored['SimilarityScore'] = scale_to_best(scored['SimilarityScore'])
                scored = scored[scored['SimilarityScore'] >= similarity_min]
                if scored.empty:
                    st.warning("No songs matched the current similarity settings.")
                else:
                    filtered = scored
                    similarity_enabled = True
                    st.caption(f"Showing similarity scores {scored['SimilarityScore'].min():.2f}–"
                               f"{scored['SimilarityScore'].max():.2f} (1.00 = closest match).")
            except Exception as e:  # keep the rest of the page usable
                st.error(f"Similarity scoring error: {e}")

    # ── Vocal range ──
    st.divider()
    st.subheader("Match your vocal range (optional)")
    st.caption("Enter the lowest and highest notes you can sing comfortably. "
               "Songs that fit come first, in the key that suits you best.")
    col_low, col_high = st.columns(2)
    with col_low:
        note_low_input = st.text_input("Your lowest comfortable note", placeholder="e.g. B3")
    with col_high:
        note_high_input = st.text_input("Your highest comfortable note", placeholder="e.g. F5")
    note_low, note_high, range_errors, range_warning = validate_vocal_range(note_low_input, note_high_input)
    for err in range_errors:
        st.error(err)
    if range_warning:
        st.warning(range_warning)
    if not range_errors and note_low is None and (note_low_input or note_high_input):
        st.caption("Enter both notes to sort by range fit.")
    range_active = note_low is not None and note_high is not None

    # ── Rank ──
    pool = filtered.copy()
    missing = missing_note_or_runtime(pool)
    if range_active:
        fit_cols = ['RangeBelow', 'RangeAbove', 'RangeOut', 'CenterDist']
        pool[fit_cols] = np.nan
        pool.loc[~missing, fit_cols] = score_range_fit(
            pool[~missing], note_low, note_high, transpose_for_men
        )[fit_cols].values
    display, missing_songs = pool[~missing].copy(), pool[missing].copy()
    if range_active:
        display['_sim'] = display['SimilarityScore'] if similarity_enabled else 0.0
        display = display.sort_values(['RangeOut', '_sim', 'CenterDist'], ascending=[True, False, True],
                                      na_position='last', kind='stable')
    elif similarity_enabled:
        display = display.sort_values('SimilarityScore', ascending=False, kind='stable')
    if similarity_enabled:
        missing_songs = missing_songs.sort_values('SimilarityScore', ascending=False, kind='stable')
    display, missing_songs = collapse_versions([display, missing_songs], pool, range_active)

    # ── Results ──
    st.divider()
    st.subheader("Recommended songs")
    top_k = st.slider("Number of songs to show", 5, 50, 10)
    if transpose_for_men:
        st.info("**Heads up!** Vocal All music is written in treble clef, so you'll need to sing it an octave "
                "lower than written. The note ranges marked \"octave lower\" below already account for this.")
    if display.empty:
        st.write("No songs match the current filters.")
    else:
        st.dataframe(build_table(display, pool, transpose_for_men, top_k, range_active, similarity_enabled),
                     width='stretch')

    st.markdown("### Also worth a look")
    st.caption("These songs match your filters, but we don't have their note range and/or runtime yet.")
    if missing_songs.empty:
        st.write("None.")
    else:
        st.dataframe(build_table(missing_songs, pool, transpose_for_men, top_k, False, similarity_enabled),
                     width='stretch')

    # ── Choose your song ──
    st.divider()
    with st.container(border=True):
        st.subheader("Choose your song")
        st.caption("Pick a song from the recommendations above. Submitting takes you back to the "
                   "registration form with your song filled in.")
        choices = []  # (label, song code); every version of each recommended song is selectable
        for frame in (display.head(top_k), missing_songs.head(top_k)):
            for idx, row in frame.iterrows():
                for i, v in enumerate([idx] + row['OtherVersions']):
                    song = pool.loc[v]
                    key = f" | Key {song['Key']}" if str(song.get('Key') or '').strip() else ""
                    choices.append((("   ↳ " if i else "") + f"{song['Title']} | {song['VocalRange']} | "
                                    f"Class {song['Class']}{key} | {song['Language']}", str(song['Song_Code'])))
        chosen = st.selectbox("Your song", range(len(choices)), index=None,
                              format_func=lambda i: choices[i][0], placeholder="— Select a song —")
        b1, b2, _ = st.columns([1, 1, 4])
        with b1:
            st.link_button("Cancel", f"{REGISTRATION_URL}?{urlencode({'StudentID': student_id})}")
        with b2:
            submit_url = (f"{REGISTRATION_URL}?{urlencode({'StudentID': student_id, 'song-code': choices[chosen][1]})}"
                          if chosen is not None else REGISTRATION_URL)
            st.link_button("Submit selection", submit_url, type="primary", disabled=chosen is None)

    st.caption("Data collected by Aves Choir members & volunteers · App developed by Shanaya Pal")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — About the song list
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("About the song list")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total songs", len(df))
    col_b.metric("Vocal ranges", df['VocalRange'].nunique())
    col_c.metric("Languages", df['Language'].nunique())
    col_d.metric("Composers", df['Composer'].nunique())

    GREEN = "#03553E"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Songs by vocal range**")
        st.bar_chart(df['VocalRange'].value_counts(), color=GREEN)
    with c2:
        st.markdown("**Songs by difficulty class**")
        st.bar_chart(df['Class'].value_counts().sort_index(), color=GREEN)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Songs by language**")
        st.bar_chart(df['Language'].value_counts(), color=GREEN)
    with c4:
        st.markdown("**Songs by era**")
        st.bar_chart(df['Era'].value_counts(), color=GREEN)

    st.markdown("### Full song list")
    full = df[['Title', 'Composer', 'VocalRange', 'Class', 'Language', 'Genre', 'Era', 'RangeSpan', 'RuntimeSeconds']]
    st.dataframe(full.rename(columns={'VocalRange': 'Vocal range', 'RangeSpan': 'Range span',
                                      'RuntimeSeconds': 'Runtime (s)'}), width='stretch')
    st.caption("Data collected by Aves Choir members & volunteers · App developed by Shanaya Pal")
