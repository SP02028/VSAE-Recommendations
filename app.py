"""
app.py  —  VSAE Song Recommender
---------------------------------
Streamlit app adapted from the original music-recommendation-app.
Structure mirrors the original app.py as closely as possible:
  - Same 3-tab layout concept (Popular → Top by Difficulty, Content-Based stays identical)
  - Spotify data loading replaced with VSAE CSV loading (via vsae_data.py)
  - Content-based feature checkboxes updated to VSAE metrics
  - Added: Repertoire Mode toggle
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re

from vsae_data import load_and_engineer, RANGE_MIDI_DEFAULTS
from vsae_contentbased import get_recommendation_scores, scale_to_best

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VSAE Song Recommender",
    page_icon="🎼",
    layout="wide",
)

# ── Styling only (UI changes) ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
      --bg-main: #020b1b;
      --bg-input: #2b3242;
      --text-main: #eef3ff;
      --text-muted: #bac3d8;
      --line: rgba(152, 168, 198, 0.28);
      --accent: #ff6a5f;
      --accent-soft: rgba(255, 106, 95, 0.26);
    }

    .stApp {
      background:
        radial-gradient(1200px 500px at 12% -10%, #14304d 0%, transparent 48%),
        radial-gradient(1200px 500px at 88% -12%, #2a1f37 0%, transparent 42%),
        var(--bg-main);
      color: var(--text-main);
    }

    [data-testid="stHeader"] {
      background: transparent;
    }

    div[data-baseweb="tab-list"] {
      border-bottom: 1px solid var(--line);
      gap: 0.25rem;
    }

    button[data-baseweb="tab"] {
      color: var(--text-main);
      font-weight: 700;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      background: transparent;
      padding: 0.5rem 0.6rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="slider"] {
      background: var(--bg-input) !important;
      border-color: transparent !important;
    }

    .stTextInput input {
      background: var(--bg-input) !important;
      color: var(--text-main) !important;
      border: 1px solid rgba(255, 106, 95, 0.55) !important;
      border-radius: 0.5rem !important;
    }

    .stCheckbox label,
    [data-baseweb="select"] input,
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stTextInput label,
    .stMarkdown,
    .stCaption {
      color: var(--text-main) !important;
    }

    [data-baseweb="tag"] {
      background: var(--accent-soft) !important;
      border: 1px solid rgba(255, 106, 95, 0.9) !important;
      color: #ffd8d3 !important;
    }

    [data-baseweb="slider"] [role="slider"] {
      background: var(--accent) !important;
      border-color: var(--accent) !important;
    }

    .stDataFrame {
      border: 1px solid var(--line);
      border-radius: 0.75rem;
      overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load data (cached — mirrors setup.py download+cache pattern) ──────────────
DATA_PATH = os.environ.get("VSAE_DATA_PATH", "VSAE_Data_Final.csv")

@st.cache_data
def load_data(path):
    return load_and_engineer(path)

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Data file not found at `{DATA_PATH}`. "
        "Set the `VSAE_DATA_PATH` environment variable or place `VSAE_Data_Final.csv` "
        "in the same directory as app.py."
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "Browse by Difficulty",
    "Dataset Overview",
])


def normalize_vocal_range(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    options = sorted(df['VocalRange'].dropna().astype(str).unique().tolist())
    for option in options:
        if option.casefold() == cleaned.casefold():
            return option
    return None


def vocal_range_bounds(vocal_range: str) -> tuple[int, int] | None:
    bounds = RANGE_MIDI_DEFAULTS.get(vocal_range)
    if not bounds:
        return None
    return bounds['low'], bounds['high']


MALE_RANGES = {"tenor", "baritone", "bass"}


def should_transpose_for_men(selected_ranges: list[str]) -> bool:
    """Vocal All songs drop an octave only when every selected voice is male;
    any female voice (e.g. Alto + Tenor for a contralto) keeps written pitch."""
    return bool(selected_ranges) and all(r.casefold() in MALE_RANGES for r in selected_ranges)


def transpose_note_label(note_str: str, octave_shift: int) -> str:
    if not isinstance(note_str, str) or octave_shift == 0:
        return str(note_str).strip()

    cleaned = note_str.strip()
    match = re.match(r'^([A-Ga-g][b#]?)(\d+)$', cleaned)
    if not match:
        return cleaned

    pitch, octave = match.group(1), int(match.group(2))
    pitch = pitch[0].upper() + pitch[1:].lower() if len(pitch) > 1 else pitch.upper()
    return f"{pitch}{octave + octave_shift}"


def note_range_label(row: pd.Series, transpose_vocal_all: bool = False) -> str:
    lowest_val = row.get('Lowest Note', None)
    highest_val = row.get('Highest Note', None)
    if pd.isna(lowest_val) or pd.isna(highest_val):
        return "Missing"
    is_vocal_all = str(row.get('VocalRange', '')).strip() == 'Vocal All'
    octave_shift = -1 if transpose_vocal_all and is_vocal_all else 0
    lowest = transpose_note_label(str(lowest_val).strip(), octave_shift)
    highest = transpose_note_label(str(highest_val).strip(), octave_shift)
    if lowest.upper() in {"N/A", "", "NAN"} or highest.upper() in {"N/A", "", "NAN"}:
        return "Missing"
    return f"{lowest} - {highest}"


def valid_note_rows(frame: pd.DataFrame) -> pd.Series:
    highest = frame['Highest Note'].fillna('').astype(str).str.strip().str.upper()
    lowest = frame['Lowest Note'].fillna('').astype(str).str.strip().str.upper()
    return (highest != 'N/A') & (lowest != 'N/A') & (highest != '') & (lowest != '')


def missing_note_or_runtime(frame: pd.DataFrame) -> pd.Series:
    highest = frame['Highest Note'].fillna('').astype(str).str.strip().str.upper()
    lowest = frame['Lowest Note'].fillna('').astype(str).str.strip().str.upper()
    runtime = frame['Runtime of Song'].fillna('').astype(str).str.strip().str.upper()
    missing_note = (highest == 'N/A') | (lowest == 'N/A') | (highest == '') | (lowest == '')
    missing_runtime = (runtime == 'N/A') | (runtime == '')
    return missing_note | missing_runtime


def parse_note_input(note_str: str) -> int | None:
    """Parse note string (e.g. 'C4', 'G#5', 'Bb3') and return MIDI number, or None if invalid."""
    from vsae_data import note_to_midi
    if not note_str or not isinstance(note_str, str):
        return None
    # Strict: a note name and one octave digit (note_to_midi alone read "C10" as C1)
    cleaned = re.sub(r'\s+', '', note_str).replace('♯', '#').replace('♭', 'b')
    if not re.fullmatch(r'[A-Ga-g][#b]?\d', cleaned):
        return None
    return note_to_midi(cleaned)


NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
VOICE_MIN, VOICE_MAX = 33, 96   # A1–C7
NARROW_SPAN, WIDE_SPAN = 12, 36


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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Home
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Browse Songs by Vocal Range & Difficulty")
    st.caption(
        "Filter by difficulty, language, and voice type. Optionally find songs similar to a reference song, "
        "or match your vocal range."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        # Vocal All isn't a voice: those songs are included whenever voices are narrowed
        range_options = [v for v in ['Soprano', 'Mezzo Soprano', 'Alto', 'Tenor', 'Baritone', 'Bass']
                         if v in set(df['VocalRange'].dropna())]
        selected_ranges = st.multiselect("Vocal Range", range_options, default=range_options)

    with col2:
        class_options = sorted(df['Class'].dropna().unique().tolist())
        selected_class = st.multiselect(
            "Difficulty Class (Required)",
            class_options,
            default=class_options,
        )

    with col3:
        lang_options = ["English", "French", "German", "Spanish", "Latin", "Italian"]
        selected_lang = st.multiselect("Language", lang_options, default=lang_options)

    # Base filter set
    filtered = df.copy()
    transpose_for_men = should_transpose_for_men(selected_ranges)
    narrowed = bool(selected_ranges) and len(selected_ranges) < len(range_options)
    if narrowed:
        filtered = filtered[filtered['VocalRange'].isin(selected_ranges + ['Vocal All'])]
        all_bounds = [b for b in (vocal_range_bounds(r) for r in selected_ranges) if b]
        if all_bounds:
            low_bound = min(b[0] for b in all_bounds)
            high_bound = max(b[1] for b in all_bounds)
            note_mask = valid_note_rows(filtered)
            filtered = filtered[note_mask]
            low_midi = filtered['LowestNote_MIDI']
            high_midi = filtered['HighestNote_MIDI']
            if transpose_for_men:
                vocal_all_mask = filtered['VocalRange'] == 'Vocal All'
                low_midi = low_midi.where(~vocal_all_mask, low_midi - 12)
                high_midi = high_midi.where(~vocal_all_mask, high_midi - 12)
            filtered = filtered[(low_midi <= high_bound) & (high_midi >= low_bound)]

    if selected_class:
        filtered = filtered[filtered['Class'].isin(selected_class)]

    if selected_lang:
        lang_casefold = filtered['Language'].fillna('').astype(str).str.strip().str.casefold()
        selected_lang_casefold = [lang.casefold() for lang in selected_lang]
        filtered = filtered[lang_casefold.isin(selected_lang_casefold)]

    st.divider()

    st.subheader("Recommend Songs based on Reference Song (Optional)")
    st.markdown(
        "Pick a reference song and we'll rank the others by how closely they connect to it through shared features. "
        "The closest match scores 1.00."
    )

    similarity_enabled = False
    use_similarity = st.checkbox("Rank songs by similarity to a reference song", value=False)

    if use_similarity:
        rec_df = df.copy()
        query_options = []
        for idx, row in rec_df.iterrows():
            title = str(row.get('Title', '')).strip()
            if not title:
                continue
            vr = str(row.get('VocalRange', '—')).strip()
            cls = str(row.get('Class', '—')).strip()
            lang = str(row.get('Language', '—')).strip()
            label = f"{title} | {vr} | Class {cls} | {lang}"
            query_options.append((label, idx, title))
        query_options = sorted(query_options, key=lambda x: x[0])

        if not query_options:
            st.warning("No songs are available for similarity scoring.")
        else:
            selected_label = st.selectbox("Reference song", [opt[0] for opt in query_options])
            selected_query = next(opt for opt in query_options if opt[0] == selected_label)
            selected_song = selected_query[2]
            selected_query_idx = selected_query[1]
            similarity_min = st.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.01)

            st.markdown("#### Similarity Features")
            use_composer = st.checkbox("Composer", value=True)
            use_vocalrange = st.checkbox("Vocal Range", value=True)
            use_language = st.checkbox("Language", value=True)
            use_genre = st.checkbox("Genre", value=True)
            use_era = st.checkbox("Era / Time Period", value=False)
            use_rangespan = st.checkbox("Note Range Span", value=False)
            use_runtime = st.checkbox("Runtime", value=False)
            exclude_transpositions = st.checkbox("Exclude other keys of the same piece", value=True)

            selected_features = []
            if use_composer: selected_features.append('Composer')
            if use_vocalrange: selected_features.append('VocalRange')
            if use_language: selected_features.append('Language')
            if use_genre: selected_features.append('Genre')
            if use_era: selected_features.append('Era')
            if use_rangespan: selected_features.append('RangeSpan')
            if use_runtime: selected_features.append('Runtime')

            if not selected_features:
                st.warning("Please select at least one feature.")
            else:
                try:
                    similarity_series, similarity_reasons = get_recommendation_scores(
                        df=rec_df,
                        selected_features=selected_features,
                        song_title=selected_song,
                        query_index=selected_query_idx,
                        repertoire_mode='similar',
                        exclude_same_base=exclude_transpositions,
                        with_reasons=True,
                    )

                    scored = filtered.copy()
                    scored['SimilarityScore'] = similarity_series.reindex(scored.index)
                    scored = scored[scored['SimilarityScore'].notna()]
                    scored['Why'] = similarity_reasons.reindex(scored.index).apply(' · '.join)
                    # Rescale within the filtered songs so the best visible match is 1.00
                    scored['SimilarityScore'] = scale_to_best(scored['SimilarityScore'])
                    scored = scored[scored['SimilarityScore'] >= similarity_min]
                    if not scored.empty:
                        filtered = scored
                        similarity_enabled = True
                        st.caption(
                            f"Showing similarity scores {scored['SimilarityScore'].min():.2f}–"
                            f"{scored['SimilarityScore'].max():.2f} (1.00 = closest match)."
                        )
                    else:
                        st.warning("No songs matched the current similarity settings.")

                
                except Exception as e:
                    st.error(f"Similarity scoring error: {e}")

    base_filtered = filtered.copy()

    st.divider()

    st.subheader("🎤 Match Your Vocal Range (Optional)")
    use_range_match = True  # applies whenever both notes are entered

    note_low = None
    note_high = None
    if use_range_match:
        col_low, col_high = st.columns(2)
        with col_low:
            note_low_input = st.text_input("Your lowest comfortable note", placeholder="e.g. B3")
        with col_high:
            note_high_input = st.text_input(
                "Your highest comfortable note",
                placeholder="Press Enter to apply",
            )
        note_low, note_high, range_errors, range_warning = validate_vocal_range(note_low_input, note_high_input)
        for err in range_errors:
            st.error(err)
        if range_warning:
            st.warning(range_warning)

    display = filtered.copy()
    display = display[~missing_note_or_runtime(display)].copy()
    if use_range_match:
        display['RangeMatchScore'] = None
        valid_rows = valid_note_rows(display)
        display = display[valid_rows].copy()
        if note_low is not None and note_high is not None:
            low_bound = min(note_low, note_high)
            high_bound = max(note_low, note_high)
            user_span = max(1, high_bound - low_bound)
            song_low = display['LowestNote_MIDI'].values.astype(float)
            song_high = display['HighestNote_MIDI'].values.astype(float)
            if transpose_for_men:
                vocal_all_mask = display['VocalRange'] == 'Vocal All'
                song_low = np.where(vocal_all_mask, song_low - 12, song_low)
                song_high = np.where(vocal_all_mask, song_high - 12, song_high)
            overlap_low = np.maximum(song_low, low_bound)
            overlap_high = np.minimum(song_high, high_bound)
            overlap = np.maximum(0, overlap_high - overlap_low)
            score = (overlap / user_span) * 100.0
            fully_contained = (song_low >= low_bound) & (song_high <= high_bound)
            score = np.where(fully_contained, 100.0, score)
            display['RangeMatchScore'] = np.round(score).astype(int)
        elif not range_errors and (note_low_input or note_high_input):
            st.caption("Enter both notes to sort by range fit.")

        sort_cols = ["RangeMatchScore"]
        if similarity_enabled and 'SimilarityScore' in display.columns:
            sort_cols.append("SimilarityScore")
        display = display.sort_values(by=sort_cols, ascending=False, na_position="last")
    else:
        display['RangeMatchScore'] = None
        if similarity_enabled and 'SimilarityScore' in display.columns:
            display = display.sort_values(by="SimilarityScore", ascending=False, na_position="last")

    st.divider()

    top_k = st.slider("Number of songs to show", 5, 50, 10)
    if transpose_for_men:
        st.info("Heads up! Vocal All music is written in treble clef, so you'll need to sing it an octave lower "
                "than written. The note ranges shown for Vocal All songs already account for this.")
    display['Note Range'] = display.apply(
        lambda row: note_range_label(row, transpose_vocal_all=transpose_for_men),
        axis=1,
    )
    display['Runtime'] = display['Runtime of Song']
    display['Runtime'] = display['Runtime'].fillna("Missing")
    display.loc[display['Runtime'].astype(str).str.strip().str.upper().isin(["N/A", "NAN", ""]), 'Runtime'] = "Missing"
    display_cols = [
        'Title',
        'Composer',
        'VocalRange',
        'Class',
        'Language',
        'Genre',
        'Era',
        'Key',
        'Note Range',
        'Runtime',
    ]
    if similarity_enabled and 'SimilarityScore' in display.columns:
        display['Similarity'] = display['SimilarityScore'].round(2)
        display_cols += ['Similarity', 'Why']
    table = display[display_cols].head(top_k).copy()
    table.index = range(1, len(table) + 1)
    st.dataframe(table, use_container_width=True)

    st.divider()
    st.subheader("Worth looking into but no music and/or tracks")
    st.caption("These are songs we think you may like based on your filters, but they are missing note and/or runtime data.")

    missing_candidates = base_filtered[missing_note_or_runtime(base_filtered)].copy()
    if similarity_enabled and 'SimilarityScore' in missing_candidates.columns:
        missing_candidates = missing_candidates.sort_values(
            by="SimilarityScore",
            ascending=False,
            na_position="last",
        )

    missing_display = missing_candidates.copy()
    missing_display['Note Range'] = missing_display.apply(
        lambda row: note_range_label(row, transpose_vocal_all=transpose_for_men),
        axis=1,
    )
    missing_display['Runtime'] = missing_display['Runtime of Song']
    missing_display['Runtime'] = missing_display['Runtime'].fillna("Missing")
    missing_display.loc[missing_display['Runtime'].astype(str).str.strip().str.upper().isin(["N/A", "NAN", ""]), 'Runtime'] = "Missing"
    missing_cols = [
        'Title',
        'Composer',
        'VocalRange',
        'Class',
        'Language',
        'Genre',
        'Era',
        'Key',
        'Note Range',
        'Runtime',
    ]
    missing_table = missing_display[missing_cols].head(top_k).copy()
    missing_table.index = range(1, len(missing_table) + 1)
    st.dataframe(missing_table, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Overview  (replaces "Predicted Top Hits" from original)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Dataset Overview")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Songs", len(df))
    col_b.metric("Vocal Ranges", df['VocalRange'].nunique())
    col_c.metric("Languages", df['Language'].nunique())
    col_d.metric("Composers", df['Composer'].nunique())

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Songs by Vocal Range**")
        vr_counts = df['VocalRange'].value_counts()
        st.bar_chart(vr_counts)

    with c2:
        st.markdown("**Songs by Difficulty Class**")
        class_counts = df['Class'].value_counts().sort_index()
        st.bar_chart(class_counts)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Songs by Language**")
        lang_counts = df['Language'].value_counts()
        st.bar_chart(lang_counts)

    with c4:
        st.markdown("**Songs by Era**")
        era_counts = df['Era'].value_counts()
        st.bar_chart(era_counts)

    st.divider()
    st.markdown("**Full Dataset**")
    st.dataframe(df[['Title', 'Composer', 'VocalRange', 'Class', 'Language',
                      'Genre', 'Era', 'RangeSpan', 'RuntimeSeconds']],
                 use_container_width=True)
