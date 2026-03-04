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

from vsae_data import load_and_engineer, build_feature_matrix
from vsae_contentbased import get_recommendations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VSAE Song Recommender",
    page_icon="🎼",
    layout="wide",
)

# ── Load data (cached — mirrors setup.py download+cache pattern) ──────────────
DATA_PATH = os.environ.get("VSAE_DATA_PATH", "VSAE_Data.csv")

@st.cache_data
def load_data(path):
    return load_and_engineer(path)

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Data file not found at `{DATA_PATH}`. "
        "Set the `VSAE_DATA_PATH` environment variable or place `VSAE_Data.csv` "
        "in the same directory as app.py."
    )
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎼 VSAE Song Recommender")
st.markdown(
    "Find the right classical vocal repertoire — filter by difficulty, "
    "language, vocal range, or discover songs similar/different to what a student already knows."
)
st.divider()

# ── Tabs (mirrors original 3-section layout) ──────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📋 Browse by Difficulty",
    "🔍 Content-Based Recommendation",
    "📊 Dataset Overview",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Browse by Difficulty  (replaces "Popular Songs" from original)
# Mirrors helper.py genre filter logic, swapped for VocalRange + Class filter
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Browse Songs by Vocal Range & Difficulty")
    st.markdown("Mirrors the original *Popular Songs* tab — filter and rank by pedagogical criteria.")

    col1, col2, col3 = st.columns(3)
    with col1:
        voice_options = sorted(df['VocalRange'].dropna().unique().tolist())
        selected_voice = st.selectbox("Vocal Range", ["All"] + voice_options)
    with col2:
        class_options = sorted(df['Class'].dropna().unique().tolist())
        selected_class = st.multiselect("Difficulty Class", class_options, default=class_options)
    with col3:
        lang_options = sorted(df['Language'].dropna().unique().tolist())
        selected_lang = st.multiselect("Language", lang_options, default=lang_options)

    top_k = st.slider("Number of songs to show", 5, 50, 10)

    filtered = df.copy()
    if selected_voice != "All":
        filtered = filtered[
            (filtered['VocalRange'] == selected_voice) |
            (filtered['VocalRange'] == 'Vocal All')
        ]
    if selected_class:
        filtered = filtered[filtered['Class'].isin(selected_class)]
    if selected_lang:
        filtered = filtered[filtered['Language'].isin(selected_lang)]

    display_cols = ['Title', 'Composer', 'VocalRange', 'Class', 'Language', 'Genre', 'Era']
    st.dataframe(
        filtered[display_cols].head(top_k).reset_index(drop=True),
        use_container_width=True,
    )
    st.caption(f"Showing {min(top_k, len(filtered))} of {len(filtered)} matching songs.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Content-Based Recommendation  (core tab — minimal changes from original)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Content-Based Song Recommendation")
    st.markdown(
        "Select a song and the features you care about. "
        "The engine uses **cosine similarity** on a weighted feature vector — "
        "same algorithm as the original app, just with VSAE features."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### 🎵 Query Song")
        song_list = sorted(df['Title'].dropna().unique().tolist())
        selected_song = st.selectbox("Select a song", song_list)

        st.markdown("#### 🎛️ Features  *(mirrors original metric checkboxes)*")
        use_vocalrange = st.checkbox("Vocal Range",  value=True)
        use_class      = st.checkbox("Difficulty (Class)", value=True)
        use_language   = st.checkbox("Language",     value=True)
        use_genre      = st.checkbox("Genre",        value=True)
        use_era        = st.checkbox("Era / Time Period", value=False)
        use_rangespan  = st.checkbox("Note Range Span (Energy proxy)", value=False)
        use_runtime    = st.checkbox("Runtime (Danceability proxy)", value=False)

        st.markdown("#### 🔄 Repertoire Mode")
        repertoire_mode = st.radio(
            "Recommend songs that are:",
            options=["similar", "different"],
            index=0,
            help=(
                "'Similar' returns highest cosine similarity scores. "
                "'Different' inverts the score (1 - cosine_sim) to find "
                "songs that contrast with the selected one."
            ),
        )

        top_n = st.slider("Top N recommendations", 1, 20, 5)
        exclude_transpositions = st.checkbox("Exclude other keys of the same piece", value=True)

        run_btn = st.button("🔍 Get Recommendations", type="primary")

    with col_right:
        if run_btn:
            selected_features = []
            if use_vocalrange:  selected_features.append('VocalRange')
            if use_class:       selected_features.append('Class')
            if use_language:    selected_features.append('Language')
            if use_genre:       selected_features.append('Genre')
            if use_era:         selected_features.append('Era')
            if use_rangespan:   selected_features.append('RangeSpan')
            if use_runtime:     selected_features.append('Runtime')

            if not selected_features:
                st.warning("Please select at least one feature.")
            else:
                with st.spinner("Computing similarity..."):
                    try:
                        feat_matrix = build_feature_matrix(df, selected_features)
                        results = get_recommendations(
                            df=df,
                            feature_matrix=feat_matrix,
                            song_title=selected_song,
                            top_n=top_n,
                            repertoire_mode=repertoire_mode,
                            exclude_same_base=exclude_transpositions,
                        )

                        mode_label = "most similar to" if repertoire_mode == "similar" else "most different from"
                        st.success(
                            f"Top {top_n} songs **{mode_label}** "
                            f"*{selected_song}* based on: {', '.join(selected_features)}"
                        )

                        # ── Display selected song info ──────────────────────
                        query_row = df[df['Title'] == selected_song].iloc[0]
                        with st.expander("ℹ️ Selected song details", expanded=True):
                            info_cols = st.columns(4)
                            info_cols[0].metric("Composer", str(query_row.get('Composer', '—')))
                            info_cols[1].metric("Vocal Range", str(query_row.get('VocalRange', '—')))
                            info_cols[2].metric("Class", str(query_row.get('Class', '—')))
                            info_cols[3].metric("Language", str(query_row.get('Language', '—')))

                        # ── Recommendation table ────────────────────────────
                        st.markdown("#### Recommendations")

                        # Score bar styling
                        def color_score(val):
                            if repertoire_mode == 'similar':
                                intensity = int(val * 200)
                                return f'background-color: rgba(46, 95, 163, {val:.2f}); color: white'
                            else:
                                intensity = int(val * 200)
                                return f'background-color: rgba(163, 46, 46, {val:.2f}); color: white'

                        display = results.copy()
                        display['Score'] = display['Score'].round(3)
                        st.dataframe(display, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("👈 Configure your settings and click **Get Recommendations**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Overview  (replaces "Predicted Top Hits" from original)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
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
