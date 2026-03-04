"""
vsae_contentbased.py
--------------------
Content-based recommendation engine for VSAE songs.
Mirrors contentbased.py from the original Spotify app:
  - Same cosine-similarity core
  - Added: repertoire_mode toggle (similar / different)
  - Changed: feature matrix comes from vsae_data.build_feature_matrix()
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from vsae_data import load_and_engineer, build_feature_matrix


def get_recommendations(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    song_title: str,
    top_n: int = 5,
    repertoire_mode: str = 'similar',   # 'similar' | 'different'
    exclude_same_base: bool = True,
) -> pd.DataFrame:
    """
    Return the top-N recommended songs given a query song title.

    Parameters
    ----------
    df               : The full VSAE DataFrame (from load_and_engineer).
    feature_matrix   : Pre-built feature matrix (from build_feature_matrix).
    song_title       : Exact title string as it appears in df['Title'].
    top_n            : Number of recommendations to return.
    repertoire_mode  : 'similar' → highest cosine sim.
                       'different' → lowest cosine sim (1 - score).
    exclude_same_base: If True, exclude transpositions of the same piece
                       (rows sharing the same base title before " (High/Low/Medium)").

    Returns
    -------
    DataFrame with columns: Title, Composer, VocalRange, Class, Language,
                             Genre, Era, RangeSpan, RuntimeSeconds, Score
    """
    # ── Find query row ────────────────────────────────────────────────────────
    matches = df[df['Title'] == song_title]
    if matches.empty:
        raise ValueError(f"Song '{song_title}' not found in dataset.")

    query_idx = matches.index[0]
    query_vec = feature_matrix[query_idx].reshape(1, -1)

    # ── Compute cosine similarity ─────────────────────────────────────────────
    # This is the identical call as in the original contentbased.py
    sims = cosine_similarity(query_vec, feature_matrix)[0]

    # ── Repertoire toggle (ONE-LINE CHANGE from original) ─────────────────────
    if repertoire_mode == 'different':
        scores = 1.0 - sims
    else:
        scores = sims

    # ── Build result DataFrame ────────────────────────────────────────────────
    results = df.copy()
    results['Score'] = scores

    # Always exclude the query song itself
    results = results[results.index != query_idx]

    # Optionally exclude other transpositions of the same piece
    if exclude_same_base:
        base = re.sub(r'\s*\((High|Low|Medium|Vocal All|Bass|Baritone|Soprano|Alto|Tenor|Mezzo Soprano)[^)]*\)\s*$',
                      '', song_title, flags=re.IGNORECASE).strip()
        results = results[~results['Title'].str.startswith(base)]

    # Sort
    results = results.sort_values('Score', ascending=False)

    output_cols = ['Title', 'Composer', 'VocalRange', 'Class', 'Language',
                   'Genre', 'Era', 'RangeSpan', 'RuntimeSeconds', 'Score']
    available = [c for c in output_cols if c in results.columns]
    return results[available].head(top_n).reset_index(drop=True)


import re  # needed for exclude_same_base — imported here to keep top clean
