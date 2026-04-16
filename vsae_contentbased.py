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
import re


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

    # ── Build weighted graph from cosine similarities ─────────────────────────
    sim_matrix = cosine_similarity(feature_matrix)
    np.fill_diagonal(sim_matrix, 0.0)

    # Keep only top-k outbound edges per node to reduce noise in the graph.
    n = sim_matrix.shape[0]
    k = min(12, max(3, n // 10))
    pruned = np.zeros_like(sim_matrix)
    if n > 1:
        for i in range(n):
            row = sim_matrix[i]
            top_idx = np.argsort(row)[-k:]
            pruned[i, top_idx] = np.clip(row[top_idx], 0.0, None)

    # Row-normalize to transition probabilities.
    row_sums = pruned.sum(axis=1, keepdims=True)
    transition = np.divide(
        pruned,
        np.where(row_sums == 0, 1.0, row_sums),
        where=True,
    )

    # Personalized PageRank centered on the selected song.
    alpha = 0.85
    teleport = np.zeros(n)
    teleport[query_idx] = 1.0
    rank = teleport.copy()
    for _ in range(100):
        next_rank = alpha * (transition.T @ rank) + (1.0 - alpha) * teleport
        if np.linalg.norm(next_rank - rank, ord=1) < 1e-10:
            rank = next_rank
            break
        rank = next_rank

    scores = rank

    if repertoire_mode == 'different':
        scores = 1.0 - scores

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

    # Sort (highest score first)
    results = results.sort_values('Score', ascending=False)

    output_cols = ['Title', 'Composer', 'VocalRange', 'Class', 'Language',
                   'Genre', 'Era', 'RangeSpan', 'RuntimeSeconds', 'Score']
    available = [c for c in output_cols if c in results.columns]
    return results[available].head(top_n).reset_index(drop=True)
