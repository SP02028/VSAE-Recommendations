"""
vsae_contentbased.py
--------------------
Content-based recommendation engine for VSAE songs.

Personalized PageRank over a song–attribute graph (mirrors the JS version in
self_contained_app.js; weights live in similarity_config.json):

  - Nodes are songs plus one node per attribute value (composer, language,
    genre tag, class, era, voice, range-span bin, runtime bin).
  - Each song links to its values; neighbouring ordinal values link to each
    other (Baroque–Classical, Soprano–Mezzo) for partial credit.
  - A walk through an attribute node splits its mass across every song that
    shares the value, so rare shared values count more (IDF-like). Rarity is
    damped by RARITY_DAMPING so one rare composer can't dominate.
  - Score = lift = PPR(song) / global PageRank(song).
"""

import json
import os
import re

import numpy as np
import pandas as pd

from vsae_data import ERA_ORDER, note_to_midi, normalize_language, runtime_to_seconds

# Weights and walk settings are shared with self_contained_app.js via
# similarity_config.json so the two versions can't drift.
CONFIG_PATH = os.environ.get(
    'VSAE_SIMILARITY_CONFIG', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'similarity_config.json')
)


def _load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)

    def num(value, name, lo, hi):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
            raise ValueError(f"{path}: {name} must be a number from {lo} to {hi}.")
        return float(value)

    relations = cfg.get('relations') or {}
    for key in ('Composer', 'Language', 'Genre', 'Era', 'VocalRange', 'RangeSpan', 'Runtime'):
        if key not in relations:
            raise ValueError(f"{path}: relations.{key} is missing.")
        num(relations[key].get('weight'), f'relations.{key}.weight', 0, 100)
    num(cfg.get('ppr_restart'), 'ppr_restart', 0.01, 0.99)
    num(cfg.get('neighbor_share'), 'neighbor_share', 0, 10)
    num(cfg.get('rarity_damping'), 'rarity_damping', 0, 1)
    return cfg


_CONFIG = _load_config(CONFIG_PATH)
RELATION_WEIGHTS = {k: float(v['weight']) for k, v in _CONFIG['relations'].items()}
RELATION_LABELS = {k: str(v.get('label') or k.lower()) for k, v in _CONFIG['relations'].items()}
ORDINAL_RELATIONS = {'Era', 'VocalRange', 'RangeSpan', 'Runtime'}
PPR_RESTART = float(_CONFIG['ppr_restart'])
NEIGHBOR_SHARE = float(_CONFIG['neighbor_share'])
RARITY_DAMPING = float(_CONFIG['rarity_damping'])

VOICE_LADDER = ['Soprano', 'Mezzo Soprano', 'Alto', 'Tenor', 'Baritone', 'Bass']
SPAN_BINS = [(9, 'narrow range'), (12, 'moderate range'), (15, 'wide range'), (99, 'very wide range')]
RUNTIME_BINS = [(119, 'under 2 min'), (179, '2–3 min'), (239, '3–4 min'), (99999, 'over 4 min')]

GENRE_TAGS = [
    ('spiritual', r'spiritual|spirtual|gospel'),
    ('sacred', r'sacred|hymn'),
    ('opera / aria', r'opera|aria|operetta'),
    ('art song', r'art|m.lodie|romance|berceuse|lied'),
    ('folk', r'folk|shanty|ballad|tonadilla'),
    ('madrigal', r'madrigal|mardrigal'),
    ('classical', r'classic'),
]

TRANSPOSITION_RE = re.compile(
    r'\s*\((High|Low|Medium|Vocal All|Bass|Baritone|Soprano|Alto|Tenor|Mezzo(?: Soprano)?)[^)]*\)\s*$',
    flags=re.IGNORECASE,
)


def genre_tags(genre) -> list:
    low = str(genre or '').lower()
    return [tag for tag, pattern in GENRE_TAGS if re.search(pattern, low)]


def _bin(bins, value) -> int:
    for i, (upper, _) in enumerate(bins):
        if value <= upper:
            return i
    return len(bins) - 1


class SongGraph:
    def __init__(self, df: pd.DataFrame, relations: list):
        self.n_songs = len(df)
        self.ids = {}
        self.nodes = []  # (relation, label); songs use relation 'song'
        edges = {}

        def node(key, rel, label):
            if key not in self.ids:
                self.ids[key] = len(self.nodes)
                self.nodes.append((rel, label))
            return self.ids[key]

        def link(a, b, w):
            edges[(a, b)] = edges.get((a, b), 0.0) + w
            edges[(b, a)] = edges.get((b, a), 0.0) + w

        for i, title in enumerate(df['Title']):
            node(f'song:{i}', 'song', title)

        for i, (_, s) in enumerate(df.iterrows()):
            w = RELATION_WEIGHTS
            if 'Composer' in relations and str(s.get('Composer') or '').strip():
                c = str(s['Composer']).strip()
                link(i, node(f'composer:{c.lower()}', 'Composer', c), w['Composer'])
            if 'Language' in relations:
                lang = normalize_language(pd.Series([s.get('Language')])).iloc[0]
                link(i, node(f'lang:{lang}', 'Language', lang.title()), w['Language'])
            if 'Genre' in relations:
                tags = genre_tags(s.get('Genre'))
                for t in tags:
                    link(i, node(f'genre:{t}', 'Genre', t.title()), w['Genre'] / len(tags))
            if 'Era' in relations and s.get('Era') in ERA_ORDER:
                link(i, node(f"era:{s['Era']}", 'Era', s['Era']), w['Era'])
            if 'VocalRange' in relations:
                vr = s.get('VocalRange')
                if vr == 'Vocal All':
                    for v in VOICE_LADDER:
                        link(i, node(f'voice:{v}', 'VocalRange', v), w['VocalRange'] / len(VOICE_LADDER))
                elif vr in VOICE_LADDER:
                    link(i, node(f'voice:{vr}', 'VocalRange', vr), w['VocalRange'])
            if ('RangeSpan' in relations and note_to_midi(s.get('Highest Note')) is not None
                    and note_to_midi(s.get('Lowest Note')) is not None):
                b = _bin(SPAN_BINS, s['RangeSpan'])
                link(i, node(f'span:{b}', 'RangeSpan', SPAN_BINS[b][1]), w['RangeSpan'])
            if 'Runtime' in relations and runtime_to_seconds(s.get('Runtime of Song')) is not None:
                b = _bin(RUNTIME_BINS, s['RuntimeSeconds'])
                link(i, node(f'runtime:{b}', 'Runtime', RUNTIME_BINS[b][1]), w['Runtime'])

        N = len(self.nodes)
        W = np.zeros((N, N))
        for (a, b), w in edges.items():
            W[a, b] = w

        # Rarity damping: scale each value's song edges by (#songs)^beta.
        n = self.n_songs
        for a in range(n, N):
            count = np.count_nonzero(W[a, :n])
            scale = count ** RARITY_DAMPING
            W[a, :n] *= scale
            W[:n, a] *= scale

        def chain(prefix, values):
            for x, y in zip(values, values[1:]):
                a, b = self.ids.get(f'{prefix}{x}'), self.ids.get(f'{prefix}{y}')
                if a is None or b is None:
                    continue
                w = NEIGHBOR_SHARE * min(W[a].sum(), W[b].sum())
                W[a, b] += w
                W[b, a] += w

        chain('era:', ERA_ORDER)
        chain('voice:', VOICE_LADDER)
        chain('span:', list(range(len(SPAN_BINS))))
        chain('runtime:', list(range(len(RUNTIME_BINS))))

        self.W = W
        self.deg = W.sum(axis=1)
        teleport = np.zeros(N)
        teleport[:n] = 1.0 / n
        self.baseline = self.random_walk(teleport)

    def random_walk(self, teleport: np.ndarray) -> np.ndarray:
        safe_deg = np.where(self.deg == 0, 1.0, self.deg)
        rank = teleport.copy()
        for _ in range(200):
            # W is symmetric, so P^T r == W @ (r / deg)
            nxt = (1.0 - PPR_RESTART) * (self.W @ (rank / safe_deg)) + PPR_RESTART * teleport
            if np.abs(nxt - rank).sum() < 1e-12:
                return nxt
            rank = nxt
        return rank

    def personalized(self, song_pos: int) -> np.ndarray:
        teleport = np.zeros(len(self.nodes))
        teleport[song_pos] = 1.0
        return self.random_walk(teleport)

    def explain(self, ppr: np.ndarray, song_pos: int, query_pos: int) -> list:
        n = self.n_songs
        shared = set(np.nonzero(self.W[query_pos])[0])
        attrs = np.nonzero(self.W[song_pos, n:])[0] + n
        flows = sorted(((ppr[a] * self.W[song_pos, a] / self.deg[a], a) for a in attrs), reverse=True)
        total = sum(f for f, _ in flows)
        out = []
        for f, a in flows:
            if len(out) >= 3 or f < 0.12 * total:
                break
            rel, label = self.nodes[a]
            if a not in shared and rel not in ORDINAL_RELATIONS:
                continue
            text = label if a in shared else f'similar {RELATION_LABELS[rel]}'
            if text not in out:
                out.append(text)
        return out


def get_recommendation_scores(
    df: pd.DataFrame,
    selected_features: list,
    song_title: str,
    query_index=None,
    repertoire_mode: str = 'similar',
    exclude_same_base: bool = True,
    with_reasons: bool = False,
):
    """Return lift scores indexed by df.index (NaN for the query and excluded
    songs). Pass the result through scale_to_best() before showing it.
    With with_reasons=True, also return a Series of "why" label lists."""
    if not selected_features:
        raise ValueError("No features selected.")
    if query_index is not None:
        if query_index not in df.index:
            raise ValueError(f"Query index '{query_index}' not found in dataset.")
        query_idx = query_index
    else:
        matches = df[df['Title'] == song_title]
        if matches.empty:
            raise ValueError(f"Song '{song_title}' not found in dataset.")
        query_idx = matches.index[0]
    query_pos = df.index.get_loc(query_idx)

    graph = SongGraph(df, selected_features)
    ppr = graph.personalized(query_pos)
    n = graph.n_songs
    base = graph.baseline[:n]
    lift = np.where(base > 0, ppr[:n] / np.where(base > 0, base, 1.0), 0.0)

    score_series = pd.Series(lift, index=df.index, dtype=float)
    score_series.loc[query_idx] = np.nan

    if exclude_same_base:
        base_title = TRANSPOSITION_RE.sub('', song_title).strip()
        base_mask = df['Title'].fillna('').astype(str).str.startswith(base_title)
        score_series.loc[base_mask] = np.nan

    if repertoire_mode == 'different':
        score_series = 1.0 / score_series.where(score_series > 0)

    if not with_reasons:
        return score_series
    reasons = pd.Series(
        [graph.explain(ppr, pos, query_pos) for pos in range(n)], index=df.index, dtype=object
    )
    return score_series, reasons


def scale_to_best(scores: pd.Series) -> pd.Series:
    """Log-scale lift to 0–1 between the weakest and strongest scores given
    (1.00 = best match among them). NaNs are left untouched."""
    valid = scores.dropna()
    if valid.empty:
        return scores
    hi = valid.max()
    if hi <= 0:
        return scores
    lo = max(valid.min(), hi * 1e-4)
    span = np.log(hi / lo)
    if span <= 0:
        return scores.where(scores.isna(), 1.0)
    return (np.log(scores.clip(lower=lo) / lo) / span).clip(0.0, 1.0)


def get_recommendations(
    df: pd.DataFrame,
    selected_features: list,
    song_title: str,
    top_n: int = 5,
    repertoire_mode: str = 'similar',   # 'similar' | 'different'
    exclude_same_base: bool = True,
) -> pd.DataFrame:
    """Return the top-N recommended songs for a query song title, with a
    0–1 Score column and a Why column."""
    scores, reasons = get_recommendation_scores(
        df=df,
        selected_features=selected_features,
        song_title=song_title,
        repertoire_mode=repertoire_mode,
        exclude_same_base=exclude_same_base,
        with_reasons=True,
    )
    results = df.copy()
    results['Score'] = scale_to_best(scores)
    results['Why'] = reasons.apply(lambda r: ' · '.join(r))
    results = results[results['Score'].notna()].sort_values('Score', ascending=False)

    output_cols = ['Title', 'Composer', 'VocalRange', 'Class', 'Language',
                   'Genre', 'Era', 'Score', 'Why']
    available = [c for c in output_cols if c in results.columns]
    return results[available].head(top_n).reset_index(drop=True)
