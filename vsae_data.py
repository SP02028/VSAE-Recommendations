"""
vsae_data.py
------------
Loads the VSAE CSV and engineers the feature matrix.
Drop-in replacement for the Spotify data loading that happened in setup.py
and the feature prep that happened in contentbased.py of the original app.
"""

import pandas as pd
import numpy as np
import re

# ── MIDI conversion ──────────────────────────────────────────────────────────
PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

def note_to_midi(note_str):
    """Convert a note string like 'G5', 'Bb3', 'C#4' to a MIDI integer.
    Returns None if unparseable."""
    if not isinstance(note_str, str):
        return None
    note_str = note_str.strip()
    # Strip parenthetical alternatives e.g. 'E5(G5)' → use first
    note_str = re.split(r'[\(\),]', note_str)[0].strip()
    if note_str in ('N/A', '', 'n/a'):
        return None
    match = re.match(r'([A-Ga-g][b#]?)(\d)', note_str)
    if not match:
        return None
    pitch, octave = match.group(1), int(match.group(2))
    # normalise capitalisation: first char upper, second lower
    pitch = pitch[0].upper() + pitch[1:].lower() if len(pitch) > 1 else pitch.upper()
    if pitch not in PITCH_CLASS:
        return None
    return (octave + 1) * 12 + PITCH_CLASS[pitch]

# Median MIDI fallbacks per vocal range (used when notes are missing)
RANGE_MIDI_DEFAULTS = {
    'Soprano':       {'high': 79, 'low': 65},   # G5 / F4
    'Mezzo Soprano': {'high': 77, 'low': 62},   # F5 / D4
    'Alto':          {'high': 77, 'low': 60},   # F5 / C4
    'Tenor':         {'high': 76, 'low': 60},   # E5 / C4
    'Baritone':      {'high': 64, 'low': 48},   # E4 / C3
    'Bass':          {'high': 62, 'low': 43},   # D4 / G2
    'Vocal All':     {'high': 77, 'low': 60},   # F5 / C4
}

ERA_ORDER = ['Renaissance', 'Baroque', 'Classical', 'Romantic', 'Modern']

def era_from_period(period_str):
    """Extract era label from a TimePeriod string like '1740 (Baroque)'."""
    if not isinstance(period_str, str):
        return 'Unknown'
    for era in ERA_ORDER:
        if era in period_str:
            return era
    # Fallback: bucket by year
    match = re.search(r'\d{4}', period_str)
    if match:
        yr = int(match.group())
        if yr < 1600: return 'Renaissance'
        if yr < 1750: return 'Baroque'
        if yr < 1820: return 'Classical'
        if yr < 1910: return 'Romantic'
        return 'Modern'
    return 'Unknown'

def runtime_to_seconds(rt):
    """Convert 'M:SS' or 'M:SS.s' string to integer seconds."""
    if not isinstance(rt, str) or rt.strip() in ('N/A', 'n/a', ''):
        return None
    match = re.match(r'(\d+):(\d+)', rt.strip())
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None

def load_and_engineer(csv_path: str) -> pd.DataFrame:
    """
    Load the VSAE CSV, compute derived columns, and return a clean DataFrame.
    This mirrors what setup.py + contentbased.py did for the Spotify dataset.
    """
    df = pd.read_csv(csv_path)

    # ── Basic cleanup ─────────────────────────────────────────────────────────
    df.columns = df.columns.str.strip()
    # Drop clearly flagged removal rows
    if df.columns[-1] == df.columns[-1]:  # trailing notes column may be unnamed
        notes_col = df.columns[-1]
        df = df[~df[notes_col].str.contains('should probably be removed', na=False)]
    df = df.dropna(subset=['Title'])
    df = df[df['Title'].str.strip() != '']
    df = df.reset_index(drop=True)

    # ── Derived: Era ──────────────────────────────────────────────────────────
    df['Era'] = df['Time Period'].apply(era_from_period)

    # ── Derived: MIDI notes & range span ──────────────────────────────────────
    df['HighestNote_MIDI'] = df['Highest Note'].apply(note_to_midi)
    df['LowestNote_MIDI']  = df['Lowest Note'].apply(note_to_midi)

    # Impute missing notes with per-VocalRange medians
    for idx, row in df.iterrows():
        vr = str(row.get('VocalRange', 'Vocal All')).strip()
        defaults = RANGE_MIDI_DEFAULTS.get(vr, RANGE_MIDI_DEFAULTS['Vocal All'])
        if pd.isna(row['HighestNote_MIDI']):
            df.at[idx, 'HighestNote_MIDI'] = defaults['high']
        if pd.isna(row['LowestNote_MIDI']):
            df.at[idx, 'LowestNote_MIDI'] = defaults['low']

    df['RangeSpan'] = (df['HighestNote_MIDI'] - df['LowestNote_MIDI']).clip(lower=0)

    # ── Derived: Runtime ─────────────────────────────────────────────────────
    df['RuntimeSeconds'] = df['Runtime of Song'].apply(runtime_to_seconds)
    df['RuntimeSeconds'] = df['RuntimeSeconds'].fillna(df['RuntimeSeconds'].median())

    # ── Ordinal: Class ────────────────────────────────────────────────────────
    class_map = {'A': 1, 'B': 2, 'C': 3}
    df['ClassOrdinal'] = df['Class'].map(class_map).fillna(2)

    # ── Ordinal: Era ──────────────────────────────────────────────────────────
    era_map = {e: i+1 for i, e in enumerate(ERA_ORDER)}
    era_map['Unknown'] = 3  # default to middle
    df['EraOrdinal'] = df['Era'].map(era_map)

    return df


def build_feature_matrix(df: pd.DataFrame, selected_features: list) -> np.ndarray:
    """
    Build the normalized numerical feature matrix for cosine similarity.
    `selected_features` is a list chosen by the user in the UI —
    mirrors the feature checkbox logic from the original contentbased.py.

    Available feature keys:
        'VocalRange', 'Class', 'Language', 'Genre', 'Era', 'RangeSpan', 'Runtime'
    """
    parts = []

    if 'VocalRange' in selected_features:
        dummies = pd.get_dummies(df['VocalRange'].fillna('Vocal All'), prefix='VR')
        parts.append(dummies.values * 3.0)   # weight boost (hard constraint)

    if 'Class' in selected_features:
        vals = df['ClassOrdinal'].values.reshape(-1, 1) / 3.0
        parts.append(vals * 2.5)              # weight boost

    if 'Language' in selected_features:
        dummies = pd.get_dummies(df['Language'].fillna('English'), prefix='Lang')
        parts.append(dummies.values * 1.5)

    if 'Genre' in selected_features:
        # Normalize genre strings a little
        genre_clean = df['Genre'].fillna('Unknown').str.strip().str.lower()
        dummies = pd.get_dummies(genre_clean, prefix='Genre')
        parts.append(dummies.values * 1.5)

    if 'Era' in selected_features:
        vals = df['EraOrdinal'].values.reshape(-1, 1) / 5.0
        parts.append(vals * 1.0)

    if 'RangeSpan' in selected_features:
        raw = df['RangeSpan'].values.reshape(-1, 1).astype(float)
        mn, mx = raw.min(), raw.max()
        normed = (raw - mn) / (mx - mn + 1e-9)
        parts.append(normed * 1.0)

    if 'Runtime' in selected_features:
        raw = df['RuntimeSeconds'].values.reshape(-1, 1).astype(float)
        mn, mx = raw.min(), raw.max()
        normed = (raw - mn) / (mx - mn + 1e-9)
        parts.append(normed * 0.5)

    if not parts:
        raise ValueError("No features selected.")

    return np.hstack(parts).astype(float)
