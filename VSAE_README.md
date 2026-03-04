# VSAE Song Recommender 🎼

Adapted from [music-recommendation-app](https://github.com/SP02028/music-recommendation-app).

## What changed vs. the original

| File | Original | VSAE Version |
|---|---|---|
| `setup.py` / data loading | Downloads Spotify CSVs from GCS | **Replaced by `vsae_data.py`** — loads local VSAE CSV, computes MIDI note ranges, era bucketing, runtime parsing |
| `contentbased.py` | Cosine sim on Spotify audio features | **Replaced by `vsae_contentbased.py`** — same cosine sim core, VSAE feature matrix, + repertoire toggle |
| `helper.py` | Genre filter + popularity sort | **Merged into Tab 1 of app.py** — VocalRange + Class + Language filter |
| `app.py` | 3-tab Streamlit app | **Updated labels/features only** — same layout, same UX pattern |
| `header.py` / `response.py` | Display helpers | Inlined into app.py (kept simple) |

## Files

```
vsae_app/
├── app.py                # Main Streamlit app  (mirrors original app.py)
├── vsae_data.py          # Data loading + feature engineering  (replaces setup.py)
├── vsae_contentbased.py  # Cosine-sim recommender  (replaces contentbased.py)
├── requirements.txt
└── README.md
```

## Setup

1. Place your VSAE CSV in the same directory as `app.py` (default name: `VSAE_Data.csv`).
   Or set the env var: `export VSAE_DATA_PATH=/path/to/your/file.csv`

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run:
   ```
   streamlit run app.py
   ```

## Swapping in the full dataset

When the data collection team delivers the complete CSV, just drop it in and restart —
no code changes required. The feature pipeline auto-handles:
- Note → MIDI conversion (with per-voice-range imputation for missing values)
- Era bucketing from time period strings
- Runtime M:SS → seconds parsing

## Repertoire Mode

The **one** logic change to the similarity engine:

```python
# similar  → highest cosine scores
scores = cosine_similarity(query_vec, feature_matrix)[0]

# different → invert scores
if repertoire_mode == 'different':
    scores = 1.0 - scores
```

This is exposed as a radio button in the UI.
