# VSAE Song Recommender 🎼

Adapted from [music-recommendation-app](https://github.com/SP02028/music-recommendation-app).

## What changed vs. the original

| File | Original | VSAE Version |
|---|---|---|
| `setup.py` / data loading | Downloads Spotify CSVs from GCS | **Replaced by `vsae_data.py`** — loads local VSAE CSV, computes MIDI note ranges, era bucketing, runtime parsing |
| `contentbased.py` | Cosine sim on Spotify audio features | **Replaced by `vsae_contentbased.py`** — personalized PageRank over a cosine-similarity graph, VSAE feature matrix, + repertoire toggle |
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

## Similarity scoring

`vsae_contentbased.py` (and the same logic in `self_contained_app.js`) ranks songs with
**personalized PageRank** seeded at the reference song:

1. Build the feature matrix. Categorical features are one-hot; ordinal/numeric features
   (Class, Era, Range span, Runtime) are encoded as an angle `[cos(πx/2), sin(πx/2)]` so
   cosine similarity depends on distance, not magnitude.
2. Connect every pair of songs with weight `cosine^4` (dense graph, no top-k pruning).
3. Run PageRank with restart probability 0.3 back to the reference song.
4. Divide each song's PageRank mass by its degree (removes "hub" bias), then scale so the
   best match among the currently filtered songs is 1.00.

The relation weights and walk settings (restart probability, neighbour share,
rarity damping β) live in **`similarity_config.json`**, which both the website
and the Python module read. Edit them there only, and deploy the file next to
`index.html`. If it is missing or invalid, the website switches similarity off
(the rest of the page keeps working); the Python module raises an error naming
the bad setting.

`repertoire_mode='different'` returns `1 - score`.
