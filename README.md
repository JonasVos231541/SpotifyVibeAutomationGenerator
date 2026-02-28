# Vibe Splitter — Refactored

Automatically clusters your Spotify library into vibe-based playlists using Last.fm tags, semantic embeddings, and optional Spotify audio features.

## Quick Start

```bash
# Install dependencies
pip install flask spotipy sentence-transformers hdbscan scikit-learn numpy apscheduler waitress

# Set your API keys (or edit config.py defaults)
export SPOTIFY_CLIENT_ID="your_id"
export SPOTIFY_CLIENT_SECRET="your_secret"
export LASTFM_API_KEY="your_key"

# Run
python app.py
# → http://127.0.0.1:5000
```

## Project Structure

```
app.py                          # Entry point — Flask app + scheduler
vibe_splitter/
├── __init__.py
├── config.py                   # All configuration — env vars + JSON files
├── state.py                    # Thread-safe state with atomic writes + job locking
├── cache.py                    # Track tag cache with atomic writes
├── spotify_client.py           # Spotify API wrapper — monkey-patches, backoff, audio features
├── lastfm.py                   # Last.fm tag fetching — backoff, genre fallback
├── embeddings.py               # Sentence construction, model loading, hybrid audio fusion
├── clustering.py               # HDBSCAN pipeline, health metric, cohesion splitting
├── naming.py                   # Energy/mood scoring, genre rules, cross-cluster naming
├── incremental.py              # Confidence classification, drift detection, auto-assign
├── playlists.py                # Playlist push, inbox with auto-assign + duplicate prevention
├── routes.py                   # All Flask routes
├── config/                     # Editable JSON config files (auto-generated on first run)
│   ├── noise_tags.json
│   ├── energy_pos.json
│   ├── energy_neg.json
│   ├── mood_pos.json
│   ├── mood_neg.json
│   ├── genre_rules.json
│   └── skip_name_tags.json
└── templates/
    └── index.html              # Web UI
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPOTIFY_CLIENT_ID` | (built-in) | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | (built-in) | Spotify app client secret |
| `SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:5000/callback` | OAuth redirect URI |
| `SPOTIFY_SCOPE` | (full scope string) | OAuth scopes to request |
| `LASTFM_API_KEY` | (built-in) | Last.fm API key |
| `FLASK_SECRET_KEY` | `vibe-splitter-v3-2026` | Flask session secret |
| `VS_STATE_FILE` | `splitter_state.json` | Path to state file |
| `VS_MODEL_FILE` | `splitter_model.pkl` | Path to model/centroid file |
| `VS_CACHE_FILE` | `track_cache.json` | Path to tag cache file |

### Numeric Thresholds

All tunable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VS_AUDIO_WEIGHT` | `0.20` | Weight of audio features vs embeddings (0-1) |
| `VS_COHESION_FLOOR` | `0.25` | Min cohesion before splitting oversized clusters |
| `VS_MAX_FRACTION` | `0.25` | Max fraction of total tracks in one cluster |
| `VS_TARGET_PLAYLIST_SIZE` | `120` | Target tracks per playlist |
| `VS_CONFIDENCE_MARGIN` | `0.12` | Min gap between best/second for confident classification |
| `VS_CONFIDENCE_MIN_SIM` | `0.30` | Min absolute similarity for confident classification |
| `VS_DRIFT_SIGMA` | `2.0` | Standard deviations for drift detection |
| `VS_RECLUSTER_DRIFT_THRESHOLD` | `0.20` | Fraction of low-confidence tracks to trigger warning |
| `VS_AUTO_ASSIGN_THRESHOLD` | `0.90` | Min confidence to auto-assign (bypass inbox) |
| `VS_AUTO_RECLUSTER_DRIFT` | `0.50` | Higher drift threshold for auto-recluster suggestion |
| `VS_LASTFM_RATE_DELAY` | `0.26` | Seconds between Last.fm API calls |
| `VS_TOP_N_TAGS` | `100` | Max tag vocabulary size |
| `VS_MIN_TAG_DF` | `2` | Min document frequency for tags |

### External JSON Config Files

Located in `vibe_splitter/config/` (auto-created on first run). Edit without touching code:

- **`noise_tags.json`** — Tags to exclude from processing (e.g., "seen live", "spotify")
- **`energy_pos.json`** — Tags indicating high energy (e.g., "energetic", "rave")
- **`energy_neg.json`** — Tags indicating low energy (e.g., "calm", "ambient")
- **`mood_pos.json`** — Tags indicating bright/positive mood
- **`mood_neg.json`** — Tags indicating dark/negative mood
- **`genre_rules.json`** — Genre matching rules: `[trigger_tags[], min_matches, name, description]`
- **`skip_name_tags.json`** — Tags too generic for cluster naming

## Key Improvements Over Original Monolith

### Architecture
- **Modular**: 11 focused modules instead of one 3100-line file
- **Thread-safe**: All file I/O protected by RLock, atomic write-then-rename
- **Job locking**: Hourly/weekly jobs can't overlap (flag in state.json)
- **Scheduler guard**: Won't double-start with Flask debug reloader

### Robustness
- **Exponential backoff**: Both Spotify (429s) and Last.fm (429s + Retry-After)
- **Atomic writes**: State and cache use temp-file + `os.replace()` — no corruption on crash
- **Spotify genre fallback**: When Last.fm tags are empty, uses Spotify artist genres
- **Multi-artist handling**: All artist names joined for embedding context
- **Duplicate prevention**: Inbox approve checks `track_ids` before adding

### Algorithmic
- **Structured embeddings**: Tags tiered by weight, artist context, energy/mood in same vector
- **Hybrid vectors**: Optional Spotify audio features fused at configurable weight
- **Cohesion-based splitting**: Replaces pure size balancing
- **Health metric**: Cohesion (60%) + inter-cluster separation (40%)
- **Confidence scoring**: Margin-based with drift detection (mean + σ-based)
- **Auto-assign**: High-confidence tracks (≥0.9) go directly to playlists
- **Cross-cluster naming**: Deprioritises shared tags for distinctive names

### UI
- **Drift warning banner**: Appears when cluster drift detected; clickable to recluster
- **Scope-aware cover upload**: Checks for `ugc-image-upload` scope before upload
- **Token metadata**: Expiry and scope info exposed to frontend via `/api/state`

### Feature Enhancements
- **Re-tag + reclassify**: Re-tagging a track also reclassifies if tags changed significantly
- **Override propagation**: Split clusters inherit parent overrides
- **Merge reclassification**: Logs outlier tracks after merge
- **Improved cover search**: Tries artist+tag queries before genre-only fallback
