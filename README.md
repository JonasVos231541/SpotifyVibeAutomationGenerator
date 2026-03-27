# Vibe Splitter

Automatically clusters your Spotify library into vibe-based playlists using Last.fm tags, semantic embeddings, and optional Spotify audio features.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:10000
```

## Project Structure

```
app.py                          # Entry point — Flask app + APScheduler (hourly jobs)
vibe_splitter/
  config.py                     # All configuration — env vars, thresholds, genre rules
  state.py                      # Thread-safe JSON state manager with atomic writes
  db.py                         # SQLite layer (tracks, config, embeddings)
  embeddings.py                 # fastembed (ONNX MiniLM) encoding + optional audio fusion
  router.py                     # Route new tracks to target playlists via embedding similarity
  lastfm.py                     # Parallel Last.fm tag fetching with global rate limiter
  naming.py                     # Cluster naming — energy/mood scoring, optional Ollama AI names
  playlists.py                  # Push clusters to Spotify playlists
  hourly.py                     # Hourly update logic (separated to avoid circular imports)
  spotify_client.py             # Spotify API wrapper — circuit breaker, rate budget
  events.py                     # SSE event bus for real-time frontend updates
  cache.py                      # Legacy JSON file cache (deprecated, use db.py)
  routes/                       # Flask Blueprints
    __init__.py                 # register_routes(), CSRF/CSP middleware, SSE endpoint
    helpers.py                  # Shared: _sanitize_name, _valid_id, rate_limit decorator
    auth.py                     # /login, /callback, /logout, /api/wipe-token, /api/token-info
    data.py                     # /api/state, /api/stats, /api/playlists, /api/cache-*, /api/overrides
    playlist.py                 # /api/update-name, /api/cleanup-playlists, /api/cover-*, /api/retag
    inbox.py                    # /api/inbox/approve, /api/inbox/dismiss
    targets.py                  # /api/targets — manage target playlists and their embeddings
    admin.py                    # /api/test-fetch, /api/test-playlist
  templates/index.html          # Frontend (vanilla JS, dark theme)
  config/                       # Editable JSON config files (auto-generated on first run)
    noise_tags.json
    energy_pos.json / energy_neg.json
    mood_pos.json / mood_neg.json
    genre_rules.json
    skip_name_tags.json
```

## Configuration

### Environment Variables

#### API Keys

| Variable | Default | Description |
|----------|---------|-------------|
| `SPOTIFY_CLIENT_ID` | (required) | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | (required) | Spotify app client secret |
| `SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:5000/callback` | OAuth redirect URI |
| `LASTFM_API_KEY` | (required) | Last.fm API key |
| `FLASK_SECRET_KEY` | auto-generated | Flask session secret (persisted to `.flask_secret`) |

#### Optional / AI Naming

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint for AI cluster names |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `VS_ENABLE_AI_NAMES` | `false` | Set `true` to use Ollama for cluster naming |

#### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `VS_STATE_FILE` | `splitter_state.json` | Path to state file |
| `VS_MODEL_FILE` | `splitter_model.npz` | Path to saved embedding model |
| `VS_CACHE_FILE` | `track_cache.json` | Path to legacy tag cache file |

#### Numeric Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `VS_AUDIO_WEIGHT` | `0.20` | Base weight of audio features vs text embeddings (0-1) |
| `VS_COHESION_FLOOR` | `0.25` | Min cohesion before splitting oversized clusters |
| `VS_MAX_FRACTION` | `0.25` | Max fraction of total tracks in one cluster |
| `VS_TARGET_PLAYLIST_SIZE` | `120` | Target tracks per playlist |
| `VS_CONFIDENCE_MARGIN` | `0.12` | Min gap between best/second similarity for confident classification |
| `VS_CONFIDENCE_MIN_SIM` | `0.30` | Min absolute similarity for confident classification |
| `VS_CONFIDENCE_TEMPERATURE` | `0.15` | Softmax temperature for confidence scoring |
| `VS_ADAPTIVE_MARGIN_SCALE` | `0.05` | Per-cluster adaptive margin adjustment |
| `VS_DRIFT_SIGMA` | `2.0` | Standard deviations for drift detection |
| `VS_RECLUSTER_DRIFT_THRESHOLD` | `0.20` | Fraction of low-confidence tracks to trigger drift warning |
| `VS_AUTO_ASSIGN_THRESHOLD` | `0.90` | Min confidence to auto-assign (bypass inbox) |
| `VS_AUTO_ASSIGN_THRESHOLD_MIN` | `0.70` | Per-cluster floor for auto-assign threshold |
| `VS_AUTO_RECLUSTER_DRIFT` | `0.50` | Higher drift threshold for auto-recluster suggestion |
| `VS_LASTFM_RATE_DELAY` | `0.26` | Seconds between Last.fm API calls |
| `VS_LASTFM_WORKERS` | `5` | Parallel Last.fm fetch workers |
| `VS_MODEL_STALE_DAYS` | `30` | Days before model is considered stale |
| `VS_MODEL_TRACK_DRIFT_PCT` | `0.25` | Track-count change fraction to flag model staleness |

### Editable JSON Config Files

Located in `vibe_splitter/config/` (auto-created on first run):

- **`noise_tags.json`** — Tags to exclude (e.g. "seen live", "spotify")
- **`energy_pos.json`** / **`energy_neg.json`** — High/low energy tag lists
- **`mood_pos.json`** / **`mood_neg.json`** — Bright/dark mood tag lists
- **`genre_rules.json`** — Genre matching rules: `[trigger_tags[], min_matches, name, description]`
- **`skip_name_tags.json`** — Tags too generic for cluster naming

## Deployment

**Render (free tier):**
- 512MB RAM limit — keep in mind when tuning batch sizes
- Auto-deploys from `main` branch via `Dockerfile`
- Set env vars in Render dashboard; `SPOTIFY_REDIRECT_URI` must point to your Render URL

**Docker:**
```bash
docker build -t vibe-splitter .
docker run -p 10000:10000 \
  -e SPOTIFY_CLIENT_ID=... \
  -e SPOTIFY_CLIENT_SECRET=... \
  -e LASTFM_API_KEY=... \
  vibe-splitter
```
