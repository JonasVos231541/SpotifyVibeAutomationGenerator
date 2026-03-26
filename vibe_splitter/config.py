"""
Centralised configuration for Vibe Splitter.

All API keys, secrets, file paths, and tunable thresholds are loaded from
environment variables with sensible defaults.  Tag vocabularies and genre
rules are loaded from external JSON files in the ``config/`` directory so
they can be edited without touching code.
"""
import json, os, logging, secrets as _secrets
from datetime import datetime

log = logging.getLogger("splitter.config")

# ─── Paths ────────────────────────────────────────────────────────────────────
_CONFIG_DIR  = os.path.join(os.path.dirname(__file__), "config")
_PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")

STATE_FILE = os.getenv("VS_STATE_FILE", "splitter_state.json")
MODEL_FILE      = os.getenv("VS_MODEL_FILE", "splitter_model.npz")
MODEL_META_FILE = os.getenv("VS_MODEL_META_FILE", "splitter_model_meta.json")
_LEGACY_PKL     = "splitter_model.pkl"
CACHE_FILE = os.getenv("VS_CACHE_FILE", "track_cache.json")

# ─── Secrets ──────────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI",  "http://127.0.0.1:5000/callback")
SPOTIFY_SCOPE         = os.getenv("SPOTIFY_SCOPE",
    "user-library-read playlist-read-private "
    "playlist-modify-public playlist-modify-private "
    "ugc-image-upload"
)

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not set — app will not function without these"
    )

SPOTIFY_CACHE_PATH = os.getenv("SPOTIPY_CACHE_PATH", ".spotify_cache")

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_BASE    = os.getenv("LASTFM_BASE",    "https://ws.audioscrobbler.com/2.0/")

if not LASTFM_API_KEY:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "LASTFM_API_KEY not set — Last.fm tag fetching will be disabled"
    )

# Flask secret key — auto-generate and persist if not configured
def _get_secret_key():
    env_key = os.getenv("FLASK_SECRET_KEY")
    if env_key:
        return env_key
    secret_file = os.path.join(_PROJECT_DIR, ".flask_secret")
    if os.path.exists(secret_file):
        try:
            with open(secret_file) as f:
                return f.read().strip()
        except OSError:
            pass
    key = _secrets.token_hex(32)
    try:
        with open(secret_file, "w") as f:
            f.write(key)
        log.info("Generated new Flask secret key (saved to .flask_secret)")
    except OSError:
        log.warning("Could not persist Flask secret key — sessions won't survive restarts")
    return key

SECRET_KEY = _get_secret_key()

# ─── Ollama (AI naming & guidance) ───────────────────────────────────────────
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://localhost:11434/api/generate")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3")
ENABLE_AI_NAMES = os.getenv("VS_ENABLE_AI_NAMES", "false").lower() == "true"

# ─── Numeric thresholds (overridable via env vars) ────────────────────────────
AUDIO_WEIGHT        = float(os.getenv("VS_AUDIO_WEIGHT",        "0.20"))
COHESION_FLOOR      = float(os.getenv("VS_COHESION_FLOOR",      "0.25"))
MAX_FRACTION        = float(os.getenv("VS_MAX_FRACTION",        "0.25"))
TARGET_PLAYLIST_SIZE = int(os.getenv("VS_TARGET_PLAYLIST_SIZE", "120"))
CONFIDENCE_MARGIN   = float(os.getenv("VS_CONFIDENCE_MARGIN",   "0.12"))
CONFIDENCE_MIN_SIM  = float(os.getenv("VS_CONFIDENCE_MIN_SIM",  "0.30"))
DRIFT_SIGMA         = float(os.getenv("VS_DRIFT_SIGMA",         "2.0"))
RECLUSTER_DRIFT_THRESHOLD = float(os.getenv("VS_RECLUSTER_DRIFT_THRESHOLD", "0.20"))
AUTO_ASSIGN_THRESHOLD = float(os.getenv("VS_AUTO_ASSIGN_THRESHOLD", "0.90"))
AUTO_RECLUSTER_DRIFT  = float(os.getenv("VS_AUTO_RECLUSTER_DRIFT", "0.50"))
CONFIDENCE_TEMPERATURE = float(os.getenv("VS_CONFIDENCE_TEMPERATURE", "0.15"))
ADAPTIVE_MARGIN_SCALE  = float(os.getenv("VS_ADAPTIVE_MARGIN_SCALE",  "0.05"))
LASTFM_RATE_DELAY   = float(os.getenv("VS_LASTFM_RATE_DELAY",  "0.26"))
LASTFM_WORKERS      = int(os.getenv("VS_LASTFM_WORKERS",      "5"))
MODEL_STALE_DAYS    = int(os.getenv("VS_MODEL_STALE_DAYS",     "30"))
MODEL_TRACK_DRIFT_PCT = float(os.getenv("VS_MODEL_TRACK_DRIFT_PCT", "0.25"))

# ─── Audio feature importance weights (applied after Z-scoring) ──────────────
AUDIO_IMPORTANCE = {
    "danceability": 1.2, "energy": 1.3, "valence": 1.1,
    "acousticness": 0.8, "instrumentalness": 0.9, "tempo": 0.7,
    "speechiness": 0.6, "mode": 0.5, "liveness": 0.4, "loudness": 0.8,
}

# ─── Audio feature type separation ───────────────────────────────────────────
# Continuous features are Z-scored; binary features are scaled to [-0.5, 0.5]
AUDIO_CONTINUOUS_KEYS = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "tempo", "speechiness", "liveness", "loudness",
]
AUDIO_BINARY_KEYS = ["mode"]  # 0=minor, 1=major — never Z-score

# ─── Tempo normalisation (logistic, replaces hard cap at 200 BPM) ─────────────
TEMPO_CENTER    = float(os.getenv("VS_TEMPO_CENTER",    "120.0"))  # inflection point
TEMPO_LOGISTIC_K = float(os.getenv("VS_TEMPO_LOGISTIC_K", "0.03"))  # steepness

# ─── Adaptive audio weight (based on tag coverage per track) ─────────────────
# Format: list of (min_tags_exclusive, weight) sorted ascending by min_tags.
# A track with N tags uses the weight from the last entry where min_tags <= N.
ADAPTIVE_AUDIO_WEIGHT_THRESHOLDS = [
    (0,  float(os.getenv("VS_AUDIO_WEIGHT_ZERO_TAGS", "0.65"))),   # 0 tags
    (5,  float(os.getenv("VS_AUDIO_WEIGHT_FEW_TAGS",  "0.40"))),   # 1–4 tags
    (15, float(os.getenv("VS_AUDIO_WEIGHT_MID_TAGS",  "0.25"))),   # 5–14 tags
    # 15+ tags → falls back to AUDIO_WEIGHT base
]

# ─── Embedding artist weight (artist context separated from semantic vector) ──
ARTIST_EMBEDDING_WEIGHT = float(os.getenv("VS_ARTIST_EMBEDDING_WEIGHT", "0.05"))

# ─── TF-IDF staleness detection ───────────────────────────────────────────────
EMBED_MODEL = os.getenv("VS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ─── Cohesion metric approximation ───────────────────────────────────────────
COHESION_SAMPLE_PAIRS = int(os.getenv("VS_COHESION_SAMPLE_PAIRS", "50"))  # random pairs for O(n) estimate

# ─── Cohesion split max sub-clusters ─────────────────────────────────────────
COHESION_SPLIT_MAX_K = int(os.getenv("VS_COHESION_SPLIT_MAX_K", "5"))  # silhouette-guided

# ─── Per-cluster confidence threshold floor ───────────────────────────────────
AUTO_ASSIGN_THRESHOLD_MIN = float(os.getenv("VS_AUTO_ASSIGN_THRESHOLD_MIN", "0.70"))

# ─── Naming: score_axis weight and zero-tag warning ──────────────────────────
SCORE_AXIS_WEIGHT  = float(os.getenv("VS_SCORE_AXIS_WEIGHT",  "0.4"))  # per-tag contribution
ZERO_TAG_WARN_PCT  = float(os.getenv("VS_ZERO_TAG_WARN_PCT",  "0.20"))  # warn if >20% tracks have zero tags

# ─── ReccoBeats (free Spotify audio_features replacement) ─────────────────
RECCOBEATS_BASE     = os.getenv("VS_RECCOBEATS_BASE", "https://api.reccobeats.com/v1")
RECCOBEATS_BATCH    = int(os.getenv("VS_RECCOBEATS_BATCH", "5"))
RECCOBEATS_DELAY    = float(os.getenv("VS_RECCOBEATS_DELAY", "0.6"))
TOP_N_TAGS          = int(os.getenv("VS_TOP_N_TAGS",            "100"))
MIN_TAG_DF          = int(os.getenv("VS_MIN_TAG_DF",            "2"))
EMBED_DIM           = int(os.getenv("VS_EMBED_DIM",            "384"))

# ─── Granularity range (for slider) ────────────────────────────────────────────
GRANULARITY_MIN     = int(os.getenv("VS_GRANULARITY_MIN",       "1"))
GRANULARITY_MAX     = int(os.getenv("VS_GRANULARITY_MAX",       "10"))
GRANULARITY_DEFAULT = int(os.getenv("VS_GRANULARITY_DEFAULT",   "5"))

# ─── Extra features (year, popularity) ────────────────────────────────────────
ENABLE_YEAR_FEATURE     = os.getenv("VS_ENABLE_YEAR_FEATURE",     "true").lower() == "true"
ENABLE_POPULARITY_FEATURE = os.getenv("VS_ENABLE_POPULARITY_FEATURE", "true").lower() == "true"
YEAR_MIN                = int(os.getenv("VS_YEAR_MIN",           "1960"))
YEAR_MAX                = int(os.getenv("VS_YEAR_MAX",           str(datetime.now().year)))

# ─── LLM Guidance for Clustering ──────────────────────────────────────────────
USE_LLM_GUIDANCE           = os.getenv("VS_USE_LLM_GUIDANCE",           "false").lower() == "true"
LLM_GUIDANCE_SAMPLE_SIZE   = int(os.getenv("VS_LLM_GUIDANCE_SAMPLE_SIZE",   "100"))
LLM_GUIDANCE_MIN_CATEGORIES = int(os.getenv("VS_LLM_GUIDANCE_MIN_CATEGORIES", "5"))
LLM_GUIDANCE_MAX_CATEGORIES = int(os.getenv("VS_LLM_GUIDANCE_MAX_CATEGORIES", "10"))

# ─── JSON-backed tag sets & genre rules ───────────────────────────────────────
def _load_json_set(filename, default):
    """Load a JSON array as a Python set, falling back to *default*."""
    path = os.path.join(_CONFIG_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            log.info(f"Loaded {len(data)} items from {filename}")
            return set(data)
        except Exception as e:
            log.warning(f"Failed to load {filename}: {e} — using defaults")
    return set(default)


def _load_genre_rules(filename="genre_rules.json", default=None):
    """Load genre rules from JSON.  Each rule is [trigger_tags[], min_matches, name, description]."""
    path = os.path.join(_CONFIG_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            rules = [(set(r[0]), r[1], r[2], r[3]) for r in raw]
            log.info(f"Loaded {len(rules)} genre rules from {filename}")
            return rules
        except Exception as e:
            log.warning(f"Failed to load {filename}: {e} — using defaults")
    return default or []


# ── Defaults (used when JSON files don't exist) ──────────────────────────────
_DEFAULT_NOISE_TAGS = [
    "seen live","favorites","favourite","love","awesome","beautiful","amazing",
    "cool","great","best","good","playlist","spotify","youtube","music",
    "all","my","the","a","an","and","or","to","in","of","for",
]

_DEFAULT_ENERGY_POS = [
    "energetic","aggressive","intense","powerful","heavy","hard","loud","brutal",
    "fast","uptempo","frantic","wild","angry","fierce","raw","explosive",
    "hype","banger","turnt","trap","drill","grime","crunk","bounce",
    "dance","danceable","club","floor-filler","rave","edm","techno","trance",
    "hard techno","gabber","drum and bass","dnb","breakbeat","jungle",
    "thrash","death metal","black metal","metalcore","hardcore","punk","grunge",
    "hard rock","heavy metal","speed metal","power metal","crossover",
    "salsa","reggaeton","cumbia","soca","afrobeats","dancehall","dembow",
    "dramatic","bombastic","symphonic","march","overture",
    "exciting","driving","stomping","headbanger","mosh",
]
_DEFAULT_ENERGY_NEG = [
    "calm","relaxed","mellow","peaceful","quiet","gentle","soothing","tranquil",
    "soft","tender","delicate","serene","still","hushed",
    "slow","mid-tempo","downtempo","ballad","lullaby",
    "acoustic","unplugged","fingerpicking","lo-fi","bedroom pop",
    "ambient","atmospheric","drone","new age","meditation","sleep","study",
    "background","elevator","easy listening",
    "cool jazz","smooth jazz","bossa nova","jazz ballad",
    "adagio","andante","nocturne","serenade","elegy",
    "dreamy","hazy","floaty","laid-back","chilled",
]
_DEFAULT_MOOD_POS = [
    "happy","upbeat","feel-good","fun","cheerful","positive","uplifting","joyful",
    "euphoric","bright","playful","carefree","hopeful","warm","blissful","sunny",
    "optimistic","elated","exuberant","gleeful",
    "romantic","love","sensual","passionate","tender","intimate","flirtatious",
    "party","celebratory","festive","carnival","fiesta","summer",
    "gospel","praise","worship","devotional",
    "tropical","beach",
    "funk","groove","soul","feel good",
    "bossa nova","samba",
    "afrobeats","highlife","juju",
    "reggae","one love","irie",
    "k-pop","idol","bubblegum",
    "folk","wholesome","heartwarming",
    "country","americana","blue skies",
    "swing","jazz","bebop","big band",
    "classical","baroque","concerto",
]
_DEFAULT_MOOD_NEG = [
    "dark","melancholic","sad","depressing","gloomy","brooding","emotional",
    "heartbreak","tragic","pain","haunting","cold","sinister","desperate","lonely",
    "sorrowful","mournful","grief","anguish","tormented","bleak","desolate",
    "hopeless","bitter","regret","loss","crying","tears",
    "blues","delta blues","soul blues",
    "black metal","doom metal","funeral doom",
    "darkwave","gothic","post-punk","industrial",
    "noise","harsh noise","drone",
    "emo","screamo","post-hardcore",
    "dark jazz","noir jazz",
    "murder ballad","tragic country",
    "minor key","dissonant","atonal",
    "horror","creepy","eerie","unsettling",
    "protest","political","angry",
    "gangsta","street","gritty",
]

_DEFAULT_GENRE_RULES = [
    (["techno","house","club","rave","edm","dance"],          3, "Club Night",      "Electronic club music — made for the dancefloor"),
    (["ambient","atmospheric","drone","experimental"],         2, "Atmosphere",      "Ambient & experimental — space to think"),
    (["drum and bass","dnb","jungle","breakbeat"],             2, "Breakneck",       "Fast-paced electronic — drum & bass energy"),
    (["synthwave","retro","80s","synth"],                      2, "Neon Drive",      "Synthwave & retro electronics — 80s nostalgia"),
    (["lo-fi","bedroom pop","chillhop","study"],               2, "Lo-Fi Study",     "Lo-fi & bedroom pop — background focus music"),
    (["rap","hip-hop","hip hop","trap","drill"],               3, "Street Level",    "Hip-hop & rap — raw lyrics, heavy beats"),
    (["grime","uk grime","uk rap","roadman"],                  2, "Grime Wave",      "UK grime & rap — cold and calculated"),
    (["afrobeats","afropop","afro","highlife","afroswing"],    2, "Afro Vibes",      "Afrobeats & Afropop — sun-soaked grooves"),
    (["heavy metal","metal","thrash","death metal"],           2, "Full Throttle",   "Metal & hard rock — no brakes"),
    (["punk","hardcore","post-hardcore","emo"],                2, "Loud & Broken",   "Punk & hardcore — raw and unfiltered"),
    (["rock","indie rock","alternative rock","grunge"],        3, "Guitar Forward",  "Rock & alternative — guitar-driven energy"),
    (["singer-songwriter","acoustic","folk","indie folk"],     2, "Just a Song",     "Singer-songwriter & folk — stripped back and honest"),
    (["k-pop","kpop","korean","idol","girl group","boy band"], 2, "K-Pop Energy",    "K-pop — polished, high-production group sound"),
    (["pop","dance pop","electropop","synth-pop"],             4, "Pop Bangers",     "Mainstream pop — catchy, polished, everywhere"),
    (["reggaeton","latin","latin pop","bachata","merengue"],   2, "Ritmo Latino",    "Latin rhythms — reggaeton, bachata, dance"),
    (["reggae","dancehall","ska","dub","roots reggae"],        2, "Roots & Riddim",  "Reggae & dancehall — slow rhythm, heavy bass"),
    (["bossa nova","samba","mpb","forro","axe"],               2, "Brazilian Heat",  "Brazilian sounds — bossa, samba, MPB"),
    (["jazz","bebop","swing","big band","cool jazz"],          2, "Jazz Hour",       "Jazz — improvised, sophisticated, timeless"),
    (["soul","r&b","rnb","neo soul","funk"],                   2, "Soul Session",    "Soul & R&B — warm, groovy, heartfelt"),
    (["blues","delta blues","chicago blues","soul blues"],     2, "The Blues",        "Blues — raw emotion, guitar and voice"),
    (["classical","orchestral","symphony","chamber","opera"],  2, "Orchestral",      "Classical & orchestral — composed and timeless"),
    (["baroque","renaissance","medieval","early music"],       2, "Early Music",     "Baroque & early music — historical grandeur"),
    (["piano","solo piano","composer","contemporary classical"],2, "Piano Works",    "Piano-forward classical and contemporary"),
    (["country","americana","bluegrass","country rock"],       2, "Country Roads",   "Country & americana — guitars, stories, space"),
    (["gospel","christian","worship","praise","spiritual"],    2, "Spirit & Soul",   "Gospel & worship — uplifting and devotional"),
    (["dutch","nederpop","nederlandstalig","levenslied"],      2, "Dutch Vibes",     "Nederlandstalig — Dutch-language music"),
]

# ── Exported sets (loaded once at import time) ────────────────────────────────
NOISE_TAGS  = _load_json_set("noise_tags.json",  _DEFAULT_NOISE_TAGS)
ENERGY_POS  = _load_json_set("energy_pos.json",  _DEFAULT_ENERGY_POS)
ENERGY_NEG  = _load_json_set("energy_neg.json",  _DEFAULT_ENERGY_NEG)
MOOD_POS    = _load_json_set("mood_pos.json",     _DEFAULT_MOOD_POS)
MOOD_NEG    = _load_json_set("mood_neg.json",     _DEFAULT_MOOD_NEG)
GENRE_RULES = _load_genre_rules("genre_rules.json", [(set(r[0]), r[1], r[2], r[3]) for r in _DEFAULT_GENRE_RULES])

# ── Skip-tags for naming ──────────────────────────────────────────────────────
SKIP_NAME_TAGS = _load_json_set("skip_name_tags.json", [
    "pop","rock","usa","american","alternative","indie","female vocalist","male vocalist",
    "seen live","british","dutch","german","french","electronic","music","spotify",
    "youtube","playlist","favorites","favourite","awesome","cool","great","good","all",
    "singer","band","artist","song","track","classic rock","male vocalists","female vocalists",
])


def write_default_configs():
    """Write default JSON config files for first-time setup.  Skips files that already exist."""
    try:
        os.makedirs(_CONFIG_DIR, exist_ok=True)
        _write_if_missing("noise_tags.json",  sorted(_DEFAULT_NOISE_TAGS))
        _write_if_missing("energy_pos.json",  sorted(_DEFAULT_ENERGY_POS))
        _write_if_missing("energy_neg.json",  sorted(_DEFAULT_ENERGY_NEG))
        _write_if_missing("mood_pos.json",    sorted(_DEFAULT_MOOD_POS))
        _write_if_missing("mood_neg.json",    sorted(_DEFAULT_MOOD_NEG))
        _write_if_missing("genre_rules.json", [list(r) for r in _DEFAULT_GENRE_RULES])
        _write_if_missing("skip_name_tags.json", sorted(SKIP_NAME_TAGS))
    except OSError:
        pass  # read-only filesystem -- use in-memory defaults


def _write_if_missing(filename, data):
    path = os.path.join(_CONFIG_DIR, filename)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"Created default config: {path}")