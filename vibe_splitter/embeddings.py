"""
Semantic embedding construction and audio-feature fusion.

Builds a structured natural-language sentence per track, encodes it via
sentence-transformers, then optionally fuses Spotify audio features to
create a hybrid vector that captures both semantic *and* sonic character.
"""
import logging
import numpy as np
from collections import Counter
from . import config
from .naming import score_axis  # energy/mood scoring

log = logging.getLogger("splitter.embeddings")

AUDIO_FEATURE_KEYS = ["danceability", "energy", "valence", "acousticness",
                      "instrumentalness", "tempo", "speechiness", "mode",
                      "liveness", "loudness"]
AUDIO_DIM = len(AUDIO_FEATURE_KEYS)

# ── Lazy-loaded model ─────────────────────────────────────────────────────────
_embed_model = None


def get_embed_model():
    """Lazy-load sentence-transformer model (~80 MB first run)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence-transformer model (first run downloads ~80 MB)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Embedding model loaded")
    return _embed_model


# ── Energy/mood descriptors (5 bands for richer semantic signal) ─────────────

def _energy_descriptor(e):
    """Map energy score to a multi-word descriptor for richer embedding context."""
    if e >= 0.75: return "explosive high energy intense"
    if e >= 0.58: return "energetic driving upbeat"
    if e >= 0.42: return "moderate tempo steady"
    if e >= 0.25: return "relaxed gentle laid-back"
    return "calm ambient still quiet"


def _mood_descriptor(m):
    """Map mood score to a multi-word descriptor for richer embedding context."""
    if m >= 0.75: return "euphoric joyful uplifting"
    if m >= 0.57: return "bright warm positive"
    if m >= 0.43: return "balanced neutral"
    if m >= 0.25: return "melancholic introspective wistful"
    return "dark brooding somber"


# ── Per-track sentence builder ────────────────────────────────────────────────

def _build_embedding_sentence(record):
    """
    Build a semantically rich sentence for one track.

    Uses percentile-based tag tiers to adapt to each track's weight
    distribution, includes track name and artist for thematic context,
    and 5-band energy/mood descriptors for finer-grained sonic character.
    """
    tags   = record.get("tags", [])
    td     = record.get("tag_dict", {})
    # Use all_artists string when available, fall back to primary
    artist = record.get("artists") or record.get("artist", "")
    name   = record.get("name", "")

    # Percentile-based tag tiers (adapts to each track's weight distribution)
    if td and len(td) > 2:
        weights = sorted(td.values(), reverse=True)
        p75 = weights[max(0, len(weights) // 4)]
        p25 = weights[min(len(weights) - 1, 3 * len(weights) // 4)]
    else:
        p75, p25 = 50, 20

    core_set = set()
    core = []
    for t in tags:
        if td.get(t, 0) >= p75 and len(core) < 5:
            core.append(t); core_set.add(t)

    style = [t for t in tags if p25 <= td.get(t, 0) < p75
             and t not in core_set][:3]
    style_set = set(style)

    flavor = [t for t in tags if td.get(t, 0) < p25
              and t not in core_set and t not in style_set][:2]

    parts = []

    # Track name for thematic/lyrical context
    if name:
        parts.append(f"'{name}'")

    # Artist for stylistic context
    if artist:
        parts.append(f"by {artist}")

    # Genre core — repeat top tag for emphasis in the transformer
    if core:
        emphasis = ", ".join(core)
        if len(core) >= 2:
            emphasis = f"{core[0]}, {emphasis}"
        parts.append(emphasis)

    # Style modifiers
    if style:
        parts.append(", ".join(style))

    # Flavour
    if flavor:
        parts.append(", ".join(flavor))

    # Energy/mood — 5-band descriptors for finer-grained matching
    tc = Counter(tags)
    e = score_axis(tc, config.ENERGY_POS, config.ENERGY_NEG)
    m = score_axis(tc, config.MOOD_POS,   config.MOOD_NEG)
    parts.append(f"{_energy_descriptor(e)}, {_mood_descriptor(m)}")

    return "; ".join(parts) if parts else "unknown music"


# ── Batch embedding ───────────────────────────────────────────────────────────

def build_embeddings(records, use_cache=True):
    """
    Encode track records into L2-normalised 384-dim embedding vectors.

    When ``use_cache=True``, looks up pre-computed embeddings from the database
    and only encodes tracks that are missing.  Saves new embeddings back to DB.
    """
    if not records:
        return np.zeros((0, 384))

    n = len(records)
    model = get_embed_model()
    dim = 384
    X = np.zeros((n, dim), dtype=np.float32)
    to_encode_idx = []  # indices in records that need encoding

    # Try loading cached embeddings from DB
    if use_cache:
        try:
            from . import db
            tids = [r.get("id", "") for r in records]
            cached = db.get_embeddings_batch(tids)
            for i, tid in enumerate(tids):
                if tid in cached:
                    X[i] = np.frombuffer(cached[tid], dtype=np.float32)
                else:
                    to_encode_idx.append(i)
            if cached:
                log.info(f"Embedding cache hit: {len(cached)}/{n} tracks")
        except Exception as e:
            log.debug(f"Embedding cache unavailable: {e}")
            to_encode_idx = list(range(n))
    else:
        to_encode_idx = list(range(n))

    # Encode missing tracks
    if to_encode_idx:
        sentences = [_build_embedding_sentence(records[i]) for i in to_encode_idx]
        X_new = model.encode(sentences, show_progress_bar=False, batch_size=256)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1) if len(sentences) == 1 else X_new.reshape(-1, dim)

        for j, idx in enumerate(to_encode_idx):
            X[idx] = X_new[j]

        # Save new embeddings to DB
        if use_cache:
            try:
                from . import db
                for j, idx in enumerate(to_encode_idx):
                    tid = records[idx].get("id", "")
                    if tid:
                        db.save_embedding(tid, X_new[j].astype(np.float32).tobytes())
            except Exception as e:
                log.debug(f"Failed to save embeddings to cache: {e}")

        log.info(f"Encoded {len(to_encode_idx)} new embeddings")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


# ── Extra features (year, popularity) ─────────────────────────────────────────

def extract_extra_features(sp, tracks, include_year=False, include_popularity=False):
    """
    Fetch year (from album release_date) and popularity for each track.
    Returns a numpy array of shape (n_tracks, n_features), or None if both disabled.
    """
    if not (include_year or include_popularity):
        return None
    n = len(tracks)
    extra = np.zeros((n, 2))  # [year_norm, popularity_norm]
    for i, t in enumerate(tracks):
        # Year
        if include_year and t.get("album") and t["album"].get("release_date"):
            try:
                year = int(t["album"]["release_date"][:4])
                year_norm = (year - config.YEAR_MIN) / (config.YEAR_MAX - config.YEAR_MIN)
                extra[i, 0] = max(0, min(1, year_norm))
            except Exception:
                extra[i, 0] = 0.5
        else:
            extra[i, 0] = 0.5

        # Popularity
        if include_popularity:
            extra[i, 1] = t.get("popularity", 50) / 100.0  # 0-100 -> 0-1
        else:
            extra[i, 1] = 0.5

    return extra


# ── Audio feature fusion ──────────────────────────────────────────────────────

def build_hybrid_vectors(embeddings, records, audio_features=None,
                         audio_weight=None, extra_features=None,
                         saved_audio_stats=None):
    """
    Fuse semantic embeddings (384-d) with normalised audio features (6-d)
    and optionally extra features (year, popularity).

    :param saved_audio_stats: dict with 'mean' and 'std' arrays for Z-scoring
                              (used during incremental classification to match
                              the normalization from the original clustering).
    :returns: tuple of (hybrid_vectors, audio_stats_dict_or_None).
    """
    if audio_weight is None:
        audio_weight = config.AUDIO_WEIGHT

    # Start with embeddings
    combined = embeddings.copy()
    audio_stats = None

    # Add audio features if available
    if audio_features:
        n = len(records)
        audio_matrix = np.full((n, AUDIO_DIM), np.nan)
        for i, r in enumerate(records):
            feat = audio_features.get(r["id"])
            if feat:
                for j, key in enumerate(AUDIO_FEATURE_KEYS):
                    val = feat.get(key, 0.0)
                    if key == "tempo":
                        val = min(val / 200.0, 1.0)
                    elif key == "loudness":
                        # Spotify loudness: typically -60 to 0 dB → map to 0-1
                        val = max(0.0, min(1.0, (val + 60.0) / 60.0))
                    # mode (0/1), speechiness, liveness already in 0-1 range
                    audio_matrix[i, j] = val

        # Fill missing values with column means
        col_means = np.nanmean(audio_matrix, axis=0)
        for j in range(AUDIO_DIM):
            mask = np.isnan(audio_matrix[:, j])
            audio_matrix[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.5

        # Z-score: use saved stats for incremental, compute fresh for clustering
        if saved_audio_stats is not None:
            means = np.array(saved_audio_stats["mean"])
            stds  = np.array(saved_audio_stats["std"])
        else:
            means = audio_matrix.mean(axis=0)
            stds  = audio_matrix.std(axis=0)

        stds_safe = stds.copy()
        stds_safe[stds_safe == 0] = 1
        audio_z = (audio_matrix - means) / stds_safe

        # Apply per-feature importance weights (use saved weights for incremental)
        if saved_audio_stats is not None and "importance" in saved_audio_stats:
            saved_imp = saved_audio_stats["importance"]
            importance = np.array([saved_imp.get(k, 1.0) for k in AUDIO_FEATURE_KEYS])
        else:
            importance = np.array([config.AUDIO_IMPORTANCE.get(k, 1.0)
                                   for k in AUDIO_FEATURE_KEYS])
        audio_z = audio_z * importance

        audio_stats = {
            "mean": means.tolist(), "std": stds.tolist(),
            "importance": {k: config.AUDIO_IMPORTANCE.get(k, 1.0)
                           for k in AUDIO_FEATURE_KEYS},
            "feature_keys": list(AUDIO_FEATURE_KEYS),
        }

        # Scale to match embedding magnitude * audio_weight
        embed_mag = np.mean(np.linalg.norm(embeddings, axis=1))
        audio_mag = np.mean(np.linalg.norm(audio_z, axis=1))
        if audio_mag > 0:
            audio_z = audio_z * (embed_mag / audio_mag) * audio_weight

        combined = np.hstack([combined, audio_z])

    # Add extra features (year, popularity) – they are already normalised 0-1
    if extra_features is not None:
        combined = np.hstack([combined, extra_features])

    # Final L2 normalisation
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return combined / norms, audio_stats
