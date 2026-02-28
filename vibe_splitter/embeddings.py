"""
Semantic embedding construction and audio-feature fusion.

Builds a structured natural-language sentence per track, encodes it via
sentence-transformers, then optionally fuses Spotify audio features to
create a hybrid vector that captures both semantic *and* sonic character.
"""
import logging
import numpy as np
from collections import Counter
from datetime import datetime
from . import config
from .naming import score_axis  # energy/mood scoring

log = logging.getLogger("splitter.embeddings")

AUDIO_FEATURE_KEYS = ["danceability", "energy", "valence", "acousticness",
                      "instrumentalness", "tempo"]
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


# ── Per-track sentence builder ────────────────────────────────────────────────

def _build_embedding_sentence(record):
    """
    Build a semantically rich sentence for one track.

    Layers:
      genre core   → high-weight tags (>65)
      style mods   → mid-weight tags (25-65)
      colour       → low-weight tags (<25)
      artist       → ``by {artist}`` context
      energy/mood  → natural language from the same scoring axis
    """
    tags   = record.get("tags", [])
    td     = record.get("tag_dict", {})
    # Use all_artists string when available, fall back to primary
    artist = record.get("artists") or record.get("artist", "")

    high = [t for t in tags if td.get(t, 0) > 65][:4]
    mid  = [t for t in tags if 25 <= td.get(t, 0) <= 65 and t not in high][:3]
    low  = [t for t in tags if td.get(t, 0) < 25 and t not in high and t not in mid][:2]

    parts = []
    if high:
        parts.append(", ".join(high))
    if mid:
        parts.append(f"style: {', '.join(mid)}")
    if low:
        parts.append(", ".join(low))
    if artist:
        parts.append(f"by {artist}")

    # Inject energy/mood from same tag scoring used in clustering/naming
    tc = Counter(tags)
    e = score_axis(tc, config.ENERGY_POS, config.ENERGY_NEG)
    m = score_axis(tc, config.MOOD_POS,   config.MOOD_NEG)
    e_word = "high energy" if e >= 0.58 else ("calm" if e < 0.42 else "moderate energy")
    m_word = "bright uplifting" if m >= 0.57 else ("dark moody" if m < 0.43 else "balanced mood")
    parts.append(f"{e_word}, {m_word}")

    return "; ".join(parts) if parts else "unknown music"


# ── Batch embedding ───────────────────────────────────────────────────────────

def build_embeddings(records):
    """Encode track records into L2-normalised 384-dim embedding vectors."""
    if not records:
        return np.zeros((0, 384))
    model = get_embed_model()
    sentences = [_build_embedding_sentence(r) for r in records]
    X = model.encode(sentences, show_progress_bar=False, batch_size=256)
    if X.ndim == 1:
        X = X.reshape(1, -1) if len(sentences) == 1 else X.reshape(-1, 384)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


# ── Extra features (year, popularity) ─────────────────────────────────────────

def extract_extra_features(sp, tracks, include_year=False, include_popularity=False):
    """
    Fetch year (from album release_date) and popularity for each track.
    Returns a numpy array of shape (n_tracks, n_features).
    """
    n = len(tracks)
    extra = np.zeros((n, 2))  # [year_norm, popularity_norm]
    for i, t in enumerate(tracks):
        # Year
        if include_year and t.get("album") and t["album"].get("release_date"):
            try:
                year = int(t["album"]["release_date"][:4])
                year_norm = (year - config.YEAR_MIN) / (config.YEAR_MAX - config.YEAR_MIN)
                year_norm = max(0, min(1, year_norm))
            except:
                year_norm = 0.5
        else:
            year_norm = 0.5
        extra[i, 0] = year_norm

        # Popularity
        if include_popularity:
            pop = t.get("popularity", 50) / 100.0  # 0-100 -> 0-1
        else:
            pop = 0.5
        extra[i, 1] = pop

    # If both disabled, return None
    if not (include_year or include_popularity):
        return None
    return extra


# ── Audio feature fusion ──────────────────────────────────────────────────────

def build_hybrid_vectors(embeddings, records, audio_features=None,
                         audio_weight=None, extra_features=None):
    """
    Fuse semantic embeddings (384-d) with normalised audio features (6-d)
    and optionally extra features (year, popularity).

    :param extra_features: (n_tracks, m) matrix of additional features, already normalised.
    """
    if audio_weight is None:
        audio_weight = config.AUDIO_WEIGHT

    # Start with embeddings
    combined = embeddings.copy()

    # Add audio features if available
    if audio_features and audio_features:
        n = len(records)
        audio_matrix = np.full((n, AUDIO_DIM), np.nan)
        for i, r in enumerate(records):
            feat = audio_features.get(r["id"])
            if feat:
                for j, key in enumerate(AUDIO_FEATURE_KEYS):
                    val = feat.get(key, 0.0)
                    if key == "tempo":
                        val = min(val / 200.0, 1.0)
                    audio_matrix[i, j] = val

        # Fill missing
        col_means = np.nanmean(audio_matrix, axis=0)
        for j in range(AUDIO_DIM):
            mask = np.isnan(audio_matrix[:, j])
            audio_matrix[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.5

        # Z‑score
        means = audio_matrix.mean(axis=0)
        stds  = audio_matrix.std(axis=0)
        stds[stds == 0] = 1
        audio_z = (audio_matrix - means) / stds

        # Scale to match embedding magnitude * audio_weight
        embed_mag = np.mean(np.linalg.norm(embeddings, axis=1))
        audio_mag = np.mean(np.linalg.norm(audio_z, axis=1))
        if audio_mag > 0:
            audio_z = audio_z * (embed_mag / audio_mag) * audio_weight

        combined = np.hstack([combined, audio_z])

    # Add extra features (year, popularity) – they are already normalised 0‑1
    if extra_features is not None:
        # Scale extra features similarly? We'll just keep them as is
        # (they are already 0‑1, no need to scale further)
        combined = np.hstack([combined, extra_features])

    # Final L2 normalisation
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return combined / norms