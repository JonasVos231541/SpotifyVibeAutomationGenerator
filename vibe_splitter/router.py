"""
Track routing engine.

Routes new tracks to the best-matching target playlist using a blended
embedding of:
  (a) the target playlist description + name + custom description text
  (b) the centroid of songs already cached with embeddings in that playlist

Blending: target_vec = normalize(BLEND_ALPHA * desc_vec + (1-BLEND_ALPHA) * song_centroid)
If no cached song embeddings exist, falls back to description-only.
"""
import logging
import numpy as np
from sklearn.preprocessing import normalize

from . import db, config
from .spotify_client import fetch_tracks_from_source

log = logging.getLogger("splitter.router")

ROUTE_THRESHOLD = 0.45   # minimum cosine similarity to auto-assign
BLEND_ALPHA = 0.35       # weight of description vs songs (0=songs only, 1=desc only)


def _embed_text(text):
    """Encode a plain text string with fastembed. Returns L2-normalized (384,) vector."""
    from .embeddings import _get_model
    model = _get_model()
    vec = list(model.embed([text]))[0]
    arr = np.array(vec, dtype=np.float32)
    return normalize(arr.reshape(1, -1))[0]


def build_target_embedding(target_config, sp):
    """
    Build and persist the blended embedding for a target playlist.

    Steps:
    1. Encode description text (name + description + custom_description).
    2. Fetch track IDs from the Spotify playlist.
    3. Look up their embeddings from the DB (no extra tagging needed).
    4. Compute song_centroid from whatever embeddings are cached.
    5. Blend: normalize(BLEND_ALPHA * desc_vec + (1-BLEND_ALPHA) * song_centroid).
    6. Save to DB.

    Returns the blended numpy vector (EMBED_DIM-dim, L2-normalized).
    """
    spotify_id = target_config["spotify_id"]
    name = target_config.get("name", "")
    description = target_config.get("description", "")
    custom_description = target_config.get("custom_description", "")

    # Build description embedding
    text_parts = [p.strip() for p in [name, description, custom_description] if p.strip()]
    full_text = ". ".join(text_parts) or "music playlist"
    desc_vec = _embed_text(full_text)

    # Fetch existing song IDs from the Spotify playlist
    song_count = 0
    song_centroid = None
    try:
        tracks = fetch_tracks_from_source(sp, {"type": "playlist", "id": spotify_id})
        track_ids = [t["id"] for t in tracks if t.get("id")]
        song_count = len(track_ids)

        if track_ids:
            cached = db.get_embeddings_batch(track_ids)
            if cached:
                vecs = []
                for raw in cached.values():
                    v = np.frombuffer(raw, dtype=np.float32)
                    if len(v) == config.EMBED_DIM:
                        vecs.append(v)
                if vecs:
                    centroid = np.mean(vecs, axis=0)
                    song_centroid = normalize(centroid.reshape(1, -1))[0]
                    log.info(f"[router] {spotify_id}: {len(vecs)}/{song_count} songs cached for centroid")
    except Exception as e:
        log.warning(f"[router] Failed to fetch tracks for target {spotify_id}: {e}")

    # Blend
    if song_centroid is not None:
        blended = BLEND_ALPHA * desc_vec + (1 - BLEND_ALPHA) * song_centroid
        blended = normalize(blended.reshape(1, -1))[0]
    else:
        blended = desc_vec

    db.save_target_embedding(spotify_id, desc_vec, song_centroid, song_count)
    log.info(f"[router] Built embedding for '{name}' (song_count={song_count})")
    return blended


def get_target_vectors(state):
    """
    Load blended target vectors from DB for all configured targets.
    Returns {spotify_id: np.ndarray (EMBED_DIM,)}.
    """
    targets = state.get("targets", [])
    if not targets:
        return {}

    result = {}
    for t in targets:
        tid = t["spotify_id"]
        row = db.get_target_embedding(tid)
        if not row:
            continue
        desc_bytes, song_bytes, song_count = row
        if not desc_bytes:
            continue

        desc_vec = np.frombuffer(desc_bytes, dtype=np.float32)
        if len(desc_vec) != config.EMBED_DIM:
            continue

        if song_bytes and song_count:
            song_centroid = np.frombuffer(song_bytes, dtype=np.float32)
            if len(song_centroid) == config.EMBED_DIM:
                blended = BLEND_ALPHA * desc_vec + (1 - BLEND_ALPHA) * song_centroid
                result[tid] = normalize(blended.reshape(1, -1))[0]
                continue

        result[tid] = normalize(desc_vec.reshape(1, -1))[0]

    return result


def route_tracks(records, embeddings_matrix, target_vectors, threshold=None):
    """
    Route records to the best-matching target playlist.

    Args:
        records: list of track dicts (must align with embeddings_matrix rows)
        embeddings_matrix: np.ndarray (n, EMBED_DIM) of L2-normalized track embeddings
        target_vectors: {spotify_id: np.ndarray} from get_target_vectors()
        threshold: cosine similarity cutoff (default: ROUTE_THRESHOLD or DB config)

    Modifies records in place:
        - auto_assigned (bool)
        - assigned_target (spotify_id string, only when auto_assigned=True)
        - route_confidence (float)

    Returns:
        (auto_list, inbox_list) — partitioned record lists
    """
    if threshold is None:
        threshold = float(db.get_config("route_threshold") or ROUTE_THRESHOLD)

    if not target_vectors or len(records) == 0:
        for rec in records:
            rec["auto_assigned"] = False
        return [], list(records)

    target_ids = list(target_vectors.keys())
    # Stack target vectors: (n_targets, dim)
    T = np.stack([target_vectors[tid] for tid in target_ids])

    auto = []
    inbox = []

    for i, rec in enumerate(records):
        vec = embeddings_matrix[i]
        # Cosine similarity (all vectors are L2-normalized)
        sims = T.dot(vec)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        rec["route_confidence"] = best_score
        if best_score >= threshold:
            rec["auto_assigned"] = True
            rec["assigned_target"] = target_ids[best_idx]
            auto.append(rec)
        else:
            rec["auto_assigned"] = False
            inbox.append(rec)

    return auto, inbox


def refresh_all_target_embeddings(targets, sp, sm, state):
    """
    Rebuild embeddings for all configured targets.
    Called at the start of each hourly job so song centroids stay current.
    """
    if not targets:
        return
    for target in targets:
        name = target.get("name", target["spotify_id"])
        try:
            build_target_embedding(target, sp)
        except Exception as e:
            log.warning(f"[router] Failed to refresh embedding for '{name}': {e}")
