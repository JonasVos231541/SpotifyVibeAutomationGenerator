"""
Incremental classification of new tracks against existing cluster centroids.

Features:
  - Softmax probability + entropy confidence scoring
  - Adaptive confidence thresholds based on cluster tightness
  - Distribution drift detection (track distance vs. cluster stats)
  - Auto-assign high-confidence tracks; inbox low-confidence ones
  - Recluster trigger when drift exceeds threshold
  - Hybrid vector support: builds full hybrid vectors when audio features
    are available, falls back to semantic-only centroids otherwise
"""
import json, os, logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from . import config
from .embeddings import build_embeddings, build_hybrid_vectors

log = logging.getLogger("splitter.incremental")


def _load_model():
    """
    Load model from npz+json.  If only a legacy .pkl exists, migrate it once.
    Returns the model dict (with numpy arrays for centroids) or None.
    """
    meta_path = config.MODEL_META_FILE

    # Migration: legacy pickle → npz+json
    if not os.path.exists(config.MODEL_FILE) and os.path.exists(config._LEGACY_PKL):
        log.warning("Migrating legacy pickle model to npz+json format")
        try:
            import pickle
            with open(config._LEGACY_PKL, "rb") as f:
                old = pickle.load(f)
            # Re-save using the new format (reuse clustering save logic inline)
            arrays = {}
            for lbl, c in old.get("centroids", {}).items():
                arrays[f"centroid_{lbl}"] = np.asarray(c)
            for lbl, c in old.get("semantic_centroids", {}).items():
                arrays[f"sem_centroid_{lbl}"] = np.asarray(c)
            np.savez(config.MODEL_FILE, **arrays)
            meta = {k: v for k, v in old.items()
                    if k not in ("centroids", "semantic_centroids")}
            meta["centroid_keys"] = [str(k) for k in old.get("centroids", {}).keys()]
            # Ensure JSON-serializable types
            meta["centroid_stats"] = {str(k): v for k, v in meta.get("centroid_stats", {}).items()}
            meta["audio_stats"] = {str(k): v for k, v in (meta.get("audio_stats") or {}).items()}
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            log.info("Migration complete — legacy .pkl can be removed")
        except Exception as e:
            log.error(f"Pickle migration failed: {e}")
            return None

    if not os.path.exists(config.MODEL_FILE) or not os.path.exists(meta_path):
        return None

    try:
        npz = np.load(config.MODEL_FILE)
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        log.error(f"Model load failed: {e}")
        return None

    # Reconstruct centroid dicts from npz arrays
    centroid_keys = meta.get("centroid_keys", [])
    centroids = {}
    semantic_centroids = {}
    for k in centroid_keys:
        arr_key = f"centroid_{k}"
        sem_key = f"sem_centroid_{k}"
        if arr_key in npz:
            centroids[int(k) if k.lstrip("-").isdigit() else k] = npz[arr_key]
        if sem_key in npz:
            semantic_centroids[int(k) if k.lstrip("-").isdigit() else k] = npz[sem_key]
    npz.close()

    # Reconstruct centroid_stats with proper key types
    raw_stats = meta.get("centroid_stats", {})
    centroid_stats = {}
    for k, v in raw_stats.items():
        key = int(k) if k.lstrip("-").isdigit() else k
        centroid_stats[key] = v

    model = dict(meta)
    model["centroids"] = centroids
    model["semantic_centroids"] = semantic_centroids
    model["centroid_stats"] = centroid_stats
    return model


def _classify_with_confidence(vec, centroids, centroid_stats=None):
    """
    Classify one embedding vector against cluster centroids using
    softmax-probability + entropy scoring.

    Returns ``(best_label, confidence, is_confident, is_drifted, probs_dict)``.
    """
    c_labels = sorted(centroids.keys())
    c_matrix = np.array([centroids[l] for l in c_labels])
    sims = cosine_similarity(vec.reshape(1, -1), c_matrix)[0]

    # Softmax probabilities over similarities (temperature-scaled)
    temperature = config.CONFIDENCE_TEMPERATURE
    shifted = (sims - sims.max()) / temperature  # subtract max for numerical stability
    exp_sims = np.exp(shifted)
    probs = exp_sims / exp_sims.sum()

    sorted_idx = np.argsort(probs)[::-1]
    best_lbl  = c_labels[sorted_idx[0]]
    best_prob = float(probs[sorted_idx[0]])
    best_sim  = float(sims[sorted_idx[0]])

    # Entropy of probability distribution (normalized to [0, 1])
    safe_probs = np.clip(probs, 1e-10, 1.0)
    entropy = -float(np.sum(safe_probs * np.log(safe_probs)))
    max_entropy = np.log(len(c_labels)) if len(c_labels) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Combined confidence: high when best_prob is high AND entropy is low
    confidence = best_prob * (1.0 - normalized_entropy)

    # Probability margin between top two candidates
    second_prob = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    margin = best_prob - second_prob

    # Adaptive confidence threshold: loose clusters require higher margin
    adaptive_margin = config.CONFIDENCE_MARGIN
    if centroid_stats and len(centroid_stats) > 1:
        all_stds = [s["std_dist"] for s in centroid_stats.values() if "std_dist" in s]
        if all_stds:
            median_std = float(np.median(all_stds))
            cluster_std = centroid_stats.get(best_lbl, {}).get("std_dist", median_std)
            adaptive_margin = (config.CONFIDENCE_MARGIN +
                               config.ADAPTIVE_MARGIN_SCALE *
                               (cluster_std / max(median_std, 1e-6)))

    is_confident = margin >= adaptive_margin and best_sim >= config.CONFIDENCE_MIN_SIM

    # Drift detection
    is_drifted = False
    if centroid_stats and best_lbl in centroid_stats:
        stats = centroid_stats[best_lbl]
        dist = 1.0 - best_sim
        if dist > stats["mean_dist"] + config.DRIFT_SIGMA * stats["std_dist"]:
            is_drifted = True

    # Build probability distribution for UI
    probs_dict = {str(c_labels[i]): round(float(probs[i]), 4)
                  for i in range(len(c_labels))}

    return best_lbl, confidence, is_confident, is_drifted, probs_dict


def classify_new_tracks(records, sm=None, state=None, sp=None):
    """
    Classify a batch of new track records against saved centroids.

    When ``sp`` is provided and the model was trained with audio features,
    fetches audio features for the new tracks and builds matching hybrid
    vectors.  Otherwise falls back to semantic-only centroids.

    Modifies each record in-place:
      - ``predicted_cluster``
      - ``classification_confidence``
      - ``cluster_probabilities``
      - ``auto_assigned`` (True if above auto-assign threshold)

    Returns ``(n_auto, n_low_conf, n_drifted)``.
    """
    if not records or not os.path.exists(config.MODEL_FILE):
        return 0, 0, 0

    model = _load_model()
    if model is None:
        return 0, 0, 0
    centroid_stats = model.get("centroid_stats", {})

    # Model version check
    model_version = model.get("version", 1)
    if model_version < 3:
        log.warning(f"Model version {model_version} detected. "
                    f"Recluster recommended for improved accuracy.")

    # Build semantic embeddings (transform mode — reuse saved TF-IDF pipeline)
    X_new = build_embeddings(records, fit=False)

    # Decide whether to build full hybrid vectors or use semantic fallback
    has_audio = model.get("has_audio", False)
    audio_stats = model.get("audio_stats")
    use_hybrid = False

    if has_audio and sp is not None and audio_stats is not None:
        # Try to build hybrid vectors matching the centroid dimensions
        try:
            from .spotify_client import fetch_audio_features
            audio_feats = fetch_audio_features(sp, [r["id"] for r in records])
            if audio_feats:
                X_hybrid, _ = build_hybrid_vectors(
                    X_new, records, audio_feats,
                    audio_weight=model.get("audio_weight", config.AUDIO_WEIGHT),
                    saved_audio_stats=audio_stats,
                )
                # Verify dimensions match hybrid centroids
                hybrid_dim = model.get("hybrid_dim", 0)
                if X_hybrid.shape[1] == hybrid_dim:
                    X_new = X_hybrid
                    use_hybrid = True
                    log.info("Incremental classification using hybrid vectors")
                else:
                    log.warning(f"Hybrid dim mismatch: {X_hybrid.shape[1]} vs {hybrid_dim}, "
                                f"falling back to semantic centroids")
        except Exception as e:
            log.warning(f"Hybrid vector construction failed, using semantic fallback: {e}")

    # Select matching centroids
    if use_hybrid:
        centroids = model.get("centroids", {})
    else:
        # Use semantic-only centroids if available, otherwise fall back
        centroids = model.get("semantic_centroids")
        if not centroids:
            # Legacy model without semantic centroids — use hybrid centroids
            centroids = model.get("centroids", {})
            embed_dim = model.get("embed_dim", config.TFIDF_DIM)
            hybrid_dim = model.get("hybrid_dim", embed_dim)
            if hybrid_dim != embed_dim and centroids:
                # Truncate hybrid centroids to semantic dimensions as best-effort
                centroids = {k: v[:embed_dim] for k, v in centroids.items()}
                log.warning("Truncating hybrid centroids to semantic dimensions (legacy model)")

    if not centroids:
        return 0, 0, 0

    n_auto, n_low_conf, n_drifted = 0, 0, 0

    for rec, vec in zip(records, X_new):
        lbl, conf, is_conf, is_drift, probs = _classify_with_confidence(
            vec, centroids, centroid_stats)
        rec["predicted_cluster"] = str(lbl)
        rec["classification_confidence"] = round(conf, 3)
        rec["cluster_probabilities"] = probs

        # Auto-assign if confidence is very high
        if conf >= config.AUTO_ASSIGN_THRESHOLD:
            rec["auto_assigned"] = True
            n_auto += 1
        else:
            rec["auto_assigned"] = False

        if not is_conf:
            n_low_conf += 1
        if is_drift:
            n_drifted += 1

    if (n_low_conf > 0 or n_drifted > 0) and sm and state:
        sm.add_log(state,
                   f"Classification: {n_auto} auto-assigned, "
                   f"{n_low_conf} low-confidence, {n_drifted} drifted")

    # Drift warning / auto-recluster flag
    total = max(1, len(records))
    drift_ratio = (n_low_conf + n_drifted) / total
    if state:
        if drift_ratio > config.AUTO_RECLUSTER_DRIFT and total >= 10:
            state["drift_warning"] = "auto_recluster"
            if sm:
                sm.add_log(state, f"High drift ({drift_ratio:.0%}) — auto-recluster recommended")
        elif drift_ratio > config.RECLUSTER_DRIFT_THRESHOLD and total >= 10:
            state["drift_warning"] = True
            if sm:
                sm.add_log(state, f"Drift detected ({drift_ratio:.0%}) — consider reclustering")
        else:
            state.pop("drift_warning", None)

    return n_auto, n_low_conf, n_drifted
