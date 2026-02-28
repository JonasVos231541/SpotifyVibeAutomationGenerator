"""
Incremental classification of new tracks against existing cluster centroids.

Features:
  - Confidence margin scoring (gap between best and second-best match)
  - Distribution drift detection (track distance vs. cluster stats)
  - Auto-assign high-confidence tracks; inbox low-confidence ones
  - Recluster trigger when drift exceeds threshold
"""
import pickle, os, logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from . import config
from .embeddings import build_embeddings

log = logging.getLogger("splitter.incremental")


def _classify_with_confidence(vec, centroids, centroid_stats=None):
    """
    Classify one embedding vector against cluster centroids.

    Returns ``(best_label, confidence, is_confident, is_drifted)``.
    """
    c_labels = sorted(centroids.keys())
    c_matrix = np.array([centroids[l] for l in c_labels])
    sims     = cosine_similarity(vec.reshape(1, -1), c_matrix)[0]
    sorted_idx = np.argsort(sims)[::-1]

    best_lbl    = c_labels[sorted_idx[0]]
    best_sim    = float(sims[sorted_idx[0]])
    second_sim  = float(sims[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    margin      = best_sim - second_sim
    is_confident = margin >= config.CONFIDENCE_MARGIN and best_sim >= config.CONFIDENCE_MIN_SIM

    is_drifted = False
    if centroid_stats and best_lbl in centroid_stats:
        stats = centroid_stats[best_lbl]
        dist  = 1.0 - best_sim
        if dist > stats["mean_dist"] + config.DRIFT_SIGMA * stats["std_dist"]:
            is_drifted = True

    return best_lbl, margin, is_confident, is_drifted


def classify_new_tracks(records, sm=None, state=None):
    """
    Classify a batch of new track records against saved centroids.

    Modifies each record in-place:
      - ``predicted_cluster``
      - ``classification_confidence``
      - ``auto_assigned`` (True if above auto-assign threshold)

    Returns ``(n_auto, n_low_conf, n_drifted)``.
    """
    if not records or not os.path.exists(config.MODEL_FILE):
        return 0, 0, 0

    with open(config.MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    centroids      = model.get("centroids", {})
    centroid_stats  = model.get("centroid_stats", {})
    if not centroids:
        return 0, 0, 0

    X_new = build_embeddings(records)
    n_auto, n_low_conf, n_drifted = 0, 0, 0

    for rec, vec in zip(records, X_new):
        lbl, conf, is_conf, is_drift = _classify_with_confidence(
            vec, centroids, centroid_stats)
        rec["predicted_cluster"] = str(lbl)
        rec["classification_confidence"] = round(conf, 3)

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
                sm.add_log(state, f"⚠ High drift ({drift_ratio:.0%}) — auto-recluster recommended")
        elif drift_ratio > config.RECLUSTER_DRIFT_THRESHOLD and total >= 10:
            state["drift_warning"] = True
            if sm:
                sm.add_log(state, f"⚠ Drift detected ({drift_ratio:.0%}) — consider reclustering")
        else:
            state.pop("drift_warning", None)

    return n_auto, n_low_conf, n_drifted
