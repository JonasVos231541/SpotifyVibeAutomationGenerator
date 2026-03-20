"""
Clustering pipeline: HDBSCAN + cohesion enforcement + health scoring.

Entry points:
  - ``cluster_records``  — full pipeline (embedding → HDBSCAN → refinement)
  - ``split_cluster``    — hierarchical sub-split
  - ``compute_health``   — cohesion + separation metric
"""
import json, logging, random
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from . import config
from .embeddings import build_embeddings, build_hybrid_vectors, AUDIO_FEATURE_KEYS
from .naming import score_axis, energy_band, mood_band, name_all_clusters, generate_vibe_categories
from .spotify_client import fetch_audio_features
from .events import publish as sse_publish

log = logging.getLogger("splitter.clustering")


# ─── Health metric (cohesion + separation) ────────────────────────────────────

def compute_health(vectors, all_centroids=None, cluster_label=None):
    """
    Cluster health 0-100 combining:
      - Cohesion  (60%): mean intra-cluster cosine similarity
      - Separation (40%): distance to nearest foreign centroid
    Falls back to cohesion-only when centroid context is unavailable.
    """
    if len(vectors) < 2:
        return 100
    X = np.array(vectors, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        return 50
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    if len(X) > 150:
        idx = np.random.choice(len(X), 150, replace=False)
        X = X[idx]

    sim   = cosine_similarity(X)
    n     = len(X)
    upper = sim[np.triu_indices(n, k=1)]
    cohesion = float(np.mean(upper))

    separation = 1.0
    if all_centroids is not None and cluster_label is not None:
        centroid = X.mean(axis=0).reshape(1, -1)
        min_sim  = 1.0
        for lbl, c in all_centroids.items():
            if lbl == cluster_label:
                continue
            s = float(cosine_similarity(centroid, c.reshape(1, -1))[0, 0])
            if s < min_sim:
                min_sim = s
        separation = max(0.0, min(1.0, (0.8 - min_sim) / 0.5))

    health = 0.6 * cohesion + 0.4 * separation
    return round(max(0, min(100, health * 100)), 1)


# ─── build_cluster_dict ──────────────────────────────────────────────────────

def build_cluster_dict(records, labels, embeddings=None, key_prefix="",
                       overrides=None, all_centroids=None):
    """Build the cluster output dict from records + integer label array."""
    if overrides:
        for i, r in enumerate(records):
            if r["id"] in overrides:
                try:
                    labels[i] = int(overrides[r["id"]])
                except (ValueError, TypeError):
                    pass

    groups = {}
    for i, (r, lbl) in enumerate(zip(records, labels)):
        groups.setdefault(lbl, []).append((i, r))

    if all_centroids is None and embeddings is not None:
        all_centroids = {}
        for lbl, idx_recs in groups.items():
            sub_idxs = [i for i, _ in idx_recs]
            if sub_idxs:
                all_centroids[lbl] = embeddings[sub_idxs].mean(axis=0)

    raw = {}
    for lbl, idx_recs in groups.items():
        sub      = [r for _, r in idx_recs]
        sub_idxs = [i for i, _ in idx_recs]
        tag_counter = Counter(tag for r in sub for tag in r.get("tags", []))
        top_tags    = [t for t, _ in tag_counter.most_common(10)]
        e_score     = score_axis(tag_counter, config.ENERGY_POS, config.ENERGY_NEG)
        m_score     = score_axis(tag_counter, config.MOOD_POS,   config.MOOD_NEG)
        key = f"{key_prefix}{lbl}" if key_prefix else str(lbl)
        raw[key] = {
            "sub": sub, "sub_idxs": sub_idxs, "top_tags": top_tags,
            "e": e_score, "m": m_score, "lbl": lbl,
            "quad": (energy_band(e_score), mood_band(m_score)),
        }

    names    = name_all_clusters(raw)
    clusters = {}
    for key, data in raw.items():
        sub = data["sub"]
        sname, desc, e, m, pinterest, ai_sugg = names[key]  # 6 elements
        if embeddings is not None and len(data["sub_idxs"]) > 1:
            X_sub  = embeddings[data["sub_idxs"]]
            health = compute_health(X_sub, all_centroids=all_centroids,
                                    cluster_label=data["lbl"])
        else:
            health = 50
        clusters[key] = {
            "cluster_id":        key,
            "suggested_name":    sname,
            "description":       desc,
            "track_count":       len(sub),
            "top_tags":          data["top_tags"],
            "energy_score":      e,
            "mood_score":        m,
            "health":            health,
            "pinterest_prompts": pinterest,
            "ai_suggestions":    ai_sugg,
            "sample_tracks":     [{"name": r["name"], "artist": r["artist"], "id": r["id"]}
                                  for r in sub[:6]],
            "track_ids":         [r["id"] for r in sub],
            "track_uris":        [r.get("uri", f"spotify:track:{r['id']}") for r in sub],
            "parent":            key_prefix.rstrip(".") if key_prefix else None,
            "can_split":         len(sub) >= 6,
        }
    return clusters


# ─── Must‑link override enforcement ───────────────────────────────────────────

def _apply_overrides(labels, records, overrides):
    """Force tracks with overrides into their target clusters."""
    if not overrides:
        return labels
    # Map override target to an existing label (if any)
    label_set = set(labels)
    for i, r in enumerate(records):
        if r["id"] in overrides:
            target = overrides[r["id"]]
            # convert to int if it's an integer string
            try:
                target_int = int(target)
            except ValueError:
                target_int = target  # keep as string (for sub-cluster keys)
            if target_int in label_set:
                labels[i] = target_int
    return labels


# ─── Cohesion-based splitting ─────────────────────────────────────────────────

def _enforce_cohesion(records, labels, X, max_fraction=None, cohesion_floor=None):
    """
    Post-process: split clusters based on size AND internal cohesion.

    A large but tight cluster stays intact.  A large incoherent cluster is split.
    """
    if max_fraction is None:
        max_fraction = config.MAX_FRACTION
    if cohesion_floor is None:
        cohesion_floor = config.COHESION_FLOOR

    labels   = list(labels)
    n_total  = len(records)
    soft_max = max(15, int(n_total * max_fraction))
    hard_max = max(30, int(n_total * max_fraction * 2))
    next_lbl = max(labels) + 1
    n_splits = 0
    MAX_SPLITS = 20

    while n_splits < MAX_SPLITS:
        counts     = Counter(labels)
        candidates = []
        for lbl, cnt in counts.items():
            if cnt < 8:
                continue
            idxs  = [i for i, l in enumerate(labels) if l == lbl]
            X_sub = X[idxs]
            X_check = X_sub[np.random.choice(len(X_sub), min(100, len(X_sub)), replace=False)] if len(X_sub) > 100 else X_sub
            sim   = cosine_similarity(X_check)
            n_c   = len(X_check)
            upper = sim[np.triu_indices(n_c, k=1)]
            coh   = float(np.mean(upper))
            should_split = (cnt > soft_max and coh < cohesion_floor) or cnt > hard_max
            if should_split:
                candidates.append((lbl, cnt, coh, idxs))

        if not candidates:
            break
        candidates.sort(key=lambda x: (-x[1], x[2]))
        lbl, cnt, coh, idxs = candidates[0]
        X_sub = X[idxs]
        km2 = KMeans(n_clusters=2, random_state=n_splits, n_init=10, max_iter=300)
        sub_labels = km2.fit_predict(X_sub)
        g0 = sum(1 for s in sub_labels if s == 0)
        g1 = sum(1 for s in sub_labels if s == 1)
        min_group = max(5, int(cnt * 0.1))
        if g0 < min_group or g1 < min_group:
            log.info(f"  Cohesion split: can't split {lbl} ({cnt}, coh={coh:.2f}) ({g0}/{g1})")
            break
        for pos, sub_lbl in zip(idxs, sub_labels):
            if sub_lbl == 1:
                labels[pos] = next_lbl
        log.info(f"  Cohesion split: {lbl} ({cnt}, coh={coh:.2f}) → {lbl}({g0}) + {next_lbl}({g1})")
        next_lbl += 1
        n_splits += 1
    return labels


# ─── Main pipeline ────────────────────────────────────────────────────────────

def cluster_records(records, n, overrides, sm=None, state=None, sp=None,
                    granularity=None, extra_features=None, use_llm_guidance=None):
    """
    Full clustering pipeline.

    :param granularity: int from 1 to 10 (optional) – influences target cluster count.
    :param extra_features: (n_tracks, m) matrix of additional features (year, popularity).
    :param use_llm_guidance: bool – if True, use Llama to suggest categories for seeding.
    """
    n_tracks = len(records)
    if not records:
        if sm and state:
            sm.add_log(state, "No tracks to cluster"); sm.save(state)
        return {}
    if n_tracks < 3:
        if sm and state:
            sm.add_log(state, f"Warning: only {n_tracks} tracks"); sm.save(state)
        return build_cluster_dict(records, [0] * n_tracks, overrides=overrides)

    # Auto‑tune based on granularity
    if granularity is not None:
        # Map 1‑10 to target cluster count (e.g., 3 to 20)
        target_k = int(3 + (granularity - 1) * (20 - 3) / 9)
        target_size = max(10, n_tracks // target_k)
        expected_k = target_k
    else:
        target_size = config.TARGET_PLAYLIST_SIZE
        expected_k  = max(3, round(n_tracks / target_size))

    min_cs = max(6, min(40, n_tracks // (expected_k * 3)))
    max_frac = max(0.08, min(0.25, 1.5 / expected_k))

    log.info(f"[cluster] {n_tracks} tracks → ~{expected_k} playlists, min_cs={min_cs}")
    if sm and state:
        sm.add_log(state, f"Building embeddings for {n_tracks} tracks...")
        sm.save(state)
    sse_publish("progress", {"step": "embedding", "pct": 10})

    X_embed = build_embeddings(records)
    sse_publish("progress", {"step": "embedding", "pct": 30})

    # Audio features (only fetch when weight > 0)
    audio_feats = None
    if sp is not None and config.AUDIO_WEIGHT > 0:
        try:
            if sm and state:
                sm.add_log(state, "Fetching audio features..."); sm.save(state)
            audio_feats = fetch_audio_features(sp, [r["id"] for r in records], sm, state)
        except Exception as e:
            log.warning(f"Audio features failed: {e}")

    # Build hybrid vectors with possible extra features
    X, audio_stats = build_hybrid_vectors(X_embed, records, audio_feats,
                                          extra_features=extra_features)
    log.info(f"Hybrid vectors: {X.shape}")

    sse_publish("progress", {"step": "hybrid_vectors", "pct": 45})
    if sm and state:
        sm.add_log(state, f"Clustering ~{expected_k} vibes..."); sm.save(state)

    # Decide whether to use LLM guidance (prefer passed value, fallback to config)
    use_llm = use_llm_guidance if use_llm_guidance is not None else config.USE_LLM_GUIDANCE

    if use_llm and len(records) >= config.LLM_GUIDANCE_SAMPLE_SIZE:
        # Take a random sample
        sample_size = min(config.LLM_GUIDANCE_SAMPLE_SIZE, len(records))
        sample_indices = random.sample(range(len(records)), sample_size)
        sample_records = [records[i] for i in sample_indices]

        # Get categories from Llama
        categories = generate_vibe_categories(
            sample_records,
            min_cats=config.LLM_GUIDANCE_MIN_CATEGORIES,
            max_cats=config.LLM_GUIDANCE_MAX_CATEGORIES
        )

        if categories and len(categories) >= 3:
            log.info(f"Llama suggested {len(categories)} vibe categories: {categories}")
            if sm and state:
                sm.add_log(state, f"Llama suggested {len(categories)} vibe categories")

            # Convert categories to embeddings
            # We need dummy records with just the category name as "tags"
            cat_records = []
            for cat in categories:
                # Create a minimal record structure that build_embeddings can process
                cat_records.append({
                    "tags": cat.split(),  # crude, but will work
                    "artist": "",
                    "name": cat,
                    "tag_dict": {tag: 50 for tag in cat.split()}
                })
            cat_embeddings = build_embeddings(cat_records)

            # Pad category embeddings to match hybrid vector dimensions
            # (build_embeddings returns TFIDF_DIM-d, but X may be wider after audio/extra fusion)
            if cat_embeddings.shape[1] < X.shape[1]:
                pad_width = X.shape[1] - cat_embeddings.shape[1]
                cat_embeddings = np.hstack([
                    cat_embeddings,
                    np.zeros((cat_embeddings.shape[0], pad_width))
                ])

            # Use these as initial centroids for KMeans
            n_cats = len(categories)
            # Adjust expected_k to the number of categories
            expected_k = n_cats
            min_cs = max(6, min(40, n_tracks // (expected_k * 3)))

            # Run KMeans with these centroids as init
            kmeans = KMeans(n_clusters=n_cats, init=cat_embeddings, n_init=3, max_iter=500, random_state=42)
            raw_labels = kmeans.fit_predict(X)
            used_llm = True
        else:
            # Fallback to normal HDBSCAN
            used_llm = False
            if sm and state:
                sm.add_log(state, "Llama category generation failed, falling back to HDBSCAN")
    else:
        used_llm = False

    if not used_llm:
        import hdbscan
        # L2-normalize so euclidean distance ∝ cosine distance
        # (avoids metric compatibility issues across sklearn/hdbscan versions)
        from sklearn.preprocessing import normalize
        X_norm = normalize(X, norm="l2")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs, min_samples=3,
            metric="euclidean", cluster_selection_method="eom",
        )
        raw_labels = clusterer.fit_predict(X_norm)
        unique = set(raw_labels)
        n_clusters = len([l for l in unique if l >= 0])
        n_noise = sum(1 for l in raw_labels if l == -1)
        log.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise")
        if sm and state:
            sm.add_log(state, f"Found {n_clusters} natural vibes ({n_noise} unassigned)")
            sm.save(state)

        # Assign noise to nearest centroid
        if n_clusters > 0 and n_noise > 0:
            centroids = {lbl: X[raw_labels == lbl].mean(axis=0) for lbl in unique if lbl >= 0}
            c_labels  = sorted(centroids.keys())
            c_matrix  = np.array([centroids[l] for l in c_labels])
            for i in range(len(raw_labels)):
                if raw_labels[i] == -1:
                    sims = cosine_similarity(X[i:i+1], c_matrix)[0]
                    raw_labels[i] = c_labels[np.argmax(sims)]

        if n_clusters < 2:
            log.warning(f"Falling back to KMeans K={expected_k}")
            if sm and state:
                sm.add_log(state, f"Falling back to KMeans K={expected_k}"); sm.save(state)
            raw_labels = KMeans(n_clusters=expected_k, random_state=42, n_init=10).fit_predict(X)

        labels = list(raw_labels)

    else:
        # already have labels from KMeans with LLM seeds
        labels = list(raw_labels)

    sse_publish("progress", {"step": "clustering", "pct": 70})

    # Apply must‑link overrides
    if overrides:
        labels = _apply_overrides(labels, records, overrides)

    # Persist centroid stats for incremental classification / drift detection
    # Build label→index mapping once (avoids repeated O(n) scans)
    label_groups = {}
    for i, lbl in enumerate(labels):
        label_groups.setdefault(lbl, []).append(i)

    centroids_save = {}
    semantic_centroids = {}
    centroid_stats = {}
    for lbl, mask in label_groups.items():
        X_sub = X[mask]
        centroid = X_sub.mean(axis=0)
        centroids_save[lbl] = centroid
        # Also save semantic-only centroids for fallback classification
        semantic_centroids[lbl] = X_embed[mask].mean(axis=0)
        dists = 1.0 - cosine_similarity(X_sub, centroid.reshape(1, -1)).flatten()
        centroid_stats[lbl] = {
            "mean_dist": float(np.mean(dists)),
            "std_dist":  float(np.std(dists)),
            "count":     len(mask),
        }
    # Save model as npz (arrays) + json (metadata) — no pickle
    _save_arrays = {}
    for lbl, c in centroids_save.items():
        _save_arrays[f"centroid_{lbl}"] = np.asarray(c)
    for lbl, c in semantic_centroids.items():
        _save_arrays[f"sem_centroid_{lbl}"] = np.asarray(c)
    np.savez(config.MODEL_FILE, **_save_arrays)

    _meta = {
        "version": 3,
        "type": "hybrid_embeddings",
        "centroid_keys": [str(k) for k in centroids_save.keys()],
        "centroid_stats": {str(k): v for k, v in centroid_stats.items()},
        "audio_weight": config.AUDIO_WEIGHT if audio_feats else 0.0,
        "audio_stats": {str(k): v for k, v in (audio_stats or {}).items()},
        "audio_feature_keys": list(AUDIO_FEATURE_KEYS),
        "embed_dim": int(X_embed.shape[1]),
        "hybrid_dim": int(X.shape[1]),
        "has_audio": bool(audio_feats),
        "has_extra": extra_features is not None,
        "hdbscan_metric": "euclidean (L2-normalized)",
    }
    with open(config.MODEL_META_FILE, "w") as f:
        json.dump(_meta, f, indent=2)

    sse_publish("progress", {"step": "model_saved", "pct": 80})
    if sm and state:
        sm.add_log(state, "Refining clusters by cohesion..."); sm.save(state)
    labels = _enforce_cohesion(records, labels, X, max_fraction=max_frac)

    # Merge tiny clusters (< 10 tracks) into nearest larger neighbor
    MIN_CLUSTER = 10
    labels = list(labels)
    merge_rounds = 0
    while merge_rounds < 20:
        # Build label→index mapping once per round
        groups = {}
        for i, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(i)
        tiny  = [(lbl, idxs) for lbl, idxs in groups.items() if len(idxs) < MIN_CLUSTER]
        large = {lbl: idxs for lbl, idxs in groups.items() if len(idxs) >= MIN_CLUSTER}
        if not tiny or not large:
            break
        large_centroids = {lbl: X[idxs].mean(axis=0) for lbl, idxs in large.items()}
        c_labels = sorted(large_centroids.keys())
        c_matrix = np.array([large_centroids[l] for l in c_labels])
        merged_any = False
        for t_lbl, t_idxs in tiny:
            t_centroid = X[t_idxs].mean(axis=0).reshape(1, -1)
            sims = cosine_similarity(t_centroid, c_matrix)[0]
            nearest = c_labels[np.argmax(sims)]
            for i in t_idxs:
                labels[i] = nearest
            log.info(f"  Merged tiny cluster {t_lbl} ({len(t_idxs)} tracks) → {nearest}")
            merged_any = True
        if not merged_any:
            break
        merge_rounds += 1

    # Recompute centroids after splits + merges
    final_groups = {}
    for i, lbl in enumerate(labels):
        final_groups.setdefault(lbl, []).append(i)
    final_centroids = {lbl: X[idxs].mean(axis=0) for lbl, idxs in final_groups.items()}

    sse_publish("progress", {"step": "finalizing", "pct": 95})
    final_k = len(set(labels))
    sizes = sorted(Counter(labels).values(), reverse=True)
    log.info(f"[cluster] Final: {final_k} playlists, sizes: {sizes}")
    if sm and state:
        sm.add_log(state, f"{final_k} vibes — sizes: {', '.join(str(s) for s in sizes[:8])}")
        sm.save(state)

    return build_cluster_dict(records, labels, embeddings=X, overrides=overrides,
                              all_centroids=final_centroids)


def split_cluster(parent_key, track_ids, cache, n_splits=None, overrides=None):
    """Re-cluster one vibe into sub-vibes.  Propagates parent overrides."""
    records = [cache[tid] for tid in track_ids if tid in cache]
    if len(records) < 6:
        return {}
    X = build_embeddings(records)
    max_s  = min(6, max(2, len(records) // 15))
    best_n = min(n_splits, max_s) if n_splits else max(2, min(max_s, 4))
    labels = list(KMeans(n_clusters=best_n, random_state=42, n_init=10).fit_predict(X))

    # Propagate overrides from parent tracks into the correct child cluster
    child_overrides = None
    if overrides:
        child_overrides = {}
        for i, r in enumerate(records):
            if r["id"] in overrides:
                child_overrides[r["id"]] = overrides[r["id"]]
    return build_cluster_dict(records, labels, embeddings=X,
                              key_prefix=f"{parent_key}.", overrides=child_overrides)


# ─── Stats ────────────────────────────────────────────────────────────────────

def compute_stats(cache):
    if not cache:
        return {}
    all_tags = Counter(tag for r in cache.values() for tag in r.get("tags", []))
    energy_scores, mood_scores = [], []
    for r in cache.values():
        tc = Counter(r.get("tags", []))
        energy_scores.append(score_axis(tc, config.ENERGY_POS, config.ENERGY_NEG))
        mood_scores.append(score_axis(tc, config.MOOD_POS,     config.MOOD_NEG))

    def bucket(scores, labels):
        buckets = Counter()
        for s in scores:
            if   s >= 0.60: buckets[labels[0]] += 1
            elif s >= 0.40: buckets[labels[1]] += 1
            else:           buckets[labels[2]] += 1
        return dict(buckets)

    return {
        "total_tracks": len(cache),
        "top_tags":     all_tags.most_common(20),
        "energy_dist":  bucket(energy_scores, ["High", "Mid", "Low"]),
        "mood_dist":    bucket(mood_scores,   ["Bright", "Neutral", "Dark"]),
        "avg_energy":   round(float(np.mean(energy_scores)), 2),
        "avg_mood":     round(float(np.mean(mood_scores)), 2),
    }