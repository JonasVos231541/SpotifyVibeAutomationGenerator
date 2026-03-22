"""
Clustering blueprint — preview, ensemble, confirm, split, merge.
"""
import threading, logging
import numpy as np
from collections import Counter
from datetime import datetime
from flask import Blueprint, request, jsonify
from concurrent.futures import ThreadPoolExecutor

from .. import config, db
from ..state import state_manager as sm
from ..spotify_client import get_sp, fetch_all_tracks
from ..lastfm import build_vectors
from ..embeddings import build_embeddings, build_hybrid_vectors, extract_extra_features
from ..clustering import cluster_records, split_cluster, compute_health
from ..naming import score_axis, energy_band, mood_band, name_all_clusters
from ..playlists import push_playlists
from ..events import publish as sse_publish
from .helpers import _ref, _sanitize_name, _valid_id, _preview_lock, rate_limit

log = logging.getLogger("splitter.routes.clustering")

clustering_bp = Blueprint("clustering", __name__)


@clustering_bp.route("/api/preview", methods=["POST"])
@rate_limit(60)
def api_preview():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    if _preview_lock.locked():
        return jsonify({"ok": True, "running": True, "message": "Preview already running"}), 202

    data = request.json
    sources = data.get("sources", [])
    granularity = data.get("granularity", None)
    include_year = data.get("include_year", False)
    include_popularity = data.get("include_popularity", False)
    use_llm_guidance = data.get("use_llm_guidance", False)

    sm.set_token(t)
    s = sm.load()
    s["preview"] = None; s["preview_running"] = True; s["sources"] = sources
    sm.add_log(s, "Loading tracks from Spotify...")
    sm.save(s)

    def _run(token, sources, granularity, include_year, include_popularity, use_llm_guidance):
        with _preview_lock:
            st = sm.load()
            try:
                sp = get_sp(token)
                sm.add_log(st, "Fetching tracks..."); sm.save(st)
                tracks = fetch_all_tracks(sp, sources, sm=sm, state=st)
                if not tracks:
                    st["preview_running"] = False
                    sm.add_log(st, "No tracks found"); sm.save(st)
                    return
                sm.add_log(st, f"{len(tracks)} tracks loaded"); sm.save(st)

                extra_features = None
                if include_year or include_popularity:
                    sm.add_log(st, "Fetching extra metadata (year, popularity)..."); sm.save(st)
                    extra_features = extract_extra_features(sp, tracks, include_year, include_popularity)

                records = build_vectors(tracks, sm, st,
                                        cb=lambda s2, m: (sm.add_log(s2, m), sm.save(s2)),
                                        sp=sp)
                sm.add_log(st, "Auto-detecting vibes..."); sm.save(st)
                clusters = cluster_records(records, 0, st.get("overrides", {}),
                                           sm=sm, state=st, sp=sp,
                                           granularity=granularity,
                                           extra_features=extra_features,
                                           use_llm_guidance=use_llm_guidance)
                st["sources"] = sources
                st["num_clusters"] = len(clusters)
                st["preview"] = clusters
                st["preview_running"] = False
                st["known_track_ids"] = [tr["id"] for tr in tracks]
                sm.add_log(st, f"DONE:{len(clusters)}")
                sm.save(st)
            except Exception as e:
                log.error(f"[preview] FAILED: {e}", exc_info=True)
                st["preview_running"] = False
                sm.add_log(st, f"Error: {e}"); sm.save(st)

    threading.Thread(target=_run, args=(t, sources, granularity, include_year, include_popularity, use_llm_guidance), daemon=True).start()
    return jsonify({"ok": True, "running": True})


@clustering_bp.route("/api/preview-ensemble", methods=["POST"])
def api_preview_ensemble():
    """Return 3 alternative clusterings (HDBSCAN, KMeans5, KMeans10) as summaries."""
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json
    sources = data.get("sources", [])
    sp = get_sp(t)
    tracks = fetch_all_tracks(sp, sources)
    records = build_vectors(tracks, sm, state=None, sp=sp)
    X_embed = build_embeddings(records)
    X, _ = build_hybrid_vectors(X_embed, records, audio_features=None)

    def _hdbscan():
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(6, len(records)//20),
                                     min_samples=3, metric="cosine", cluster_selection_method="eom")
        labels = clusterer.fit_predict(X)
        return _summarize_clusters(records, labels)

    def _kmeans(k):
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        return _summarize_clusters(records, labels)

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "hdbscan": executor.submit(_hdbscan),
            "kmeans5": executor.submit(_kmeans, 5),
            "kmeans10": executor.submit(_kmeans, 10),
        }
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=60)
            except Exception as e:
                results[name] = {"error": str(e)}
    return jsonify(results)


def _summarize_clusters(records, labels):
    """Return a compact summary for ensemble preview: list of {size, top_tags}."""
    groups = {}
    for i, (r, lbl) in enumerate(zip(records, labels)):
        groups.setdefault(lbl, []).append(r)
    summary = []
    for lbl, recs in groups.items():
        tag_counter = Counter(tag for r in recs for tag in r.get("tags", []))
        top_tags = [t for t, _ in tag_counter.most_common(5)]
        summary.append({
            "size": len(recs),
            "top_tags": top_tags,
        })
    return summary


@clustering_bp.route("/api/confirm-selected", methods=["POST"])
def api_confirm_selected():
    """
    User selected a subset of detected vibes.
    Only create Spotify playlists for selected clusters.
    Tracks from unselected clusters are distributed to their nearest selected cluster.
    """
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    s = sm.load()
    if not s.get("preview"):
        return jsonify({"error": "No preview"}), 400

    data = request.json or {}
    selected_keys = [str(k) for k in data.get("selected", [])]
    confirmed_names = {str(k): _sanitize_name(v) for k, v in data.get("names", {}).items()}
    redistribute = data.get("redistribute", False)
    sm.set_token(t)

    if not selected_keys:
        return jsonify({"error": "Select at least one vibe"}), 400

    full_preview = s["preview"]

    # Only redistribute rejected tracks into selected vibes if explicitly asked
    rejected = [k for k in full_preview if k not in selected_keys]
    if redistribute and rejected and config.MODEL_FILE and __import__("os").path.exists(config.MODEL_FILE):
        all_tids = []
        for k in selected_keys:
            all_tids.extend(full_preview[k].get("track_ids", []))
        for rk in rejected:
            all_tids.extend(full_preview[rk].get("track_ids", []))
        cache_batch = db.get_tracks_batch(all_tids)

        selected_centroids = {}
        for k in selected_keys:
            tids = full_preview[k].get("track_ids", [])
            recs = [cache_batch[tid] for tid in tids if tid in cache_batch]
            if recs:
                X_c = build_embeddings(recs)
                selected_centroids[k] = X_c.mean(axis=0)

        for rk in rejected:
            tids = full_preview[rk].get("track_ids", [])
            recs_to_embed = [cache_batch[tid] for tid in tids if tid in cache_batch]
            if not recs_to_embed or not selected_centroids:
                continue
            X_rejected = build_embeddings(recs_to_embed)
            for tid, vec in zip(tids, X_rejected):
                best_k, best_sim = selected_keys[0], -1
                for sk, centroid in selected_centroids.items():
                    n1 = np.linalg.norm(vec); n2 = np.linalg.norm(centroid)
                    sim = float(np.dot(vec, centroid)/(n1*n2)) if n1 > 0 and n2 > 0 else 0
                    if sim > best_sim:
                        best_sim = sim; best_k = sk
                full_preview[best_k]["track_ids"].append(tid)
                full_preview[best_k]["track_count"] += 1

    final_clusters = {k: full_preview[k] for k in selected_keys}

    try:
        sm.add_log(s, "Creating selected playlists..."); sm.save(s)
        for k, c in final_clusters.items():
            sm.add_log(s, f"  Cluster {k}: {len(c.get('track_ids', []))} ids")
        sm.save(s)
        push_playlists(get_sp(t), sm, s, final_clusters, confirmed_names)
        s["last_weekly"] = s["last_hourly"] = datetime.now().isoformat()
        s["preview"] = None
        s["num_clusters"] = len(selected_keys)
        sm.save(s)
        sm.add_log(s, f"{len(selected_keys)} playlists live on Spotify!"); sm.save(s)
        return jsonify({"ok": True})
    except Exception as e:
        sm.add_log(s, f"Error: {str(e)}"); sm.save(s)
        return jsonify({"error": str(e)}), 500


@clustering_bp.route("/api/confirm", methods=["POST"])
def api_confirm():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    s = sm.load()
    if not s.get("preview"):
        return jsonify({"error": "No preview"}), 400
    names = {str(k): _sanitize_name(v) for k, v in (request.json or {}).get("names", {}).items()}
    sm.set_token(t)
    try:
        sm.add_log(s, "Creating playlists..."); sm.save(s)
        push_playlists(get_sp(t), sm, s, s["preview"], names)
        s["last_weekly"] = s["last_hourly"] = datetime.now().isoformat()
        s["preview"] = None; sm.save(s)
        sm.add_log(s, "All playlists live!"); sm.save(s)
        return jsonify({"ok": True})
    except Exception as e:
        sm.add_log(s, f"Error: {e}"); sm.save(s)
        return jsonify({"error": str(e)}), 500


@clustering_bp.route("/api/split", methods=["POST"])
def api_split():
    """HIERARCHICAL: Split one vibe card into sub-vibes."""
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json
    parent_key = str(data.get("cluster_key", ""))
    n_splits = data.get("n_splits")
    if not _valid_id(parent_key):
        return jsonify({"error": "Invalid cluster_key"}), 400

    with sm.atomic_update() as s:
        preview = s.get("preview", {})
        if parent_key not in preview:
            return jsonify({"error": "Cluster not found"}), 404

        track_ids = preview[parent_key]["track_ids"]
        if len(track_ids) < 6:
            return jsonify({"error": "Too few tracks to split (need >=6)"}), 400

        cache_batch = db.get_tracks_batch(track_ids)
        children = split_cluster(parent_key, track_ids, cache_batch, n_splits, overrides=s.get("overrides", {}))
        if not children:
            return jsonify({"error": "Split produced no results"}), 500

        del preview[parent_key]
        preview.update(children)
        s["preview"] = preview
        sm.add_log(s, f"Split cluster {parent_key} -> {len(children)} sub-vibes")
    return jsonify({"ok": True, "children": children, "removed_key": parent_key})


@clustering_bp.route("/api/merge", methods=["POST"])
def api_merge():
    """HIERARCHICAL: Merge two or more vibe cards into one."""
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json
    keys = [str(k) for k in data.get("keys", [])]
    if len(keys) < 2:
        return jsonify({"error": "Need at least 2 clusters to merge"}), 400
    if not all(_valid_id(k) for k in keys):
        return jsonify({"error": "Invalid cluster key"}), 400

    with sm.atomic_update() as s:
        preview = s.get("preview", {})

        merged_ids = []
        merged_uris = []
        merged_tags = []
        merged_samp = []
        for k in keys:
            c = preview.get(k, {})
            merged_ids += c.get("track_ids", [])
            merged_uris += c.get("track_uris", [])
            merged_tags += c.get("top_tags", [])
            merged_samp += c.get("sample_tracks", [])

        seen_ids, deduped_ids, deduped_uris = set(), [], []
        for tid, uri in zip(merged_ids, merged_uris):
            if tid not in seen_ids:
                seen_ids.add(tid); deduped_ids.append(tid); deduped_uris.append(uri)

        cache_batch = db.get_tracks_batch(deduped_ids)
        vecs_for_health = []
        recs_all = [cache_batch[tid] for tid in deduped_ids if tid in cache_batch]
        if recs_all:
            try:
                X_m = build_embeddings(recs_all)
                vecs_for_health = X_m
            except Exception:
                pass
        tag_counter = Counter(merged_tags)
        top_tags = [t for t, _ in tag_counter.most_common(10)]
        e_score = score_axis(tag_counter, config.ENERGY_POS, config.ENERGY_NEG)
        m_score = score_axis(tag_counter, config.MOOD_POS, config.MOOD_NEG)
        quad = (energy_band(e_score), mood_band(m_score))
        health = compute_health(vecs_for_health) if len(vecs_for_health) > 1 else 50
        new_key = keys[0]
        merged_name_meta = {new_key: {"e": e_score, "m": m_score, "top_tags": top_tags, "quad": quad}}
        named = name_all_clusters(merged_name_meta)
        sname, desc2, e, m, pinterest, ai_sugg = named[new_key]

        new_cluster = {
            "cluster_id":        new_key,
            "suggested_name":    sname,
            "description":       desc2,
            "track_count":       len(deduped_ids),
            "top_tags":          top_tags,
            "energy_score":      e,
            "mood_score":        m,
            "health":            health,
            "pinterest_prompts": pinterest,
            "ai_suggestions":    ai_sugg,
            "sample_tracks":     merged_samp[:6],
            "track_ids":         deduped_ids,
            "track_uris":        deduped_uris,
            "parent":            None,
            "can_split":         len(deduped_ids) >= 6,
        }

        for k in keys:
            preview.pop(k, None)
        preview[new_key] = new_cluster
        s["preview"] = preview
        sm.add_log(s, f"Merged {len(keys)} clusters -> '{sname}' ({len(deduped_ids)} tracks)")
    return jsonify({"ok": True, "cluster": new_cluster, "new_key": new_key, "removed_keys": keys})
