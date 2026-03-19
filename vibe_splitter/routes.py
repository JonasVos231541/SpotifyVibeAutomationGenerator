"""
All Flask routes for the Vibe Splitter web UI and API.
"""
import os, time, threading, logging, glob, json
from urllib.parse import urlparse
import numpy as np
import requests as req
from collections import Counter
from datetime import datetime
from flask import request, session, jsonify, redirect, render_template, Response
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from spotipy.oauth2 import SpotifyOAuth
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from . import cache as cache_mod
from .state import state_manager as sm
from .spotify_client import (
    get_sp, refresh_token, fetch_all_tracks, get_user_playlists, fetch_audio_features,
)
from .lastfm import build_vectors, fetch_and_cache_track
from .embeddings import build_embeddings, build_hybrid_vectors, extract_extra_features
from .clustering import (
    cluster_records, split_cluster, compute_health, compute_stats,
    build_cluster_dict,
)
from .naming import score_axis, energy_band, mood_band, name_all_clusters, generate_ai_names
from .playlists import push_playlists, push_to_inbox, approve_inbox
from .incremental import classify_new_tracks
from .events import publish as sse_publish, stream_events

log = logging.getLogger("splitter.routes")

_preview_lock = threading.Lock()

import re
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-:.]+$")


def _sanitize_name(name, max_len=100):
    """Strip HTML tags and cap length for playlist/cluster names."""
    if not isinstance(name, str):
        return ""
    clean = re.sub(r"<[^>]+>", "", name).strip()
    return clean[:max_len]


def _valid_id(val):
    """Return True if val looks like a safe Spotify / cluster ID."""
    return isinstance(val, str) and bool(_SAFE_ID_RE.match(val))


def _ref(_app=None):
    """Shorthand for refresh_token with current session & state manager."""
    return refresh_token(session, sm)


def _collect_active_ids(state):
    """Gather all track IDs that should be retained (playlists, inbox, known)."""
    active = set(state.get("known_track_ids", []))
    for pl in (state.get("playlists") or {}).values():
        active.update(pl.get("track_ids", []))
    for item in state.get("inbox", []):
        active.add(item.get("id", ""))
    return active


def register_routes(app):
    """Attach all routes to the Flask app."""

    @app.errorhandler(Exception)
    def _handle_error(e):
        log.exception(f"Unhandled error: {e}")
        code = getattr(e, "code", 500)
        return jsonify({"error": str(e), "code": code}), code

    @app.after_request
    def _security_headers(resp):
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' https://i.scdn.co data:; "
            "connect-src 'self'"
        )
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        return resp

    # ── Pages ─────────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html", logged_in="token_info" in session)

    @app.route("/api/events")
    def api_events():
        return Response(stream_events(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    @app.route("/login")
    def login():
        for f in [".cache", config.SPOTIFY_CACHE_PATH] + glob.glob(".cache-*"):
            if os.path.exists(f):
                os.remove(f)
        o = SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=config.SPOTIFY_REDIRECT_URI,
            scope=config.SPOTIFY_SCOPE,
            cache_path=config.SPOTIFY_CACHE_PATH,
        )
        return redirect(o.get_authorize_url() + "&show_dialog=true")

    @app.route("/callback")
    def callback():
        o = SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=config.SPOTIFY_REDIRECT_URI,
            scope=config.SPOTIFY_SCOPE,
            cache_path=config.SPOTIFY_CACHE_PATH,
        )
        t = o.get_access_token(request.args.get("code"), as_dict=True, check_cache=False)
        granted = set(t.get("scope", "").split()) if isinstance(t, dict) else set()
        required = {"user-library-read", "playlist-modify-public", "playlist-modify-private"}
        missing = required - granted
        if missing:
            log.warning(f"Token MISSING scopes: {missing}")
        session["token_info"] = t
        sm.set_token(t)
        s = sm.load()
        s["preview"] = None
        sm.save(s)
        return redirect("/")

    @app.route("/logout")
    def logout():
        session.clear()
        sm.clear_token()
        for f in [".cache", config.SPOTIFY_CACHE_PATH]:
            if os.path.exists(f):
                os.remove(f)
        return redirect("/")

    # ── API: Data ─────────────────────────────────────────────────────────────
    @app.route("/api/playlists")
    def api_playlists():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        try:
            sp = get_sp(t)
            user = sp.current_user()
            log.info(f"Fetching playlists for user: {user['id']}")
            playlists = get_user_playlists(sp)
            log.info(f"Found {len(playlists)} playlists")
            return jsonify(playlists)
        except Exception as e:
            log.error(f"Error in /api/playlists: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/state")
    def api_state():
        s = sm.load()
        # Inject token metadata for UI
        token = session.get("token_info")
        if token:
            s["has_ugc_scope"] = "ugc-image-upload" in token.get("scope", "")
            s["has_playlist_read"] = "playlist-read-private" in token.get("scope", "")
            s["token_expires_at"] = token.get("expires_at")
        return jsonify(s)

    @app.route("/api/token-info")
    def api_token_info():
        token = session.get("token_info")
        if not token:
            return jsonify({"error": "not logged in"}), 401
        sp = get_sp(token)
        try:
            me = sp.current_user()
            return jsonify({
                "user": me.get("id"),
                "scope": token.get("scope", "(not set)"),
                "expires_at": token.get("expires_at"),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/stats")
    def api_stats():
        return jsonify(compute_stats(cache_mod.load()))

    @app.route("/api/cache-stats")
    def api_cache_stats():
        cache = cache_mod.load()
        return jsonify({"cached_tracks": len(cache), "cache_size_kb": cache_mod.size_kb()})

    @app.route("/api/clear-cache", methods=["POST"])
    def api_clear_cache():
        cache_mod.clear()
        s = sm.load()
        sm.add_log(s, "Cache cleared")
        sm.save(s)
        return jsonify({"ok": True})

    @app.route("/api/prune-cache", methods=["POST"])
    def api_prune_cache():
        """Remove cached tracks that aren't in any active source or playlist."""
        s = sm.load()
        active_ids = _collect_active_ids(s)
        removed = cache_mod.prune(active_ids)
        if removed:
            sm.add_log(s, f"Pruned {removed} stale tracks from cache")
            sm.save(s)
        remaining = len(cache_mod.load())
        return jsonify({"ok": True, "removed": removed, "remaining": remaining})

    # ── Playlist history, duplicates, overrides ──────────────────────────────

    @app.route("/api/playlist-history/<key>")
    def api_playlist_history(key):
        s = sm.load()
        pl = s.get("playlists", {}).get(key)
        if not pl:
            return jsonify({"error": "Playlist not found"}), 404
        last_push = pl.get("last_push")
        if not last_push:
            return jsonify({"message": "No push history yet"})
        cache = cache_mod.load()
        added_info = [{"id": tid, "name": cache.get(tid, {}).get("name", "Unknown")}
                      for tid in last_push.get("added_ids", [])]
        removed_info = [{"id": tid, "name": cache.get(tid, {}).get("name", "Unknown")}
                        for tid in last_push.get("removed_ids", [])]
        return jsonify({
            "timestamp": last_push["timestamp"],
            "added": added_info,
            "removed": removed_info,
            "added_count": last_push["added_count"],
            "removed_count": last_push["removed_count"],
            "total": last_push["total"],
        })

    @app.route("/api/check-duplicates")
    def api_check_duplicates():
        s = sm.load()
        cache = cache_mod.load()
        track_locations = {}
        for key, pl in s.get("playlists", {}).items():
            for tid in pl.get("track_ids", []):
                track_locations.setdefault(tid, []).append(key)
        duplicates = []
        for tid, keys in track_locations.items():
            if len(keys) > 1:
                entry = cache.get(tid, {})
                duplicates.append({
                    "track_id": tid,
                    "name": entry.get("name", "Unknown"),
                    "artist": entry.get("artist", "Unknown"),
                    "found_in": [{"key": k, "name": s["playlists"][k].get("name", k)}
                                 for k in keys],
                })
        return jsonify({"duplicates": duplicates, "count": len(duplicates)})

    @app.route("/api/overrides")
    def api_overrides():
        s = sm.load()
        cache = cache_mod.load()
        overrides = s.get("overrides", {})
        result = []
        for tid, target_key in overrides.items():
            entry = cache.get(tid, {})
            target_name = s.get("playlists", {}).get(str(target_key), {}).get(
                "name", f"Cluster {target_key}")
            result.append({
                "track_id": tid,
                "name": entry.get("name", "Unknown"),
                "artist": entry.get("artist", "Unknown"),
                "target_key": str(target_key),
                "target_name": target_name,
            })
        return jsonify({"overrides": result, "count": len(result)})

    @app.route("/api/overrides/<track_id>", methods=["DELETE"])
    def api_delete_override(track_id):
        s = sm.load()
        overrides = s.get("overrides", {})
        if track_id not in overrides:
            return jsonify({"error": "Override not found"}), 404
        del overrides[track_id]
        sm.add_log(s, f"Removed override for track {track_id}")
        sm.save(s)
        return jsonify({"ok": True})

    @app.route("/api/toggle-auto-recluster", methods=["POST"])
    def api_toggle_auto_recluster():
        s = sm.load()
        s["auto_recluster_enabled"] = not s.get("auto_recluster_enabled", True)
        sm.save(s)
        return jsonify({"ok": True, "enabled": s["auto_recluster_enabled"]})

    @app.route("/api/cache-sources")
    def api_cache_sources():
        cache = cache_mod.load()
        s     = sm.load()
        total = len(cache)
        sources = [{
            "type": "cache_all", "id": "cache_all",
            "name": "All Cached Tracks",
            "track_count": total,
            "description": f"{total} tracks already tagged — instant, no API calls",
        }]
        for src in s.get("sources", []):
            src_type = src.get("type")
            src_name = src.get("name") or ("Liked Songs" if src_type == "liked" else src.get("id", "?"))
            sources.append({
                "type": "cache_playlist", "id": f"prev_{src.get('id', 'liked')}",
                "spotify_id": src.get("id"), "spotify_type": src_type,
                "name": src_name, "track_count": None,
                "description": "Re-fetch IDs from Spotify, use cached tags",
            })
        for key, pl in (s.get("playlists") or {}).items():
            tids   = pl.get("track_ids", [])
            cached = sum(1 for tid in tids if tid in cache)
            if not tids:
                continue
            sources.append({
                "type": "cache_playlist", "id": f"vibe_{key}",
                "name": pl.get("name", "Vibe " + key),
                "track_ids": tids, "track_count": cached,
                "description": f"{cached}/{len(tids)} tracks cached",
            })
        return jsonify({"total_cached": total, "sources": sources})

    # ── API: Token ────────────────────────────────────────────────────────────
    @app.route("/api/wipe-token", methods=["POST"])
    def api_wipe_token():
        session.clear()
        sm.clear_token()
        for f in [".cache", config.SPOTIFY_CACHE_PATH, ".cache-anonymous"] + glob.glob(".cache-*"):
            if os.path.exists(f):
                os.remove(f)
        return jsonify({"ok": True, "message": "All tokens wiped"})

    # ── API: Diagnostics ──────────────────────────────────────────────────────
    @app.route("/api/test-fetch")
    def api_test_fetch():
        token = session.get("token_info")
        if not token:
            return jsonify({"error": "not logged in"}), 401
        t0 = time.time()
        try:
            resp = req.get("https://api.spotify.com/v1/me/tracks",
                        headers={"Authorization": f"Bearer {token['access_token']}"},
                        params={"limit": 5, "offset": 0}, timeout=15)
            elapsed = time.time() - t0
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                names = [(item.get("track") or item.get("item", {})).get("name", "?") for item in items[:5]]
                return jsonify({"status": "OK", "http_status": 200, "elapsed": round(elapsed, 2),
                                "total": data.get("total", "?"), "sample": names})
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "unknown")
                log.error(f"test-fetch 429 — Retry-After: {retry_after} seconds")
                return jsonify({"status": "FAILED", "http_status": 429,
                                "retry_after": retry_after,
                                "body": resp.text[:500]}), 429
            else:
                return jsonify({"status": "FAILED", "http_status": resp.status_code,
                                "body": resp.text[:500]}), resp.status_code
        except Exception as e:
            return jsonify({"status": "ERROR", "error": str(e)}), 500

    @app.route("/api/test-playlist/<playlist_id>")
    def api_test_playlist(playlist_id):
        token = session.get("token_info")
        if not token:
            return jsonify({"error": "not logged in"}), 401
        try:
            resp = req.get(f"https://api.spotify.com/v1/playlists/{playlist_id}/items",
                           headers={"Authorization": f"Bearer {token['access_token']}"},
                           params={"limit": 2}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return jsonify({"status": "OK", "total": data.get("total"),
                                "first_item": data.get("items", [None])[0]})
            return jsonify({"status": "FAILED", "http": resp.status_code}), resp.status_code
        except Exception as e:
            return jsonify({"status": "ERROR", "error": str(e)}), 500

    # ── API: Track operations ─────────────────────────────────────────────────
    @app.route("/api/retag", methods=["POST"])
    def api_retag():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        tid = request.json.get("track_id")
        sp  = get_sp(t)
        cache = cache_mod.load()
        old = cache.get(tid)
        if not old:
            return jsonify({"error": "Track not in cache"}), 404
        try:
            track_obj = sp.track(tid)
        except Exception:
            return jsonify({"error": "Spotify lookup failed"}), 500
        new_entry = fetch_and_cache_track(track_obj, sp=sp)
        cache[tid] = new_entry
        cache_mod.save(cache)
        s = sm.load()
        sm.add_log(s, f"Re-tagged '{new_entry['name']}' — {', '.join(new_entry['tags'][:5])}")

        # Reclassify if tags changed significantly
        old_tags = set(old.get("tags", []))
        new_tags = set(new_entry.get("tags", []))
        if len(old_tags ^ new_tags) >= 3:
            classify_new_tracks([new_entry], sm, s, sp=sp)
            predicted = new_entry.get("predicted_cluster")
            if predicted:
                sm.add_log(s, f"Reclassified to cluster {predicted}")
        sm.save(s)
        return jsonify({"ok": True, "tags": new_entry["tags"],
                        "predicted_cluster": new_entry.get("predicted_cluster")})

    @app.route("/api/override", methods=["POST"])
    def api_override():
        data = request.json or {}
        tid  = data.get("track_id", "")
        key  = str(data.get("cluster_key", ""))
        if not _valid_id(tid) or not key:
            return jsonify({"error": "Invalid track_id or cluster_key"}), 400
        s    = sm.load()
        s.setdefault("overrides", {})[tid] = key
        t = _ref()
        if t and s["playlists"]:
            sp    = get_sp(t)
            cache = cache_mod.load()
            rec   = cache.get(tid)
            if rec:
                for k, pl in s["playlists"].items():
                    pid = pl.get("spotify_id")
                    if not pid:
                        continue
                    if k == key:
                        if tid not in pl.get("track_ids", []):
                            sp.playlist_add_items(pid, [rec["uri"]])
                            pl.setdefault("track_ids", []).append(tid)
                    else:
                        if tid in pl.get("track_ids", []):
                            try:
                                sp.playlist_remove_all_occurrences_of_items(pid, [{"uri": rec["uri"]}])
                            except Exception:
                                pass
                            pl["track_ids"] = [i for i in pl["track_ids"] if i != tid]
        sm.add_log(s, f"Override saved — track pinned to cluster {key}")
        sm.save(s)
        return jsonify({"ok": True})

    # ── API: Inbox ────────────────────────────────────────────────────────────
    @app.route("/api/inbox/approve", methods=["POST"])
    def api_inbox_approve():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        s = sm.load()
        approve_inbox(get_sp(t), sm, s, request.json.get("approvals", []))
        return jsonify({"ok": True})

    @app.route("/api/inbox/dismiss", methods=["POST"])
    def api_inbox_dismiss():
        tids = set(request.json.get("track_ids", []))
        s = sm.load()
        s["inbox"] = [t for t in s.get("inbox", []) if t["id"] not in tids]
        sm.save(s)
        return jsonify({"ok": True})

    # ── API: Preview + Confirm ────────────────────────────────────────────────
    @app.route("/api/preview", methods=["POST"])
    def api_preview():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        if _preview_lock.locked():
            return jsonify({"ok": True, "running": True, "message": "Preview already running"}), 202

        data    = request.json
        sources = data.get("sources", [])
        granularity = data.get("granularity", None)
        include_year = data.get("include_year", False)
        include_popularity = data.get("include_popularity", False)
        use_llm_guidance = data.get("use_llm_guidance", False)   # <-- new

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

                    # Fetch extra features if enabled
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
                                            use_llm_guidance=use_llm_guidance)  # <-- pass it
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

    @app.route("/api/preview-ensemble", methods=["POST"])
    def api_preview_ensemble():
        """Return 3 alternative clusterings (HDBSCAN, KMeans5, KMeans10) as summaries."""
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        data = request.json
        sources = data.get("sources", [])
        # Quick fetch (no extra features for ensemble to keep it fast)
        sp = get_sp(t)
        tracks = fetch_all_tracks(sp, sources)
        records = build_vectors(tracks, sm, state=None, sp=sp)  # no state for speed
        X_embed = build_embeddings(records)
        X, _ = build_hybrid_vectors(X_embed, records, audio_features=None)  # no audio

        def _hdbscan():
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(6, len(records)//20),
                                         min_samples=3, metric="cosine", cluster_selection_method="eom")
            labels = clusterer.fit_predict(X)
            return _summarize_clusters(records, labels)

        def _kmeans(k):
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

    @app.route("/api/confirm-selected", methods=["POST"])
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

        data            = request.json or {}
        selected_keys   = [str(k) for k in data.get("selected", [])]
        confirmed_names = {str(k): _sanitize_name(v) for k, v in data.get("names", {}).items()}
        redistribute    = data.get("redistribute", False)  # default: DON'T redistribute
        sm.set_token(t)

        if not selected_keys:
            return jsonify({"error": "Select at least one vibe"}), 400

        full_preview = s["preview"]

        # Only redistribute rejected tracks into selected vibes if explicitly asked
        rejected = [k for k in full_preview if k not in selected_keys]
        if redistribute and rejected and os.path.exists(config.MODEL_FILE):
            cache = cache_mod.load()

            # Build embedding centroids for selected clusters
            selected_centroids = {}
            for k in selected_keys:
                tids = full_preview[k].get("track_ids", [])
                recs = [cache[tid] for tid in tids if tid in cache]
                if recs:
                    X_c = build_embeddings(recs)
                    selected_centroids[k] = X_c.mean(axis=0)

            for rk in rejected:
                tids = full_preview[rk].get("track_ids", [])
                recs_to_embed = [cache[tid] for tid in tids if tid in cache]
                if not recs_to_embed or not selected_centroids:
                    continue
                X_rejected = build_embeddings(recs_to_embed)
                for tid, vec in zip(tids, X_rejected):
                    best_k, best_sim = selected_keys[0], -1
                    for sk, centroid in selected_centroids.items():
                        n1 = np.linalg.norm(vec); n2 = np.linalg.norm(centroid)
                        sim = float(np.dot(vec, centroid)/(n1*n2)) if n1>0 and n2>0 else 0
                        if sim > best_sim:
                            best_sim = sim; best_k = sk
                    full_preview[best_k]["track_ids"].append(tid)
                    full_preview[best_k]["track_count"] += 1

        # Build final clusters dict with only selected keys
        final_clusters = {k: full_preview[k] for k in selected_keys}

        try:
            sm.add_log(s, "Creating selected playlists..."); sm.save(s)
            # Debug: log track counts going into push
            for k, c in final_clusters.items():
                sm.add_log(s, f"  Cluster {k}: {len(c.get('track_ids', []))} ids")
            sm.save(s)
            push_playlists(get_sp(t), sm, s, final_clusters, confirmed_names)
            s["last_weekly"] = s["last_hourly"] = datetime.now().isoformat()
            s["preview"]     = None
            s["num_clusters"]= len(selected_keys)
            sm.save(s)
            sm.add_log(s, f"{len(selected_keys)} playlists live on Spotify!"); sm.save(s)
            return jsonify({"ok": True})
        except Exception as e:
            sm.add_log(s, f"Error: {str(e)}"); sm.save(s)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/confirm", methods=["POST"])
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

    # ── API: Split / Merge ────────────────────────────────────────────────────
    @app.route("/api/split", methods=["POST"])
    def api_split():
        """HIERARCHICAL: Split one vibe card into sub-vibes."""
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        data       = request.json
        parent_key = str(data.get("cluster_key"))
        n_splits   = data.get("n_splits")   # optional override
        s          = sm.load()
        preview    = s.get("preview", {})
        if parent_key not in preview:
            return jsonify({"error": "Cluster not found"}), 404

        track_ids = preview[parent_key]["track_ids"]
        if len(track_ids) < 6:
            return jsonify({"error": "Too few tracks to split (need ≥6)"}), 400

        cache    = cache_mod.load()
        children = split_cluster(parent_key, track_ids, cache, n_splits, overrides=s.get("overrides", {}))
        if not children:
            return jsonify({"error": "Split produced no results"}), 500

        # Remove parent, add children into preview
        del preview[parent_key]
        preview.update(children)
        s["preview"] = preview
        sm.add_log(s, f"Split cluster {parent_key} → {len(children)} sub-vibes")
        sm.save(s)
        return jsonify({"ok": True, "children": children, "removed_key": parent_key})

    @app.route("/api/merge", methods=["POST"])
    def api_merge():
        """HIERARCHICAL: Merge two or more vibe cards into one."""
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        data    = request.json
        keys    = [str(k) for k in data.get("keys", [])]
        s       = sm.load()
        preview = s.get("preview", {})
        if len(keys) < 2:
            return jsonify({"error": "Need at least 2 clusters to merge"}), 400

        # Combine all tracks from selected keys
        merged_ids  = []
        merged_uris = []
        merged_tags = []
        merged_samp = []
        for k in keys:
            c = preview.get(k, {})
            merged_ids  += c.get("track_ids", [])
            merged_uris += c.get("track_uris", [])
            merged_tags += c.get("top_tags", [])
            merged_samp += c.get("sample_tracks", [])

        # Dedupe
        seen_ids, deduped_ids, deduped_uris = set(), [], []
        for tid, uri in zip(merged_ids, merged_uris):
            if tid not in seen_ids:
                seen_ids.add(tid); deduped_ids.append(tid); deduped_uris.append(uri)

        # Compute merged stats
        cache = cache_mod.load()
        vecs_for_health = []
        recs_all = [cache[tid] for tid in deduped_ids if tid in cache]
        if recs_all:
            try:
                X_m = build_embeddings(recs_all)
                vecs_for_health = X_m
            except Exception:
                pass
        tag_counter = Counter(merged_tags)
        top_tags    = [t for t,_ in tag_counter.most_common(10)]
        e_score     = score_axis(tag_counter, config.ENERGY_POS, config.ENERGY_NEG)
        m_score     = score_axis(tag_counter, config.MOOD_POS,   config.MOOD_NEG)
        quad        = (energy_band(e_score), mood_band(m_score))
        health      = compute_health(vecs_for_health) if len(vecs_for_health) > 1 else 50
        new_key     = keys[0]  # keep first key
        merged_name_meta = {new_key: {"e": e_score, "m": m_score, "top_tags": top_tags, "quad": quad}}
        named = name_all_clusters(merged_name_meta)
        sname, desc2, e, m, pinterest, ai_sugg = named[new_key]  # unpack 6 items

        new_cluster = {
            "cluster_id":       new_key,
            "suggested_name":   sname,
            "description":      desc2,
            "track_count":      len(deduped_ids),
            "top_tags":         top_tags,
            "energy_score":     e,
            "mood_score":       m,
            "health":           health,
            "pinterest_prompts": pinterest,
            "ai_suggestions":   ai_sugg,
            "sample_tracks":    merged_samp[:6],
            "track_ids":        deduped_ids,
            "track_uris":       deduped_uris,
            "parent":           None,
            "can_split":        len(deduped_ids) >= 6,
        }

        # Remove merged keys, insert combined
        for k in keys:
            preview.pop(k, None)
        preview[new_key] = new_cluster
        s["preview"] = preview
        sm.add_log(s, f"Merged {len(keys)} clusters → '{sname}' ({len(deduped_ids)} tracks)")
        sm.save(s)
        return jsonify({"ok": True, "cluster": new_cluster, "new_key": new_key, "removed_keys": keys})

    # ── API: Hourly / Names / Covers / Cleanup ────────────────────────────────
    @app.route("/api/run-hourly", methods=["POST"])
    def api_run_hourly():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        s = sm.load()
        if not s.get("last_weekly"):
            return jsonify({"error": "Run full cluster first"}), 400
        sm.set_token(t)
        hourly_update(get_sp(t), s)
        return jsonify({"ok": True})

    @app.route("/api/trigger-weekly", methods=["POST"])
    def api_trigger_weekly():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        s = sm.load()
        if not s.get("last_weekly"):
            return jsonify({"error": "Run full cluster first"}), 400
        if s.get("job_running"):
            return jsonify({"error": f"Job '{s['job_running']}' already running"}), 409
        sm.set_token(t)
        # Run in background thread to avoid blocking the request
        def _run():
            try:
                from app import _weekly
                _weekly()
            except Exception as e:
                log.error(f"Manual weekly trigger failed: {e}")
        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"ok": True, "message": "Weekly recluster started in background"})

    @app.route("/api/update-name", methods=["POST"])
    def api_update_name():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        d = request.json or {}
        s = sm.load()
        key = str(d.get("cluster_id", ""))
        name = _sanitize_name(d.get("name", ""))
        if not name:
            return jsonify({"error": "Name is required"}), 400
        if key in s["playlists"]:
            s["playlists"][key]["name"] = name
            pid = s["playlists"][key].get("spotify_id")
            if pid:
                get_sp(t).playlist_change_details(pid, name=name)
        sm.save(s)
        return jsonify({"ok": True})

    @app.route("/api/cleanup-playlists", methods=["POST"])
    def api_cleanup_playlists():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        s  = sm.load()
        sp = get_sp(t)
        try:
            live_ids = set()
            r = sp.current_user_playlists(limit=50)
            while r:
                for p in r["items"]:
                    if p:
                        live_ids.add(p["id"])
                r = sp.next(r) if r.get("next") else None
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        removed, kept = [], {}
        for key, pl in (s.get("playlists") or {}).items():
            pid = pl.get("spotify_id")
            if pid and pid not in live_ids:
                removed.append(pl.get("name", key))
            else:
                kept[key] = pl
        s["playlists"] = kept
        if removed:
            sm.add_log(s, f"Removed {len(removed)} deleted playlist(s)")
        sm.save(s)
        return jsonify({"ok": True, "removed": removed, "kept": len(kept)})

    @app.route("/api/cover-search/<cluster_key>")
    def api_cover_search(cluster_key):
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        s       = sm.load()
        cluster = (s.get("preview") or {}).get(cluster_key, {})
        top_tags = cluster.get("top_tags", [])
        sample   = cluster.get("sample_tracks", [])
        if not top_tags:
            return jsonify({"images": []})
        sp = get_sp(t)
        images, seen_urls = [], set()

        # Strategy 1: search by artist + tag (more relevant)
        for track in sample[:3]:
            artist = track.get("artist", "")
            if not artist:
                continue
            for tag in top_tags[:2]:
                try:
                    results = sp.search(q=f'{artist} {tag}', type="track", limit=5, market="NL")
                    for item in (results.get("tracks") or {}).get("items", []):
                        if not item:
                            continue
                        imgs = item.get("album", {}).get("images", [])
                        if imgs:
                            url = imgs[0]["url"]
                            if url not in seen_urls:
                                seen_urls.add(url)
                                images.append({"url": url,
                                    "album": item["album"].get("name", ""),
                                    "artist": (item.get("artists") or [{}])[0].get("name", "")})
                    if len(images) >= 12:
                        break
                    time.sleep(0.1)
                except Exception:
                    continue
            if len(images) >= 12:
                break

        # Strategy 2: tag-based genre search (fallback)
        if len(images) < 6:
            for tag in top_tags[:5]:
                try:
                    results = sp.search(q=f'genre:"{tag}"', type="track", limit=10, market="NL")
                    for item in (results.get("tracks") or {}).get("items", []):
                        if not item:
                            continue
                        imgs = item.get("album", {}).get("images", [])
                        if imgs:
                            url = imgs[0]["url"]
                            if url not in seen_urls:
                                seen_urls.add(url)
                                images.append({"url": url,
                                    "album": item["album"].get("name", ""),
                                    "artist": (item.get("artists") or [{}])[0].get("name", "")})
                    if len(images) >= 12:
                        break
                    time.sleep(0.1)
                except Exception:
                    continue
        return jsonify({"images": images[:12]})

    @app.route("/api/upload-cover", methods=["POST"])
    def api_upload_cover():
        t = _ref()
        if not t:
            return jsonify({"error": "Not logged in"}), 401
        # Check scope
        scope = t.get("scope", "")
        if "ugc-image-upload" not in scope:
            return jsonify({"error": "Missing ugc-image-upload scope — re-login to fix"}), 403
        data = request.json
        playlist_id = data.get("playlist_id"); image_url = data.get("image_url")
        if not playlist_id or not image_url:
            return jsonify({"error": "playlist_id and image_url required"}), 400
        # SSRF protection: only allow HTTPS URLs to public hosts
        parsed = urlparse(image_url)
        if parsed.scheme != "https":
            return jsonify({"error": "Only HTTPS image URLs are allowed"}), 400
        hostname = (parsed.hostname or "").lower()
        _blocked = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}
        if hostname in _blocked or hostname.startswith("169.254.") or hostname.startswith("10."):
            return jsonify({"error": "Internal URLs are not allowed"}), 400
        try:
            img_resp = req.get(image_url, timeout=15); img_resp.raise_for_status()
        except Exception as e:
            return jsonify({"error": f"Could not download image: {e}"}), 400
        # Spotify requires base64-encoded JPEG, max ~256KB
        import base64
        img_b64 = base64.b64encode(img_resp.content)
        if len(img_b64) > 256 * 1024:
            try:
                from PIL import Image
                from io import BytesIO
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                img.thumbnail((600, 600))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=75)
                img_b64 = base64.b64encode(buf.getvalue())
            except ImportError:
                return jsonify({"error": "Image > 256KB and Pillow not installed"}), 400
        resp = req.put(
            f"https://api.spotify.com/v1/playlists/{playlist_id}/images",
            headers={"Authorization": f"Bearer {t['access_token']}", "Content-Type": "image/jpeg"},
            data=img_b64, timeout=30,
        )
        if resp.status_code in (200, 202):
            s = sm.load(); sm.add_log(s, f"Cover updated"); sm.save(s)
            return jsonify({"ok": True})
        if resp.status_code == 403:
            return jsonify({"error": "403 — missing ugc-image-upload scope"}), 403
        return jsonify({"error": f"Spotify returned {resp.status_code}"}), 400


# ─── Removed track detection & model staleness ────────────────────────────────

def _handle_removed_tracks(sp, sm, state, removed_ids):
    """Remove tracks from vibe playlists, inbox, and overrides when they
    disappear from source playlists."""
    _cache = cache_mod.load()
    total_removed = 0

    # Remove from each vibe playlist
    for key, pl in state.get("playlists", {}).items():
        pl_tids = set(pl.get("track_ids", []))
        to_remove = pl_tids & removed_ids
        if not to_remove:
            continue
        pid = pl.get("spotify_id")
        if pid:
            uris = []
            for tid in to_remove:
                entry = _cache.get(tid)
                uri = entry["uri"] if entry and entry.get("uri") else f"spotify:track:{tid}"
                uris.append({"uri": uri})
            # Batch remove from Spotify (100 per call)
            for i in range(0, len(uris), 100):
                try:
                    sp.playlist_remove_all_occurrences_of_items(pid, uris[i:i+100])
                except Exception as e:
                    log.warning(f"Failed to remove tracks from {pl.get('name', key)}: {e}")
                time.sleep(0.1)
        pl["track_ids"] = [tid for tid in pl.get("track_ids", []) if tid not in to_remove]
        total_removed += len(to_remove)

    # Remove from inbox
    inbox_removed = [t for t in state.get("inbox", []) if t["id"] in removed_ids]
    if inbox_removed:
        state["inbox"] = [t for t in state.get("inbox", []) if t["id"] not in removed_ids]
        inbox_pid = state.get("inbox_playlist_id")
        if inbox_pid:
            uris = []
            for t in inbox_removed:
                uri = t.get("uri") or _cache.get(t["id"], {}).get("uri") or f"spotify:track:{t['id']}"
                uris.append({"uri": uri})
            try:
                sp.playlist_remove_all_occurrences_of_items(inbox_pid, uris)
            except Exception as e:
                log.warning(f"Failed to remove tracks from Inbox: {e}")

    # Remove overrides for deleted tracks
    overrides = state.get("overrides", {})
    for tid in removed_ids:
        overrides.pop(tid, None)

    if total_removed > 0 or inbox_removed:
        sm.add_log(state, f"Removed {total_removed + len(inbox_removed)} tracks "
                          f"no longer in sources")


def _check_model_staleness(state, current_track_count, sm):
    """Check if the model is stale and set drift_warning accordingly."""
    model_built = state.get("model_built_at")
    if model_built:
        try:
            from datetime import datetime as dt
            built_dt = dt.fromisoformat(model_built)
            age_days = (dt.now() - built_dt).days
            if age_days > config.MODEL_STALE_DAYS:
                if not state.get("drift_warning"):
                    state["drift_warning"] = True
                    sm.add_log(state, f"Model is {age_days} days old — consider reclustering")
        except (ValueError, TypeError):
            pass

    model_count = state.get("model_track_count", 0)
    if model_count > 0 and current_track_count > 0:
        drift = abs(current_track_count - model_count) / model_count
        if drift > config.MODEL_TRACK_DRIFT_PCT:
            state["drift_warning"] = "auto_recluster"
            sm.add_log(state, f"Library changed {drift:.0%} since last cluster — "
                              f"recluster recommended")


# ─── Hourly update (used by scheduler and manual trigger) ─────────────────────

def hourly_update(sp, state):
    if not os.path.exists(config.MODEL_FILE):
        sm.add_log(state, "No model yet"); sm.save(state); return
    if not sm.try_acquire_job(state, "hourly"):
        return
    try:
        sm.add_log(state, "Hourly scan...")
        tracks = fetch_all_tracks(sp, state.get("sources", []), sm=sm, state=state)
        current_ids = {t["id"] for t in tracks}
        known  = set(state.get("known_track_ids", []))
        new_t  = [t for t in tracks if t["id"] not in known]
        removed_ids = known - current_ids

        # Handle removed tracks
        if removed_ids:
            _handle_removed_tracks(sp, sm, state, removed_ids)

        # Check model staleness
        _check_model_staleness(state, len(tracks), sm)

        if not new_t and not removed_ids:
            sm.add_log(state, "All up to date")
            state["last_hourly"] = datetime.now().isoformat()
            sm.save(state); return

        if new_t:
            records = build_vectors(new_t, sm, state, sp=sp)
            if records:
                classify_new_tracks(records, sm, state, sp=sp)
            push_to_inbox(sp, sm, state, records)

        state["known_track_ids"] = [t["id"] for t in tracks]
        state["last_hourly"] = datetime.now().isoformat()
        sm.save(state)
    finally:
        sm.release_job(state)