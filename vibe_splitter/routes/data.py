"""
Data blueprint — state, stats, cache, playlists, overrides, history.
"""
import os, logging
from datetime import datetime
from flask import Blueprint, request, session, jsonify

from .. import config, db
from ..state import state_manager as sm
from ..spotify_client import get_sp, get_user_playlists
from ..clustering import compute_stats
from .helpers import _ref, _sanitize_name, _valid_id, _collect_active_ids

log = logging.getLogger("splitter.routes.data")

data_bp = Blueprint("data", __name__)


@data_bp.route("/api/playlists")
def api_playlists():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    try:
        sp = get_sp(t)
        playlists = get_user_playlists(sp)
        log.info(f"Found {len(playlists)} playlists")
        return jsonify(playlists)
    except Exception as e:
        log.error(f"Error in /api/playlists: {e}")
        return jsonify({"error": str(e)}), 500


def _inject_drift_info(s):
    """Compute drift_warning and recluster_age_days from state fields (read-only)."""
    model_built = s.get("model_built_at")
    if not model_built:
        return
    try:
        built_dt = datetime.fromisoformat(model_built)
        age_days = (datetime.now() - built_dt).days
        s["recluster_age_days"] = age_days
        if not s.get("drift_warning"):
            if age_days > config.MODEL_STALE_DAYS:
                s["drift_warning"] = True
            model_count = s.get("model_track_count", 0)
            current_count = len(s.get("known_track_ids", []))
            if model_count > 0 and current_count > 0:
                drift_pct = abs(current_count - model_count) / model_count
                if drift_pct > config.MODEL_TRACK_DRIFT_PCT:
                    s["drift_warning"] = "auto_recluster"
    except (ValueError, TypeError):
        pass


@data_bp.route("/api/state")
def api_state():
    s = sm.load()
    token = session.get("token_info")
    if token:
        s["has_ugc_scope"] = "ugc-image-upload" in token.get("scope", "")
        s["has_playlist_read"] = "playlist-read-private" in token.get("scope", "")
        s["token_expires_at"] = token.get("expires_at")
    _inject_drift_info(s)
    return jsonify(s)


@data_bp.route("/api/stats")
def api_stats():
    return jsonify(compute_stats(db.get_all_tracks()))


@data_bp.route("/api/cache-stats")
def api_cache_stats():
    count = db.track_count()
    try:
        size_kb = round(os.path.getsize(db.DB_FILE) / 1024)
    except OSError:
        size_kb = 0
    tagged = db.tagged_track_count()
    af_count = db.audio_features_count()
    return jsonify({
        "cached_tracks": count,
        "cache_size_kb": size_kb,
        "tagged_tracks": tagged,
        "audio_features_tracks": af_count,
    })


@data_bp.route("/api/clear-cache", methods=["POST"])
def api_clear_cache():
    db.clear_tracks()
    s = sm.load()
    sm.add_log(s, "Cache cleared")
    sm.save(s)
    return jsonify({"ok": True})


@data_bp.route("/api/prune-cache", methods=["POST"])
def api_prune_cache():
    """Remove cached tracks that aren't in any active source or playlist."""
    s = sm.load()
    active_ids = _collect_active_ids(s)
    removed = db.prune_tracks(active_ids)
    if removed:
        sm.add_log(s, f"Pruned {removed} stale tracks from cache")
        sm.save(s)
    remaining = db.track_count()
    return jsonify({"ok": True, "removed": removed, "remaining": remaining})


@data_bp.route("/api/playlist-history/<key>")
def api_playlist_history(key):
    s = sm.load()
    pl = s.get("playlists", {}).get(key)
    if not pl:
        return jsonify({"error": "Playlist not found"}), 404
    last_push = pl.get("last_push")
    if not last_push:
        return jsonify({"message": "No push history yet"})
    all_ids = last_push.get("added_ids", []) + last_push.get("removed_ids", [])
    batch = db.get_tracks_batch(all_ids)
    added_info = [{"id": tid, "name": batch.get(tid, {}).get("name", "Unknown")}
                  for tid in last_push.get("added_ids", [])]
    removed_info = [{"id": tid, "name": batch.get(tid, {}).get("name", "Unknown")}
                    for tid in last_push.get("removed_ids", [])]
    return jsonify({
        "timestamp": last_push["timestamp"],
        "added": added_info,
        "removed": removed_info,
        "added_count": last_push["added_count"],
        "removed_count": last_push["removed_count"],
        "total": last_push["total"],
    })


@data_bp.route("/api/check-duplicates")
def api_check_duplicates():
    s = sm.load()
    track_locations = {}
    for key, pl in s.get("playlists", {}).items():
        for tid in pl.get("track_ids", []):
            track_locations.setdefault(tid, []).append(key)
    dup_ids = [tid for tid, keys in track_locations.items() if len(keys) > 1]
    batch = db.get_tracks_batch(dup_ids)
    duplicates = []
    for tid in dup_ids:
        entry = batch.get(tid, {})
        duplicates.append({
            "track_id": tid,
            "name": entry.get("name", "Unknown"),
            "artist": entry.get("artist", "Unknown"),
            "found_in": [{"key": k, "name": s["playlists"][k].get("name", k)}
                         for k in track_locations[tid]],
        })
    return jsonify({"duplicates": duplicates, "count": len(duplicates)})


@data_bp.route("/api/overrides")
def api_overrides():
    s = sm.load()
    overrides = s.get("overrides", {})
    batch = db.get_tracks_batch(list(overrides.keys()))
    result = []
    for tid, target_key in overrides.items():
        entry = batch.get(tid, {})
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


@data_bp.route("/api/overrides/<track_id>", methods=["DELETE"])
def api_delete_override(track_id):
    s = sm.load()
    overrides = s.get("overrides", {})
    if track_id not in overrides:
        return jsonify({"error": "Override not found"}), 404
    del overrides[track_id]
    sm.add_log(s, f"Removed override for track {track_id}")
    sm.save(s)
    return jsonify({"ok": True})


@data_bp.route("/api/toggle-auto-recluster", methods=["POST"])
def api_toggle_auto_recluster():
    s = sm.load()
    s["auto_recluster_enabled"] = not s.get("auto_recluster_enabled", True)
    sm.save(s)
    return jsonify({"ok": True, "enabled": s["auto_recluster_enabled"]})


@data_bp.route("/api/cache-sources")
def api_cache_sources():
    cached_ids = db.track_ids_set()
    total = len(cached_ids)
    s = sm.load()
    sources = [{
        "type": "cache_all", "id": "cache_all",
        "name": "All Cached Tracks",
        "track_count": total,
        "description": f"{total} tracks already tagged -- instant, no API calls",
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
        tids = pl.get("track_ids", [])
        cached = sum(1 for tid in tids if tid in cached_ids)
        if not tids:
            continue
        sources.append({
            "type": "cache_playlist", "id": f"vibe_{key}",
            "name": pl.get("name", "Vibe " + key),
            "track_ids": tids, "track_count": cached,
            "description": f"{cached}/{len(tids)} tracks cached",
        })
    return jsonify({"total_cached": total, "sources": sources})
