"""
Data blueprint -- state, stats, cache, playlists, overrides.
"""
import os, logging
from flask import Blueprint, request, session, jsonify

from .. import config, db
from ..state import state_manager as sm
from ..spotify_client import get_sp, get_user_playlists
from .helpers import _ref, _sanitize_name, _valid_id

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


@data_bp.route("/api/state")
def api_state():
    s = sm.load()
    token = session.get("token_info")
    if token:
        s["has_ugc_scope"] = "ugc-image-upload" in token.get("scope", "")
        s["has_playlist_read"] = "playlist-read-private" in token.get("scope", "")
        s["token_expires_at"] = token.get("expires_at")
    return jsonify(s)


@data_bp.route("/api/stats")
def api_stats():
    tracks = db.get_all_tracks()
    # Basic stats without clustering
    artists = {}
    tags = {}
    for t in tracks.values():
        a = t.get("artist", "Unknown")
        artists[a] = artists.get(a, 0) + 1
        for tag in t.get("tags", []):
            tags[tag] = tags.get(tag, 0) + 1
    top_artists = sorted(artists.items(), key=lambda x: x[1], reverse=True)[:20]
    top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:30]
    return jsonify({
        "total_tracks": len(tracks),
        "top_artists": [{"name": a, "count": c} for a, c in top_artists],
        "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
    })


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
    """Remove cached tracks that aren't in any active source or known_track_ids."""
    s = sm.load()
    active_ids = set(s.get("known_track_ids", []))
    for item in s.get("inbox", []):
        active_ids.add(item.get("id", ""))
    removed = db.prune_tracks(active_ids)
    if removed:
        sm.add_log(s, f"Pruned {removed} stale tracks from cache")
        sm.save(s)
    remaining = db.track_count()
    return jsonify({"ok": True, "removed": removed, "remaining": remaining})


@data_bp.route("/api/overrides")
def api_overrides():
    s = sm.load()
    overrides = s.get("overrides", {})
    batch = db.get_tracks_batch(list(overrides.keys()))
    targets_by_id = {tg["spotify_id"]: tg for tg in s.get("targets", [])}
    result = []
    for tid, target_id in overrides.items():
        entry = batch.get(tid, {})
        target_name = targets_by_id.get(str(target_id), {}).get("name", str(target_id))
        result.append({
            "track_id": tid,
            "name": entry.get("name", "Unknown"),
            "artist": entry.get("artist", "Unknown"),
            "target_id": str(target_id),
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


@data_bp.route("/api/sources", methods=["POST"])
def api_save_sources():
    """Save the user's source configuration to state."""
    body = request.json or {}
    sources = body.get("sources", [])
    # Basic validation: each source needs type and id
    validated = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        src_type = src.get("type", "")
        if src_type not in ("liked", "playlist", "cache_all", "cache_playlist"):
            continue
        validated.append({
            "type": src_type,
            "id": src.get("id"),
            "name": _sanitize_name(src.get("name", ""), max_len=100),
        })
    with sm.atomic_update() as s:
        s["sources"] = validated
        sm.add_log(s, f"Sources updated: {len(validated)} source(s)")
    return jsonify({"ok": True, "count": len(validated)})


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
    return jsonify({"total_cached": total, "sources": sources})
