"""
Targets blueprint -- manage target playlists and their embeddings.

Endpoints:
  GET    /api/targets                   -- list configured targets
  POST   /api/targets                   -- add a target {spotify_id}
  PUT    /api/targets/<spotify_id>      -- update custom_description
  DELETE /api/targets/<spotify_id>      -- remove target
  POST   /api/targets/<spotify_id>/refresh -- force-rebuild embedding
  GET    /api/suggest-targets           -- score user playlists for target suitability
"""
import logging
from flask import Blueprint, request, jsonify

from .. import db
from ..state import state_manager as sm
from ..spotify_client import get_sp
from ..router import build_target_embedding
from .helpers import _ref, _sanitize_name, _valid_id, rate_limit

log = logging.getLogger("splitter.routes.targets")

targets_bp = Blueprint("targets", __name__)


def _get_playlist_info(sp, playlist_id):
    """Fetch name + description from Spotify for a playlist."""
    try:
        pl = sp.playlist(playlist_id, fields="id,name,description,tracks(total)")
        return {
            "name": pl.get("name", ""),
            "description": pl.get("description", "") or "",
            "song_count": pl.get("tracks", {}).get("total", 0),
        }
    except Exception as e:
        log.warning(f"Failed to fetch playlist info for {playlist_id}: {e}")
        return {"name": playlist_id, "description": "", "song_count": 0}


@targets_bp.route("/api/targets")
def api_targets_list():
    s = sm.load()
    return jsonify({"targets": s.get("targets", [])})


@targets_bp.route("/api/targets", methods=["POST"])
@rate_limit(5)
def api_targets_add():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    body = request.json or {}
    spotify_id = body.get("spotify_id", "")
    if not _valid_id(spotify_id):
        return jsonify({"error": "Invalid spotify_id"}), 400

    s = sm.load()
    existing_ids = {tg["spotify_id"] for tg in s.get("targets", [])}
    if spotify_id in existing_ids:
        return jsonify({"error": "Target already added"}), 409

    sp = get_sp(t)
    info = _get_playlist_info(sp, spotify_id)

    target_config = {
        "spotify_id": spotify_id,
        "name": _sanitize_name(info["name"]),
        "description": _sanitize_name(info["description"], max_len=300),
        "custom_description": "",
        "song_count": info["song_count"],
    }

    # Persist metadata and build embedding
    db.upsert_target_meta(
        spotify_id,
        target_config["name"],
        target_config["description"],
        "",
    )
    try:
        build_target_embedding(target_config, sp)
    except Exception as e:
        log.warning(f"Failed to build embedding for new target {spotify_id}: {e}")

    s.setdefault("targets", []).append(target_config)
    sm.add_log(s, f"Added target playlist: '{target_config['name']}'")
    sm.save(s)
    return jsonify({"ok": True, "target": target_config})


@targets_bp.route("/api/targets/<spotify_id>", methods=["PUT"])
def api_targets_update(spotify_id):
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    if not _valid_id(spotify_id):
        return jsonify({"error": "Invalid spotify_id"}), 400

    body = request.json or {}
    custom_desc = _sanitize_name(body.get("custom_description", ""), max_len=300)

    s = sm.load()
    targets = s.get("targets", [])
    target = next((tg for tg in targets if tg["spotify_id"] == spotify_id), None)
    if not target:
        return jsonify({"error": "Target not found"}), 404

    target["custom_description"] = custom_desc
    db.upsert_target_meta(spotify_id, target["name"], target["description"], custom_desc)

    # Rebuild embedding with updated description
    sp = get_sp(t)
    try:
        build_target_embedding(target, sp)
    except Exception as e:
        log.warning(f"Failed to rebuild embedding for {spotify_id}: {e}")

    sm.add_log(s, f"Updated target '{target['name']}' custom description")
    sm.save(s)
    return jsonify({"ok": True})


@targets_bp.route("/api/targets/<spotify_id>", methods=["DELETE"])
def api_targets_remove(spotify_id):
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    if not _valid_id(spotify_id):
        return jsonify({"error": "Invalid spotify_id"}), 400

    s = sm.load()
    before = len(s.get("targets", []))
    s["targets"] = [tg for tg in s.get("targets", []) if tg["spotify_id"] != spotify_id]
    if len(s["targets"]) == before:
        return jsonify({"error": "Target not found"}), 404

    db.delete_target(spotify_id)
    sm.add_log(s, f"Removed target playlist {spotify_id}")
    sm.save(s)
    return jsonify({"ok": True})


@targets_bp.route("/api/suggest-targets")
@rate_limit(15)
def api_suggest_targets():
    """Score the user's playlists for suitability as routing targets."""
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    sp = get_sp(t)

    # Fetch playlists with description included
    playlists = []
    try:
        r = sp.current_user_playlists(limit=50)
        while r:
            for p in (r.get("items") or []):
                if not p:
                    continue
                playlists.append({
                    "id": p["id"],
                    "name": p.get("name", ""),
                    "description": (p.get("description") or "").strip(),
                    "track_count": (p.get("tracks") or {}).get("total", 0),
                })
            r = sp.next(r) if r.get("next") else None
    except Exception as e:
        log.warning(f"suggest-targets: failed to fetch playlists: {e}")
        return jsonify({"error": "Could not fetch playlists"}), 500

    s = sm.load()
    existing_ids = {tg["spotify_id"] for tg in s.get("targets", [])}

    result = []
    for pl in playlists:
        has_desc = bool(pl["description"])
        track_count = pl["track_count"]
        # Score: more songs + has description = better target candidate
        score = min(track_count, 100) + (15 if has_desc else 0)
        stars = 3 if track_count >= 20 or (track_count >= 10 and has_desc) else (2 if track_count >= 5 else 1)
        result.append({
            "id": pl["id"],
            "name": pl["name"],
            "description": pl["description"],
            "track_count": track_count,
            "has_description": has_desc,
            "stars": stars,
            "score": score,
            "already_target": pl["id"] in existing_ids,
        })

    result.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"playlists": result})


@targets_bp.route("/api/targets/<spotify_id>/refresh", methods=["POST"])
@rate_limit(30)
def api_targets_refresh(spotify_id):
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    if not _valid_id(spotify_id):
        return jsonify({"error": "Invalid spotify_id"}), 400

    s = sm.load()
    target = next((tg for tg in s.get("targets", []) if tg["spotify_id"] == spotify_id), None)
    if not target:
        return jsonify({"error": "Target not found"}), 404

    sp = get_sp(t)
    # Refresh playlist metadata from Spotify
    info = _get_playlist_info(sp, spotify_id)
    target["name"] = _sanitize_name(info["name"])
    target["description"] = _sanitize_name(info["description"], max_len=300)
    target["song_count"] = info["song_count"]
    db.upsert_target_meta(spotify_id, target["name"], target["description"], target.get("custom_description", ""))

    try:
        build_target_embedding(target, sp)
        sm.add_log(s, f"Refreshed embedding for '{target['name']}'")
    except Exception as e:
        log.warning(f"Refresh embedding failed for {spotify_id}: {e}")
        sm.add_log(s, f"Failed to refresh '{target['name']}': {e}")

    sm.save(s)
    return jsonify({"ok": True, "target": target})
