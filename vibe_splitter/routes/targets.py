"""
Targets blueprint -- manage target playlists and their embeddings.

Endpoints:
  GET    /api/targets                   -- list configured targets
  POST   /api/targets                   -- add a target {spotify_id}
  PUT    /api/targets/<spotify_id>      -- update custom_description
  DELETE /api/targets/<spotify_id>      -- remove target
  POST   /api/targets/<spotify_id>/refresh -- force-rebuild embedding
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
        pl = sp.playlist(playlist_id, fields="id,name,description,tracks.total")
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
