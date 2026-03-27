"""
Playlist blueprint -- retag, override, manual hourly trigger, cover upload.
"""
import base64, ipaddress, logging, socket, time
from urllib.parse import urlparse
from flask import Blueprint, request, jsonify
import requests as req

from .. import db
from ..state import state_manager as sm
from ..spotify_client import get_sp
from ..lastfm import fetch_and_cache_track
from ..hourly import hourly_update
from .helpers import _ref, _valid_id, _sanitize_name, rate_limit

log = logging.getLogger("splitter.routes.playlist")

playlist_bp = Blueprint("playlist", __name__)


@playlist_bp.route("/api/playlists/create", methods=["POST"])
def api_create_playlist():
    """Create a new Spotify playlist and return its ID. Used by the Vibe Builder."""
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    scope = t.get("scope", "")
    if "playlist-modify-public" not in scope and "playlist-modify-private" not in scope:
        return jsonify({"error": "Missing playlist-modify scope -- re-login to fix"}), 403
    data = request.json or {}
    name = _sanitize_name(data.get("name", ""), max_len=100)
    description = _sanitize_name(data.get("description", ""), max_len=300)
    if not name:
        return jsonify({"error": "name is required"}), 400
    sp = get_sp(t)
    try:
        # Use /me/playlists (current endpoint) instead of deprecated /users/{id}/playlists
        pl = sp._post("me/playlists", payload={
            "name": name,
            "description": description,
            "public": True,
        })
        s = sm.load()
        sm.add_log(s, f"Created playlist '{name}' via Vibe Builder")
        sm.save(s)
        return jsonify({"ok": True, "spotify_id": pl["id"], "name": pl["name"],
                        "description": pl.get("description", "")})
    except Exception as e:
        log.error(f"Failed to create playlist '{name}': {e}")
        err = str(e)
        if "403" in err:
            return jsonify({"error": "403 from Spotify -- re-login to grant playlist-modify access"}), 403
        return jsonify({"error": err}), 500


@playlist_bp.route("/api/retag", methods=["POST"])
@rate_limit(5)
def api_retag():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    tid = (request.json or {}).get("track_id")
    sp = get_sp(t)
    old = db.get_track(tid)
    if not old:
        return jsonify({"error": "Track not in cache"}), 404
    try:
        track_obj = sp.track(tid)
    except Exception:
        return jsonify({"error": "Spotify lookup failed"}), 500
    new_entry = fetch_and_cache_track(track_obj, sp=sp)
    db.upsert_track(new_entry)
    # Clear cached embedding so it gets rebuilt on next hourly run
    db.save_embedding(tid, None)
    s = sm.load()
    sm.add_log(s, f"Re-tagged '{new_entry['name']}' -- {', '.join(new_entry['tags'][:5])}")
    sm.save(s)
    return jsonify({"ok": True, "tags": new_entry["tags"]})


@playlist_bp.route("/api/override", methods=["POST"])
def api_override():
    """Pin a track to a specific target playlist, overriding the router's decision."""
    data = request.json or {}
    tid = data.get("track_id", "")
    target_id = str(data.get("target_spotify_id", ""))
    if not _valid_id(tid) or not _valid_id(target_id):
        return jsonify({"error": "Invalid track_id or target_spotify_id"}), 400

    t = _ref()
    with sm.atomic_update() as s:
        s.setdefault("overrides", {})[tid] = target_id
        s.setdefault("track_assignments", {})[tid] = target_id
        # Move on Spotify if we have a token and the track is cached
        if t:
            sp = get_sp(t)
            rec = db.get_track(tid)
            if rec and rec.get("uri"):
                try:
                    sp.playlist_add_items(target_id, [rec["uri"]])
                except Exception as e:
                    log.warning(f"Override add failed: {e}")
        target_name = next(
            (tg["name"] for tg in s.get("targets", []) if tg["spotify_id"] == target_id),
            target_id
        )
        sm.add_log(s, f"Override: track pinned to '{target_name}'")
    return jsonify({"ok": True})


@playlist_bp.route("/api/run-hourly", methods=["POST"])
def api_run_hourly():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    s = sm.load()
    if not s.get("targets"):
        return jsonify({"error": "No targets configured -- add a target playlist first"}), 400
    sm.set_token(t)
    hourly_update(get_sp(t), s)
    return jsonify({"ok": True})


@playlist_bp.route("/api/upload-cover", methods=["POST"])
def api_upload_cover():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    scope = t.get("scope", "")
    if "ugc-image-upload" not in scope:
        return jsonify({"error": "Missing ugc-image-upload scope -- re-login to fix"}), 403
    data = request.json
    playlist_id = data.get("playlist_id")
    image_url = data.get("image_url")
    if not playlist_id or not image_url:
        return jsonify({"error": "playlist_id and image_url required"}), 400
    # SSRF protection
    parsed = urlparse(image_url)
    if parsed.scheme != "https":
        return jsonify({"error": "Only HTTPS image URLs are allowed"}), 400
    hostname = (parsed.hostname or "").lower().strip("[]")
    _blocked = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}
    if hostname in _blocked:
        return jsonify({"error": "Internal URLs are not allowed"}), 400
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _, _, _, _, addr in resolved:
            ip = ipaddress.ip_address(addr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return jsonify({"error": "Internal URLs are not allowed"}), 400
    except (socket.gaierror, ValueError):
        return jsonify({"error": "Could not resolve hostname"}), 400
    try:
        img_resp = req.get(image_url, timeout=15)
        img_resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Could not download image: {e}"}), 400
    img_b64 = base64.b64encode(img_resp.content).decode("utf-8")
    if len(img_b64) > 256 * 1024:
        try:
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            img.thumbnail((600, 600))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=75)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except ImportError:
            return jsonify({"error": "Image > 256KB and Pillow not installed"}), 400
    resp = req.put(
        f"https://api.spotify.com/v1/playlists/{playlist_id}/images",
        headers={"Authorization": f"Bearer {t['access_token']}", "Content-Type": "image/jpeg"},
        data=img_b64, timeout=30,
    )
    if resp.status_code in (200, 202):
        s = sm.load()
        sm.add_log(s, "Cover updated")
        sm.save(s)
        return jsonify({"ok": True})
    if resp.status_code == 403:
        return jsonify({"error": "403 -- missing ugc-image-upload scope"}), 403
    return jsonify({"error": f"Spotify returned {resp.status_code}"}), 400
