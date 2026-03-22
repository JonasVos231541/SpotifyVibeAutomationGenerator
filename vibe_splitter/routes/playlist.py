"""
Playlist blueprint — names, covers, cleanup, retag, override, hourly/weekly triggers.
"""
import os, time, logging, threading, base64
from urllib.parse import urlparse
from flask import Blueprint, request, session, jsonify
import requests as req

from .. import config, db
from ..state import state_manager as sm
from ..spotify_client import get_sp
from ..lastfm import fetch_and_cache_track
from ..incremental import classify_new_tracks
from ..hourly import hourly_update
from .helpers import _ref, _sanitize_name, _valid_id, rate_limit

log = logging.getLogger("splitter.routes.playlist")

playlist_bp = Blueprint("playlist", __name__)


@playlist_bp.route("/api/retag", methods=["POST"])
@rate_limit(5)
def api_retag():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    tid = request.json.get("track_id")
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
    s = sm.load()
    sm.add_log(s, f"Re-tagged '{new_entry['name']}' -- {', '.join(new_entry['tags'][:5])}")

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


@playlist_bp.route("/api/override", methods=["POST"])
def api_override():
    data = request.json or {}
    tid = data.get("track_id", "")
    key = str(data.get("cluster_key", ""))
    if not _valid_id(tid) or not _valid_id(key):
        return jsonify({"error": "Invalid track_id or cluster_key"}), 400

    with sm.atomic_update() as s:
        s.setdefault("overrides", {})[tid] = key
        t = _ref()
        if t and s.get("playlists"):
            sp = get_sp(t)
            rec = db.get_track(tid)
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
        sm.add_log(s, f"Override saved -- track pinned to cluster {key}")
    return jsonify({"ok": True})


@playlist_bp.route("/api/run-hourly", methods=["POST"])
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


@playlist_bp.route("/api/trigger-weekly", methods=["POST"])
@rate_limit(300)
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

    def _run():
        try:
            from app import _weekly
            _weekly()
        except Exception as e:
            log.error(f"Manual weekly trigger failed: {e}")
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "Weekly recluster started in background"})


@playlist_bp.route("/api/update-name", methods=["POST"])
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


@playlist_bp.route("/api/cleanup-playlists", methods=["POST"])
def api_cleanup_playlists():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    s = sm.load()
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


_cover_cache = {}  # {cluster_key: {"images": [...], "ts": time.time()}}
_COVER_CACHE_TTL = 86400  # 24 hours
_COVER_CACHE_MAX = 50  # max entries to prevent unbounded growth


@playlist_bp.route("/api/cover-search/<cluster_key>")
def api_cover_search(cluster_key):
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    if not _valid_id(cluster_key):
        return jsonify({"error": "Invalid cluster_key"}), 400

    cached = _cover_cache.get(cluster_key)
    if cached and time.time() - cached["ts"] < _COVER_CACHE_TTL:
        return jsonify({"images": cached["images"]})

    s = sm.load()
    cluster = (s.get("preview") or {}).get(cluster_key, {})
    top_tags = cluster.get("top_tags", [])
    sample = cluster.get("sample_tracks", [])
    if not top_tags:
        return jsonify({"images": []})
    sp = get_sp(t)
    images, seen_urls = [], set()

    # Strategy 1: search by artist + tag
    for track in sample[:2]:
        artist = track.get("artist", "")
        if not artist:
            continue
        tag = top_tags[0]
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

    # Strategy 2: tag-based genre search
    if len(images) < 6:
        for tag in top_tags[:3]:
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

    result = images[:12]
    # Evict expired entries and cap size
    now = time.time()
    expired = [k for k, v in _cover_cache.items() if now - v["ts"] > _COVER_CACHE_TTL]
    for k in expired:
        del _cover_cache[k]
    if len(_cover_cache) >= _COVER_CACHE_MAX:
        oldest_key = min(_cover_cache, key=lambda k: _cover_cache[k]["ts"])
        del _cover_cache[oldest_key]
    _cover_cache[cluster_key] = {"images": result, "ts": now}
    return jsonify({"images": result})


@playlist_bp.route("/api/upload-cover", methods=["POST"])
def api_upload_cover():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    scope = t.get("scope", "")
    if "ugc-image-upload" not in scope:
        return jsonify({"error": "Missing ugc-image-upload scope -- re-login to fix"}), 403
    data = request.json
    playlist_id = data.get("playlist_id"); image_url = data.get("image_url")
    if not playlist_id or not image_url:
        return jsonify({"error": "playlist_id and image_url required"}), 400
    # SSRF protection
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
        s = sm.load(); sm.add_log(s, "Cover updated"); sm.save(s)
        return jsonify({"ok": True})
    if resp.status_code == 403:
        return jsonify({"error": "403 -- missing ugc-image-upload scope"}), 403
    return jsonify({"error": f"Spotify returned {resp.status_code}"}), 400
