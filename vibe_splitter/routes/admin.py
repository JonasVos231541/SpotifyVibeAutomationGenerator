"""
Admin/diagnostic blueprint — test-fetch, test-playlist.
"""
import time, logging
from flask import Blueprint, session, jsonify
import requests as req

admin_bp = Blueprint("admin", __name__)

log = logging.getLogger("splitter.routes.admin")


@admin_bp.route("/api/test-fetch")
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
            log.error(f"test-fetch 429 -- Retry-After: {retry_after} seconds")
            return jsonify({"status": "FAILED", "http_status": 429,
                            "retry_after": retry_after,
                            "body": resp.text[:500]}), 429
        else:
            return jsonify({"status": "FAILED", "http_status": resp.status_code,
                            "body": resp.text[:500]}), resp.status_code
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500


@admin_bp.route("/api/test-playlist/<playlist_id>")
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
