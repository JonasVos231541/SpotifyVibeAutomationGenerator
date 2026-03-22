"""
Inbox blueprint — approve and dismiss inbox tracks.
"""
from flask import Blueprint, request, jsonify

from ..state import state_manager as sm
from ..spotify_client import get_sp
from ..playlists import approve_inbox
from .helpers import _ref

inbox_bp = Blueprint("inbox", __name__)


@inbox_bp.route("/api/inbox/approve", methods=["POST"])
def api_inbox_approve():
    t = _ref()
    if not t:
        return jsonify({"error": "Not logged in"}), 401
    s = sm.load()
    approve_inbox(get_sp(t), sm, s, request.json.get("approvals", []))
    return jsonify({"ok": True})


@inbox_bp.route("/api/inbox/dismiss", methods=["POST"])
def api_inbox_dismiss():
    tids = set(request.json.get("track_ids", []))
    s = sm.load()
    s["inbox"] = [t for t in s.get("inbox", []) if t["id"] not in tids]
    sm.save(s)
    return jsonify({"ok": True})
