"""
Auth blueprint — login, callback, logout, token management.
"""
import os, glob, logging
from flask import Blueprint, request, session, jsonify, redirect
from spotipy.oauth2 import SpotifyOAuth

from .. import config
from ..state import state_manager as sm
from ..spotify_client import get_sp

log = logging.getLogger("splitter.routes.auth")

auth_bp = Blueprint("auth", __name__)


def _make_oauth():
    return SpotifyOAuth(
        client_id=config.SPOTIFY_CLIENT_ID,
        client_secret=config.SPOTIFY_CLIENT_SECRET,
        redirect_uri=config.SPOTIFY_REDIRECT_URI,
        scope=config.SPOTIFY_SCOPE,
        cache_path=config.SPOTIFY_CACHE_PATH,
    )


@auth_bp.route("/login")
def login():
    for f in [".cache", config.SPOTIFY_CACHE_PATH] + glob.glob(".cache-*"):
        if os.path.exists(f):
            os.remove(f)
    o = _make_oauth()
    return redirect(o.get_authorize_url() + "&show_dialog=true")


@auth_bp.route("/callback")
def callback():
    o = _make_oauth()
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


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    sm.clear_token()
    for f in [".cache", config.SPOTIFY_CACHE_PATH]:
        if os.path.exists(f):
            os.remove(f)
    return redirect("/")


@auth_bp.route("/api/wipe-token", methods=["POST"])
def api_wipe_token():
    session.clear()
    sm.clear_token()
    for f in [".cache", config.SPOTIFY_CACHE_PATH, ".cache-anonymous"] + glob.glob(".cache-*"):
        if os.path.exists(f):
            os.remove(f)
    return jsonify({"ok": True, "message": "All tokens wiped"})


@auth_bp.route("/api/token-info")
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
