"""
Blueprint registration and core routes (index, SSE, error handler, security headers).

Exports register_routes and hourly_update for backward compat with app.py.
"""
import logging
from flask import jsonify, render_template, Response, session, request

from ..events import stream_events
from ..hourly import hourly_update  # noqa: F401 — re-exported for app.py
from .auth import auth_bp
from .data import data_bp
from .targets import targets_bp
from .playlist import playlist_bp
from .inbox import inbox_bp
from .admin import admin_bp

log = logging.getLogger("splitter.routes")


def register_routes(app):
    """Register all blueprints and attach core middleware."""

    @app.errorhandler(Exception)
    def _handle_error(e):
        log.exception(f"Unhandled error: {e}")
        code = getattr(e, "code", 500)
        return jsonify({"error": str(e), "code": code}), code

    @app.before_request
    def _csrf_check():
        if request.method in ("POST", "PUT", "DELETE"):
            if not request.headers.get("X-Requested-With"):
                return jsonify({"error": "Missing X-Requested-With header"}), 403

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

    @app.route("/")
    def index():
        return render_template("index.html", logged_in="token_info" in session)

    @app.route("/api/events")
    def api_events():
        return Response(stream_events(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    app.register_blueprint(auth_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(targets_bp)
    app.register_blueprint(playlist_bp)
    app.register_blueprint(inbox_bp)
    app.register_blueprint(admin_bp)
