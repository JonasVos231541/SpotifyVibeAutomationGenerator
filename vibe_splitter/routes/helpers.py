"""
Shared helpers for route blueprints.
"""
import re
import time
import threading
from functools import wraps
from flask import jsonify

from .. import db
from ..state import state_manager as sm
from ..spotify_client import refresh_token

_preview_lock = threading.Lock()

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-:.]+$")


def _sanitize_name(name, max_len=100):
    """Strip HTML tags and cap length for playlist/cluster names."""
    if not isinstance(name, str):
        return ""
    clean = re.sub(r"<[^>]+>", "", name).strip()
    return clean[:max_len]


def _valid_id(val):
    """Return True if val looks like a safe Spotify / cluster ID."""
    return isinstance(val, str) and bool(_SAFE_ID_RE.match(val))


def _ref():
    """Shorthand for refresh_token with current session & state manager."""
    from flask import session
    return refresh_token(session, sm)


def _collect_active_ids(state):
    """Gather all track IDs that should be retained (known, assignments, inbox)."""
    active = set(state.get("known_track_ids", []))
    active.update(state.get("track_assignments", {}).keys())
    for item in state.get("inbox", []):
        active.add(item.get("id", ""))
    return active


# ── Rate limiter ─────────────────────────────────────────────────────────────

_rate_limits = {}  # {endpoint_name: last_call_time}
_rate_lock = threading.Lock()


def rate_limit(cooldown_seconds):
    """Decorator that returns 429 if the endpoint is called again within cooldown."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = fn.__name__
            with _rate_lock:
                last = _rate_limits.get(key, 0)
                remaining = cooldown_seconds - (now - last)
                if remaining > 0:
                    return jsonify({
                        "error": "Too many requests",
                        "retry_after": round(remaining),
                    }), 429
                _rate_limits[key] = now
            return fn(*args, **kwargs)
        return wrapper
    return decorator
