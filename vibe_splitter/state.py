"""
Thread-safe state manager with atomic file writes.

All reads and writes go through a single ``StateManager`` instance which
wraps file I/O with a ``threading.RLock`` and uses write-to-temp + rename
to avoid corruption when the app crashes mid-write.
"""
import copy, json, os, tempfile, threading, logging
from contextlib import contextmanager
from datetime import datetime
from . import config

log = logging.getLogger("splitter.state")

_DEFAULT_STATE = {
    "sources":           [],
    "source_snapshots":  {},
    "targets":           [],
    "known_track_ids":   [],
    "track_assignments": {},
    "overrides":         {},
    "inbox":             [],
    "inbox_playlist_id": None,
    "logs":              [],
    "last_hourly":       None,
    "job_running":       None,
}


class StateManager:
    """
    Singleton-style state wrapper.  Usage::

        sm = StateManager()
        s  = sm.load()
        s["preview"] = clusters
        sm.save(s)
        sm.add_log(s, "Done!")
    """

    def __init__(self, path=None):
        self._path = path or config.STATE_FILE
        self._lock = threading.RLock()
        self._token = None
        self._token_lock = threading.RLock()

    # ── Read ──────────────────────────────────────────────────────────────────
    def load(self):
        with self._lock:
            if os.path.exists(self._path):
                try:
                    with open(self._path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log.warning("State file corrupted — resetting to defaults")
            return copy.deepcopy(_DEFAULT_STATE)

    # ── Atomic write ──────────────────────────────────────────────────────────
    def save(self, s):
        with self._lock:
            try:
                dir_name = os.path.dirname(self._path) or "."
                fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(s, f, indent=2)
                    os.replace(tmp, self._path)
                except Exception:
                    # Clean up temp file on failure
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
            except Exception as e:
                log.error(f"State save failed: {e}")

    # ── Logging helper ────────────────────────────────────────────────────────
    def add_log(self, s, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        s.setdefault("logs", []).insert(0, f"[{ts}] {msg}")
        s["logs"] = s["logs"][:80]
        log.info(msg)
        # Push to SSE clients (lazy import to avoid circular deps)
        try:
            from .events import publish
            publish("log", {"message": msg, "timestamp": ts})
        except Exception:
            pass

    # ── Job locking (prevents hourly/weekly overlap) ──────────────────────────
    def try_acquire_job(self, s, job_name):
        """Return True if no other job is running and mark *job_name* active."""
        if s.get("job_running"):
            log.warning(f"Job '{job_name}' skipped — '{s['job_running']}' is running")
            return False
        s["job_running"] = job_name
        self.save(s)
        return True

    def release_job(self, s):
        s["job_running"] = None
        self.save(s)

    # ── Token management (thread-safe in-memory) ─────────────────────────────
    def get_token(self):
        """Return the current Spotify token dict, or None."""
        with self._token_lock:
            return self._token

    def set_token(self, token):
        """Store a Spotify token dict (thread-safe)."""
        with self._token_lock:
            self._token = token

    def clear_token(self):
        """Clear the stored token."""
        with self._token_lock:
            self._token = None

    @contextmanager
    def atomic_update(self):
        """Context manager for atomic read-modify-write.

        Usage::

            with sm.atomic_update() as s:
                s["preview"] = clusters
            # auto-saved on exit
        """
        with self._lock:
            s = self.load()
            yield s
            self.save(s)


# Module-level singleton — imported by other modules
state_manager = StateManager()
