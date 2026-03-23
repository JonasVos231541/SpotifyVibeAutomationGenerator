"""
Vibe Splitter — entry point.

Creates the Flask app, registers routes, starts the background scheduler.
Run with:  python app.py
"""
import os, sys, logging, json as _json
from logging.handlers import RotatingFileHandler
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

from vibe_splitter import config
from vibe_splitter.config import write_default_configs
from vibe_splitter.state import state_manager as sm
from vibe_splitter.spotify_client import get_sp
from vibe_splitter.routes import register_routes, hourly_update
from vibe_splitter.clustering import cluster_records
from vibe_splitter.lastfm import build_vectors
from vibe_splitter.playlists import push_playlists
from vibe_splitter.db import init_db, check_embedding_version

# ─── Logging ──────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for production."""
    def format(self, record):
        return _json.dumps({
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
        }, default=str)

_log_format = os.getenv("VS_LOG_FORMAT", "text")
_log_level = getattr(logging, os.getenv("VS_LOG_LEVEL", "INFO").upper(), logging.INFO)
_root = logging.getLogger()
_root.setLevel(_log_level)

# Console handler
_console = logging.StreamHandler(sys.stderr)
if _log_format == "json":
    _console.setFormatter(_JsonFormatter())
else:
    _console.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s",
                                             datefmt="%H:%M:%S"))
_root.addHandler(_console)

# File handler with rotation (10MB, 5 backups)
_log_file = os.getenv("VS_LOG_FILE", "vibe_splitter.log")
try:
    _file_handler = RotatingFileHandler(_log_file, maxBytes=10*1024*1024, backupCount=5)
    _file_handler.setFormatter(_JsonFormatter() if _log_format == "json"
                               else logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    _root.addHandler(_file_handler)
except Exception:
    pass  # Skip file logging if can't write (e.g., read-only filesystem)

log = logging.getLogger("splitter")

# ─── Write default config JSON files if they don't exist ──────────────────────
write_default_configs()

# ─── Initialize SQLite database (auto-migrates from JSON on first run) ───────
init_db()
check_embedding_version(expected_version=2)  # TF-IDF v2 — clears old transformer embeddings

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__),
                                                    "vibe_splitter", "templates"))
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max request body
register_routes(app)


# ─── Scheduled jobs ───────────────────────────────────────────────────────────

def _hourly():
    s = sm.load()
    if s.get("job_running"):
        log.debug("Hourly skipped — another job is running")
        return
    t = sm.get_token()
    if t and s.get("last_weekly") and s.get("sources"):
        try:
            hourly_update(get_sp(t), s)
        except Exception as e:
            log.error(f"Hourly failed: {e}")
            sm.add_log(s, f"Hourly sync failed: {e}")
            sm.save(s)


def _weekly():
    s = sm.load()
    t = sm.get_token()
    if not t or not s.get("last_weekly"):
        return
    if not sm.try_acquire_job(s, "weekly"):
        return
    try:
        sp = get_sp(t)
        sm.add_log(s, "Weekly recluster triggered")
        from vibe_splitter.spotify_client import fetch_all_tracks
        tracks  = fetch_all_tracks(sp, s.get("sources", []), sm=sm, state=s)
        records = build_vectors(tracks, sm, s,
                                cb=lambda st, m: (sm.add_log(st, m), sm.save(st)),
                                sp=sp)
        extra_features = None
        if config.ENABLE_YEAR_FEATURE or config.ENABLE_POPULARITY_FEATURE:
            from vibe_splitter.embeddings import extract_extra_features
            extra_features = extract_extra_features(
                sp, tracks, config.ENABLE_YEAR_FEATURE, config.ENABLE_POPULARITY_FEATURE)
        clusters = cluster_records(records, s["num_clusters"], s.get("overrides", {}),
                                   sm=sm, state=s, sp=sp, extra_features=extra_features)
        names = {k: v["name"] for k, v in s["playlists"].items()}
        push_playlists(sp, sm, s, clusters, names)
        s["known_track_ids"] = [t_obj["id"] for t_obj in tracks]
        s["last_weekly"] = s["last_hourly"] = datetime.now().isoformat()
        s["model_built_at"] = datetime.now().isoformat()
        s["model_track_count"] = len(tracks)
        s["drift_warning"] = False
        s["preview"] = None
        sm.add_log(s, "Weekly recluster complete!")
        sm.save(s)
        # Auto-prune stale cache entries
        try:
            from vibe_splitter import db as _db
            active_ids = set(s.get("known_track_ids", []))
            for pl in (s.get("playlists") or {}).values():
                active_ids.update(pl.get("track_ids", []))
            removed = _db.prune_tracks(active_ids)
            if removed:
                sm.add_log(s, f"Auto-pruned {removed} stale cache entries")
                sm.save(s)
        except Exception as prune_e:
            log.warning(f"Auto-prune failed: {prune_e}")
    except Exception as e:
        log.error(f"Weekly failed: {e}")
        sm.add_log(s, f"Weekly recluster failed: {e}")
        sm.save(s)
    finally:
        sm.release_job(s)


# ─── Startup time (for health endpoint) ──────────────────────────────────────
_start_time = datetime.now()

# ─── Health endpoint ─────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    from flask import jsonify
    uptime = (datetime.now() - _start_time).total_seconds()
    sched_running = hasattr(globals().get("scheduler", None), "running") and scheduler.running
    # Memory usage (Linux /proc, fallback for other OS)
    mem_mb = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_mb = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass
    # DB connectivity check
    db_ok = False
    try:
        from vibe_splitter.db import track_count
        db_tracks = track_count()
        db_ok = True
    except Exception:
        db_tracks = None
    result = {
        "status": "ok" if db_ok else "degraded",
        "uptime_seconds": round(uptime),
        "scheduler": "running" if sched_running else "stopped",
        "db": "ok" if db_ok else "error",
        "db_tracks": db_tracks,
    }
    if mem_mb is not None:
        result["memory_mb"] = mem_mb
    return jsonify(result)


# Guard: only start scheduler once (avoids double-start with Flask debug reloader)
scheduler = None
if not os.environ.get("WERKZEUG_RUN_MAIN"):
    scheduler = BackgroundScheduler()
    scheduler.add_job(_hourly, "interval", hours=1, id="hourly")
    scheduler.add_job(_weekly, "interval", weeks=1, id="weekly")
    scheduler.start()

    import atexit
    atexit.register(lambda: scheduler.shutdown(wait=True) if scheduler and scheduler.running else None)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _port = int(os.environ.get("PORT", 10000))
    _host = os.environ.get("HOST", "0.0.0.0")
    try:
        from waitress import serve
        log.info(f"Starting with waitress on http://{_host}:{_port}")
        print(f"\n  Vibe Splitter running at: http://{_host}:{_port}\n", flush=True)
        serve(app, host=_host, port=_port, threads=16,
              channel_timeout=600, recv_bytes=65536)
    except ImportError:
        log.warning("waitress not installed — using Flask dev server")
        app.run(debug=False, port=_port, host=_host, threaded=True)
