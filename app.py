"""
Vibe Splitter — entry point.

Creates the Flask app, registers routes, starts the background scheduler.
Run with:  python app.py
"""
import os, sys, logging
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

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
log = logging.getLogger("splitter")

# ─── Write default config JSON files if they don't exist ──────────────────────
write_default_configs()

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__),
                                                    "vibe_splitter", "templates"))
app.secret_key = config.SECRET_KEY
register_routes(app)


# ─── Scheduled jobs ───────────────────────────────────────────────────────────

def _hourly():
    s = sm.load()
    t = app.config.get("STORED_TOKEN")
    if t and s.get("last_weekly") and s.get("sources"):
        try:
            hourly_update(get_sp(t), s)
        except Exception as e:
            log.error(f"Hourly failed: {e}")


def _weekly():
    s = sm.load()
    t = app.config.get("STORED_TOKEN")
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
        clusters = cluster_records(records, s["num_clusters"], s.get("overrides", {}),
                                   sm=sm, state=s, sp=sp)
        names = {k: v["name"] for k, v in s["playlists"].items()}
        push_playlists(sp, sm, s, clusters, names)
        s["known_track_ids"] = [t_obj["id"] for t_obj in tracks]
        s["last_weekly"] = s["last_hourly"] = datetime.now().isoformat()
        s["preview"] = None
        sm.save(s)
        sm.add_log(s, "Weekly recluster complete!")
    except Exception as e:
        log.error(f"Weekly failed: {e}")
    finally:
        sm.release_job(s)


# Guard: only start scheduler once (avoids double-start with Flask debug reloader)
if os.environ.get("WERKZEUG_RUN_MAIN") != "true" or not os.environ.get("WERKZEUG_RUN_MAIN"):
    scheduler = BackgroundScheduler()
    scheduler.add_job(_hourly, "interval", hours=1, id="hourly")
    scheduler.add_job(_weekly, "interval", weeks=1, id="weekly")
    scheduler.start()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from waitress import serve
        log.info("Starting with waitress on http://127.0.0.1:5000")
        print("\n  Vibe Splitter running at: http://127.0.0.1:5000\n", flush=True)
        serve(app, host="127.0.0.1", port=5000, threads=16,
              channel_timeout=600, recv_bytes=65536)
    except ImportError:
        log.warning("waitress not installed — using Flask dev server")
        app.run(debug=False, port=5000, host="127.0.0.1", threaded=True)
