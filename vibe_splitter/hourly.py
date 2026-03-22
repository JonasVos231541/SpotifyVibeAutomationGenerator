"""
Hourly update logic and helpers (removed-track handling, model staleness).

Extracted from routes.py to break circular imports between the routes.py
compat layer and the routes/ blueprint package.
"""
import os, time, logging
from datetime import datetime

from . import config, db
from .state import state_manager as sm
from .spotify_client import get_sp, fetch_all_tracks, check_sources_changed
from .lastfm import build_vectors
from .playlists import push_to_inbox
from .incremental import classify_new_tracks

log = logging.getLogger("splitter.routes")


def _handle_removed_tracks(sp, sm, state, removed_ids):
    """Remove tracks from vibe playlists, inbox, and overrides when they
    disappear from source playlists."""
    _cache = db.get_tracks_batch(list(removed_ids))
    total_removed = 0

    for key, pl in state.get("playlists", {}).items():
        pl_tids = set(pl.get("track_ids", []))
        to_remove = pl_tids & removed_ids
        if not to_remove:
            continue
        pid = pl.get("spotify_id")
        if pid:
            uris = []
            for tid in to_remove:
                entry = _cache.get(tid)
                uri = entry["uri"] if entry and entry.get("uri") else f"spotify:track:{tid}"
                uris.append({"uri": uri})
            for i in range(0, len(uris), 100):
                try:
                    sp.playlist_remove_all_occurrences_of_items(pid, uris[i:i+100])
                except Exception as e:
                    log.warning(f"Failed to remove tracks from {pl.get('name', key)}: {e}")
                time.sleep(0.1)
        pl["track_ids"] = [tid for tid in pl.get("track_ids", []) if tid not in to_remove]
        total_removed += len(to_remove)

    inbox_removed = [t for t in state.get("inbox", []) if t["id"] in removed_ids]
    if inbox_removed:
        state["inbox"] = [t for t in state.get("inbox", []) if t["id"] not in removed_ids]
        inbox_pid = state.get("inbox_playlist_id")
        if inbox_pid:
            uris = []
            for t in inbox_removed:
                uri = t.get("uri") or _cache.get(t["id"], {}).get("uri") or f"spotify:track:{t['id']}"
                uris.append({"uri": uri})
            try:
                sp.playlist_remove_all_occurrences_of_items(inbox_pid, uris)
            except Exception as e:
                log.warning(f"Failed to remove tracks from Inbox: {e}")

    overrides = state.get("overrides", {})
    for tid in removed_ids:
        overrides.pop(tid, None)

    if total_removed > 0 or inbox_removed:
        sm.add_log(state, f"Removed {total_removed + len(inbox_removed)} tracks "
                          f"no longer in sources")


def _check_model_staleness(state, current_track_count, sm):
    """Check if the model is stale and set drift_warning accordingly."""
    model_built = state.get("model_built_at")
    if model_built:
        try:
            from datetime import datetime as dt
            built_dt = dt.fromisoformat(model_built)
            age_days = (dt.now() - built_dt).days
            if age_days > config.MODEL_STALE_DAYS:
                if not state.get("drift_warning"):
                    state["drift_warning"] = True
                    sm.add_log(state, f"Model is {age_days} days old -- consider reclustering")
        except (ValueError, TypeError):
            pass

    model_count = state.get("model_track_count", 0)
    if model_count > 0 and current_track_count > 0:
        drift = abs(current_track_count - model_count) / model_count
        if drift > config.MODEL_TRACK_DRIFT_PCT:
            state["drift_warning"] = "auto_recluster"
            sm.add_log(state, f"Library changed {drift:.0%} since last cluster -- "
                              f"recluster recommended")


def hourly_update(sp, state):
    if not os.path.exists(config.MODEL_FILE):
        sm.add_log(state, "No model yet"); sm.save(state); return
    if not sm.try_acquire_job(state, "hourly"):
        return
    try:
        sources = state.get("sources", [])
        saved_snapshots = state.get("source_snapshots", {})

        changed, new_snapshots = check_sources_changed(sp, sources, saved_snapshots)
        state["source_snapshots"] = new_snapshots
        if not changed:
            sm.add_log(state, "No sources changed, skipping")
            state["last_hourly"] = datetime.now().isoformat()
            sm.save(state)
            return

        sm.add_log(state, "Hourly scan -- changes detected...")
        tracks = fetch_all_tracks(sp, sources, sm=sm, state=state)
        current_ids = {t["id"] for t in tracks}
        known  = set(state.get("known_track_ids", []))
        new_t  = [t for t in tracks if t["id"] not in known]
        removed_ids = known - current_ids

        if removed_ids:
            _handle_removed_tracks(sp, sm, state, removed_ids)

        _check_model_staleness(state, len(tracks), sm)

        if not new_t and not removed_ids:
            sm.add_log(state, "All up to date")
            state["last_hourly"] = datetime.now().isoformat()
            sm.save(state); return

        if new_t:
            records = build_vectors(new_t, sm, state, sp=sp)
            if records:
                classify_new_tracks(records, sm, state, sp=sp)
            push_to_inbox(sp, sm, state, records)

        state["known_track_ids"] = [t["id"] for t in tracks]
        state["last_hourly"] = datetime.now().isoformat()
        sm.save(state)
    finally:
        sm.release_job(state)
