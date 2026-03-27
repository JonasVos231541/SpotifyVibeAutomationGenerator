"""
Hourly update: detect source changes, route new tracks to target playlists.

Flow:
  1. Skip if no targets configured.
  2. Refresh all target embeddings (picks up songs added directly in Spotify).
  3. Check if any sources changed (snapshot comparison).
  4. Fetch all tracks from sources.
  5. Find new tracks (not in known_track_ids).
  6. Build Last.fm tag vectors for new tracks.
  7. Build fastembed embeddings.
  8. Route: auto-assign to best-matching target or send to Inbox.
  9. Push routed tracks to Spotify playlists.
 10. Update known_track_ids.
"""
import logging
from datetime import datetime

from . import config, db
from .state import state_manager as sm
from .spotify_client import get_sp, fetch_all_tracks, check_sources_changed
from .lastfm import build_vectors
from .embeddings import build_embeddings
from .router import refresh_all_target_embeddings, get_target_vectors, route_tracks
from .playlists import push_to_inbox

log = logging.getLogger("splitter.hourly")


def hourly_update(sp, state):
    targets = state.get("targets", [])
    if not targets:
        sm.add_log(state, "No targets configured -- skipping hourly sync")
        sm.save(state)
        return

    if not sm.try_acquire_job(state, "hourly"):
        return

    try:
        # Refresh target embeddings so song centroids stay current
        refresh_all_target_embeddings(targets, sp, sm, state)

        sources = state.get("sources", [])
        if not sources:
            sm.add_log(state, "No sources configured -- skipping")
            state["last_hourly"] = datetime.now().isoformat()
            sm.save(state)
            return

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
        known = set(state.get("known_track_ids", []))
        new_t = [t for t in tracks if t["id"] not in known]

        if not new_t:
            sm.add_log(state, "All up to date")
            state["known_track_ids"] = list(current_ids)
            state["last_hourly"] = datetime.now().isoformat()
            sm.save(state)
            return

        sm.add_log(state, f"Processing {len(new_t)} new tracks...")

        # Build tag vectors (Last.fm)
        records = build_vectors(new_t, sm, state, sp=sp)
        if not records:
            state["known_track_ids"] = list(current_ids)
            state["last_hourly"] = datetime.now().isoformat()
            sm.save(state)
            return

        # Build fastembed embeddings
        X = build_embeddings(records)

        # Route tracks to targets
        target_vectors = get_target_vectors(state)
        auto, inbox_recs = route_tracks(records, X, target_vectors)

        # Push auto-assigned to target playlists + inbox to Vibe Inbox
        push_to_inbox(sp, sm, state, records)

        state["known_track_ids"] = list(current_ids)
        state["last_hourly"] = datetime.now().isoformat()
        sm.save(state)

    finally:
        sm.release_job(state)
