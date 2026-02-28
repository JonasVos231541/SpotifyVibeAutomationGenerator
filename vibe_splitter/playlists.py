"""
Playlist management: push clusters to Spotify, inbox workflow.

Handles:
  - Creating / updating Spotify playlists from cluster data
  - Auto-assigning high-confidence tracks directly to playlists
  - Routing low-confidence tracks to the Inbox playlist
  - Approving inbox items with duplicate-track prevention
"""
import os, time, logging
from . import config
from . import cache as cache_mod
from .spotify_client import create_playlist_me
from .incremental import classify_new_tracks
import pickle

log = logging.getLogger("splitter.playlists")


def push_playlists(sp, sm, state, clusters, confirmed_names):
    """Create or update Spotify playlists for each cluster."""
    _cache = cache_mod.load()

    for key, c in clusters.items():
        name   = confirmed_names.get(key) or c["suggested_name"]
        health = c.get("health", 50)
        track_ids = c.get("track_ids", [])
        uris = []
        for tid in track_ids:
            entry = _cache.get(tid)
            if entry and entry.get("uri"):
                uris.append(entry["uri"])
        log.info(f"push: {key} '{name}' — {len(track_ids)} ids → {len(uris)} uris")
        if not uris:
            sm.add_log(state, f"Warning: '{name}' has no URIs — skipping")
            continue
        exist = state["playlists"].get(key, {}).get("spotify_id")
        if exist:
            sp.playlist_replace_items(exist, [])
            for i in range(0, len(uris), 100):
                sp.playlist_add_items(exist, uris[i:i+100]); time.sleep(0.1)
            sm.add_log(state, f"Updated '{name}' — {len(uris)} tracks")
        else:
            pl  = create_playlist_me(sp, name)
            pid = pl["id"]
            for i in range(0, len(uris), 100):
                sp.playlist_add_items(pid, uris[i:i+100]); time.sleep(0.1)
            state["playlists"][key] = {"name": name, "spotify_id": pid,
                                       "track_ids": track_ids, "health": health}
            sm.add_log(state, f"Created '{name}' — {len(uris)} tracks")
        state["playlists"][key]["name"]      = name
        state["playlists"][key]["track_ids"] = track_ids
        state["playlists"][key]["health"]    = health


def push_to_inbox(sp, sm, state, new_records):
    """
    Route new tracks: auto-assign high-confidence tracks directly to playlists,
    send low-confidence ones to the Inbox.
    """
    state.setdefault("inbox", [])
    existing_inbox_ids = {t["id"] for t in state["inbox"]}

    auto_assigned = [r for r in new_records if r.get("auto_assigned")]
    inbox_bound   = [r for r in new_records if not r.get("auto_assigned")
                     and r["id"] not in existing_inbox_ids]

    # Auto-assign high-confidence tracks
    n_auto = 0
    if auto_assigned and state.get("playlists"):
        _cache = cache_mod.load()
        for rec in auto_assigned:
            key = rec.get("predicted_cluster")
            pl  = state["playlists"].get(key, {})
            pid = pl.get("spotify_id")
            if not pid or not rec.get("uri"):
                inbox_bound.append(rec)
                continue
            # Prevent duplicates
            existing_tids = set(pl.get("track_ids", []))
            if rec["id"] in existing_tids:
                continue
            try:
                sp.playlist_add_items(pid, [rec["uri"]])
                pl.setdefault("track_ids", []).append(rec["id"])
                n_auto += 1
            except Exception as e:
                log.warning(f"Auto-assign failed for {rec.get('name')}: {e}")
                inbox_bound.append(rec)
    elif auto_assigned:
        # No playlists yet — send to inbox
        inbox_bound.extend(auto_assigned)

    if n_auto > 0:
        sm.add_log(state, f"{n_auto} tracks auto-assigned to playlists")

    # Route remaining to inbox
    if inbox_bound:
        state["inbox"].extend(inbox_bound)
        inbox_pid = state.get("inbox_playlist_id")
        if not inbox_pid:
            pl = create_playlist_me(sp, "Vibe Inbox",
                                    description="New liked songs waiting to be sorted — Vibe Splitter")
            inbox_pid = pl["id"]
            state["inbox_playlist_id"] = inbox_pid
        uris = [r.get("uri") for r in inbox_bound if r.get("uri")]
        for i in range(0, len(uris), 100):
            sp.playlist_add_items(inbox_pid, uris[i:i+100])
        sm.add_log(state, f"{len(inbox_bound)} new tracks in Inbox — review & approve in app")


def approve_inbox(sp, sm, state, approvals):
    """
    Approve inbox tracks into cluster playlists.

    Prevents duplicates by checking ``track_ids`` before adding.
    """
    if not os.path.exists(config.MODEL_FILE):
        return
    _cache    = cache_mod.load()
    inbox_pid = state.get("inbox_playlist_id")

    for a in approvals:
        tid = a["track_id"]
        key = str(a["cluster_key"])
        rec = _cache.get(tid)
        pl  = state["playlists"].get(key, {})
        pid = pl.get("spotify_id")
        if rec and pid:
            # Duplicate prevention
            existing = set(pl.get("track_ids", []))
            if tid in existing:
                sm.add_log(state, f"'{rec['name']}' already in {pl.get('name', key)} — skipped")
                continue
            sp.playlist_add_items(pid, [rec["uri"]])
            pl.setdefault("track_ids", []).append(tid)
            if inbox_pid:
                try:
                    sp.playlist_remove_all_occurrences_of_items(inbox_pid, [{"uri": rec["uri"]}])
                except Exception:
                    pass
            sm.add_log(state, f"'{rec['name']}' → {pl.get('name', 'cluster ' + key)}")

    approved_ids = {a["track_id"] for a in approvals}
    state["inbox"] = [t for t in state.get("inbox", []) if t["id"] not in approved_ids]
    sm.save(state)
