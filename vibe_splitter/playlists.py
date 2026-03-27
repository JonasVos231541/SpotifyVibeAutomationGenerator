"""
Playlist management: route tracks to targets, inbox workflow.

Handles:
  - Auto-routing high-confidence tracks directly to target playlists
  - Sending low-confidence tracks to the Vibe Inbox playlist
  - Approving inbox items to a chosen target with duplicate-track prevention
"""
import time, logging
from . import db
from .spotify_client import create_playlist_me

log = logging.getLogger("splitter.playlists")


def push_to_inbox(sp, sm, state, new_records):
    """
    Push routed tracks to their destinations.

    Records are expected to already have auto_assigned / assigned_target set
    by router.route_tracks().

    Auto-assigned records go directly to the target Spotify playlist.
    Others go to the Vibe Inbox playlist for manual review.
    """
    state.setdefault("inbox", [])
    state.setdefault("track_assignments", {})
    existing_inbox_ids = {t["id"] for t in state["inbox"]}

    auto_assigned = [r for r in new_records if r.get("auto_assigned")]
    inbox_bound = [r for r in new_records
                   if not r.get("auto_assigned") and r["id"] not in existing_inbox_ids]

    # Push auto-assigned tracks to their target Spotify playlists
    n_auto = 0
    if auto_assigned:
        by_target = {}
        for rec in auto_assigned:
            tid = rec.get("assigned_target")
            if not tid:
                inbox_bound.append(rec)
                continue
            by_target.setdefault(tid, []).append(rec)

        for target_id, recs in by_target.items():
            already = {k for k, v in state["track_assignments"].items() if v == target_id}
            uris_to_add = []
            for rec in recs:
                if rec["id"] in already or not rec.get("uri"):
                    if not rec.get("uri"):
                        inbox_bound.append(rec)
                    continue
                uris_to_add.append(rec["uri"])
                state["track_assignments"][rec["id"]] = target_id
                n_auto += 1

            if uris_to_add:
                for i in range(0, len(uris_to_add), 100):
                    try:
                        sp.playlist_add_items(target_id, uris_to_add[i:i+100])
                    except Exception as e:
                        log.warning(f"Failed to add tracks to target {target_id}: {e}")
                    time.sleep(0.1)

    if n_auto > 0:
        sm.add_log(state, f"{n_auto} tracks auto-routed to target playlists")

    # Send remaining to Vibe Inbox
    if inbox_bound:
        state["inbox"].extend(inbox_bound)
        inbox_pid = state.get("inbox_playlist_id")
        if not inbox_pid:
            pl = create_playlist_me(sp, "Vibe Inbox",
                                    description="New liked songs waiting to be sorted -- Vibe Splitter")
            inbox_pid = pl["id"]
            state["inbox_playlist_id"] = inbox_pid
        uris = [r.get("uri") for r in inbox_bound if r.get("uri")]
        for i in range(0, len(uris), 100):
            sp.playlist_add_items(inbox_pid, uris[i:i+100])
        sm.add_log(state, f"{len(inbox_bound)} new tracks in Inbox -- review & approve in app")


def approve_inbox(sp, sm, state, approvals):
    """
    Approve inbox tracks into target playlists.

    approvals: [{"track_id": "...", "target_spotify_id": "..."}]

    Prevents duplicates by checking track_assignments before adding.
    """
    _approve_ids = [a["track_id"] for a in approvals]
    _cache = db.get_tracks_batch(_approve_ids)
    inbox_pid = state.get("inbox_playlist_id")
    state.setdefault("track_assignments", {})

    targets_by_id = {t["spotify_id"]: t for t in state.get("targets", [])}

    for a in approvals:
        tid = a["track_id"]
        target_id = str(a["target_spotify_id"])
        rec = _cache.get(tid)
        if not rec or not rec.get("uri"):
            continue

        target_name = targets_by_id.get(target_id, {}).get("name", target_id)

        # Prevent duplicates
        if state["track_assignments"].get(tid) == target_id:
            sm.add_log(state, f"'{rec['name']}' already in '{target_name}' -- skipped")
            continue

        try:
            sp.playlist_add_items(target_id, [rec["uri"]])
            state["track_assignments"][tid] = target_id
            if inbox_pid:
                try:
                    sp.playlist_remove_all_occurrences_of_items(inbox_pid, [{"uri": rec["uri"]}])
                except Exception:
                    pass
            sm.add_log(state, f"'{rec['name']}' -> '{target_name}'")
        except Exception as e:
            log.warning(f"Failed to approve '{rec.get('name')}': {e}")

    approved_ids = {a["track_id"] for a in approvals}
    state["inbox"] = [t for t in state.get("inbox", []) if t["id"] not in approved_ids]
    sm.save(state)
