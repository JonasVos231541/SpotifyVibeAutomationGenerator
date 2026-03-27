"""
Last.fm tag fetching with exponential backoff and Spotify genre fallback.

Strategy per track:
  1. Try track-level tags from Last.fm
  2. If < 3 useful tags, supplement with artist-level tags (downweighted)
  3. If still empty, fall back to Spotify artist genres (if provided)
"""
import time, logging, threading, requests as req
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config
from .events import publish as sse_publish

log = logging.getLogger("splitter.lastfm")

# Reuse HTTP connections for Last.fm API calls (keep-alive)
_session = req.Session()

# Global rate limiter — ensures API calls are spaced at least LASTFM_RATE_DELAY
# apart even when multiple threads are fetching concurrently.
_rate_lock = threading.Lock()
_last_api_time = 0.0


def _api_call(params, max_retries=3):
    """Call Last.fm API with global rate limiting and exponential backoff on 429."""
    global _last_api_time

    # Reserve a time slot (threads wait their turn, then sleep outside the lock)
    with _rate_lock:
        now = time.time()
        wait = max(0, config.LASTFM_RATE_DELAY - (now - _last_api_time))
        _last_api_time = now + wait
    if wait > 0:
        time.sleep(wait)

    backoff = 0.5
    for attempt in range(max_retries):
        try:
            r = _session.get(config.LASTFM_BASE, params=params, timeout=8)
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", backoff))
                wait = max(retry_after, backoff)
                log.warning(f"Last.fm 429 — waiting {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
                backoff = min(backoff * 2, 30)
                continue
            return r.json()
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)
    return {}


def get_track_tags(artist, track):
    """Fetch top tags for one track.  Returns list of ``(tag, count)``."""
    data = _api_call({
        "method": "track.getTopTags", "artist": artist, "track": track,
        "api_key": config.LASTFM_API_KEY, "format": "json",
    })
    tags = data.get("toptags", {}).get("tag", [])
    if isinstance(tags, dict):
        tags = [tags]
    return [(t["name"].lower(), int(t.get("count") or 0)) for t in tags[:15]]


def get_artist_tags(artist):
    """Fetch top tags for one artist.  Returns list of ``(tag, count)``."""
    data = _api_call({
        "method": "artist.getTopTags", "artist": artist,
        "api_key": config.LASTFM_API_KEY, "format": "json",
    })
    tags = data.get("toptags", {}).get("tag", [])
    if isinstance(tags, dict):
        tags = [tags]
    return [(t["name"].lower(), int(t.get("count") or 0)) for t in tags[:10]]


def fetch_and_cache_track(track_obj, sp=None, artist_cache=None, artist_lock=None):
    """
    Build a cache entry for one track.

    Multi-artist handling: joins all artist names (e.g. "Artist1 & Artist2").
    Fallback: if Last.fm returns <3 tags *and* ``sp`` is provided, uses Spotify
    artist genres as low-weight tags.

    ``artist_cache``: optional dict to cache artist tags across tracks
    in the same batch, avoiding duplicate Last.fm calls.
    ``artist_lock``: optional threading.Lock for thread-safe artist_cache access.
    """
    artists = track_obj.get("artists", [{}])
    # Multi-artist: primary for API lookups, joined for embedding context
    primary_artist = artists[0].get("name", "Unknown") if artists else "Unknown"
    all_artists    = " & ".join(a.get("name", "") for a in artists if a.get("name"))
    name           = track_obj.get("name", "")

    # 1. Track-level tags
    track_tags = get_track_tags(primary_artist, name)
    useful = [(tg, cnt) for tg, cnt in track_tags if tg not in config.NOISE_TAGS and len(tg) > 1]

    # 2. Artist-level tags (downweighted) — use cache if available
    if len(useful) < 3:
        _lock = artist_lock or threading.Lock()
        with _lock:
            cached = artist_cache.get(primary_artist) if artist_cache is not None else None
        if cached is not None:
            artist_tags = cached
        else:
            artist_tags = get_artist_tags(primary_artist)
            if artist_cache is not None:
                with _lock:
                    artist_cache[primary_artist] = artist_tags
        artist_useful = [(tg, int(cnt * 0.7)) for tg, cnt in artist_tags
                         if tg not in config.NOISE_TAGS and len(tg) > 1]
        existing = {tg for tg, _ in useful}
        useful += [(tg, cnt) for tg, cnt in artist_useful if tg not in existing]

    # 3. Spotify artist genre fallback — gated by rate budget to prevent bans
    if len(useful) < 2 and sp is not None:
        try:
            from .spotify_client import _rate_budget
            if _rate_budget.can_call():
                for a in artists[:2]:
                    aid = a.get("id")
                    if not aid:
                        continue
                    if not _rate_budget.can_call():
                        break
                    _rate_budget.record()
                    artist_data = sp.artist(aid)
                    for genre in artist_data.get("genres", [])[:5]:
                        g = genre.lower()
                        if g not in config.NOISE_TAGS and g not in {tg for tg, _ in useful}:
                            useful.append((g, 30))  # low weight
        except Exception as e:
            log.debug(f"Spotify genre fallback failed for {primary_artist}: {e}")

    tag_dict = {tg: cnt for tg, cnt in useful[:20]}

    return {
        "id":       track_obj["id"],
        "uri":      track_obj.get("uri", f"spotify:track:{track_obj['id']}"),
        "name":     name,
        "artist":   primary_artist,
        "artists":  all_artists,
        "tags":     [tg for tg, _ in useful[:8]],
        "tag_dict": tag_dict,
    }


def build_vectors(tracks, sm, state, cb=None, sp=None):
    """
    Build feature records using the tag cache.  Fetches new tracks from Last.fm
    using parallel workers for ~5x speedup.  Saves every 50 tracks so progress
    is never lost if the run is interrupted.
    """
    from . import db
    SAVE_EVERY = 50
    cached_ids = db.track_ids_set()
    to_fetch = [t for t in tracks if t["id"] not in cached_ids]
    total_new = len(to_fetch)
    total_cached = len(cached_ids)

    if total_new == 0:
        if state is not None:
            sm.add_log(state, f"All {len(tracks)} tracks cached — no Last.fm fetch needed!")
    else:
        if state is not None:
            sm.add_log(state, f"{total_cached} cached · fetching {total_new} new tracks from Last.fm ({config.LASTFM_WORKERS} workers)...")

    if total_new > 0:
        artist_cache = {}
        artist_lock = threading.Lock()
        completed = 0
        completed_lock = threading.Lock()
        batch_entries = []
        batch_lock = threading.Lock()

        def _fetch_one(track_obj):
            """Fetch tags for one track (rate limiting handled by _api_call)."""
            return fetch_and_cache_track(
                track_obj, sp=sp,
                artist_cache=artist_cache, artist_lock=artist_lock,
            )

        workers = min(config.LASTFM_WORKERS, total_new)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in to_fetch}
            for future in as_completed(futures):
                try:
                    entry = future.result()
                except Exception as e:
                    t = futures[future]
                    log.warning(f"Failed to fetch tags for {t.get('name', '?')}: {e}")
                    continue

                to_save = None
                with batch_lock:
                    batch_entries.append(entry)
                    if len(batch_entries) >= SAVE_EVERY:
                        to_save = batch_entries[:]
                        batch_entries.clear()
                if to_save:
                    db.upsert_tracks_batch(to_save)
                    log.info(f"[cache] Incremental save — {len(to_save)} tracks")

                with completed_lock:
                    completed += 1
                    if cb and completed % 25 == 0:
                        pct = round((completed / total_new) * 100)
                        cb(state, f"Tagging {completed}/{total_new} ({pct}%)...")
                        sse_publish("progress", {"step": "tagging", "pct": pct,
                                                 "current": completed, "total": total_new})

        # Flush remaining entries
        if batch_entries:
            db.upsert_tracks_batch(batch_entries)

        if state is not None:
            sm.add_log(state, f"Cache saved — {total_cached + total_new} tracks total")

    # Load results from DB for all requested tracks
    all_ids = [t["id"] for t in tracks]
    all_cached = db.get_tracks_batch(all_ids)
    results = []
    for t in tracks:
        r = all_cached.get(t["id"])
        if r:
            if "tag_dict" not in r:
                r["tag_dict"] = {tag: 50 for tag in r.get("tags", [])}
            results.append(r)
    return results
