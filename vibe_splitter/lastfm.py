"""
Last.fm tag fetching with exponential backoff and Spotify genre fallback.

Strategy per track:
  1. Try track-level tags from Last.fm
  2. If < 3 useful tags, supplement with artist-level tags (downweighted)
  3. If still empty, fall back to Spotify artist genres (if provided)
"""
import time, logging, requests as req
from . import config
from . import cache as cache_mod
from .events import publish as sse_publish

log = logging.getLogger("splitter.lastfm")

# Reuse HTTP connections for Last.fm API calls (keep-alive)
_session = req.Session()


def _api_call(params, max_retries=3):
    """Call Last.fm API with exponential backoff on 429 / transient errors."""
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
    return [(t["name"].lower(), int(t["count"])) for t in tags[:15]]


def get_artist_tags(artist):
    """Fetch top tags for one artist.  Returns list of ``(tag, count)``."""
    data = _api_call({
        "method": "artist.getTopTags", "artist": artist,
        "api_key": config.LASTFM_API_KEY, "format": "json",
    })
    tags = data.get("toptags", {}).get("tag", [])
    if isinstance(tags, dict):
        tags = [tags]
    return [(t["name"].lower(), int(t["count"])) for t in tags[:10]]


def fetch_and_cache_track(track_obj, sp=None):
    """
    Build a cache entry for one track.

    Multi-artist handling: joins all artist names (e.g. "Artist1 & Artist2").
    Fallback: if Last.fm returns <3 tags *and* ``sp`` is provided, uses Spotify
    artist genres as low-weight tags.
    """
    artists = track_obj.get("artists", [{}])
    # Multi-artist: primary for API lookups, joined for embedding context
    primary_artist = artists[0].get("name", "Unknown") if artists else "Unknown"
    all_artists    = " & ".join(a.get("name", "") for a in artists if a.get("name"))
    name           = track_obj.get("name", "")

    # 1. Track-level tags
    track_tags = get_track_tags(primary_artist, name)
    useful = [(tg, cnt) for tg, cnt in track_tags if tg not in config.NOISE_TAGS and len(tg) > 1]

    # 2. Artist-level tags (downweighted)
    if len(useful) < 3:
        artist_tags = get_artist_tags(primary_artist)
        artist_useful = [(tg, int(cnt * 0.7)) for tg, cnt in artist_tags
                         if tg not in config.NOISE_TAGS and len(tg) > 1]
        existing = {tg for tg, _ in useful}
        useful += [(tg, cnt) for tg, cnt in artist_useful if tg not in existing]

    # 3. Spotify artist genre fallback
    if len(useful) < 2 and sp is not None:
        try:
            for a in artists[:2]:
                aid = a.get("id")
                if not aid:
                    continue
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
    Build feature records using the tag cache.  Saves every 50 tracks so
    progress is never lost if the run is interrupted.
    """
    SAVE_EVERY = 50
    cache = cache_mod.load()
    to_fetch = [t for t in tracks if t["id"] not in cache]
    total_new = len(to_fetch)

    if total_new == 0:
        sm.add_log(state, f"All {len(tracks)} tracks cached — no Last.fm fetch needed!")
    else:
        sm.add_log(state, f"{len(cache)} cached · fetching {total_new} new tracks from Last.fm...")
        cache_mod.save(cache)

    unsaved = 0
    for i, t in enumerate(to_fetch):
        entry = fetch_and_cache_track(t, sp=sp)
        cache[t["id"]] = entry
        unsaved += 1
        if unsaved >= SAVE_EVERY:
            cache_mod.save(cache)
            unsaved = 0
            log.info(f"[cache] Incremental save at {i+1}/{total_new}")
        if cb and i % 25 == 0:
            pct = round((i / total_new) * 100)
            cb(state, f"Tagging {i+1}/{total_new} ({pct}%)...")
            sse_publish("progress", {"step": "tagging", "pct": pct,
                                     "current": i + 1, "total": total_new})
        time.sleep(config.LASTFM_RATE_DELAY)

    if total_new > 0:
        cache_mod.save(cache)
        sm.add_log(state, f"Cache saved — {len(cache)} tracks total")

    results = []
    for t in tracks:
        r = cache.get(t["id"])
        if r:
            if "tag_dict" not in r:
                r["tag_dict"] = {tag: 50 for tag in r.get("tags", [])}
            results.append(r)
    return results
