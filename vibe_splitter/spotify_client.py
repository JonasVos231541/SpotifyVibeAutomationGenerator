"""
Spotify API wrapper.

Handles:
  - Feb 2026 endpoint migration (monkey-patches spotipy)
  - Token refresh with session/cache fallback
  - Track fetching with exponential backoff on 429s
  - Audio-feature batch retrieval with detailed 429 logging
  - Playlist fetching with 429 handling
"""
import json, os, time, logging
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException
from . import config

log = logging.getLogger("splitter.spotify")


# ─── Simple circuit breaker ──────────────────────────────────────────────────
class _CircuitBreaker:
    """Tracks consecutive API failures. Raises early when circuit is open."""
    def __init__(self, threshold=3, cooldown=60):
        self._failures = 0
        self._threshold = threshold
        self._cooldown = cooldown
        self._open_until = 0

    def check(self):
        if self._failures >= self._threshold and time.time() < self._open_until:
            raise ConnectionError(
                f"Spotify API circuit open — {self._failures} consecutive failures. "
                f"Retry after {int(self._open_until - time.time())}s")

    def record_success(self):
        self._failures = 0

    def record_failure(self):
        self._failures += 1
        if self._failures >= self._threshold:
            self._open_until = time.time() + self._cooldown
            log.warning(f"Circuit breaker OPEN — {self._failures} failures, "
                        f"cooldown {self._cooldown}s")

_spotify_cb = _CircuitBreaker()


# ─── Rate budget tracker ──────────────────────────────────────────────────────
class _RateBudget:
    """Track API calls in a sliding window to prevent rate limit bans."""
    def __init__(self, max_calls_per_hour=150):
        self._calls = []
        self._max = max_calls_per_hour

    def can_call(self):
        now = time.time()
        self._calls = [t for t in self._calls if now - t < 3600]
        return len(self._calls) < self._max

    def record(self):
        self._calls.append(time.time())

    def remaining(self):
        now = time.time()
        self._calls = [t for t in self._calls if now - t < 3600]
        return self._max - len(self._calls)


_rate_budget = _RateBudget()


def get_sp(token):
    """Build a ``spotipy.Spotify`` instance with Feb 2026 endpoint patches."""
    sp = spotipy.Spotify(auth=token["access_token"], requests_timeout=30)

    # ── Feb 2026 patch: /playlists/{id}/tracks → /items ──
    def patched_add(playlist_id, items, position=None):
        plid = sp._get_id("playlist", playlist_id)
        uris = [sp._get_uri("track", tid) for tid in items]
        payload = {"uris": uris}
        if position is not None:
            payload["position"] = position
        return sp._post("playlists/%s/items" % plid, payload=payload)

    def patched_replace(playlist_id, items):
        plid = sp._get_id("playlist", playlist_id)
        uris = [sp._get_uri("track", tid) for tid in items]
        return sp._put("playlists/%s/items" % plid, payload={"uris": uris})

    def patched_remove(playlist_id, items):
        plid = sp._get_id("playlist", playlist_id)
        tracks = [{"uri": item["uri"]} if isinstance(item, dict) else {"uri": item} for item in items]
        return sp._delete("playlists/%s/items" % plid, payload={"tracks": tracks})

    def patched_get_items(playlist_id, fields=None, limit=100, offset=0,
                          market=None, additional_types=("track", "episode")):
        plid = sp._get_id("playlist", playlist_id)
        return sp._get(
            "playlists/%s/items" % plid,
            limit=limit, offset=offset, fields=fields, market=market,
            additional_types=",".join(additional_types) if additional_types else None,
        )

    sp.playlist_add_items = patched_add
    sp.playlist_replace_items = patched_replace
    sp.playlist_remove_all_occurrences_of_items = patched_remove
    sp.playlist_items = patched_get_items
    sp.playlist_tracks = patched_get_items
    return sp


def is_healthy(sp):
    """Quick connectivity check — returns True if Spotify API responds."""
    try:
        sp.current_user()
        _spotify_cb.record_success()
        return True
    except Exception as e:
        _spotify_cb.record_failure()
        log.warning(f"Spotify health check failed: {e}")
        return False


def refresh_token(session, sm):
    """Return a valid token dict, refreshing if needed.  Falls back to stored/cached tokens."""
    token = session.get("token_info")
    if not token:
        token = sm.get_token()
        if token:
            session["token_info"] = token
    if not token and os.path.exists(config.SPOTIFY_CACHE_PATH):
        try:
            with open(config.SPOTIFY_CACHE_PATH) as f:
                token = json.load(f)
            session["token_info"] = token
            sm.set_token(token)
        except Exception:
            token = None
    if not token:
        return None
    oauth = SpotifyOAuth(
        client_id=config.SPOTIFY_CLIENT_ID,
        client_secret=config.SPOTIFY_CLIENT_SECRET,
        redirect_uri=config.SPOTIFY_REDIRECT_URI,
        scope=config.SPOTIFY_SCOPE,
        cache_path=config.SPOTIFY_CACHE_PATH,
    )
    if oauth.is_token_expired(token):
        try:
            token = oauth.refresh_access_token(token["refresh_token"])
            session["token_info"] = token
            sm.set_token(token)
        except Exception as e:
            log.error(f"Token refresh failed: {e}")
            return None
    return token


# ─── Track fetching with exponential backoff ──────────────────────────────────

def _fetch_page(sp, src, offset):
    if src["type"] == "liked":
        return sp.current_user_saved_tracks(limit=50, offset=offset)
    return sp.playlist_items(src["id"], limit=100, offset=offset)


def fetch_tracks_from_source(sp, src, sm=None, state=None):
    """Fetch all tracks from a single source dict ``{type, id}`` with exponential backoff."""
    tracks, offset = [], 0
    MAX_PAGES = 500
    backoff = 1  # seconds — doubles after each consecutive 429

    for _page in range(MAX_PAGES):
        _spotify_cb.check()
        if not _rate_budget.can_call():
            log.warning(f"[fetch] Rate budget exhausted ({_rate_budget.remaining()} remaining) — stopping fetch")
            break
        _rate_budget.record()
        t0 = time.time()
        try:
            r = _fetch_page(sp, src, offset)
            _spotify_cb.record_success()
            backoff = 1  # reset on success
        except SpotifyException as e:
            elapsed = time.time() - t0
            if e.http_status == 429:
                retry_header = e.headers.get("Retry-After") if e.headers else None
                retry_after = int(retry_header) if retry_header else backoff
                # Cap wait at 120s — if Spotify says wait longer, abort instead
                if retry_after > 120:
                    log.error(f"[fetch] 429 Retry-After={retry_after}s (>{120}s cap) — aborting")
                    _spotify_cb.record_failure()
                    raise ConnectionError(f"Spotify rate limit too long: {retry_after}s. Try again later.")
                wait = max(retry_after, backoff) + 1
                log.warning(f"[fetch] 429 page={_page} — Retry-After: {retry_header}, waiting {wait}s (backoff={backoff})")
                if sm and state:
                    sm.add_log(state, f"Spotify rate limit — waiting {wait}s...")
                    sm.save(state)
                time.sleep(wait)
                backoff = min(backoff * 2, 120)
                try:
                    r = _fetch_page(sp, src, offset)
                except Exception as retry_e:
                    log.error(f"[fetch] retry also failed: {retry_e}")
                    raise
            else:
                _spotify_cb.record_failure()
                log.error(f"[fetch] HTTP {e.http_status} page={_page}: {e}")
                raise
        except ConnectionError:
            raise
        except Exception as e:
            _spotify_cb.record_failure()
            log.error(f"[fetch] page={_page}: {type(e).__name__}: {e}")
            raise

        items = r.get("items", [])
        for item in items:
            if not isinstance(item, dict):
                continue
            t = item.get("track") or item.get("item")
            if t and isinstance(t, dict) and t.get("id"):
                tracks.append(t)
            elif item.get("id") and item.get("uri") and item.get("name"):
                tracks.append(item)

        if not r.get("next") or not items:
            break
        offset += len(items)
        if sm and state and offset % 50 == 0:
            sm.add_log(state, f"Fetching tracks from Spotify... {len(tracks)} so far")
            sm.save(state)
        time.sleep(0.15)

    log.info(f"[fetch] Done: {len(tracks)} tracks from source")
    return tracks


def fetch_tracks_from_cache_source(src, cache):
    """Build track-like dicts from the local tag cache (no API call)."""
    if src["type"] == "cache_all":
        return [{"id": tid, "uri": e.get("uri", f"spotify:track:{tid}"),
                 "name": e.get("name", ""), "artists": [{"name": e.get("artist", "")}]}
                for tid, e in cache.items()]
    if src["type"] == "cache_playlist":
        wanted = set(src.get("track_ids", []))
        return [{"id": tid, "uri": e.get("uri", f"spotify:track:{tid}"),
                 "name": e.get("name", ""), "artists": [{"name": e.get("artist", "")}]}
                for tid in wanted if (e := cache.get(tid))]
    return []


def fetch_all_tracks(sp, sources, sm=None, state=None):
    """Merge tracks from all sources, deduplicating by track ID."""
    from . import cache as cache_mod
    seen, merged = set(), []
    cache_srcs   = [s for s in sources if s["type"] in ("cache_all", "cache_playlist")]
    spotify_srcs = [s for s in sources if s["type"] not in ("cache_all", "cache_playlist")]
    cache = cache_mod.load() if cache_srcs else {}

    for src in cache_srcs:
        for t in fetch_tracks_from_cache_source(src, cache):
            if t["id"] not in seen:
                seen.add(t["id"]); merged.append(t)
        if sm and state:
            sm.add_log(state, f"Loaded {len(merged)} tracks from cache"); sm.save(state)

    for i, src in enumerate(spotify_srcs):
        log.info(f"[fetch_all] source {i+1}/{len(spotify_srcs)}: {src}")
        for t in fetch_tracks_from_source(sp, src, sm=sm, state=state):
            if t["id"] not in seen:
                seen.add(t["id"]); merged.append(t)
    log.info(f"[fetch_all] {len(merged)} unique tracks total")
    return merged


def get_user_playlists(sp):
    """Return a list of ``{id, name, tracks}`` for the current user's playlists with 429 handling."""
    pls = []
    try:
        r = sp.current_user_playlists(limit=50)
    except SpotifyException as e:
        if e.http_status == 429:
            retry_after = e.headers.get("Retry-After") if e.headers else "unknown"
            log.error(f"get_user_playlists: 429 — Retry-After: {retry_after}")
        raise

    while r:
        for p in r["items"]:
            if not p:
                continue
            tracks_info = p.get("tracks") or {}
            pls.append({"id": p["id"], "name": p["name"],
                        "tracks": tracks_info.get("total", 0)})
        try:
            r = sp.next(r) if r.get("next") else None
        except SpotifyException as e:
            if e.http_status == 429:
                retry_after = e.headers.get("Retry-After") if e.headers else "unknown"
                log.error(f"get_user_playlists pagination: 429 — Retry-After: {retry_after}")
            raise
    return pls


def fetch_audio_features(sp, track_ids, sm=None, state=None):
    """Fetch Spotify audio features in batches of 100, with SQLite caching.

    Checks the DB cache first and only fetches missing tracks from Spotify.
    Returns ``{tid: {key: val}}``.
    """
    KEYS = ["danceability", "energy", "valence", "acousticness",
            "instrumentalness", "tempo", "speechiness", "mode",
            "liveness", "loudness"]

    # Check cache first
    features = {}
    to_fetch = list(track_ids)
    try:
        from . import db
        cached = db.get_audio_features_batch(track_ids)
        if cached:
            features.update(cached)
            to_fetch = [tid for tid in track_ids if tid not in cached]
            log.info(f"Audio features cache hit: {len(cached)}/{len(track_ids)}, "
                     f"fetching {len(to_fetch)} from API")
    except Exception as e:
        log.debug(f"Audio feature cache unavailable: {e}")

    if not to_fetch:
        if sm and state:
            sm.add_log(state, f"Audio features: {len(features)}/{len(track_ids)} from cache")
        return features

    # Fetch missing from Spotify API
    new_features = {}
    backoff = 1
    for i in range(0, len(to_fetch), 100):
        batch = to_fetch[i:i+100]
        try:
            results = sp.audio_features(batch)
            if results:
                for tid, feat in zip(batch, results):
                    if feat:
                        new_features[tid] = {k: float(feat.get(k, 0.0)) for k in KEYS}
            backoff = 1  # reset on success
        except SpotifyException as e:
            if e.http_status == 429:
                retry_header = e.headers.get("Retry-After") if e.headers else None
                retry_after = int(retry_header) if retry_header else backoff
                if retry_after > 120:
                    log.error(f"audio_features batch {i}: 429 Retry-After={retry_after}s — skipping remaining")
                    break
                wait = max(retry_after, backoff) + 1
                log.warning(f"audio_features batch {i}: 429 — waiting {wait}s")
                if sm and state:
                    sm.add_log(state, f"Audio features rate limited — waiting {wait}s...")
                    sm.save(state)
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
                try:
                    results = sp.audio_features(batch)
                    if results:
                        for tid, feat in zip(batch, results):
                            if feat:
                                new_features[tid] = {k: float(feat.get(k, 0.0)) for k in KEYS}
                    backoff = 1
                except Exception as retry_e:
                    log.warning(f"audio_features batch {i} retry failed: {retry_e}")
            else:
                log.warning(f"audio_features batch {i}: {e}")
        except Exception as e:
            log.warning(f"audio_features batch {i}: {e}")
        time.sleep(0.12)

    # Save newly fetched features to DB cache
    if new_features:
        try:
            from . import db
            db.save_audio_features_batch(new_features)
        except Exception as e:
            log.debug(f"Failed to cache audio features: {e}")

    features.update(new_features)
    if sm and state:
        sm.add_log(state, f"Audio features: {len(features)}/{len(track_ids)} "
                          f"({len(cached) if 'cached' in dir() else 0} cached, "
                          f"{len(new_features)} fetched)")
    return features


def check_sources_changed(sp, sources, saved_snapshots):
    """Check if any source playlists changed since last check.

    Uses Spotify snapshot_id for playlists (1 API call each) and
    total count for liked songs. Returns (changed: bool, new_snapshots: dict).
    """
    new_snapshots = {}
    changed = False
    for src in sources:
        src_type = src.get("type", "")
        src_id = src.get("id", "liked")

        if src_type in ("cache_all", "cache_playlist"):
            continue  # no API call needed for cache sources

        saved = saved_snapshots.get(src_id, {})

        try:
            if src_type == "liked":
                # Check total count for liked songs (no snapshot_id available)
                r = sp.current_user_saved_tracks(limit=1)
                total = r.get("total", 0)
                new_snapshots[src_id] = {"total": total}
                if total != saved.get("total"):
                    changed = True
            else:
                # Check playlist snapshot_id
                r = sp.playlist(src_id, fields="snapshot_id")
                snap = r.get("snapshot_id", "")
                new_snapshots[src_id] = {"snapshot_id": snap}
                if snap != saved.get("snapshot_id"):
                    changed = True
        except Exception as e:
            log.warning(f"Snapshot check failed for {src_id}: {e}")
            changed = True  # assume changed on error
            new_snapshots[src_id] = saved

    return changed, new_snapshots


def create_playlist_me(sp, name, description="Auto-split by Vibe Splitter"):
    """Create a playlist via ``POST /me/playlists`` (Feb 2026 endpoint)."""
    try:
        result = sp._post("me/playlists", payload={
            "name": name, "public": False, "description": description
        })
        log.info(f"Created playlist '{name}' — id={result.get('id')}")
        return result
    except SpotifyException as e:
        if e.http_status == 429:
            retry_after = e.headers.get("Retry-After") if e.headers else "unknown"
            log.error(f"create_playlist_me: 429 — Retry-After: {retry_after}")
        raise
    except Exception as e:
        log.error(f"Playlist creation failed: {e}")
        raise