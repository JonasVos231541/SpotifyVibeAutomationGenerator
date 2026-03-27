"""
SQLite database layer with WAL mode for concurrent reads.

Provides a drop-in replacement for JSON file-based state and cache storage.
Tables:
  - config    — key/value pairs (replaces splitter_state.json)
  - tracks    — track cache (replaces track_cache.json)
  - logs      — activity log entries
  - playlists — playlist metadata
  - playlist_tracks — M2M relationship

Migration: On first run, imports existing JSON files into SQLite automatically.
"""
import json, os, sqlite3, threading, logging
from datetime import datetime
from . import config as app_config

log = logging.getLogger("splitter.db")

DB_FILE = os.getenv("VS_DB_FILE", "vibe_splitter.db")
DB_BATCH_SIZE = 500  # max IDs per IN clause (SQLite limit is 999)

_local = threading.local()


def _get_conn():
    """Per-thread SQLite connection with WAL mode."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db():
    """Create tables if they don't exist and run migration."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tracks (
            id        TEXT PRIMARY KEY,
            uri       TEXT,
            name      TEXT,
            artist    TEXT,
            artists   TEXT,
            tags_json TEXT,
            tag_dict_json TEXT,
            embedding BLOB
        );
        CREATE TABLE IF NOT EXISTS logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            message   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS playlists (
            key        TEXT PRIMARY KEY,
            name       TEXT,
            spotify_id TEXT,
            health     REAL,
            meta_json  TEXT
        );
        CREATE TABLE IF NOT EXISTS playlist_tracks (
            playlist_key TEXT NOT NULL,
            track_id     TEXT NOT NULL,
            PRIMARY KEY (playlist_key, track_id)
        );
        CREATE INDEX IF NOT EXISTS idx_pt_track ON playlist_tracks(track_id);
        CREATE TABLE IF NOT EXISTS audio_features (
            track_id     TEXT PRIMARY KEY,
            features_json TEXT NOT NULL,
            fetched_at   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS targets (
            spotify_id          TEXT PRIMARY KEY,
            name                TEXT,
            description         TEXT DEFAULT '',
            custom_description  TEXT DEFAULT '',
            desc_embedding      BLOB,
            song_centroid       BLOB,
            song_count          INTEGER DEFAULT 0,
            updated_at          TEXT
        );
    """)
    conn.commit()

    # Auto-migrate from JSON files if DB is empty
    row = conn.execute("SELECT COUNT(*) FROM config").fetchone()
    if row[0] == 0:
        _migrate_from_json(conn)


def _migrate_from_json(conn):
    """Import existing JSON state and cache into SQLite."""
    # Migrate state
    if os.path.exists(app_config.STATE_FILE):
        try:
            with open(app_config.STATE_FILE) as f:
                state = json.load(f)
            for k, v in state.items():
                if k == "logs":
                    for entry in (v or [])[:80]:
                        conn.execute("INSERT INTO logs (timestamp, message) VALUES (?, ?)",
                                     (datetime.now().isoformat(), str(entry)))
                elif k == "playlists":
                    for pk, pv in (v or {}).items():
                        conn.execute(
                            "INSERT OR REPLACE INTO playlists (key, name, spotify_id, health, meta_json) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (pk, pv.get("name"), pv.get("spotify_id"),
                             pv.get("health", 0), json.dumps(pv)))
                        for tid in pv.get("track_ids", []):
                            conn.execute(
                                "INSERT OR IGNORE INTO playlist_tracks (playlist_key, track_id) "
                                "VALUES (?, ?)", (pk, tid))
                else:
                    conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                                 (k, json.dumps(v)))
            conn.commit()
            log.info(f"Migrated state from {app_config.STATE_FILE}")
        except Exception as e:
            log.warning(f"State migration failed: {e}")

    # Migrate cache
    if os.path.exists(app_config.CACHE_FILE):
        try:
            with open(app_config.CACHE_FILE) as f:
                cache = json.load(f)
            for tid, entry in cache.items():
                conn.execute(
                    "INSERT OR REPLACE INTO tracks (id, uri, name, artist, artists, tags_json, tag_dict_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (tid, entry.get("uri"), entry.get("name"), entry.get("artist"),
                     entry.get("artists"), json.dumps(entry.get("tags", [])),
                     json.dumps(entry.get("tag_dict", {}))))
            conn.commit()
            log.info(f"Migrated {len(cache)} tracks from {app_config.CACHE_FILE}")
        except Exception as e:
            log.warning(f"Cache migration failed: {e}")


# ─── Config (state) helpers ──────────────────────────────────────────────────

def get_config(key, default=None):
    """Read a config value."""
    conn = _get_conn()
    row = conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    if row:
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return row[0]
    return default


def set_config(key, value):
    """Write a config value."""
    conn = _get_conn()
    conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                 (key, json.dumps(value)))
    conn.commit()


def get_all_config():
    """Return all config as a dict (matches old state.load() shape)."""
    conn = _get_conn()
    rows = conn.execute("SELECT key, value FROM config").fetchall()
    result = {}
    for row in rows:
        try:
            result[row[0]] = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            result[row[0]] = row[1]
    return result


# ─── Track cache helpers ─────────────────────────────────────────────────────

def get_track(tid):
    """Get a single cached track."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
    if not row:
        return None
    return _row_to_track(row)


def get_all_tracks():
    """Get all cached tracks as {tid: entry} dict."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM tracks").fetchall()
    return {row["id"]: _row_to_track(row) for row in rows}


def upsert_track(entry):
    """Insert or update a track in the cache."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO tracks (id, uri, name, artist, artists, tags_json, tag_dict_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (entry["id"], entry.get("uri"), entry.get("name"), entry.get("artist"),
         entry.get("artists"), json.dumps(entry.get("tags", [])),
         json.dumps(entry.get("tag_dict", {}))))
    conn.commit()


def upsert_tracks_batch(entries):
    """Batch insert/update tracks."""
    conn = _get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO tracks (id, uri, name, artist, artists, tags_json, tag_dict_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(e["id"], e.get("uri"), e.get("name"), e.get("artist"),
          e.get("artists"), json.dumps(e.get("tags", [])),
          json.dumps(e.get("tag_dict", {}))) for e in entries])
    conn.commit()


def get_tracks_batch(track_ids):
    """Get multiple tracks by ID as {tid: entry} dict without loading entire table."""
    if not track_ids:
        return {}
    track_ids = list(track_ids)  # convert once (handles sets, generators)
    conn = _get_conn()
    result = {}
    for i in range(0, len(track_ids), DB_BATCH_SIZE):
        batch = track_ids[i:i+DB_BATCH_SIZE]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT * FROM tracks WHERE id IN ({placeholders})", batch).fetchall()
        for row in rows:
            result[row["id"]] = _row_to_track(row)
    return result


def iter_all_tracks():
    """Yield (tid, entry) pairs one at a time — memory-efficient full scan."""
    conn = _get_conn()
    cursor = conn.execute("SELECT * FROM tracks")
    while True:
        row = cursor.fetchone()
        if row is None:
            break
        yield row["id"], _row_to_track(row)


def track_ids_set():
    """Return set of all cached track IDs (lightweight — no full row data)."""
    conn = _get_conn()
    rows = conn.execute("SELECT id FROM tracks").fetchall()
    return {row["id"] for row in rows}


def delete_tracks(track_ids):
    """Remove tracks from cache by IDs."""
    if not track_ids:
        return
    conn = _get_conn()
    conn.executemany("DELETE FROM tracks WHERE id=?", [(tid,) for tid in track_ids])
    conn.commit()


def clear_tracks():
    """Delete all cached tracks."""
    conn = _get_conn()
    conn.execute("DELETE FROM tracks")
    conn.commit()
    log.info("Cleared all cached tracks from DB")


def prune_tracks(active_ids):
    """Remove tracks not in active_ids. Returns count of removed tracks."""
    if not active_ids:
        return 0
    conn = _get_conn()
    active_ids = set(active_ids)
    all_ids = {row["id"] for row in conn.execute("SELECT id FROM tracks").fetchall()}
    stale = all_ids - active_ids
    if stale:
        conn.executemany("DELETE FROM tracks WHERE id=?", [(tid,) for tid in stale])
        conn.commit()
        log.info(f"Pruned {len(stale)} stale tracks from DB")
    return len(stale)


def track_count():
    """Return number of cached tracks."""
    conn = _get_conn()
    return conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]


def tagged_track_count():
    """Count tracks that have at least one tag."""
    conn = _get_conn()
    return conn.execute(
        "SELECT COUNT(*) FROM tracks WHERE tags_json IS NOT NULL AND tags_json != '[]'"
    ).fetchone()[0]


def audio_features_count():
    """Count tracks with cached audio features."""
    conn = _get_conn()
    return conn.execute("SELECT COUNT(*) FROM audio_features").fetchone()[0]


def _row_to_track(row):
    """Convert a DB row to a track dict matching the old cache format."""
    tags = json.loads(row["tags_json"]) if row["tags_json"] else []
    tag_dict = json.loads(row["tag_dict_json"]) if row["tag_dict_json"] else {}
    return {
        "id": row["id"],
        "uri": row["uri"],
        "name": row["name"],
        "artist": row["artist"],
        "artists": row["artists"],
        "tags": tags,
        "tag_dict": tag_dict,
    }


# ─── Log helpers ─────────────────────────────────────────────────────────────

def add_log(message):
    """Insert a log entry."""
    conn = _get_conn()
    conn.execute("INSERT INTO logs (timestamp, message) VALUES (?, ?)",
                 (datetime.now().isoformat(), message))
    conn.commit()


def get_recent_logs(limit=80):
    """Return recent log entries as formatted strings."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT timestamp, message FROM logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [f"[{row['timestamp'][-8:]}] {row['message']}" for row in rows]


# ─── Embedding storage ──────────────────────────────────────────────────────

def save_embedding(tid, embedding_bytes):
    """Store a precomputed embedding blob for a track."""
    conn = _get_conn()
    conn.execute("UPDATE tracks SET embedding=? WHERE id=?", (embedding_bytes, tid))
    conn.commit()


def get_embedding(tid):
    """Return raw embedding bytes for a track, or None."""
    conn = _get_conn()
    row = conn.execute("SELECT embedding FROM tracks WHERE id=?", (tid,)).fetchone()
    return row["embedding"] if row and row["embedding"] else None


def get_embeddings_batch(track_ids):
    """Return {tid: bytes} for tracks that have cached embeddings."""
    if not track_ids:
        return {}
    track_ids = list(track_ids)
    conn = _get_conn()
    result = {}
    for i in range(0, len(track_ids), DB_BATCH_SIZE):
        batch = track_ids[i:i+DB_BATCH_SIZE]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT id, embedding FROM tracks WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
            batch).fetchall()
        for row in rows:
            result[row["id"]] = row["embedding"]
    return result


def clear_embeddings():
    """Clear all cached embeddings (used when embedding method changes)."""
    conn = _get_conn()
    conn.execute("UPDATE tracks SET embedding = NULL")
    conn.commit()
    log.info("Cleared all cached embeddings")


def check_embedding_version(expected_version=2):
    """Check embedding version, clear cache if mismatched."""
    current = get_config("embedding_version", 0)
    if current != expected_version:
        log.info(f"Embedding version changed ({current} -> {expected_version}), clearing cache")
        clear_embeddings()
        set_config("embedding_version", expected_version)


# ─── Audio feature cache ──────────────────────────────────────────────────────

def get_audio_features_batch(track_ids):
    """Return {tid: {feature_dict}} for tracks with cached audio features."""
    if not track_ids:
        return {}
    conn = _get_conn()
    result = {}
    # Process in chunks to avoid SQLite variable limit
    for i in range(0, len(track_ids), DB_BATCH_SIZE):
        batch = track_ids[i:i+DB_BATCH_SIZE]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT track_id, features_json FROM audio_features WHERE track_id IN ({placeholders})",
            batch).fetchall()
        for row in rows:
            try:
                result[row["track_id"]] = json.loads(row["features_json"])
            except (json.JSONDecodeError, TypeError):
                pass
    return result


def save_audio_features_batch(features_dict):
    """Save {tid: {feature_dict}} to the audio_features cache."""
    if not features_dict:
        return
    conn = _get_conn()
    now = datetime.now().isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO audio_features (track_id, features_json, fetched_at) VALUES (?, ?, ?)",
        [(tid, json.dumps(feats), now) for tid, feats in features_dict.items()])
    conn.commit()
    log.info(f"Cached audio features for {len(features_dict)} tracks")


# ─── Target helpers ────────────────────────────────────────────────────────────

def save_target_embedding(spotify_id, desc_embedding, song_centroid, song_count):
    """Persist a target playlist's embedding data."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO targets
           (spotify_id, desc_embedding, song_centroid, song_count, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            spotify_id,
            desc_embedding.tobytes() if desc_embedding is not None else None,
            song_centroid.tobytes() if song_centroid is not None else None,
            song_count,
            datetime.now().isoformat(),
        )
    )
    conn.commit()


def get_target_embedding(spotify_id):
    """Return (desc_embedding_bytes, song_centroid_bytes, song_count) or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT desc_embedding, song_centroid, song_count FROM targets WHERE spotify_id=?",
        (spotify_id,)
    ).fetchone()
    if not row:
        return None
    return row["desc_embedding"], row["song_centroid"], row["song_count"]


def delete_target(spotify_id):
    """Remove a target's embedding row."""
    conn = _get_conn()
    conn.execute("DELETE FROM targets WHERE spotify_id=?", (spotify_id,))
    conn.commit()


def upsert_target_meta(spotify_id, name, description, custom_description):
    """Insert or update target metadata (name, descriptions) without touching embeddings."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO targets (spotify_id, name, description, custom_description)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(spotify_id) DO UPDATE SET
             name=excluded.name,
             description=excluded.description,
             custom_description=excluded.custom_description""",
        (spotify_id, name, description, custom_description)
    )
    conn.commit()
