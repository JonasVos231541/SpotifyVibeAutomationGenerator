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


def delete_tracks(track_ids):
    """Remove tracks from cache by IDs."""
    if not track_ids:
        return
    conn = _get_conn()
    conn.executemany("DELETE FROM tracks WHERE id=?", [(tid,) for tid in track_ids])
    conn.commit()


def track_count():
    """Return number of cached tracks."""
    conn = _get_conn()
    return conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]


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
    conn = _get_conn()
    placeholders = ",".join("?" * len(track_ids))
    rows = conn.execute(
        f"SELECT id, embedding FROM tracks WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
        track_ids).fetchall()
    return {row["id"]: row["embedding"] for row in rows}
