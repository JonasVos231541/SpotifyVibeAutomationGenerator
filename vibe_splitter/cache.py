"""
Thread-safe tag cache with atomic file writes.

The cache maps ``track_id → {id, uri, name, artist, tags, tag_dict, ...}``
and is persisted as a single JSON file.  Atomic rename prevents corruption.
"""
import json, os, tempfile, threading, logging
from . import config

log = logging.getLogger("splitter.cache")

_lock = threading.RLock()


def load():
    """Read cache from disk.  Returns empty dict if missing or corrupt."""
    with _lock:
        if os.path.exists(config.CACHE_FILE):
            try:
                with open(config.CACHE_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                log.warning("Cache file corrupted — starting fresh")
        return {}


def save(cache):
    """Atomically write the entire cache to disk."""
    with _lock:
        try:
            dir_name = os.path.dirname(config.CACHE_FILE) or "."
            fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(cache, f)
                os.replace(tmp, config.CACHE_FILE)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            log.error(f"Cache save failed: {e}")


def clear():
    """Delete the cache file from disk."""
    with _lock:
        if os.path.exists(config.CACHE_FILE):
            os.remove(config.CACHE_FILE)
            log.info("Cache cleared")


def size_kb():
    """Return on-disk size in KB."""
    try:
        return round(os.path.getsize(config.CACHE_FILE) / 1024, 1) if os.path.exists(config.CACHE_FILE) else 0
    except OSError:
        return 0
