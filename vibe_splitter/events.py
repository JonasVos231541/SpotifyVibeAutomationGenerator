"""
Thread-safe Server-Sent Events (SSE) event bus.

Supports multiple concurrent browser clients.  Each client gets its own
``queue.Queue`` so slow clients don't block others.

Usage::

    from .events import publish, stream_events

    # Publish from anywhere (thread-safe)
    publish("progress", {"step": "embedding", "pct": 45})
    publish("log", {"message": "Fetching tags..."})

    # Flask route
    @app.route("/api/events")
    def sse():
        return Response(stream_events(), mimetype="text/event-stream")
"""
import json, threading, time, logging
from queue import Queue, Empty

log = logging.getLogger("splitter.events")

_subscribers = []  # list of (queue, created_at) tuples
_lock = threading.Lock()
_MAX_SUBSCRIBER_AGE = 3600  # 1 hour max per SSE connection

# Event types: progress, log, state_change, error
HEARTBEAT_INTERVAL = 15  # seconds


def subscribe():
    """Register a new client queue. Returns the queue."""
    q = Queue(maxsize=256)
    with _lock:
        _subscribers.append((q, time.time()))
    log.debug(f"SSE subscriber added (total: {len(_subscribers)})")
    return q


def unsubscribe(q):
    """Remove a client queue."""
    with _lock:
        _subscribers[:] = [(sq, t) for sq, t in _subscribers if sq is not q]
    log.debug(f"SSE subscriber removed (total: {len(_subscribers)})")


def publish(event_type, data):
    """Broadcast an event to all connected clients (non-blocking)."""
    payload = json.dumps(data, default=str)
    msg = f"event: {event_type}\ndata: {payload}\n\n"
    now = time.time()
    with _lock:
        alive = []
        for q, created_at in _subscribers:
            # Evict stale subscribers (older than max age)
            if now - created_at > _MAX_SUBSCRIBER_AGE:
                continue
            try:
                q.put_nowait(msg)
                alive.append((q, created_at))
            except Exception:
                pass  # drop dead/full queues
        _subscribers[:] = alive


def stream_events():
    """Generator that yields SSE messages for one client."""
    q = subscribe()
    try:
        while True:
            try:
                msg = q.get(timeout=HEARTBEAT_INTERVAL)
                yield msg
            except Empty:
                # Heartbeat to keep connection alive
                yield ": heartbeat\n\n"
    except GeneratorExit:
        pass
    finally:
        unsubscribe(q)
