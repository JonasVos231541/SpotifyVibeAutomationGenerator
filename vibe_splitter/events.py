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

_subscribers = []
_lock = threading.Lock()

# Event types: progress, log, state_change, error
HEARTBEAT_INTERVAL = 15  # seconds


def subscribe():
    """Register a new client queue. Returns the queue."""
    q = Queue(maxsize=256)
    with _lock:
        _subscribers.append(q)
    log.debug(f"SSE subscriber added (total: {len(_subscribers)})")
    return q


def unsubscribe(q):
    """Remove a client queue."""
    with _lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass
    log.debug(f"SSE subscriber removed (total: {len(_subscribers)})")


def publish(event_type, data):
    """Broadcast an event to all connected clients (non-blocking)."""
    payload = json.dumps(data, default=str)
    msg = f"event: {event_type}\ndata: {payload}\n\n"
    with _lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _subscribers.remove(q)
            except ValueError:
                pass


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
