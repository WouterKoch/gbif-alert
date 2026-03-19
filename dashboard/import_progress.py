"""Real-time import progress tracking via Redis.

Uses the same Redis connection as django-rq to store progress info
that can be polled from the admin interface during an import.
"""

import json
import time

import django_rq  # type: ignore

REDIS_KEY = "gbif_alert:import_progress"


def set_progress(status: str, message: str) -> None:
    """Update the current import progress in Redis."""
    conn = django_rq.get_connection()
    conn.set(
        REDIS_KEY,
        json.dumps(
            {
                "status": status,
                "message": message,
                "timestamp": time.time(),
            }
        ),
    )


def get_progress() -> dict | None:
    """Read the current import progress from Redis. Returns None if no import info exists."""
    conn = django_rq.get_connection()
    data = conn.get(REDIS_KEY)
    if data:
        return json.loads(data)
    return None


def clear_progress() -> None:
    """Remove the import progress key from Redis."""
    conn = django_rq.get_connection()
    conn.delete(REDIS_KEY)
