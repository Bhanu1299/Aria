"""
memory.py — Aria Phase 3D: Persistent session state.

In-memory dict for fast intra-session access.
Every write is also persisted to SQLite (db.py) so state survives restarts.

Session facts (last_jobs): expire after 24 hours.
Long-term facts (last_search_query): never expire.

Public API:
  store_jobs(results)          store job search results (session + SQLite)
  get_job_by_index(n)          1-based job lookup from session
  store_last_search(query)     persist last search query (no expiry)
  get_last_search()            retrieve last search query
  get_persistent(key)          read any persistent value from session
  set_persistent(key, value)   write any persistent value (no expiry)
  store_session_notes(notes)   append turn summary to running session log (no expiry)
  get_session_notes()          retrieve full session log, "" if empty
  clear_session_notes()        delete session log from memory + SQLite
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta

import db

logger = logging.getLogger(__name__)

_SESSION_TTL_HOURS = 24

_lock = threading.Lock()
session: dict = {}


# ---------------------------------------------------------------------------
# Internal persistence helpers
# ---------------------------------------------------------------------------

def _save(key: str, value, expires_hours: int | None = _SESSION_TTL_HOURS) -> None:
    """Persist key → JSON value to SQLite. expires_hours=None means no expiry."""
    expires_at = None
    if expires_hours is not None:
        expires_at = (datetime.utcnow() + timedelta(hours=expires_hours)).isoformat()
    conn = None
    try:
        conn = db.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO memory (key, value, updated_at, expires_at) "
            "VALUES (?, ?, datetime('now'), ?)",
            (key, json.dumps(value), expires_at),
        )
        conn.commit()
    except Exception as exc:
        logger.error("memory._save(%r) failed: %s", key, exc)
    finally:
        if conn is not None:
            conn.close()


def _load_from_db() -> None:
    """Load all non-expired rows from SQLite into the session dict at startup."""
    conn = None
    try:
        conn = db.get_connection()
        now = datetime.utcnow().isoformat()
        rows = conn.execute(
            "SELECT key, value FROM memory WHERE expires_at IS NULL OR expires_at > ?",
            (now,),
        ).fetchall()
        with _lock:
            for row in rows:
                try:
                    session[row["key"]] = json.loads(row["value"])
                except Exception:
                    pass
        logger.info("memory: loaded %d keys from SQLite", len(rows))
    except Exception as exc:
        logger.error("memory._load_from_db failed: %s", exc)
    finally:
        if conn is not None:
            conn.close()


# Load persisted state immediately on import (i.e. on every Aria startup)
_load_from_db()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_jobs(results: list[dict]) -> None:
    """Overwrite last_jobs in session and SQLite (24-hour TTL)."""
    with _lock:
        session["last_jobs"] = results
    _save("last_jobs", results, expires_hours=_SESSION_TTL_HOURS)


def get_job_by_index(n: int) -> dict | None:
    """Return the 1-based nth job from last_jobs, or None if out of range."""
    with _lock:
        jobs = session.get("last_jobs", [])
    return jobs[n - 1] if 0 < n <= len(jobs) else None


def store_last_search(query: str) -> None:
    """Persist the last job search query. Never expires."""
    with _lock:
        session["last_search_query"] = query
    _save("last_search_query", query, expires_hours=None)


def get_last_search() -> str:
    """Return the last stored search query, or empty string if none."""
    with _lock:
        return session.get("last_search_query", "")


def get_persistent(key: str):
    """Read any persistent value from session. Returns None if not set."""
    with _lock:
        return session.get(key)


def set_persistent(key: str, value) -> None:
    """Write any persistent value. Never expires."""
    with _lock:
        session[key] = value
    _save(key, value, expires_hours=None)


# ---------------------------------------------------------------------------
# Job search cache (30-minute TTL, in-memory only)
# ---------------------------------------------------------------------------

_JOB_CACHE_TTL = 1800  # 30 minutes in seconds
# Format: {"query_normalized": {"results": [...], "ts": float}}
_job_cache: dict = {}
_job_cache_lock = threading.Lock()


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def get_cached_jobs(query: str) -> list[dict] | None:
    """
    Return cached job results for query if they are less than 30 minutes old.
    Returns None if no valid cache entry exists.
    """
    key = _normalize_query(query)
    with _job_cache_lock:
        entry = _job_cache.get(key)
    if entry is None:
        return None
    if time.time() - entry["ts"] > _JOB_CACHE_TTL:
        return None
    return entry["results"]


def store_cached_jobs(query: str, results: list[dict]) -> None:
    """Cache job results for query with current timestamp."""
    key = _normalize_query(query)
    with _job_cache_lock:
        _job_cache[key] = {"results": results, "ts": time.time()}


# ---------------------------------------------------------------------------
# Last plan (24-hour TTL — for restart recovery)
# ---------------------------------------------------------------------------


def store_last_plan(plan_dict: dict) -> None:
    """Persist the current plan context dict. 24-hour TTL."""
    with _lock:
        session["last_plan"] = plan_dict
    _save("last_plan", plan_dict, expires_hours=_SESSION_TTL_HOURS)


def get_last_plan() -> dict | None:
    """Return the last stored plan dict, or None if not set / expired."""
    with _lock:
        return session.get("last_plan")


# ---------------------------------------------------------------------------
# Session notes (no expiry — persists across restarts)
# ---------------------------------------------------------------------------

_SESSION_NOTES_KEY = "session_notes"


def store_session_notes(notes: str) -> None:
    """Append bullet-point notes for this turn to the running session log. Never expires."""
    import compact
    with _lock:
        existing = session.get(_SESSION_NOTES_KEY, "")
    combined = (existing + "\n\n" + notes).strip() if existing else notes
    if compact.needs_compaction(combined):
        combined = compact.compress(combined)
    with _lock:
        session[_SESSION_NOTES_KEY] = combined
    _save(_SESSION_NOTES_KEY, combined, expires_hours=None)


def get_session_notes() -> str:
    """Return the last stored session notes, or empty string if none."""
    with _lock:
        return session.get(_SESSION_NOTES_KEY, "")


def clear_session_notes() -> None:
    """Remove session notes from both in-memory session and SQLite."""
    with _lock:
        session.pop(_SESSION_NOTES_KEY, None)
    conn = None
    try:
        conn = db.get_connection()
        conn.execute("DELETE FROM memory WHERE key = ?", (_SESSION_NOTES_KEY,))
        conn.commit()
    except Exception as exc:
        logger.error("memory.clear_session_notes failed: %s", exc)
    finally:
        if conn is not None:
            conn.close()
