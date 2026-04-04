"""
tracker.py — Phase 3B: Job application persistence for Aria.

Logs every submitted application to SQLite via db.py (aria.db).
Previously used applications.db — now consolidated to aria.db.

Public API:
  log_application(company, role, platform, url, status)
  get_applications() → list[dict]
"""

from __future__ import annotations

import logging

import db

logger = logging.getLogger(__name__)


def log_application(
    company: str,
    role: str,
    platform: str,
    url: str,
    status: str = "applied",
) -> None:
    """Insert one application row. Commits immediately."""
    conn = None
    try:
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO applications (company, role, platform, url, status) VALUES (?,?,?,?,?)",
            (company, role, platform, url, status),
        )
        conn.commit()
        logger.info("Logged application: %s @ %s (%s)", role, company, status)
    except Exception as exc:
        logger.error("log_application failed: %s", exc)
    finally:
        if conn is not None:
            conn.close()


def get_applications() -> list[dict]:
    """Return all rows as a list of dicts, newest first."""
    conn = None
    try:
        conn = db.get_connection()
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY applied_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.error("get_applications failed: %s", exc)
        return []
    finally:
        if conn is not None:
            conn.close()
