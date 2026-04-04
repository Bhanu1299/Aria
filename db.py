"""
db.py — Aria Phase 3D: Centralised SQLite connection and schema management.

All Aria components that need SQLite import get_connection() from here.
Single database file: ~/.aria/aria.db

Tables owned here:
  applications  — job application history (migrated from tracker.py)
  memory        — persistent key-value store (owned by memory.py)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH: str = str(Path.home() / ".aria" / "aria.db")

_CREATE_APPLICATIONS = """
CREATE TABLE IF NOT EXISTS applications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    company    TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    platform   TEXT    NOT NULL DEFAULT '',
    url        TEXT    NOT NULL DEFAULT '',
    status     TEXT    NOT NULL DEFAULT 'applied',
    applied_at TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_MEMORY = """
CREATE TABLE IF NOT EXISTS memory (
    key        TEXT    PRIMARY KEY,
    value      TEXT    NOT NULL,
    updated_at TEXT    NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT
)
"""


def get_connection() -> sqlite3.Connection:
    """
    Open (creating if needed) ~/.aria/aria.db, ensure all tables exist,
    and return a connection with row_factory set to sqlite3.Row.

    Never raises — callers should handle sqlite3.Error if needed.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_APPLICATIONS)
    conn.execute(_CREATE_MEMORY)
    conn.commit()
    return conn
