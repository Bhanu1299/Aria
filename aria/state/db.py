"""
db.py — Aria Phase 3D: Centralised SQLite connection and schema management.

All Aria components that need SQLite import get_connection() from here.
Single database file: ~/.aria/aria.db

Tables owned here:
  applications     — job application history (migrated from tracker.py)
  memory           — persistent key-value store (owned by memory.py)
  learned_answers  — form answers the user gave by voice / in review (features/answer_bank.py)
  application_runs — every apply-pipeline step: fill / submit / skip / error, with cost
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

_CREATE_FACTS = """
CREATE TABLE IF NOT EXISTS facts (
    id           TEXT    PRIMARY KEY,
    text         TEXT    NOT NULL,
    source       TEXT    NOT NULL DEFAULT 'conversation',
    session_ids  TEXT    NOT NULL DEFAULT '[]',
    timestamp    REAL    NOT NULL,
    recall_count INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_LEARNED_ANSWERS = """
CREATE TABLE IF NOT EXISTS learned_answers (
    question   TEXT    PRIMARY KEY,
    answer     TEXT    NOT NULL,
    source     TEXT    NOT NULL DEFAULT 'user',
    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_APPLICATION_RUNS = """
CREATE TABLE IF NOT EXISTS application_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sheet_row   INTEGER,
    company     TEXT    NOT NULL DEFAULT '',
    role        TEXT    NOT NULL DEFAULT '',
    url         TEXT    NOT NULL DEFAULT '',
    ats         TEXT    NOT NULL DEFAULT '',
    stage       TEXT    NOT NULL,
    needs_me    INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL    NOT NULL DEFAULT 0,
    review_path TEXT    NOT NULL DEFAULT '',
    note        TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
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
    conn.execute(_CREATE_FACTS)
    conn.execute(_CREATE_LEARNED_ANSWERS)
    conn.execute(_CREATE_APPLICATION_RUNS)
    conn.commit()
    return conn
