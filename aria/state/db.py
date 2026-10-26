"""
db.py — Aria Phase 3D: Centralised SQLite connection and schema management.

All Aria components that need SQLite import get_connection() from here.
Single database file: ~/.aria/aria.db

Tables owned here:
  applications     — job application history (migrated from tracker.py)
  memory           — persistent key-value store (owned by memory.py)
  learned_answers  — form answers the user gave by voice / in review (features/answer_bank.py)
  application_runs — every apply-pipeline step: fill / submit / skip / error, with cost
  answer_changes   — every edit to the answer bank (dashboard / review), for history + undo
  generated_answers — LLM-written open-text form answers, reused on the same question
  apply_corrections — every apply failure (unknown ATS, unmapped field, wall, timeout), to fix after a batch
  postings_seen    — every job posting Aria read or prepared, to spot reposts (features/jobprep/reposts.py)
  do_not_apply     — companies / domains never applied to (features/jobprep/blocklist.py)
  followups        — follow-up emails due after an application (features/jobprep/followups.py)
  reply_suggestions — employer replies classified, sheet status waiting for the user's Approve
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


_CREATE_ANSWER_CHANGES = """
CREATE TABLE IF NOT EXISTS answer_changes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT    NOT NULL DEFAULT (datetime('now')),
    target     TEXT    NOT NULL,          -- answer | learned | job | always_ask | salary_floor
    key        TEXT    NOT NULL,          -- canonical key, or the normalized question
    job        TEXT    NOT NULL DEFAULT '',
    old        TEXT,                      -- JSON value before (null = unset)
    new        TEXT,                      -- JSON value after
    source     TEXT    NOT NULL DEFAULT 'dashboard',
    undone_by  INTEGER
)
"""

_CREATE_GENERATED_ANSWERS = """
CREATE TABLE IF NOT EXISTS generated_answers (
    question   TEXT    PRIMARY KEY,       -- normalized field label
    answer     TEXT    NOT NULL,
    company    TEXT    NOT NULL DEFAULT '',
    max_chars  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""


_CREATE_APPLY_CORRECTIONS = """
CREATE TABLE IF NOT EXISTS apply_corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_url    TEXT    NOT NULL DEFAULT '',
    company    TEXT    NOT NULL DEFAULT '',
    row        INTEGER,
    ats        TEXT    NOT NULL DEFAULT '',
    step       TEXT    NOT NULL DEFAULT '',   -- resolve | login | fill | wall
    kind       TEXT    NOT NULL DEFAULT '',   -- unknown_ats | unmapped_field | failed_field | login_failed | captcha | timeout | …
    detail     TEXT    NOT NULL DEFAULT '',
    page_url   TEXT    NOT NULL DEFAULT '',
    snapshot   TEXT    NOT NULL DEFAULT '',   -- ~/.aria/corrections/<stem> (.png + .html)
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""


_CREATE_POSTINGS_SEEN = """
CREATE TABLE IF NOT EXISTS postings_seen (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    company    TEXT    NOT NULL DEFAULT '',
    role       TEXT    NOT NULL DEFAULT '',
    url        TEXT    NOT NULL DEFAULT '',
    url_key    TEXT    NOT NULL UNIQUE,      -- host + path, lowercased (no query)
    first_seen TEXT    NOT NULL DEFAULT (datetime('now')),
    last_seen  TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_DO_NOT_APPLY = """
CREATE TABLE IF NOT EXISTS do_not_apply (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    value      TEXT    NOT NULL UNIQUE,      -- normalized company name, or a domain
    label      TEXT    NOT NULL DEFAULT '',  -- as the user said / typed it
    scope      TEXT    NOT NULL DEFAULT 'company',   -- company | domain
    reason     TEXT    NOT NULL DEFAULT '',
    since      TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_FOLLOWUPS = """
CREATE TABLE IF NOT EXISTS followups (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT    NOT NULL,             -- sheet Job ID, else "company|role"
    company    TEXT    NOT NULL DEFAULT '',
    role       TEXT    NOT NULL DEFAULT '',
    n          INTEGER NOT NULL DEFAULT 1,   -- 1st or 2nd follow-up (max 2)
    applied_at TEXT    NOT NULL,
    next_due   TEXT    NOT NULL,
    done_at    TEXT,
    draft_link TEXT    NOT NULL DEFAULT '',
    UNIQUE (job_id, n)
)
"""

_CREATE_REPLY_SUGGESTIONS = """
CREATE TABLE IF NOT EXISTS reply_suggestions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT    NOT NULL DEFAULT '',
    company          TEXT    NOT NULL DEFAULT '',
    role             TEXT    NOT NULL DEFAULT '',
    row              INTEGER,
    thread_id        TEXT    NOT NULL UNIQUE,  -- Gmail message / thread id
    subject          TEXT    NOT NULL DEFAULT '',
    sender           TEXT    NOT NULL DEFAULT '',
    kind             TEXT    NOT NULL,         -- interview | rejected | offer | noise
    by               TEXT    NOT NULL DEFAULT 'rules',   -- rules | llm
    suggested_status TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    approved_at      TEXT,
    rejected_at      TEXT
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
    conn.execute(_CREATE_ANSWER_CHANGES)
    conn.execute(_CREATE_GENERATED_ANSWERS)
    conn.execute(_CREATE_APPLY_CORRECTIONS)
    conn.execute(_CREATE_POSTINGS_SEEN)
    conn.execute(_CREATE_DO_NOT_APPLY)
    conn.execute(_CREATE_FOLLOWUPS)
    conn.execute(_CREATE_REPLY_SUGGESTIONS)
    conn.commit()
    return conn
