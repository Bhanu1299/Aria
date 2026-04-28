# Phase 3D Session 1 — Persistent Memory + Gap Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite-backed persistent memory with thread safety, fix identity injection, add capability intent, add salary job filter, add last-search persistence, and add two seed skills (apply_status, calculate).

**Architecture:** All SQLite access centralised through `db.py`. `memory.py` gains a `threading.Lock` and writes every mutation to the `memory` table in `~/.aria/aria.db`. `tracker.py` migrates to the same DB file. Two new skills live under `skills/` and are auto-discovered by `skills/skill_loader.py`, which is checked in `router.py` before the Groq classifier. A regex pre-check in `router.py` routes capability questions to a hardcoded handler, bypassing Groq entirely.

**Tech Stack:** Python 3.11+, SQLite (stdlib), threading.Lock (stdlib), ast (stdlib for safe math eval), Groq (existing), Playwright (existing).

---

## Pre-flight Notes

**Fixes C and D are already implemented** — do not re-implement them:
- Resume upload: `linkedin_applicator._upload_resume()` exists at line 524.
- Debug screenshots: `dom_browser.save_debug_screenshot()` is called throughout `linkedin_applicator.py`.

**summarizer.py** already injects the user's name into `_KNOWLEDGE_SYSTEM_PROMPT` (line 47-49) but the identity context is name-only and the capability list (lines 61-63) is outdated. Both need updating — not rewriting from scratch.

**Phase naming:** New files get "Phase 3D" in their docstrings. Existing files keep their "Phase 3B/3C" labels — those are accurate historical labels. `project-state.md` gains a Phase 3D section.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `db.py` | All SQLite table schemas + `get_connection()` |
| Modify | `tracker.py` | Use `db.get_connection()` instead of own connection |
| Rewrite | `memory.py` | Thread lock + SQLite persistence + load on startup |
| Modify | `jobs.py` | Add salary filter to `_parse_filters()` |
| Modify | `main.py` | Call `memory.store_last_search()` after job search; add capability handler |
| Modify | `summarizer.py` | Expand identity context; update capability list |
| Modify | `router.py` | Add capability pre-check; call `skill_loader.match_skill()` before Groq |
| Create | `skills/__init__.py` | Empty package marker |
| Create | `skills/skill_loader.py` | Auto-discover skills; expose `match_skill()` and `load_skills()` |
| Create | `skills/apply_status/__init__.py` | TRIGGERS + `handle()` — reads from tracker.py |
| Create | `skills/calculate/__init__.py` | TRIGGERS + `handle()` — safe AST math eval |
| Modify | `.claude/project-state.md` | Phase 3D section |
| Create/Modify | `.claude/session-log.md` | Session record |
| Create | `tests/test_db.py` | DB schema creation |
| Create | `tests/test_memory.py` | Persistence, thread lock, expiry |
| Create | `tests/test_jobs_filters.py` | Salary filter parsing |
| Create | `tests/test_router_capability.py` | Capability pre-check routing |
| Create | `tests/test_skill_loader.py` | Skill discovery and matching |
| Create | `tests/test_skills.py` | Both seed skills |

---

## Task 1: Create db.py — centralised SQLite module

**Files:**
- Create: `db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
import os
import sqlite3
import tempfile
import pytest

# Patch DB_PATH before importing db so tests don't touch ~/.aria/aria.db
@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    db_file = str(tmp_path / "test_aria.db")
    import db
    monkeypatch.setattr(db, "DB_PATH", db_file)
    yield db_file

def test_get_connection_creates_file():
    import db
    conn = db.get_connection()
    conn.close()
    assert os.path.exists(db.DB_PATH)

def test_applications_table_exists():
    import db
    conn = db.get_connection()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "applications" in tables

def test_memory_table_exists():
    import db
    conn = db.get_connection()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "memory" in tables

def test_get_connection_idempotent():
    import db
    conn1 = db.get_connection()
    conn1.close()
    conn2 = db.get_connection()
    conn2.close()
    # Should not raise on repeated calls
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria
source venv/bin/activate
pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Create db.py**

```python
# /Users/bhanuteja/Documents/trae_projects/Aria/db.py
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
```

- [ ] **Step 4: Run tests to confirm passing**

```bash
pytest tests/test_db.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat(3D): add db.py — centralised SQLite schema management"
```

---

## Task 2: Migrate tracker.py to use db.py

**Files:**
- Modify: `tracker.py`

The only changes are: import `db`, replace `DB_PATH` with `db.DB_PATH`, replace `_get_connection()` with `db.get_connection()`, and remove the inline `_CREATE_TABLE` and `_get_connection` definitions.

- [ ] **Step 1: Read the current tracker.py tests (if any)**

There are no tracker-specific test files currently. The existing API is `log_application()` and `get_applications()` — these must continue to work identically.

- [ ] **Step 2: Rewrite tracker.py**

```python
# /Users/bhanuteja/Documents/trae_projects/Aria/tracker.py
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
    try:
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO applications (company, role, platform, url, status) VALUES (?,?,?,?,?)",
            (company, role, platform, url, status),
        )
        conn.commit()
        conn.close()
        logger.info("Logged application: %s @ %s (%s)", role, company, status)
    except Exception as exc:
        logger.error("log_application failed: %s", exc)


def get_applications() -> list[dict]:
    """Return all rows as a list of dicts, newest first."""
    try:
        conn = db.get_connection()
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY applied_at DESC"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.error("get_applications failed: %s", exc)
        return []
```

- [ ] **Step 3: Smoke test**

```bash
python -c "
import tracker
tracker.log_application('Acme', 'SWE', 'LinkedIn', 'https://example.com')
apps = tracker.get_applications()
assert len(apps) >= 1
assert apps[0]['company'] == 'Acme'
print('tracker OK')
"
```

Expected: `tracker OK`

- [ ] **Step 4: Commit**

```bash
git add tracker.py
git commit -m "feat(3D): migrate tracker.py to use centralised db.py"
```

---

## Task 3: Rewrite memory.py — thread lock + SQLite persistence

**Files:**
- Rewrite: `memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory.py
"""
Tests for memory.py persistence and thread safety.
Uses a temp DB so tests never touch ~/.aria/aria.db.
"""
import json
import threading
import time
import importlib
import pytest


@pytest.fixture(autouse=True)
def isolated_memory(monkeypatch, tmp_path):
    """Patch db.DB_PATH and reload memory module fresh for each test."""
    import db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test_aria.db"))
    # Force memory to reload so _load_from_db() runs against the temp DB
    import memory
    importlib.reload(memory)
    yield
    importlib.reload(memory)  # clean up after test


def test_store_and_retrieve_jobs():
    import memory
    jobs = [{"title": "SWE", "company": "Acme", "url": "https://example.com"}]
    memory.store_jobs(jobs)
    assert memory.get_job_by_index(1)["title"] == "SWE"
    assert memory.get_job_by_index(2) is None


def test_store_last_search_and_get():
    import memory
    memory.store_last_search("SWE jobs in New York")
    assert memory.get_last_search() == "SWE jobs in New York"


def test_persistent_memory_survives_reload(tmp_path, monkeypatch):
    """Simulate a restart: store, reload module, confirm value is still there."""
    import db
    db_file = str(tmp_path / "persist_test.db")
    monkeypatch.setattr(db, "DB_PATH", db_file)

    import memory
    importlib.reload(memory)
    memory.store_last_search("backend engineer remote")

    # Simulate restart by reloading the module
    importlib.reload(memory)
    assert memory.get_last_search() == "backend engineer remote"


def test_jobs_expire_after_24h(tmp_path, monkeypatch):
    """Jobs written with past expiry should not load on restart."""
    import db
    db_file = str(tmp_path / "expire_test.db")
    monkeypatch.setattr(db, "DB_PATH", db_file)

    import memory
    importlib.reload(memory)

    # Manually write an expired job row
    conn = db.get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO memory (key, value, updated_at, expires_at) VALUES (?,?,datetime('now'),?)",
        ("last_jobs", json.dumps([{"title": "Old Job"}]), "2020-01-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    # Reload — expired row should not load
    importlib.reload(memory)
    assert memory.get_job_by_index(1) is None


def test_thread_safety():
    """Multiple threads writing simultaneously must not corrupt the session dict."""
    import memory
    errors = []

    def writer(i):
        try:
            memory.store_last_search(f"query {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    # Final state should be a valid query string (some thread won the race)
    result = memory.get_last_search()
    assert result.startswith("query ")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_memory.py -v
```

Expected: failures on `store_last_search`, `get_last_search`, persistence tests.

- [ ] **Step 3: Rewrite memory.py**

```python
# /Users/bhanuteja/Documents/trae_projects/Aria/memory.py
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
"""

from __future__ import annotations

import json
import logging
import threading
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
    try:
        conn = db.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO memory (key, value, updated_at, expires_at) "
            "VALUES (?, ?, datetime('now'), ?)",
            (key, json.dumps(value), expires_at),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.error("memory._save(%r) failed: %s", key, exc)


def _load_from_db() -> None:
    """Load all non-expired rows from SQLite into the session dict at startup."""
    try:
        conn = db.get_connection()
        now = datetime.utcnow().isoformat()
        rows = conn.execute(
            "SELECT key, value FROM memory WHERE expires_at IS NULL OR expires_at > ?",
            (now,),
        ).fetchall()
        conn.close()
        with _lock:
            for row in rows:
                try:
                    session[row["key"]] = json.loads(row["value"])
                except Exception:
                    pass
        logger.info("memory: loaded %d keys from SQLite", len(rows))
    except Exception as exc:
        logger.error("memory._load_from_db failed: %s", exc)


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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_memory.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add memory.py tests/test_memory.py
git commit -m "feat(3D): rewrite memory.py with thread lock and SQLite persistence"
```

---

## Task 4: Add salary filter to jobs.py + store last search in main.py

**Files:**
- Modify: `jobs.py`
- Modify: `main.py`
- Create: `tests/test_jobs_filters.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jobs_filters.py
"""Tests for _parse_salary_filter() and _parse_filters() salary branch."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from jobs import _parse_salary_filter, _parse_filters


def test_salary_100k():
    assert _parse_salary_filter("find SWE jobs paying 100k") == "4"

def test_salary_120k():
    assert _parse_salary_filter("find jobs 120k or more") == "5"

def test_salary_140k():
    assert _parse_salary_filter("roles paying 140k") == "6"

def test_salary_80k():
    assert _parse_salary_filter("jobs paying around 80k") == "3"

def test_salary_six_figures():
    assert _parse_salary_filter("six figure jobs") == "4"

def test_salary_no_mention():
    assert _parse_salary_filter("find me remote SWE jobs") == ""

def test_parse_filters_includes_salary():
    filters = _parse_filters("find remote SWE jobs paying 120k")
    assert filters.get("f_WT") == "2"    # remote
    assert filters.get("f_SB2") == "5"   # 120k+

def test_parse_filters_no_salary():
    filters = _parse_filters("find remote SWE jobs")
    assert "f_SB2" not in filters
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_jobs_filters.py -v
```

Expected: `ImportError` or failures on `_parse_salary_filter` (function doesn't exist yet).

- [ ] **Step 3: Add `_parse_salary_filter()` to jobs.py and call it from `_parse_filters()`**

Find the `_parse_filters()` function in `jobs.py` (starts at line 46). Add the new helper above it, then call it inside `_parse_filters()`.

Add this function directly above `_parse_filters()`:

```python
def _parse_salary_filter(query: str) -> str:
    """
    Extract LinkedIn salary filter value (f_SB2) from a voice query.

    LinkedIn salary tiers (f_SB2):
      1=$40k+  2=$60k+  3=$80k+  4=$100k+  5=$120k+  6=$140k+

    Returns the string filter value, or '' if no salary mentioned.
    """
    q = query.lower()

    # "six figures" / "six-figure" → $100k+
    if "six figure" in q:
        return "4"

    # Pattern: "100k", "120,000", "$140k", "80 thousand"
    m = re.search(r'\$?(\d{2,3})[\s,]?(?:k|,000|thousand)', q)
    if m:
        amount = int(m.group(1))
        if amount >= 140:
            return "6"
        if amount >= 120:
            return "5"
        if amount >= 100:
            return "4"
        if amount >= 80:
            return "3"
        if amount >= 60:
            return "2"
        if amount >= 40:
            return "1"

    return ""
```

Then add the salary filter call inside the existing `_parse_filters()` function, after the existing date-filter block:

```python
def _parse_filters(query: str) -> dict:
    """Extract job filter params from voice query for LinkedIn URL."""
    filters = {}
    q = query.lower()
    if "remote" in q:
        filters["f_WT"] = "2"
    elif "hybrid" in q:
        filters["f_WT"] = "3"
    elif "on-site" in q or "onsite" in q or "in person" in q or "in office" in q:
        filters["f_WT"] = "1"
    if "today" in q or "past day" in q or "last 24" in q:
        filters["f_TPR"] = "r86400"
    elif "this week" in q or "past week" in q or "last week" in q:
        filters["f_TPR"] = "r604800"
    elif "this month" in q or "past month" in q:
        filters["f_TPR"] = "r2592000"
    salary_tier = _parse_salary_filter(query)
    if salary_tier:
        filters["f_SB2"] = salary_tier
    return filters
```

- [ ] **Step 4: Add store_last_search call in main.py**

In `main.py`, find the "jobs" intent handler block inside `_handle_intent()` (around line 347-353):

```python
    # --- Jobs: LinkedIn + Indeed search via Google → vision pipeline ---
    if intent_type == "jobs":
        print(f"[Aria] Job search: {intent['query']!r}")
        speaker.say("Searching for jobs, one moment.")
        results = jobs.search_jobs(intent["query"])
        memory.store_jobs(results)
        memory.store_last_search(intent["query"])   # ADD THIS LINE
        return jobs.format_spoken_results(results)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_jobs_filters.py -v
```

Expected: 8 PASSED

- [ ] **Step 6: Commit**

```bash
git add jobs.py main.py tests/test_jobs_filters.py
git commit -m "feat(3D): add salary filter to jobs.py; persist last search query via memory.py"
```

---

## Task 5: Expand identity injection in summarizer.py

**Files:**
- Modify: `summarizer.py`

The current `_IDENTITY_CONTEXT` only injects the user's name. Expand it to include email, location, and a skills summary. Also update the stale capability list in `_KNOWLEDGE_SYSTEM_PROMPT` to reflect all current Aria capabilities.

- [ ] **Step 1: Write a manual smoke test first (no pytest for LLM calls)**

After the change, run:

```bash
python -c "
from summarizer import answer_knowledge
print(answer_knowledge(\"what's my name\"))
print(answer_knowledge(\"where am i based\"))
"
```

Before the change this returns vague answers. After, it should use the identity values.

- [ ] **Step 2: Update summarizer.py**

Replace the `_load_identity`, `_IDENTITY`, `_IDENTITY_CONTEXT`, and `_KNOWLEDGE_SYSTEM_PROMPT` block (lines 37-64) with:

```python
def _load_identity() -> dict:
    path = os.path.join(os.path.dirname(__file__), "identity.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


_IDENTITY = _load_identity()


def _build_identity_context(identity: dict) -> str:
    """Build a compact natural-language identity description from identity.json."""
    parts = []
    if identity.get("name"):
        parts.append(f"The user's name is {identity['name']}.")
    if identity.get("email"):
        parts.append(f"Their email is {identity['email']}.")
    if identity.get("location"):
        parts.append(f"They are based in {identity['location']}.")
    if identity.get("linkedin"):
        parts.append(f"LinkedIn: {identity['linkedin']}.")
    if identity.get("github"):
        parts.append(f"GitHub: {identity['github']}.")
    skills = identity.get("skills", [])
    if skills:
        top = skills[:6]
        parts.append(f"Their top skills include: {', '.join(top)}.")
    education = identity.get("education", "")
    if education:
        parts.append(f"Education: {education}.")
    return " ".join(parts)


_IDENTITY_CONTEXT = _build_identity_context(_IDENTITY)

_KNOWLEDGE_SYSTEM_PROMPT = (
    "You are Aria, a voice assistant on Mac. "
    + (_IDENTITY_CONTEXT + " " if _IDENTITY_CONTEXT else "")
    + "Answer concisely in 2 to 4 sentences maximum. "
    "No markdown, no bullet points, no lists — plain spoken sentences only. "
    "The response will be read aloud. "
    "If the question asks about personal details you don't have "
    "(item locations, the user's schedule, personal files, physical surroundings), "
    "say you don't have that information. "
    "Use the user information above to answer personal questions like "
    "'what's my name', 'what's my email', 'where am I based', 'what are my skills'. "
    "If asked what you can do or what your capabilities are, list only these: "
    "answer general knowledge questions, search the web, open apps and websites, "
    "control media playback (Apple Music and YouTube), control Mac settings via voice, "
    "give a morning briefing (weather, calendar, email, news), "
    "search for jobs on LinkedIn and Indeed, help apply to jobs, "
    "track submitted applications, check system info (battery, wifi, disk), "
    "take notes, set reminders, and run calculations. "
    "Do not claim capabilities beyond this list."
)
```

- [ ] **Step 3: Run smoke test**

```bash
python -c "
from summarizer import answer_knowledge, _IDENTITY
print('Identity loaded:', bool(_IDENTITY))
print('Name:', _IDENTITY.get('name', 'MISSING'))
ans = answer_knowledge(\"what's my name\")
print('Answer:', ans)
assert _IDENTITY.get('name', '').lower() in ans.lower() or 'don\\'t have' in ans.lower()
print('PASS')
"
```

Expected: The answer includes the name from identity.json.

- [ ] **Step 4: Commit**

```bash
git add summarizer.py
git commit -m "feat(3D): expand identity injection in summarizer.py; update capability list"
```

---

## Task 6: Add capability pre-check to router.py + handler in main.py

**Files:**
- Modify: `router.py`
- Modify: `main.py`
- Create: `tests/test_router_capability.py`

Capability questions must NEVER reach Groq — they get intercepted by a regex pre-check and routed to a hardcoded response. This prevents Groq from hallucinating features.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router_capability.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from unittest.mock import patch
from router import route


def _route_no_groq(command):
    """Route without making a real Groq API call."""
    with patch("router._classify") as mock_classify:
        mock_classify.side_effect = AssertionError("Groq should not be called")
        return route(command)


def test_what_can_you_do():
    result = _route_no_groq("what can you do")
    assert result["type"] == "capability"

def test_capabilities():
    result = _route_no_groq("what are your capabilities")
    assert result["type"] == "capability"

def test_what_are_you_capable_of():
    result = _route_no_groq("what are you capable of")
    assert result["type"] == "capability"

def test_aria_features():
    result = _route_no_groq("what features do you have")
    assert result["type"] == "capability"

def test_help_me():
    result = _route_no_groq("help me")
    assert result["type"] == "capability"

def test_non_capability_still_goes_to_groq():
    """A normal question must NOT be intercepted by the capability pre-check."""
    result = route("what is the capital of France")
    # Will call Groq — just confirm it didn't return capability
    assert result["type"] != "capability"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_router_capability.py -v
```

Expected: first 5 tests fail (capability type doesn't exist).

- [ ] **Step 3: Add capability pre-check to router.py**

Add the regex constant after the `_APPLY_INTENT_RE` block (around line 90):

```python
# Capability intent pre-check — routes capability questions without an LLM call.
# Covers: "what can you do", "what are your capabilities", "help me", etc.
_CAPABILITY_RE = re.compile(
    r"\b(?:"
    r"what\s+can\s+you\s+do"
    r"|what\s+are\s+(?:your\s+)?capabilities"
    r"|what\s+are\s+you\s+capable\s+of"
    r"|what\s+(?:features?|skills?|abilities?)\s+do\s+you\s+have"
    r"|help\s+me$"
    r"|what\s+do\s+you\s+(?:do|know)"
    r")\b",
    re.IGNORECASE,
)
```

Add the capability pre-check inside `route()`, after the apply pre-check (around line 293):

```python
    # Capability pre-check — never needs an LLM call
    if _CAPABILITY_RE.search(command):
        logger.info("Capability pre-check matched: %r", command)
        return {
            "type": "capability",
            "query": command.strip(),
            "url": "",
            "instructions": "",
            "app_name": "",
            "contact": "",
            "site_name": "",
        }
```

Also add `"capability"` to the valid types list in `_classify()` (around line 345):

```python
    if "type" not in parsed or parsed["type"] not in (
        "knowledge", "web_search", "web_direct", "app", "media", "navigate",
        "app_control", "briefing", "jobs", "apply", "capability",
    ):
```

And add the capability passthrough to `_build_intent()` (before the final fallback):

```python
    if intent_type == "capability":
        return {**_base}
```

- [ ] **Step 4: Add capability handler to main.py**

Add this function after `_vision_fallback()` (around line 247):

```python
def _get_capability_response() -> str:
    return (
        "I can answer general knowledge questions, search the web, open websites and apps, "
        "control media playback on Apple Music and YouTube, adjust Mac system settings by voice, "
        "give you a morning briefing with weather, calendar, email, and news, "
        "search for jobs on LinkedIn and Indeed, help you apply to jobs and track your applications, "
        "check your system info like battery and wifi, take notes, set reminders, "
        "and run calculations. Just tell me what you need."
    )
```

Add the handler inside `_handle_intent()`, after the `briefing` block:

```python
    # --- Capability: hardcoded response, no LLM call ---
    if intent_type == "capability":
        return _get_capability_response()
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_router_capability.py -v
```

Expected: 5/6 PASSED (the last test `test_non_capability_still_goes_to_groq` requires a real Groq key — skip it if running offline: `pytest tests/test_router_capability.py -v -k "not non_capability"`)

- [ ] **Step 6: Commit**

```bash
git add router.py main.py tests/test_router_capability.py
git commit -m "feat(3D): add capability pre-check to router.py; hardcoded handler in main.py"
```

---

## Task 7: Create skills infrastructure

**Files:**
- Create: `skills/__init__.py`
- Create: `skills/skill_loader.py`
- Create: `tests/test_skill_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skill_loader.py
import sys, os, importlib, types
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_match_skill_returns_callable():
    """After loading, matching a trigger returns the skill's handle function."""
    from skills import skill_loader
    importlib.reload(skill_loader)
    skill_loader.load_skills()
    fn = skill_loader.match_skill("what jobs have I applied to")
    assert callable(fn)


def test_match_skill_returns_none_for_unknown():
    from skills import skill_loader
    importlib.reload(skill_loader)
    skill_loader.load_skills()
    assert skill_loader.match_skill("open YouTube") is None


def test_match_skill_case_insensitive():
    from skills import skill_loader
    importlib.reload(skill_loader)
    skill_loader.load_skills()
    fn = skill_loader.match_skill("CALCULATE 5 times 8")
    assert callable(fn)


def test_load_skills_handles_broken_skill(tmp_path, monkeypatch):
    """A broken skill module must not prevent other skills from loading."""
    from skills import skill_loader
    importlib.reload(skill_loader)

    # Patch skills dir to a tmp dir with one bad skill and one good skill
    bad_dir = tmp_path / "broken_skill"
    bad_dir.mkdir()
    (bad_dir / "__init__.py").write_text("raise ImportError('intentional')\n")

    good_dir = tmp_path / "good_skill"
    good_dir.mkdir()
    (good_dir / "__init__.py").write_text(
        "TRIGGERS = ['good trigger']\ndef handle(c): return 'ok'\n"
    )

    monkeypatch.setattr(skill_loader, "_SKILLS_DIR", tmp_path)
    skill_loader.load_skills()

    assert skill_loader.match_skill("good trigger") is not None
    assert skill_loader.match_skill("broken") is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_skill_loader.py -v
```

Expected: ImportError (module doesn't exist yet).

- [ ] **Step 3: Create skills/__init__.py**

```python
# /Users/bhanuteja/Documents/trae_projects/Aria/skills/__init__.py
```

(Empty — marks the directory as a Python package.)

- [ ] **Step 4: Create skills/skill_loader.py**

```python
# /Users/bhanuteja/Documents/trae_projects/Aria/skills/skill_loader.py
"""
skills/skill_loader.py — Aria Phase 3D: Skill auto-discovery and matching.

Scans the skills/ directory on startup. Each skill folder must contain:
  __init__.py  with TRIGGERS: list[str] and handle(command: str) -> str

Skills are matched before the Groq classifier in router.py — matched skills
execute with zero API latency.

Public API:
  load_skills()                          call once at startup
  match_skill(transcript: str)           returns handle fn or None
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_SKILLS_DIR: Path = Path(__file__).parent
_registry: dict[str, Callable] = {}


def load_skills() -> None:
    """
    Auto-discover and register all skills in the skills/ directory.
    Skips __pycache__ and any directory without an __init__.py.
    A skill that fails to import is logged and skipped — it does not
    prevent other skills from loading.
    """
    global _registry
    _registry = {}

    for skill_dir in _SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        if skill_dir.name.startswith("_"):
            continue
        init_file = skill_dir / "__init__.py"
        if not init_file.exists():
            continue

        module_name = f"skills.{skill_dir.name}"
        try:
            module = importlib.import_module(module_name)
            triggers: list[str] = getattr(module, "TRIGGERS", [])
            handle_fn: Optional[Callable] = getattr(module, "handle", None)

            if not callable(handle_fn):
                logger.warning("Skill %s has no callable handle() — skipping", skill_dir.name)
                continue
            if not triggers:
                logger.warning("Skill %s has no TRIGGERS — skipping", skill_dir.name)
                continue

            for trigger in triggers:
                _registry[trigger.lower()] = handle_fn

            logger.info(
                "Loaded skill: %s  triggers: %s",
                skill_dir.name, triggers,
            )
        except Exception as exc:
            logger.error("Failed to load skill %s: %s", skill_dir.name, exc)


def match_skill(transcript: str) -> Optional[Callable]:
    """
    Return the handle function of the first skill whose trigger phrase
    appears anywhere in the transcript (case-insensitive). Returns None
    if no skill matches.

    Called in router.py BEFORE the Groq classifier so matched commands
    never incur an API call.
    """
    lower = transcript.lower()
    for trigger, fn in _registry.items():
        if trigger in lower:
            return fn
    return None
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_skill_loader.py -v
```

Note: `test_match_skill_returns_callable` and `test_match_skill_case_insensitive` require the seed skills to exist (Tasks 8 and 9). Run the remaining two now:

```bash
pytest tests/test_skill_loader.py -v -k "none_for_unknown or broken_skill"
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add skills/__init__.py skills/skill_loader.py tests/test_skill_loader.py
git commit -m "feat(3D): add skills/__init__.py and skills/skill_loader.py"
```

---

## Task 8: Create skills/apply_status skill

**Files:**
- Create: `skills/apply_status/__init__.py`

- [ ] **Step 1: Create skills/apply_status/__init__.py**

```python
# /Users/bhanuteja/Documents/trae_projects/Aria/skills/apply_status/__init__.py
"""
skills/apply_status — Aria Phase 3D: Read application history from tracker.py.

Handles: "what jobs have I applied to", "application history", "show my applications"
"""

from __future__ import annotations

TRIGGERS = [
    "what jobs have i applied to",
    "application history",
    "show my applications",
    "jobs i've applied to",
    "jobs ive applied to",
    "applied to any",
    "what have i applied for",
]


def handle(command: str) -> str:
    """Return a spoken summary of the most recent job applications."""
    import tracker

    apps = tracker.get_applications()
    if not apps:
        return "You haven't applied to any jobs through Aria yet."

    recent = apps[:5]
    total = len(apps)
    parts = []
    for app in recent:
        date = app.get("applied_at", "")[:10]  # "2026-04-04"
        parts.append(
            f"{app['role']} at {app['company']}"
            + (f" on {date}" if date else "")
        )

    summary = f"You've applied to {total} job{'s' if total != 1 else ''} through Aria. "
    summary += "Most recent: " + "; ".join(parts) + "."
    return summary
```

- [ ] **Step 2: Write tests for this skill**

Add to `tests/test_skills.py` (create it if it doesn't exist):

```python
# tests/test_skills.py
"""Tests for seed skills — apply_status and calculate."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestApplyStatus:
    def test_no_applications(self, monkeypatch):
        import skills.apply_status as skill
        import tracker
        monkeypatch.setattr(tracker, "get_applications", lambda: [])
        result = skill.handle("what jobs have I applied to")
        assert "haven't applied" in result

    def test_with_applications(self, monkeypatch):
        import skills.apply_status as skill
        import tracker
        monkeypatch.setattr(tracker, "get_applications", lambda: [
            {"role": "SWE", "company": "Acme", "applied_at": "2026-04-01 10:00:00", "platform": "LinkedIn", "url": ""},
            {"role": "Backend Engineer", "company": "Globex", "applied_at": "2026-03-28 09:00:00", "platform": "Indeed", "url": ""},
        ])
        result = skill.handle("application history")
        assert "Acme" in result
        assert "Globex" in result
        assert "2" in result  # total count

    def test_caps_at_five(self, monkeypatch):
        import skills.apply_status as skill
        import tracker
        apps = [
            {"role": f"Role {i}", "company": f"Co {i}", "applied_at": "2026-04-01", "platform": "", "url": ""}
            for i in range(10)
        ]
        monkeypatch.setattr(tracker, "get_applications", lambda: apps)
        result = skill.handle("show my applications")
        # Should mention total=10 but only show 5
        assert "10" in result
        assert result.count("Role") <= 5
```

- [ ] **Step 3: Run skill tests**

```bash
pytest tests/test_skills.py::TestApplyStatus -v
```

Expected: 3 PASSED

- [ ] **Step 4: Commit**

```bash
git add skills/apply_status/__init__.py tests/test_skills.py
git commit -m "feat(3D): add skills/apply_status — application history voice command"
```

---

## Task 9: Create skills/calculate skill

**Files:**
- Create: `skills/calculate/__init__.py`

Uses Python's `ast` module for safe mathematical expression evaluation — no `eval()`.

- [ ] **Step 1: Create skills/calculate/__init__.py**

```python
# /Users/bhanuteja/Documents/trae_projects/Aria/skills/calculate/__init__.py
"""
skills/calculate — Aria Phase 3D: Safe math expression evaluation.

Handles: "calculate 5 times 8", "how much is 120 divided by 4", "compute 2 to the power 10"

Uses AST parsing — never calls eval(). Only supports arithmetic operators.
"""

from __future__ import annotations

import ast
import operator
import re

TRIGGERS = [
    "calculate",
    "compute",
    "how much is",
    "what is the result of",
]

_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
}

_SPOKEN_MAP = [
    (r'\btimes\b',          '*'),
    (r'\bmultiplied\s+by\b', '*'),
    (r'\bdivided\s+by\b',   '/'),
    (r'\bover\b',           '/'),
    (r'\bplus\b',           '+'),
    (r'\bminus\b',          '-'),
    (r'\bsquared\b',        '**2'),
    (r'\bcubed\b',          '**3'),
    (r'\bto\s+the\s+power\s+of\b', '**'),
    (r'\bpercent\s+of\b',   '* 0.01 *'),
]


def _safe_eval(node) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def handle(command: str) -> str:
    """Extract a math expression from the command and evaluate it safely."""
    expr = command.lower()

    # Strip trigger phrases
    for trigger in TRIGGERS:
        expr = expr.replace(trigger, "")

    # Translate spoken operators to symbols
    for pattern, replacement in _SPOKEN_MAP:
        expr = re.sub(pattern, replacement, expr)

    # Keep only math-safe characters
    expr = re.sub(r'[^0-9\s\+\-\*\/\.\(\)\*]', '', expr).strip()

    if not expr:
        return "I couldn't find a math expression to calculate. Try saying 'calculate 5 times 8'."

    try:
        tree = ast.parse(expr, mode='eval')
        result = _safe_eval(tree.body)

        # Format: drop .0 from whole numbers
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        else:
            formatted = f"{result:.6g}"  # up to 6 significant figures

        return f"The result is {formatted}."
    except (SyntaxError, ZeroDivisionError, ValueError, OverflowError) as exc:
        return "I couldn't calculate that. Try saying something like 'calculate 5 times 8'."
```

- [ ] **Step 2: Add calculate tests to tests/test_skills.py**

```python
class TestCalculate:
    def _handle(self, cmd):
        import skills.calculate as skill
        return skill.handle(cmd)

    def test_multiplication(self):
        assert "40" in self._handle("calculate 5 times 8")

    def test_division(self):
        result = self._handle("how much is 120 divided by 4")
        assert "30" in result

    def test_addition(self):
        assert "15" in self._handle("calculate 7 plus 8")

    def test_subtraction(self):
        assert "3" in self._handle("calculate 10 minus 7")

    def test_power(self):
        assert "1024" in self._handle("compute 2 to the power of 10")

    def test_no_expression(self):
        result = self._handle("calculate")
        assert "couldn't find" in result.lower()

    def test_division_by_zero(self):
        result = self._handle("calculate 5 divided by 0")
        assert "couldn't calculate" in result.lower()

    def test_whole_number_result_no_decimal(self):
        result = self._handle("calculate 10 divided by 2")
        assert "5" in result
        assert "5.0" not in result  # should not show trailing .0
```

- [ ] **Step 3: Run all skill tests**

```bash
pytest tests/test_skills.py -v
```

Expected: all PASSED

- [ ] **Step 4: Run full skill_loader tests now that both skills exist**

```bash
pytest tests/test_skill_loader.py -v
```

Expected: all 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add skills/calculate/__init__.py tests/test_skills.py
git commit -m "feat(3D): add skills/calculate — safe AST math evaluation via voice"
```

---

## Task 10: Wire skill_loader into router.py

**Files:**
- Modify: `router.py`
- Modify: `main.py`

Skills are checked BEFORE the Groq classifier. If a skill matches, `route()` returns `type: "skill"` along with the matched handle function stored in the routing dict. `main.py` calls the handle function directly.

- [ ] **Step 1: Import skill_loader in router.py**

Add this import at the top of `router.py`, after the existing imports:

```python
from skills import skill_loader as _skill_loader
```

- [ ] **Step 2: Call load_skills() once and add skill pre-check in route()**

Add `load_skills()` call at module level (after imports, before `_get_client()`):

```python
# Load skills on module import — happens once when router is first imported
_skill_loader.load_skills()
```

Add the skill pre-check inside `route()`, AFTER the contact pre-check and BEFORE the apply pre-check:

```python
    # Skill pre-check — runs before Groq, zero API latency for matched skills
    skill_fn = _skill_loader.match_skill(command)
    if skill_fn is not None:
        logger.info("Skill matched for: %r", command)
        return {
            "type": "skill",
            "query": command.strip(),
            "url": "",
            "instructions": "",
            "app_name": "",
            "contact": "",
            "site_name": "",
            "_skill_fn": skill_fn,   # passed through to _handle_intent
        }
```

Also add `"skill"` to the valid types check in `_classify()` and add the passthrough in `_build_intent()`:

In `_classify()`:
```python
    if "type" not in parsed or parsed["type"] not in (
        "knowledge", "web_search", "web_direct", "app", "media", "navigate",
        "app_control", "briefing", "jobs", "apply", "capability", "skill",
    ):
```

In `_build_intent()` (add before the final fallback):
```python
    if intent_type == "skill":
        return {**_base}
```

- [ ] **Step 3: Add skill handler in main.py**

Inside `_handle_intent()`, add after the capability handler block:

```python
    # --- Skill: local skill matched by skill_loader ---
    if intent_type == "skill":
        skill_fn = intent.get("_skill_fn")
        if callable(skill_fn):
            try:
                return skill_fn(original_question)
            except Exception as exc:
                print(f"[Aria] Skill execution failed: {exc}")
                return "I ran into a problem with that command. Please try again."
        return answer_knowledge(original_question)
```

- [ ] **Step 4: Integration test**

```bash
python -c "
from router import route
r = route('what jobs have I applied to')
print('type:', r['type'])
print('has skill_fn:', callable(r.get('_skill_fn')))
assert r['type'] == 'skill'
assert callable(r.get('_skill_fn'))
print('PASS')
"
```

Expected:
```
type: skill
has skill_fn: True
PASS
```

- [ ] **Step 5: Commit**

```bash
git add router.py main.py
git commit -m "feat(3D): wire skill_loader into router.py — skills matched before Groq"
```

---

## Task 11: Update project-state.md to Phase 3D

**Files:**
- Modify: `.claude/project-state.md`

- [ ] **Step 1: Update project-state.md**

Replace the header and add a Phase 3D section. The "PHASE 3C COMPLETE" line stays — Phase 3C is done and its code is unchanged.

Open `.claude/project-state.md` and:

1. Change the status line from:
   ```
   ## Status: PHASE 3C COMPLETE — visible browser + coordinate-based computer use + stealth
   ```
   to:
   ```
   ## Status: PHASE 3D IN PROGRESS (Session 1 complete) — persistent memory + gap fixes
   ## Phase 3C: COMPLETE — visible browser + coordinate-based computer use + stealth
   ```

2. Add a new section after the "What to do next" block:

```markdown
## Phase 3D — What was added (Session 1)

### New files
- `db.py` — centralised SQLite (aria.db). All DB access goes here.
- `skills/__init__.py` — package marker
- `skills/skill_loader.py` — auto-discover + match skills before Groq
- `skills/apply_status/__init__.py` — "what jobs have I applied to"
- `skills/calculate/__init__.py` — "calculate 5 times 8" (safe AST eval)

### Modified files
- `tracker.py` — now uses db.get_connection() (aria.db). Old applications.db deprecated.
- `memory.py` — threading.Lock + SQLite persistence. State survives restart.
- `jobs.py` — salary filter (f_SB2). "Find jobs paying 100k" works.
- `main.py` — store_last_search() called after job search; capability + skill handlers added.
- `router.py` — capability pre-check (no Groq); skill_loader check before Groq classifier.
- `summarizer.py` — full identity injection (name, email, location, skills, education).

### Known gaps fixed
- "What's my name?" → now uses identity.json (name, email, location, skills)
- "What can you do?" → hardcoded response, never hallucinates features
- Salary filter for job search ("find jobs paying 100k") added to jobs.py
- Last search query persists across restarts via memory.py → SQLite

### What to do next (Session 2)
- Phase 3D Session 2: Channel adapter layer (Telegram, iMessage, Discord)
  - Create channels/base.py, channels/telegram_channel.py, etc.
  - Create channels/channel_manager.py
  - Extract handle_command() from main.py _process_release()
```

- [ ] **Step 2: Commit**

```bash
git add .claude/project-state.md
git commit -m "docs(3D): update project-state.md with Phase 3D Session 1 progress"
```

---

## Task 12: Write session-log.md

**Files:**
- Modify: `.claude/session-log.md`

- [ ] **Step 1: Prepend session entry to session-log.md**

Add at the top of `.claude/session-log.md`:

```markdown
## Session: 2026-04-04 — Phase 3D Session 1

**What we built:**
- db.py: centralised SQLite, all tables in ~/.aria/aria.db
- memory.py: threading.Lock + SQLite persistence + startup load from DB
- tracker.py: migrated to db.py (aria.db instead of applications.db)
- jobs.py: salary filter added (_parse_salary_filter, f_SB2 LinkedIn param)
- main.py: store_last_search() after job search; capability + skill handlers
- router.py: capability pre-check (regex, no Groq); skill_loader check before Groq
- summarizer.py: full identity injection (name, email, location, skills, education); updated capability list
- skills/skill_loader.py: auto-discovery, trigger matching
- skills/apply_status/: application history voice command
- skills/calculate/: safe AST math eval

**What we discovered was already done (stale gaps list):**
- Resume upload: already in linkedin_applicator._upload_resume() (line 524)
- Debug screenshots: already called throughout linkedin_applicator.py

**What's next:**
- Phase 3D Session 2: Channel adapter layer (Telegram, iMessage, Discord)
- Prerequisite: extract handle_command() from main.py _process_release() first
```

- [ ] **Step 2: Commit**

```bash
git add .claude/session-log.md
git commit -m "docs(3D): write session-log.md for Phase 3D Session 1"
```

---

## Verification Sequence

Run these in order after all tasks complete. Stop on first failure and fix before continuing.

```bash
# 1. Full test suite
cd /Users/bhanuteja/Documents/trae_projects/Aria
source venv/bin/activate
pytest tests/test_db.py tests/test_memory.py tests/test_jobs_filters.py \
       tests/test_router_capability.py tests/test_skill_loader.py tests/test_skills.py -v
```

Expected: all PASSED

```bash
# 2. Identity injection smoke test
python -c "
from summarizer import answer_knowledge, _IDENTITY
assert _IDENTITY.get('name'), 'identity.json not loading name'
ans = answer_knowledge(\"what's my name\")
print('Name answer:', ans)
"
```

Expected: response includes your actual name from identity.json

```bash
# 3. Capability pre-check (no Groq call)
python -c "
from unittest.mock import patch
from router import route
with patch('router._classify') as m:
    m.side_effect = AssertionError('Groq called for capability — FAIL')
    r = route('what can you do')
    assert r['type'] == 'capability', f'got {r[\"type\"]}'
    print('Capability pre-check OK')
"
```

Expected: `Capability pre-check OK`

```bash
# 4. Skill routing smoke test
python -c "
from router import route
r = route('calculate 5 times 8')
assert r['type'] == 'skill'
result = r['_skill_fn']('calculate 5 times 8')
assert '40' in result, f'got: {result}'
print('Calculate skill OK:', result)
"
```

Expected: `Calculate skill OK: The result is 40.`

```bash
# 5. Persistent memory smoke test
python -c "
import importlib
import db, memory
importlib.reload(memory)

memory.store_last_search('backend engineer remote')
q = memory.get_last_search()
assert q == 'backend engineer remote', f'got: {q}'

# Simulate restart
importlib.reload(memory)
q2 = memory.get_last_search()
assert q2 == 'backend engineer remote', f'after reload got: {q2}'
print('Persistent memory OK:', q2)
"
```

Expected: `Persistent memory OK: backend engineer remote`

```bash
# 6. Full Aria startup (live run)
# Start Aria:
python main.py

# Then manually verify:
# - Hold ⌥ Space → say "what's my name" → should speak your name
# - Hold ⌥ Space → say "what can you do" → hardcoded capability list
# - Hold ⌥ Space → say "calculate 10 times 7" → "The result is 70"
# - Hold ⌥ Space → say "find SWE jobs remote" → LinkedIn search runs
# - Ctrl+C to stop
# - python main.py again
# - Hold ⌥ Space → say "find me jobs" (without specifying)
#   → Aria should still remember the last search query is available
#     (verify with: python -c "import memory; print(memory.get_last_search())")
```

---

## Self-Review Notes

**Spec coverage check:**
- Part 4 (persistent memory): ✓ Tasks 1-3
- Fix A (identity injection): ✓ Task 5
- Fix B (capability handler): ✓ Task 6
- Fix C (resume upload): Already implemented — no task needed
- Fix D (debug screenshots): Already implemented — no task needed
- Fix E (salary filter): ✓ Task 4
- Addition F (apply_status skill): ✓ Task 8
- Addition G (calculate skill): ✓ Task 9
- Addition J (store last search): ✓ Task 4 (main.py change)
- Phase 3D rename: ✓ Task 11 (project-state.md; new files use Phase 3D in docstrings)
- thread lock: ✓ Task 3
- drop apscheduler: ✓ Not added anywhere
- centralized db.py: ✓ Task 1

**Type consistency:**
- `db.get_connection()` → `sqlite3.Connection` — used in Task 2 (tracker) and Task 3 (memory)
- `memory.store_last_search(query: str)` called in Task 4 (main.py)
- `memory.get_last_search() -> str` used in verification
- `skill_loader.match_skill(transcript) -> Callable | None` — returns `_skill_fn` stored in routing dict
- `intent["_skill_fn"]` accessed in main.py Task 10 — matches what router.py stores

**No placeholders confirmed:** All steps have complete code, exact commands, expected output.
