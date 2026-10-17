"""plugins/context/store.py — the cross-app context index (local SQLite).

items        one row per email / calendar event / iMessage conversation-day
items_fts    FTS5 over title + body, for topic lookups ("the Macro interview")
people       one row per real person
aliases      typed handles → person ("email:…", "tel:…", "name:…")
item_people  which people each item involves

Linking rule: aliases that arrive together (a "Name <addr>" header, a
Contacts card, a calendar attendee) belong to one person; if they already
point at two different people, those people are merged. Single-word
names never link on their own — two different "John"s must not merge.
"""
from __future__ import annotations

import difflib
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path

from aria.plugins.context.people import email_alias, handle_alias, name_alias, phone_alias

logger = logging.getLogger(__name__)

DEFAULT_DB = Path.home() / ".aria" / "context.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, ts REAL NOT NULL,
    title TEXT, body TEXT, ref TEXT
);
CREATE INDEX IF NOT EXISTS items_ts ON items(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(item_id UNINDEXED, title, body);
CREATE TABLE IF NOT EXISTS people (id INTEGER PRIMARY KEY, name TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS aliases (alias TEXT PRIMARY KEY, person_id INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS aliases_person ON aliases(person_id);
CREATE TABLE IF NOT EXISTS item_people (
    item_id TEXT NOT NULL, person_id INTEGER NOT NULL, PRIMARY KEY (item_id, person_id)
);
CREATE INDEX IF NOT EXISTS item_people_person ON item_people(person_id);
CREATE TABLE IF NOT EXISTS sync_state (
    source TEXT PRIMARY KEY, last_ts REAL, last_count INTEGER, last_error TEXT
);
"""

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "on", "in", "to", "for", "with", "about",
    "what", "whats", "is", "going", "my", "me", "i", "up", "catch", "anything",
}


class ContextStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = str(path or DEFAULT_DB)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    # ------------------------------------------------------------------
    # people
    # ------------------------------------------------------------------

    def link(self, name: str, handles: list, authoritative: bool = False) -> int | None:
        """Link a display name + raw handles to one person; returns person id."""
        with self._lock, self._connect() as conn:
            return self._link(conn, name, handles, authoritative)

    def _link(self, conn, name: str, handles: list, authoritative: bool) -> int | None:
        aliases = [a for a in (handle_alias(h) for h in handles) if a]
        full = name_alias(name)
        if full and len(full.split()) >= 2:
            aliases.append(full)
        if not aliases:
            return None
        marks = ",".join("?" * len(aliases))
        pids = sorted({r[0] for r in conn.execute(
            f"SELECT person_id FROM aliases WHERE alias IN ({marks})", aliases)})
        if pids:
            pid = pids[0]
            for other in pids[1:]:
                self._merge(conn, pid, other)
        else:
            pid = conn.execute("INSERT INTO people(name) VALUES (?)", (name or "",)).lastrowid
        conn.executemany("INSERT OR IGNORE INTO aliases(alias, person_id) VALUES (?, ?)",
                         [(a, pid) for a in aliases])
        if name and name_alias(name):
            current = conn.execute("SELECT name FROM people WHERE id=?", (pid,)).fetchone()[0]
            if authoritative or not current or len(name) > len(current) and " " in name:
                conn.execute("UPDATE people SET name=? WHERE id=?", (name, pid))
        return pid

    @staticmethod
    def _merge(conn, keep: int, other: int) -> None:
        conn.execute("UPDATE aliases SET person_id=? WHERE person_id=?", (keep, other))
        conn.execute("UPDATE OR IGNORE item_people SET person_id=? WHERE person_id=?",
                     (keep, other))
        conn.execute("DELETE FROM item_people WHERE person_id=?", (other,))
        row = conn.execute("SELECT name FROM people WHERE id=?", (other,)).fetchone()
        if row and row[0]:
            conn.execute("UPDATE people SET name=? WHERE id=? AND name=''", (row[0], keep))
        conn.execute("DELETE FROM people WHERE id=?", (other,))

    def resolve_person(self, query: str) -> tuple:
        """Best person for a spoken name/handle → (person_id, name, [other close names], how).

        Order: exact handle → exact full name → every query word matches a
        name word → fuzzy (voice transcripts misspell names). Ties go to
        whoever has the most indexed items. `how` is "exact", "words" or
        "fuzzy"; (None, '', [], '') if nobody fits.
        """
        none = (None, "", [], "")
        q = (query or "").strip()
        if not q:
            return none
        with self._connect() as conn:
            for alias in (email_alias(q) if "@" in q else phone_alias(q), name_alias(q)):
                if alias:
                    row = conn.execute("SELECT person_id FROM aliases WHERE alias=?",
                                       (alias,)).fetchone()
                    if row:
                        return row[0], self._name(conn, row[0]), [], "exact"

            people = conn.execute(
                "SELECT p.id, p.name, COUNT(ip.item_id) FROM people p "
                "LEFT JOIN item_people ip ON ip.person_id = p.id "
                "WHERE p.name != '' GROUP BY p.id").fetchall()
        words = re.findall(r"[a-z0-9]+", q.lower())
        if not words or not people:
            return none

        def words_match(name: str) -> bool:
            parts = re.findall(r"[a-z0-9]+", name.lower())
            return all(any(p == w or (len(w) >= 3 and p.startswith(w)) for p in parts)
                       for w in words)

        how = "words"
        hits = [p for p in people if words_match(p[1])]
        if not hits:
            how = "fuzzy"
            by_key: dict = {}
            for p in people:
                by_key.setdefault(p[1].lower(), p)
                first = p[1].lower().split()[0]
                by_key.setdefault(first, p)
            close = difflib.get_close_matches(q.lower(), list(by_key), n=3, cutoff=0.75)
            hits = list({by_key[c][0]: by_key[c] for c in close}.values())
        if not hits:
            return none
        hits.sort(key=lambda p: -p[2])
        best = hits[0]
        return best[0], best[1], [p[1] for p in hits[1:4]], how

    def person_handles(self, person_id: int) -> list:
        with self._connect() as conn:
            rows = conn.execute("SELECT alias FROM aliases WHERE person_id=? ORDER BY alias",
                                (person_id,)).fetchall()
        return [r[0].split(":", 1)[1] for r in rows if not r[0].startswith("name:")]

    @staticmethod
    def _name(conn, person_id: int) -> str:
        row = conn.execute("SELECT name FROM people WHERE id=?", (person_id,)).fetchone()
        return row[0] if row else ""

    # ------------------------------------------------------------------
    # items
    # ------------------------------------------------------------------

    def upsert_items(self, items: list) -> int:
        """Insert or replace items. Each: {id, source, ts, title, body, ref,
        participants: [{name, handles}]}. Returns how many were written."""
        written = 0
        with self._lock, self._connect() as conn:
            for it in items:
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO items(id, source, ts, title, body, ref) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (it["id"], it["source"], float(it["ts"]), it.get("title", ""),
                         it.get("body", ""), it.get("ref", "")))
                    conn.execute("DELETE FROM items_fts WHERE item_id=?", (it["id"],))
                    conn.execute("INSERT INTO items_fts(item_id, title, body) VALUES (?, ?, ?)",
                                 (it["id"], it.get("title", ""), it.get("body", "")))
                    conn.execute("DELETE FROM item_people WHERE item_id=?", (it["id"],))
                    for p in it.get("participants", []):
                        pid = self._link(conn, p.get("name", ""), p.get("handles", []), False)
                        if pid is not None:
                            conn.execute("INSERT OR IGNORE INTO item_people VALUES (?, ?)",
                                         (it["id"], pid))
                    written += 1
                except Exception as exc:
                    logger.warning("context: skipping item %r: %s", it.get("id"), exc)
        return written

    def known_ids(self, source: str) -> set:
        with self._connect() as conn:
            return {r[0] for r in conn.execute("SELECT id FROM items WHERE source=?", (source,))}

    def person_items(self, person_id: int, since: float, until: float, limit: int = 20) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT i.id, i.source, i.ts, i.title, i.body, i.ref FROM items i "
                "JOIN item_people ip ON ip.item_id = i.id "
                "WHERE ip.person_id=? AND i.ts BETWEEN ? AND ? ORDER BY i.ts DESC LIMIT ?",
                (person_id, since, until, limit)).fetchall()
        return [_row(r) for r in rows]

    def topic_items(self, query: str, since: float, until: float, limit: int = 20) -> list:
        words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower())
                 if w not in _STOPWORDS]
        if not words:
            return []
        quoted = [f'"{w}"' for w in words]
        with self._connect() as conn:
            for match in (" AND ".join(quoted), " OR ".join(quoted)):
                rows = conn.execute(
                    "SELECT i.id, i.source, i.ts, i.title, i.body, i.ref FROM items_fts f "
                    "JOIN items i ON i.id = f.item_id "
                    "WHERE items_fts MATCH ? AND i.ts BETWEEN ? AND ? "
                    "ORDER BY i.ts DESC LIMIT ?",
                    (match, since, until, limit)).fetchall()
                if rows:
                    return [_row(r) for r in rows]
        return []

    def item_people_names(self, item_id: str) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT p.name FROM people p JOIN item_people ip ON ip.person_id = p.id "
                "WHERE ip.item_id=? AND p.name != ''", (item_id,)).fetchall()
        return [r[0] for r in rows]

    def prune(self, older_than: float) -> None:
        self._delete_where("ts < ?", (older_than,))

    def delete_source_since(self, source: str, since: float) -> None:
        """Drop a source's items from `since` on (calendar re-sync: removes cancelled events)."""
        self._delete_where("source = ? AND ts >= ?", (source, since))

    def _delete_where(self, where: str, args: tuple) -> None:
        sub = f"(SELECT id FROM items WHERE {where})"
        with self._lock, self._connect() as conn:
            conn.execute(f"DELETE FROM items_fts WHERE item_id IN {sub}", args)
            conn.execute(f"DELETE FROM item_people WHERE item_id IN {sub}", args)
            conn.execute(f"DELETE FROM items WHERE {where}", args)

    def counts(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT source, COUNT(*) FROM items GROUP BY source").fetchall()
            people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        out = {r[0]: r[1] for r in rows}
        out["people"] = people
        return out

    # ------------------------------------------------------------------
    # sync bookkeeping
    # ------------------------------------------------------------------

    def set_sync_state(self, source: str, count: int, error: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO sync_state VALUES (?, ?, ?, ?)",
                         (source, time.time(), count, error))

    def sync_states(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT source, last_ts, last_count, last_error "
                                "FROM sync_state").fetchall()
        return {r[0]: {"last_ts": r[1], "last_count": r[2], "last_error": r[3] or ""}
                for r in rows}


def _row(r) -> dict:
    return {"id": r[0], "source": r[1], "ts": r[2], "title": r[3] or "",
            "body": r[4] or "", "ref": r[5] or ""}
