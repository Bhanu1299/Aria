"""applying/tabs.py — every application tab Aria opened, persisted across restarts.

One JSON list in ~/.aria/apply_tabs.json. Each entry:
  id            "apply-<row>-<epoch>": also the agent_browser tab name
  row, company, role, url, location     from the sheet row
  ats, form_url, filled, needs_me (labels), review_path, screenshot
  state         filling | filled | submitted | open | closed
  submitted_by  "aria" (confirmation seen after "submit it") | "user" ("I submitted it")
  sheet_marked  True once the sheet says Applied for it
  note, opened, updated

States: filling → filled → submitted. open = the tab is open but the fill didn't finish
(stopped, error, a wall). closed = the user closed it; kept for history, never shown.
Reading a missing or corrupt file gives an empty registry; writing never raises.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path.home() / ".aria" / "apply_tabs.json"

FILLING, FILLED, SUBMITTED, OPEN, CLOSED = "filling", "filled", "submitted", "open", "closed"

_lock = threading.RLock()


def _load() -> list:
    try:
        data = json.loads(REGISTRY_PATH.read_text())
        return [e for e in data if isinstance(e, dict) and e.get("id")] if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning("apply tabs: unreadable registry (%s) — starting empty", exc)
        return []


def _save(entries: list) -> None:
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1))
        os.replace(tmp, REGISTRY_PATH)
    except Exception as exc:
        logger.error("apply tabs: write failed: %s", exc)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:20] or "job"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def all_entries() -> list:
    with _lock:
        return _load()


def add(job: dict, ats: str = "") -> dict:
    now = time.time()
    entry = {
        "id": f"apply-{job.get('row') or _slug(job.get('company', ''))}-{int(now * 1000)}",
        **{k: job.get(k) for k in ("row", "company", "role", "url", "location")},
        "ats": ats, "form_url": "", "filled": {}, "needs_me": [], "review_path": "", "screenshot": "",
        "state": FILLING, "submitted_by": "", "sheet_marked": False, "note": "",
        "opened": now, "updated": now,
    }
    with _lock:
        entries = _load()
        entries.append(entry)
        _save(entries)
    return dict(entry)


def update(entry_id: str, **fields) -> Optional[dict]:
    with _lock:
        entries = _load()
        for e in entries:
            if e["id"] == entry_id:
                e.update(fields)
                e["updated"] = time.time()
                _save(entries)
                return dict(e)
    return None


def get(entry_id: str) -> Optional[dict]:
    return next((e for e in all_entries() if e["id"] == entry_id), None)


def live() -> list:
    """Every tab not closed, oldest first."""
    return sorted((e for e in all_entries() if e.get("state") != CLOSED), key=lambda e: e.get("opened", 0))


_NOT_COMPANIES = {"it", "this", "that", "one", "me", "them", "all", "everything", "everyone", "the", "a", "an",
                  "now", "here", "there", "up", "out", "off"}


def _starts(company: str, want: str) -> bool:
    words = re.findall(r"[a-z0-9]+", (company or "").lower())
    return any("".join(words[i:]).startswith(want) for i in range(len(words)))


def by_company(text: str) -> list:
    """Live tabs whose company starts with these words (at a word start). Short or filler
    words ("it", "me", "ad") never match, so everyday phrases don't hit a tab."""
    want = _norm(text)
    if len(want) < 3 or (text or "").strip().lower() in _NOT_COMPANIES:
        return []
    return [e for e in live() if _starts(e.get("company", ""), want)]


def is_open_for(job: dict) -> bool:
    row, url = job.get("row"), (job.get("url") or "").strip()
    return any((row and e.get("row") == row) or (url and url in (e.get("url"), e.get("form_url")))
               for e in live())


def mark_closed(entry_id: str) -> Optional[dict]:
    """closed_from keeps what the tab was, so a submitted tab still gets its Applied write."""
    e = get(entry_id)
    if e is None:
        return None
    return update(entry_id, state=CLOSED, closed_from=e.get("state", "") if e.get("state") != CLOSED
                  else e.get("closed_from", ""))
