"""dashboard/gmail_cache.py — the dashboard's private copy of Gmail metadata it already read.

Keyed by Gmail's change markers (a thread's historyId, a draft's message id), so a reload or
a dashboard restart only re-reads what changed instead of every thread — reading them all at
once is what tripped Gmail's per-user rate limit. This file is the dashboard's only local
write besides settings/answers; it never touches the inbox, the sheet or any database.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PATH = Path.home() / ".aria" / "dashboard_gmail_cache.json"


def read(path: Path, me: str) -> dict:
    """{me, threads: {id: {h, t}}, drafts: {message_id: draft}}; empty for another account or a bad file."""
    try:
        data = json.loads(path.read_text())
        if data.get("me") == me and isinstance(data.get("threads"), dict) and isinstance(data.get("drafts"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"me": me, "threads": {}, "drafts": {}}


def write(path: Path, data: dict) -> None:
    """Atomic, owner-only (subjects and addresses from the jobs inbox). Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.chmod(tmp, 0o600)
        tmp.replace(path)
    except OSError as exc:
        logger.warning("dashboard: couldn't save the Gmail cache: %s", exc)
