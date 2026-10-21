"""features/outreach/feedback.py — replies and bounces in the jobs Gmail → Coldstart.

    sync(gmail_data=None) → {"replied": [...], "bounced": [...], "error": ""}   never raises

Outreach threads are found the same way the dashboard finds them (model.thread_info:
first message mine, to a company address). A thread with a reply from the recipient's
domain records "replied"; one with a bounce and no reply records "bounced". Coldstart's
record_feedback moves format weight every time it's called, so an (address, outcome)
already in its feedback table is skipped. The contact's name is passed when Coldstart
knows it, so the outcome can join its eval set.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from aria.dashboard import model, sources

logger = logging.getLogger(__name__)

REPLIED, BOUNCED = "replied", "bounced"


def outcomes(gmail_data: dict) -> dict[str, str]:
    """{address: replied | bounced} for every outreach thread in the jobs Gmail data."""
    me, out = gmail_data.get("me", ""), {}
    for t in gmail_data.get("threads", []):
        info = model.thread_info(t, me)
        if info is None:
            continue
        if info["replied"]:
            out[info["to"]] = REPLIED          # a reply beats an earlier bounce on the same address
        elif info["bounced"] and out.get(info["to"]) != REPLIED:
            out[info["to"]] = BOUNCED
    return out


def _coldstart_state(path: Optional[Path] = None) -> tuple[dict, dict]:
    """(recorded {email: outcome}, names {email: name}) from Coldstart's DB, read-only."""
    if path is None:
        import coldstart.config as cs_config
        path = cs_config.db_path()
    if not Path(path).exists():
        return {}, {}
    conn = sources._ro_connect(Path(path))
    try:
        recorded = {r["email"]: r["outcome"] for r in conn.execute("SELECT email, outcome FROM feedback")}
        names = {r["email"]: r["name"] for r in conn.execute(
            "SELECT email, name FROM contacts WHERE email IS NOT NULL AND name IS NOT NULL AND name != ''")}
    finally:
        conn.close()
    return recorded, names


def sync(gmail_data: Optional[dict] = None, record_fn=None, coldstart_db: Optional[Path] = None) -> dict:
    """Record new replies/bounces with Coldstart. Returns what was recorded (or would be)."""
    result: dict = {"replied": [], "bounced": [], "error": ""}
    try:
        if gmail_data is None:
            src = sources.load_gmail()
            if not src.ok:
                result["error"] = f"jobs Gmail unavailable: {src.error}"
                return result
            gmail_data = src.data or {}
        found = outcomes(gmail_data)
        if not found:
            return result
        recorded, names = _coldstart_state(coldstart_db)
        if record_fn is None:
            from coldstart import api as coldstart
            record_fn = coldstart.record_feedback
        for email, outcome in sorted(found.items()):
            if recorded.get(email) == outcome:
                continue
            try:
                record_fn(email, outcome, name=names.get(email))
            except Exception as exc:
                logger.warning("outreach: record_feedback(%s, %s) failed: %s", email, outcome, exc)
                continue
            result[outcome].append(email)
    except Exception as exc:
        logger.error("outreach: feedback sync failed: %s", exc)
        result["error"] = str(exc)
    return result


def spoken(result: dict) -> str:
    r, b = len(result.get("replied", [])), len(result.get("bounced", []))
    if not r and not b:
        return ""
    parts = [f"{r} repl{'y' if r == 1 else 'ies'}" if r else "", f"{b} bounce{'s' if b != 1 else ''}" if b else ""]
    return "Told Coldstart about " + " and ".join(p for p in parts if p) + "."
