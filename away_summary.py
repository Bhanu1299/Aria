"""
away_summary.py — Aria Phase 4: Away Summary (spoken startup greeting + gap detection).

On startup, loads session notes and last search from prior sessions and
generates a short spoken greeting via Groq, then speaks it aloud.

Public API:
  generate(session_notes: str, last_search: str) -> str
      Synchronous. Returns a 1-2 sentence spoken greeting string.
      Returns "Ready when you are." if both inputs are empty or on any error.

  speak_greeting(speaker) -> None
      Loads data from memory, calls generate(), speaks the result.
      Wraps everything in try/except — never raises.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone, timedelta

from llm import llm_client
import memory

_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")
_AWAY_GAP_MINUTES = 30

logger = logging.getLogger(__name__)

_FALLBACK = "Ready when you are."


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are Aria, a concise voice assistant. "
    "Generate a short spoken greeting for the user returning to their computer."
)

_USER_TEMPLATE = """\
The user has just returned to their computer. Here is context from their last session:

Session notes:
{session_notes}

Last search query:
{last_search}

Generate a 1-2 spoken sentence greeting that:
- Welcomes them back
- Briefly describes what was happening last session
- Hints at what they might want to do next

Return ONLY the spoken greeting sentences, nothing else.
"""


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate(session_notes: str, last_search: str) -> str:
    """
    Call Groq synchronously and return a 1-2 sentence spoken greeting.
    Returns "Ready when you are." if both inputs are empty or on any error.
    """
    if not session_notes and not last_search:
        return _FALLBACK

    try:
        prompt = _USER_TEMPLATE.format(
            session_notes=session_notes.strip() if session_notes else "(none)",
            last_search=last_search.strip() if last_search else "(none)",
        )
        resp = llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            tier="fast",
            system=_SYSTEM_PROMPT,
            max_tokens=128,
        )
        greeting = resp.text.strip()
        logger.debug("away_summary.generate: %d chars generated", len(greeting))
        return greeting if greeting else _FALLBACK
    except Exception as exc:
        logger.warning("away_summary.generate failed: %s", exc)
        return _FALLBACK


# ---------------------------------------------------------------------------
# Speak greeting
# ---------------------------------------------------------------------------

def check_and_speak(speaker) -> None:
    """
    If more than 30 minutes have passed since last_active_at in identity.json,
    speak a 1-sentence recap of the last task. Never raises.
    """
    try:
        with open(_IDENTITY_PATH) as f:
            identity = json.load(f)
        last_active_str = identity.get("last_active_at")
        if not last_active_str:
            return
        last_active = datetime.fromisoformat(last_active_str)
        if last_active.tzinfo is None:
            last_active = last_active.replace(tzinfo=timezone.utc)
        gap = datetime.now(timezone.utc) - last_active
        if gap < timedelta(minutes=_AWAY_GAP_MINUTES):
            return
        summary = identity.get("last_task_summary", "")
        if summary:
            speaker.say(f"Welcome back. Last time: {summary}")
    except Exception as exc:
        logger.warning("away_summary.check_and_speak failed: %s", exc)


def update_last_active(task_summary: str) -> None:
    """Write last_active_at and last_task_summary to identity.json. Never raises."""
    try:
        try:
            with open(_IDENTITY_PATH) as f:
                identity = json.load(f)
        except Exception:
            identity = {}
        identity["last_active_at"] = datetime.now(timezone.utc).isoformat()
        identity["last_task_summary"] = task_summary
        dir_ = os.path.dirname(_IDENTITY_PATH)
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(identity, f, indent=2)
            tmp = f.name
        os.replace(tmp, _IDENTITY_PATH)
    except Exception as exc:
        logger.warning("away_summary.update_last_active failed: %s", exc)


def speak_greeting(speaker) -> None:
    """
    Load session notes and last search from memory, generate a greeting,
    and speak it. Never raises — all errors are caught and logged.
    """
    try:
        session_notes = memory.get_session_notes()
        last_search = memory.get_last_search()
        greeting = generate(session_notes, last_search)
        speaker.say(greeting)
    except Exception as exc:
        logger.warning("away_summary.speak_greeting failed: %s", exc)
