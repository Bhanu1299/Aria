"""
away_summary.py — Aria Phase 4: Away Summary (spoken startup greeting).

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

import logging

from groq import Groq

import config
import memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq client (lazy singleton)
# ---------------------------------------------------------------------------

_CLIENT: Groq | None = None

_FALLBACK = "Ready when you are."


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set — away summary disabled")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


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
        client = _get_client()
        prompt = _USER_TEMPLATE.format(
            session_notes=session_notes.strip() if session_notes else "(none)",
            last_search=last_search.strip() if last_search else "(none)",
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=128,
        )
        greeting = response.choices[0].message.content or ""
        greeting = greeting.strip()
        logger.debug("away_summary.generate: %d chars generated", len(greeting))
        return greeting if greeting else _FALLBACK
    except Exception as exc:
        logger.warning("away_summary.generate failed: %s", exc)
        return _FALLBACK


# ---------------------------------------------------------------------------
# Speak greeting
# ---------------------------------------------------------------------------

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
