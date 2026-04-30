"""
session_notes.py — Aria Phase 4: Session Notes extraction.

After every command Aria handles, this module fires a Groq call in a daemon
thread to summarise the turn into 3-5 bullet points and persists the result
to SQLite via memory.py.

On the next Aria startup the notes are loaded automatically (memory._load_from_db
runs on import) and are available via memory.get_session_notes().

Public API:
  extract(transcript: str, answer: str) -> str
      Synchronous extraction.  Returns the bullet-point summary string.
      Returns "" on any error so callers never have to handle exceptions.

  extract_async(transcript: str, answer: str) -> None
      Fire-and-forget: spawns a daemon thread and returns immediately.
      The thread calls extract() then stores the result.
"""

from __future__ import annotations

import logging
import threading

from groq import Groq

import config
import memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq client (lazy singleton)
# ---------------------------------------------------------------------------

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a concise session-notes assistant for a voice agent called Aria. "
    "You extract the key facts from a single conversation turn and return them as "
    "a short bullet-point list.  No prose, no filler, no extra commentary."
)

_USER_TEMPLATE = """\
Here is a single Aria conversation turn.

User said:
{transcript}

Aria answered:
{answer}

Summarise this turn into 3-5 bullet points covering:
- What the user wanted
- What Aria did or found
- Any key data mentioned (URLs, prices, job titles, names, numbers)
- Anything that was left unfinished or should be followed up

Return ONLY the bullet points, each starting with "- ".
"""


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract(transcript: str, answer: str) -> str:
    """
    Call Groq synchronously and return a bullet-point summary string.
    Returns "" if extraction fails for any reason.
    """
    if not transcript or not answer:
        return ""
    try:
        client = _get_client()
        prompt = _USER_TEMPLATE.format(
            transcript=transcript.strip(),
            answer=answer.strip(),
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        notes = response.choices[0].message.content or ""
        notes = notes.strip()
        logger.debug("session_notes.extract: %d chars extracted", len(notes))
        return notes
    except Exception as exc:
        logger.warning("session_notes.extract failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Async (daemon thread) helper
# ---------------------------------------------------------------------------

def _worker(transcript: str, answer: str) -> None:
    """Target for the daemon thread: extract notes and persist to memory."""
    try:
        notes = extract(transcript, answer)
        if notes:
            memory.store_session_notes(notes)
            logger.debug("session_notes: stored %d chars", len(notes))
    except Exception as exc:
        logger.warning("session_notes._worker unhandled error: %s", exc)


def extract_async(transcript: str, answer: str) -> None:
    """
    Fire-and-forget: spawn a daemon thread that calls extract() then
    persists the result.  Returns to the caller immediately — never blocks.
    """
    t = threading.Thread(
        target=_worker,
        args=(transcript, answer),
        daemon=True,
        name="session-notes-extractor",
    )
    t.start()
