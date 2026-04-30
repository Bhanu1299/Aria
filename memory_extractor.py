"""
memory_extractor.py — Aria Phase 4: Persistent User Model.

After every command, this module fires a Groq call in a daemon thread to extract
durable facts about the user from the conversation turn and persists them to
identity.json under the "learned_facts" key.

Facts are deduplicated (case-insensitive) and capped at 50 entries — when over
the cap, the oldest entries are displaced.

Public API:
  extract(transcript: str, answer: str) -> list[str]
      Synchronous extraction.  Returns list of NEW facts found this turn.
      Returns [] on any error so callers never have to handle exceptions.

  extract_async(transcript: str, answer: str) -> None
      Fire-and-forget: spawns a daemon thread and returns immediately.
      The thread calls extract() then persists facts to identity.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading

from groq import Groq

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq client (lazy singleton)
# ---------------------------------------------------------------------------

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set — memory extractor disabled")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


# ---------------------------------------------------------------------------
# Identity file helpers
# ---------------------------------------------------------------------------

_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")
_identity_lock = threading.Lock()


def _load_identity() -> dict:
    with _identity_lock:
        try:
            with open(_IDENTITY_PATH) as f:
                return json.load(f)
        except Exception:
            return {}


def _save_identity(identity: dict) -> None:
    with _identity_lock:
        try:
            with open(_IDENTITY_PATH, "w") as f:
                json.dump(identity, f, indent=2)
        except Exception as exc:
            logger.warning("memory_extractor: failed to save identity: %s", exc)


# ---------------------------------------------------------------------------
# Deduplication + cap logic
# ---------------------------------------------------------------------------

def _merge_facts(existing: list[str], new_facts: list[str]) -> list[str]:
    """
    Merge new_facts into existing, deduplicating case-insensitively.
    Caps the result at 50 entries, keeping the newest.
    """
    existing_lower = {f.lower() for f in existing}
    for fact in new_facts:
        if fact.lower() not in existing_lower:
            existing.append(fact)
            existing_lower.add(fact.lower())
    if len(existing) > 50:
        existing = existing[-50:]  # keep newest 50
    return existing


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a user-modeling assistant for a voice agent called Aria. "
    "Your job is to extract DURABLE facts about the user from a single conversation turn. "
    "Durable facts persist over time and reveal who the user is. "
    "Ephemeral observations (e.g. 'user asked about weather today') are NOT durable facts. "
    "Focus on: name, location, job preferences, skills mentioned, frequently used apps/tools, "
    "personal preferences (dark mode, keyboard shortcuts, workflows), recurring topics, "
    "people they mention, career goals, education, and any other stable personal attributes. "
    "Return ONLY a JSON array of strings. Each string must be a complete sentence. "
    "Example: [\"User prefers dark mode\", \"User is looking for senior Python roles\"] "
    "If no durable facts are present, return an empty array: []"
)

_USER_TEMPLATE = """\
Here is a single Aria conversation turn.

User said:
{transcript}

Aria answered:
{answer}

Extract all durable facts about the user from this turn.
Return a JSON array of strings. Each fact must be a complete sentence.
If there are no durable facts, return [].
"""


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract(transcript: str, answer: str) -> list[str]:
    """
    Call Groq synchronously and return a list of new durable facts found this turn.
    Returns [] if extraction fails for any reason or if there are no new facts.
    Does NOT persist — callers or _worker handle persistence.
    """
    if not transcript or not answer:
        return []
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
            temperature=0.2,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        logger.debug("memory_extractor.extract raw response: %s", raw[:200])

        # Parse JSON array from the response
        facts = json.loads(raw)
        if not isinstance(facts, list):
            return []
        # Filter to strings only
        return [f for f in facts if isinstance(f, str) and f.strip()]

    except Exception as exc:
        logger.warning("memory_extractor.extract failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Async (daemon thread) helper
# ---------------------------------------------------------------------------

def _worker(transcript: str, answer: str) -> None:
    """Target for the daemon thread: extract facts, merge, and persist to identity.json."""
    try:
        new_facts = extract(transcript, answer)
        if not new_facts:
            return
        identity = _load_identity()
        existing = identity.get("learned_facts", [])
        if not isinstance(existing, list):
            existing = []
        merged = _merge_facts(existing, new_facts)
        identity["learned_facts"] = merged
        _save_identity(identity)
        logger.debug(
            "memory_extractor: stored %d new facts (%d total)",
            len(new_facts),
            len(merged),
        )
    except Exception as exc:
        logger.warning("memory_extractor._worker unhandled error: %s", exc)


def extract_async(transcript: str, answer: str) -> None:
    """
    Fire-and-forget: spawn a daemon thread that calls extract() then
    persists new facts to identity.json.  Returns to the caller immediately.
    """
    t = threading.Thread(
        target=_worker,
        args=(transcript, answer),
        daemon=True,
        name="memory-extractor",
    )
    t.start()
