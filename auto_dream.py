"""
auto_dream.py — Aria Phase 4: Background Memory Consolidation (Task 10)

After every 5 commands, fires a background Groq call that reads session_notes
and learned_facts from identity.json, consolidates them into cleaner organized
versions, and writes both back. Deduplicates facts, trims noise.
Counter is persisted in SQLite via memory.py.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading

from groq import Groq
import config
import memory

logger = logging.getLogger(__name__)

_CONSOLIDATE_EVERY = 5
_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")
_CLIENT: Groq | None = None

_SYSTEM_PROMPT = (
    "You are a memory consolidation assistant for a voice agent called Aria. "
    "Given session notes and known user facts, return a JSON object with two keys: "
    "'session_notes' (3-5 bullet points of the most important recent activity) and "
    "'learned_facts' (deduplicated array of durable user facts, max 50, newest kept). "
    "Return ONLY valid JSON. No markdown."
)


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def _load_identity() -> dict:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_identity(identity: dict) -> None:
    try:
        dir_ = os.path.dirname(_IDENTITY_PATH)
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(identity, f, indent=2)
            tmp = f.name
        os.replace(tmp, _IDENTITY_PATH)
    except Exception as exc:
        logger.warning("auto_dream: failed to save identity: %s", exc)


def consolidate() -> None:
    """Synchronous consolidation. Reads notes + facts, rewrites both. Never raises."""
    try:
        notes = memory.get_session_notes()
        identity = _load_identity()
        facts = identity.get("learned_facts", [])

        if not notes and not facts:
            return

        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Session notes:\n{notes}\n\n"
                    f"Known facts:\n{json.dumps(facts)}"
                )},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)

        new_notes = result.get("session_notes", "")
        new_facts = result.get("learned_facts", facts)

        if isinstance(new_notes, list):
            new_notes = "\n".join(f"- {n}" if not n.startswith("-") else n for n in new_notes)

        if new_notes:
            memory.clear_session_notes()
            memory.store_session_notes(new_notes)

        if isinstance(new_facts, list) and identity:
            identity["learned_facts"] = new_facts[:50]
            _save_identity(identity)

        memory.reset_command_count()
        logger.debug("auto_dream.consolidate: complete")

    except Exception as exc:
        logger.warning("auto_dream.consolidate failed: %s", exc)


def maybe_consolidate_async(transcript: str, answer: str) -> None:
    """Increment counter; if interval reached, consolidate in a daemon thread."""
    count = memory.increment_command_count()
    if count % _CONSOLIDATE_EVERY == 0:
        t = threading.Thread(target=consolidate, daemon=True, name="auto-dream")
        t.start()
