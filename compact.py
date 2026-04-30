"""
compact.py — Aria Phase 4: Session notes compaction.

When accumulated session notes exceed 3000 chars, a Groq call compresses them
to ~500 chars. memory.py calls needs_compaction() and compress() inside
store_session_notes() when the threshold is crossed.
"""

from __future__ import annotations

import logging
from groq import Groq
import config

logger = logging.getLogger(__name__)

_NOTES_MAX_CHARS = 3000
_CLIENT: Groq | None = None

_SYSTEM_PROMPT = (
    "You are a concise notes compressor for a voice agent called Aria. "
    "Given a running session log, compress it to the most essential facts "
    "in 3-7 bullet points. Keep key data: names, URLs, prices, job titles. "
    "Return ONLY bullet points starting with '- '."
)


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def needs_compaction(notes: str) -> bool:
    return len(notes) > _NOTES_MAX_CHARS


def compress(notes: str) -> str:
    """Call Groq to compress notes. Returns original on any failure."""
    if not notes.strip():
        return notes
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Compress these session notes:\n\n{notes}"},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        result = response.choices[0].message.content or ""
        result = result.strip()
        logger.debug("compact.compress: %d → %d chars", len(notes), len(result))
        return result if result else notes
    except Exception as exc:
        logger.warning("compact.compress failed: %s", exc)
        return notes
