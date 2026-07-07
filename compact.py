"""
compact.py — Aria Phase 4/5D: Session notes compaction + agent message history compaction.

Notes: memory.py calls needs_compaction() / compress() to keep session notes short.
Messages: agent.py calls should_compact_messages() / compact_messages() to keep
          multi-turn conversation history within the model context window.
"""

from __future__ import annotations

import json
import logging

from llm import llm_client

logger = logging.getLogger(__name__)

_NOTES_MAX_CHARS = 3000
_CLAUDE_CONTEXT_TOKENS = 200_000
_COMPACT_THRESHOLD = 0.8          # fire at 80% of context window
_KEEP_RECENT_MESSAGES = 20        # keep last 10 user+assistant pairs verbatim

_NOTES_SYSTEM_PROMPT = (
    "You are a concise notes compressor for a voice agent called Aria. "
    "Given a running session log, compress it to the most essential facts "
    "in 3-7 bullet points. Keep key data: names, URLs, prices, job titles. "
    "Return ONLY bullet points starting with '- '."
)

_MESSAGES_SYSTEM_PROMPT = (
    "You are summarizing an earlier part of a conversation between a user and Aria, "
    "a voice assistant. Capture the key facts, decisions, and context in 5-10 bullet "
    "points. Keep names, numbers, and action items. Return ONLY bullet points starting "
    "with '- '."
)


# --- Session notes compaction (Phase 4) ---

def needs_compaction(notes: str) -> bool:
    return len(notes) > _NOTES_MAX_CHARS


def compress(notes: str) -> str:
    """Call LLM to compress notes. Returns original on any failure."""
    if not notes.strip():
        return notes
    try:
        resp = llm_client.complete(
            messages=[
                {"role": "user", "content": f"Compress these session notes:\n\n{notes}"},
            ],
            tier="fast",
            system=_NOTES_SYSTEM_PROMPT,
            max_tokens=300,
        )
        result = resp.text.strip()
        logger.debug("compact.compress: %d → %d chars", len(notes), len(result))
        return result if result else notes
    except Exception as exc:
        logger.warning("compact.compress failed: %s", exc)
        return notes


# --- Agent message history compaction (Phase 5D) ---

def _estimate_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages)) // 4


def should_compact_messages(messages: list[dict], context_tokens: int = _CLAUDE_CONTEXT_TOKENS) -> bool:
    """Return True if messages exceed 80% of the model context window."""
    return _estimate_tokens(messages) > _COMPACT_THRESHOLD * context_tokens


def compact_messages(messages: list[dict]) -> list[dict]:
    """
    Keep the last 20 messages verbatim; summarize everything older into one
    Prior context message prepended to the list. Returns original on failure.
    """
    if len(messages) <= _KEEP_RECENT_MESSAGES:
        return messages

    older = messages[:-_KEEP_RECENT_MESSAGES]
    recent = messages[-_KEEP_RECENT_MESSAGES:]
    tokens_before = _estimate_tokens(messages)

    try:
        older_text = "\n".join(
            f"{m['role'].upper()}: {m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])}"
            for m in older
        )
        resp = llm_client.complete(
            messages=[{"role": "user", "content": f"Summarize this conversation:\n\n{older_text}"}],
            tier="fast",
            system=_MESSAGES_SYSTEM_PROMPT,
            max_tokens=400,
        )
        summary = resp.text.strip() or "(no summary)"
        summary_msg = {"role": "user", "content": f"Prior context (summarized):\n{summary}"}
        result = [summary_msg] + recent
        tokens_after = _estimate_tokens(result)
        logger.debug(
            "Compacted %d messages into summary (%d → %d tokens)",
            len(older), tokens_before, tokens_after,
        )
        return result
    except Exception as exc:
        logger.warning("compact.compact_messages failed: %s", exc)
        return messages
