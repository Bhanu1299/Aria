"""plugins/memory/context_injector.py — Inject relevant memories into agent system prompt."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MAX_TOKENS = 300
_APPROX_CHARS_PER_TOKEN = 4


def build(query: str) -> str:
    """
    Search vector store for facts relevant to query.
    Returns a formatted string to append to the system prompt, or "" if nothing relevant.
    """
    try:
        from plugins.memory.vector_store import ChromaStore
        store = ChromaStore.get()
        facts = store.search(query, k=5, min_score=0.6)
        if not facts:
            return ""
        # Increment recall for retrieved facts
        for f in facts:
            store.increment_recall(f["id"])
        lines = [f["text"] for f in facts]
        # Token budget: cap at 300 tokens (~1200 chars)
        budget = _MAX_TOKENS * _APPROX_CHARS_PER_TOKEN
        selected = []
        used = 0
        for line in lines:
            if used + len(line) > budget:
                break
            selected.append(line)
            used += len(line)
        if not selected:
            return ""
        bullet_lines = "\n".join(f"- {l}" for l in selected)
        return f"Relevant things I know about you:\n{bullet_lines}"
    except Exception as exc:
        logger.warning("context_injector.build failed: %s", exc)
        return ""
