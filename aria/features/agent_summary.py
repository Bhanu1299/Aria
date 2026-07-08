"""
agent_summary.py — Post-task spoken summary for Aria.

After every completed command, summarize_async() generates a natural 1-sentence
spoken recap via Claude API in a daemon thread, then speaks it.
Falls back to the raw answer on any error.
"""
from __future__ import annotations

import logging
import threading

from aria.llm import llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are Aria, a concise voice assistant. "
    "Given what you just did for the user, write ONE natural spoken sentence "
    "summarising the outcome. Start with 'Done —'. Max 20 words. "
    "Return ONLY the sentence."
)


def summarize(intent_type: str, raw_answer: str) -> str:
    """
    Call LLM synchronously and return a 1-sentence spoken summary.
    Returns raw_answer on any error or if raw_answer is empty.
    """
    if not raw_answer.strip():
        return raw_answer
    try:
        resp = llm_client.complete(
            messages=[
                {"role": "user", "content": f"Intent: {intent_type}\nResult: {raw_answer[:500]}"},
            ],
            tier="cheap",
            system=_SYSTEM_PROMPT,
            max_tokens=60,
        )
        result = resp.text.strip()
        return result if result else raw_answer
    except Exception as exc:
        logger.warning("agent_summary.summarize failed: %s", exc)
        return raw_answer


def summarize_async(intent_type: str, raw_answer: str, speaker) -> None:
    """Generate summary in a daemon thread and speak it. Never raises."""
    def _run():
        summary = summarize(intent_type, raw_answer)
        try:
            speaker.say(summary)
        except Exception as exc:
            logger.warning("agent_summary: speaker.say failed: %s", exc)

    t = threading.Thread(target=_run, daemon=True, name="agent-summary")
    t.start()
