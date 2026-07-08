"""
prompt_suggester.py — Fire-and-forget follow-up suggestion after Aria answers.

After speaker.say(answer) in handle_command(), call suggest_async() to generate
and speak a single follow-up prompt in a daemon thread. Only activates for
browser_task, jobs, and web_search intents with answers >= 20 words.
"""

from __future__ import annotations

import logging
import threading

from aria.llm import llm_client

logger = logging.getLogger(__name__)

_TRIGGER_INTENTS = frozenset({"browser_task", "jobs", "web_search"})
_MIN_ANSWER_WORDS = 20

_SYSTEM_PROMPT = (
    "You are Aria, a sharp voice assistant. "
    "Given what you just did for the user, suggest ONE short follow-up action "
    "they might want next. Return a single spoken sentence starting with 'Also —'. "
    "Keep it under 15 words. If no obvious follow-up exists, return empty string."
)


def suggest(intent_type: str, answer: str) -> str:
    """Return a follow-up suggestion string, or '' if none applies. Never raises."""
    if intent_type not in _TRIGGER_INTENTS:
        return ""
    if len(answer.split()) < _MIN_ANSWER_WORDS:
        return ""
    try:
        resp = llm_client.complete(
            messages=[{"role": "user", "content": f"You just told the user: {answer}"}],
            tier="fast",
            system=_SYSTEM_PROMPT,
            max_tokens=60,
        )
        result = resp.text.strip()
        return result if result.lower().lstrip("*_ ").startswith("also") else ""
    except Exception as exc:
        logger.warning("prompt_suggester.suggest failed: %s", exc)
        return ""


def _worker(intent_type: str, answer: str, speaker) -> None:
    suggestion = suggest(intent_type, answer)
    if suggestion:
        try:
            speaker.say(suggestion)
        except Exception as exc:
            logger.warning("prompt_suggester: speaker.say failed: %s", exc)


def suggest_async(intent_type: str, answer: str, speaker) -> None:
    """Fire-and-forget: speak a follow-up suggestion in a daemon thread."""
    t = threading.Thread(
        target=_worker,
        args=(intent_type, answer, speaker),
        daemon=True,
        name="prompt-suggester",
    )
    t.start()
