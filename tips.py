"""
tips.py — Periodic voice tips for Aria.

Speaks one rotating tip every 10 commands to teach the user new capabilities.
Call maybe_speak_tip(command_count, speaker) after every completed command.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

TIPS: list[str] = [
    "Try saying: new project called my-app",
    "Try saying: write a Flask hello world",
    "Try saying: what's on Hacker News today",
    "Try saying: find me remote Python jobs posted this week",
    "Try saying: compare flights from Buffalo to New York next Friday",
    "Try saying: open YouTube and play lo-fi beats",
    "Try saying: summarize my LinkedIn feed",
    "Try saying: run the tests in my project",
    "Try saying: switch to my scraper project",
    "Try saying: what can you do",
]

_tip_index: int = 0


def maybe_speak_tip(command_count: int, speaker) -> None:
    """Speak the next tip if command_count is a multiple of 10."""
    global _tip_index
    if command_count % 10 != 0:
        return
    tip = TIPS[_tip_index % len(TIPS)]
    _tip_index += 1
    try:
        speaker.say(tip)
        logger.debug("tips: spoke tip %d: %r", _tip_index, tip)
    except Exception as exc:
        logger.warning("tips.maybe_speak_tip failed: %s", exc)
