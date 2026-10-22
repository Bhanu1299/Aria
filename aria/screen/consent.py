"""screen/consent.py — Aria looks at the user's screen only when the user asks about it.

Everything else Aria does happens inside its own Chrome (agent_browser), the way a
browser agent works on the tabs it was given. The dispatcher records the words of the
command being handled (set_request); every whole-screen capture (screen_qa, mac_controller's
screen read, a saved screenshot) checks asked() first, so no tool and no LLM choice can
read the screen on a request that never mentioned it. Live log 2026-10-22: a garbled
wake-word transcript ("What's going on? What happened?") led the agent to screenshot the
screen and describe the user's game.
"""
from __future__ import annotations

import re

# The user's words must point at the screen itself, not at a web page or the browser.
_ASKED = re.compile(
    r"(?i)\b(?:screens?|display|monitor|desktop|screenshots?|screen ?shots?|"
    r"what (?:am i|i'?m) (?:looking at|watching|seeing|reading)|"
    r"(?:this|that|the) (?:window|error|dialog|popup|pop-up)|on my mac|"
    r"highlighted|selected(?: text)?|selection|"
    r"show me where|point (?:out|to|at)|circle (?:it|that|where|the))\b")

REFUSAL = "I only look at your screen when you ask me to. Say \"what's on my screen\" and I'll look."

_request = ""


def set_request(text: str) -> None:
    """Called by the dispatcher with the user's words, once per command (follow-ups too)."""
    global _request
    _request = text or ""


def asked(text: str = None) -> bool:
    """True if the command being handled (or `text`) explicitly asks about the screen."""
    return bool(_ASKED.search(_request if text is None else text))
