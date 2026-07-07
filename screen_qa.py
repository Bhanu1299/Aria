"""
screen_qa.py — answer questions about what's currently on the user's screen.

answer(query) -> str

Silently captures the main display with macOS `screencapture`, resizes it
with `sips` to stay within Groq's image limits, then asks the Groq vision
model to answer the user's spoken question from the screenshot.

Never raises — all error paths return a spoken error string.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess

from groq import Groq

import config

logger = logging.getLogger(__name__)

_SCREENSHOT_PATH = "/tmp/aria_screen_qa.jpg"
_MAX_PX = "1920"            # max dimension for sips resize (keeps JPEG under ~500 KB)
_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_CLIENT: Groq | None = None

_SYSTEM = (
    "You are Aria, a voice assistant. "
    "You will be shown a screenshot of the user's current screen and their spoken question. "
    "Answer using only what is visible on the screen. "
    "If they ask about an error, explain it and suggest the likely fix. "
    "2 to 4 sentences maximum. "
    "Plain text only — no markdown, no bullet points, no dashes, no asterisks. "
    "The answer will be read aloud."
)


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def _capture(path: str = _SCREENSHOT_PATH) -> str | None:
    """Silently screenshot the main display and resize it. Returns path or None."""
    try:
        subprocess.run(
            ["screencapture", "-x", "-m", "-t", "jpg", path],
            capture_output=True,
            timeout=10,
        )
        if not os.path.exists(path):
            return None
        subprocess.run(
            ["sips", "-Z", _MAX_PX, path],
            capture_output=True,
            timeout=10,
        )
        return path
    except Exception as exc:
        logger.error("Screen capture failed: %s", exc)
        return None


def answer(query: str) -> str:
    """
    Capture the current screen and answer the user's question about it.

    Args:
        query: The user's original spoken question.

    Returns:
        Spoken answer string. Never raises.
    """
    shot = _capture()
    if shot is None:
        return (
            "I couldn't capture your screen. Make sure this app has "
            "Screen Recording permission in System Settings."
        )
    try:
        with open(shot, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = _get_client().chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The user asked: {query}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            max_tokens=300,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or "I looked at your screen but couldn't come up with an answer."
    except Exception as exc:
        logger.error("Screen QA vision call failed: %s", exc)
        return "I captured your screen but couldn't analyze it right now."
