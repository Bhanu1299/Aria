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
import json
import logging
import os
import re
import subprocess

from groq import Groq

import config
import overlay
import selection

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


_EXPLAIN_SYSTEM = (
    "You are Aria, a voice assistant that can draw on the user's screen. "
    "You will be shown a screenshot of their current screen and their spoken question. "
    "Locate what they asked about and respond with STRICT JSON only, no markdown fences, "
    "in this exact shape:\n"
    '{"answer": "<2-3 spoken sentences, plain text>", '
    '"regions": [{"x": <int>, "y": <int>, "w": <int>, "h": <int>, "label": "<short label>"}]}\n'
    "Coordinates are normalized 0-1000 with (0,0) at the TOP-LEFT of the screenshot. "
    "Include 1-3 regions for the most relevant spots. "
    "If you cannot locate anything, return an empty regions list and explain in the answer."
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


# Queries that reference a selection — only these trigger the selection lookup,
# because the clipboard fallback synthesizes a cmd+C keystroke.
_SELECTION_HINT_RE = re.compile(
    r"\b(?:highlighted|selected|selection|this|that|it)\b", re.IGNORECASE
)


def _build_question(query: str) -> str:
    """Compose the text part of the vision prompt, with selection context if relevant."""
    text = f"The user asked: {query}"
    if not _SELECTION_HINT_RE.search(query):
        return text
    try:
        sel = selection.get_selected_text()
    except Exception as exc:
        logger.debug("Selection lookup failed: %s", exc)
        sel = ""
    if sel:
        text += (
            "\n\nThe user currently has this text highlighted on screen:\n"
            f'"""\n{sel}\n"""\n'
            "If the question is about the highlighted text, answer about it specifically."
        )
    return text


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
                        {"type": "text", "text": _build_question(query)},
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


def _strip_fences(raw: str) -> str:
    """Remove ```json ... ``` fences the model sometimes adds despite instructions."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def explain_visual(query: str) -> str:
    """
    Answer a "show me where / point out" question: speak the answer AND draw
    labeled boxes on the screen over the relevant spots.

    Returns the spoken answer string. Never raises; a failed overlay never
    blocks the spoken answer.
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
                {"role": "system", "content": _EXPLAIN_SYSTEM},
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
            max_tokens=500,
            temperature=0.2,
        )
        raw = _strip_fences((resp.choices[0].message.content or ""))
    except Exception as exc:
        logger.error("Screen explain vision call failed: %s", exc)
        return "I captured your screen but couldn't analyze it right now."

    try:
        data = json.loads(raw)
        answer_text = str(data.get("answer") or "").strip()
        regions = data.get("regions") or []
    except (json.JSONDecodeError, AttributeError):
        # model ignored the JSON contract — still give the user its answer
        return raw or "I looked at your screen but couldn't come up with an answer."

    if regions:
        try:
            w, h = overlay.screen_size()
            scaled = overlay.scale_normalized_regions(regions, w, h)
            if scaled:
                overlay.draw_boxes(scaled)
        except Exception as exc:
            logger.error("Overlay failed (answer still spoken): %s", exc)

    return answer_text or "I looked at your screen but couldn't come up with an answer."
