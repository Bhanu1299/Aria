"""
vision.py — Aria vision fallback (Phase 2B-3)

read_screen(url, query) -> str

Opens a URL in the real browser, waits for it to load, takes a silent
screenshot, resizes it to stay within Groq's image limits, then sends it
to the Groq vision model to extract the answer to the user's query.

Never raises — all error paths return a spoken error string.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import time
from pathlib import Path

from groq import Groq

import config

logger = logging.getLogger(__name__)

_SCREENSHOT_PATH = "/tmp/aria_screen.jpg"
_SETTLE_SECS = 8.0          # seconds to wait after `open url` before screenshotting
_MAX_PX = "1920"            # max dimension for sips resize (keeps JPEG under ~500 KB)
_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_CLIENT: Groq | None = None

_VISION_SYSTEM = (
    "You are Aria, a voice assistant. "
    "You will be shown a screenshot of a webpage and told what the user asked. "
    "Answer the question using only what is visible on the screen. "
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


def read_screen(url: str, query: str) -> str:
    """
    Open url in the real browser, screenshot, and extract an answer to query
    using a Groq vision model.

    Args:
        url:   Page to open and capture.
        query: The user's original spoken question — used as context for extraction.

    Returns:
        A spoken answer string. Never raises.
    """
    try:
        return _read_screen_impl(url, query)
    except subprocess.CalledProcessError as exc:
        cmd = " ".join(exc.cmd) if isinstance(exc.cmd, list) else str(exc.cmd)
        logger.error("Vision subprocess failed (%s): %s", cmd, exc)
        return "I opened the page in your browser but couldn't take a screenshot."
    except FileNotFoundError as exc:
        logger.error("Vision tool not found: %s", exc)
        return "I opened the page in your browser but the screenshot tool was not available."
    except Exception as exc:
        logger.error("Vision fallback failed: %s", exc)
        return "I opened the page in your browser, but couldn't read the results. Please check your browser."


def _read_screen_impl(url: str, query: str) -> str:
    """Inner implementation — may raise; caller wraps with error handling."""

    # Step 1 — open in real browser (brings browser to front)
    logger.info("Vision: opening %s in real browser", url)
    subprocess.run(["open", url], check=True, capture_output=True, text=True)

    # Step 2 — wait for page to load
    time.sleep(_SETTLE_SECS)

    # Step 3 — screenshot (silent: -x suppresses the shutter sound)
    subprocess.run(
        ["screencapture", "-x", "-t", "jpg", _SCREENSHOT_PATH],
        check=True,
        capture_output=True,
        text=True,
    )

    # Step 4 — resize to max _MAX_PX to stay within Groq's ~4 MB image limit
    # sips is a built-in macOS image-processing tool; -Z scales largest dimension
    subprocess.run(
        ["sips", "-Z", _MAX_PX, _SCREENSHOT_PATH],
        capture_output=True,   # suppress sips progress output
    )

    # Step 5 — read resized image as base64
    img_bytes = Path(_SCREENSHOT_PATH).read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    logger.info("Vision screenshot: %d bytes after resize", len(img_bytes))

    # Step 6 — send to Groq vision model
    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f'The user asked: "{query}"\n'
                            "This is a screenshot of the webpage they were directed to. "
                            "Extract only the information relevant to the query. "
                            "Be concise. Return plain text only — no markdown, no bullet "
                            "symbols — this will be spoken aloud."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    },
                ],
            },
        ],
        temperature=0.2,
        max_tokens=300,
    )

    answer = response.choices[0].message.content.strip()
    logger.debug("Vision answer: %s", answer)
    return answer
