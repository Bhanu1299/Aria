"""
vision.py — Aria vision fallback (Phase 2B-3)

read_screen(url, query) -> str

Opens the URL in a background tab of Aria's own Chrome, screenshots that page (not the
user's screen, and without opening their browser or taking focus), then sends it to the
vision model to extract the answer to the user's query.

Never raises — all error paths return a spoken error string.
"""

from __future__ import annotations
from aria.llm import gateway_client
from aria.llm.tasks import tier_for

import base64
import logging
from pathlib import Path



logger = logging.getLogger(__name__)

_SCREENSHOT_PATH = "/tmp/aria_screen.jpg"
_SETTLE_SECS = 4.0          # seconds for the page to render before the screenshot
VISION_TAB = "vision"       # Aria's Chrome tab these lookups reuse
_MODEL = tier_for("page_vision")  # aria/llm/tasks.py


_VISION_SYSTEM = (
    "You are Aria, a voice assistant. "
    "You will be shown a screenshot of a webpage and told what the user asked. "
    "Answer the question using only what is visible on the screen. "
    "2 to 4 sentences maximum. "
    "Plain text only — no markdown, no bullet points, no dashes, no asterisks. "
    "The answer will be read aloud."
)


def _get_client():
    """GateLLM client — every LLM call in Aria goes through the gateway."""
    return gateway_client()


def read_screen(url: str, query: str) -> str:
    """
    Open url in a background tab of Aria's Chrome, screenshot the page, and extract an
    answer to query with the vision model.

    Args:
        url:   Page to open and capture.
        query: The user's original spoken question — used as context for extraction.

    Returns:
        A spoken answer string. Never raises.
    """
    try:
        return _read_screen_impl(url, query)
    except Exception as exc:
        logger.error("Vision fallback failed: %s", exc)
        return "I opened the page in my browser, but couldn't read it."


def _read_screen_impl(url: str, query: str) -> str:
    import aria.web.agent_browser as agent_browser
    agent_browser.navigate(url, settle_secs=_SETTLE_SECS, tab=VISION_TAB)
    shot = agent_browser.run(lambda page: page.screenshot(full_page=False, type="jpeg", quality=75), tab=VISION_TAB)
    Path(_SCREENSHOT_PATH).write_bytes(shot)

    img_bytes = Path(_SCREENSHOT_PATH).read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    logger.info("Vision screenshot: %d bytes after resize", len(img_bytes))

    # Step 6 — send to Groq vision model
    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        task="page_vision",
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
