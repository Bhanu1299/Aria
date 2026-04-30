"""
computer_use.py — Phase 3C: Coordinate-based browser automation for Aria

See → think → act loop using screenshot coordinates, exactly like Claude
computer use. No DOM selectors, no accessibility tree. Works on any website.

All Playwright calls are routed through agent_browser.run() so they execute
on the dedicated browser worker thread (Playwright sync_api requirement).

Public API:
  take_screenshot() → str | None       base64 JPEG, None on error
  decide(b64, goal, context_data, step, max_steps) → dict   next action
  execute(action) → None               run one action on the page
  run_loop(goal, context_data, max_steps) → (str, dict|None)  full loop
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
import re
import subprocess
import time
from pathlib import Path

from groq import Groq

import agent_browser
import config
import dom_browser

logger = logging.getLogger(__name__)

_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_SCREENSHOT_PATH = "/tmp/aria_cu.jpg"
_ACTION_SLEEP = 1.2  # base — actual sleep randomised via _human_sleep()

_CLIENT: Groq | None = None
_VALID_ACTIONS = {"click", "type", "scroll", "key", "confirm", "stuck", "needs_input"}
_VALID_GENERAL_ACTIONS = {"click", "click_text", "type", "scroll", "key", "navigate", "extract", "confirm", "needs_input", "done", "stuck"}
_VALID_DOM_FORM_ACTIONS = {"click", "click_text", "type", "scroll", "key", "confirm", "stuck", "needs_input"}
_VALID_DOM_RESEARCH_ACTIONS = _VALID_GENERAL_ACTIONS | {"click_text"}
_DOM_TEXT_MODEL = "llama-3.3-70b-versatile"

_CU_GENERAL_SYSTEM = """You control a Chrome browser at 1280x900 resolution.
Complete ANY browser task: research, shopping, messaging, form submission, booking, or anything else.

Given a screenshot, the goal, previously collected data, and action history, decide the SINGLE next action.

Return ONLY a JSON object — no markdown, no explanation:

{"action": "navigate",    "url": "https://example.com", "reason": "going to Amazon product page"}
{"action": "click",       "x": 640, "y": 450, "reason": "clicking Add to Cart button"}
{"action": "type",        "text": "AirPods Pro", "reason": "typing search query"}
{"action": "scroll",      "direction": "down", "amount": 400, "reason": "revealing more content"}
{"action": "key",         "key": "Enter", "reason": "submitting search"}
{"action": "extract",     "label": "product price", "value": "$249", "reason": "recording price for user"}
{"action": "confirm",     "summary": "AirPods Pro are in your cart for $249. Should I proceed to checkout?", "reason": "about to do something irreversible — asking user first"}
{"action": "needs_input", "field": "delivery address", "reason": "required field not in context"}
{"action": "done",        "summary": "Done. AirPods Pro added to your Amazon cart.", "reason": "task complete, nothing irreversible pending"}
{"action": "stuck",       "reason": "Login required and no session available"}

Coordinate rules:
- x: pixels from left edge, 0–1280. y: pixels from top edge, 0–900.
- Always click the CENTER of the target element.

Behaviour rules:
- Use "navigate" to jump to any URL directly.
- Use "extract" to record any data point worth remembering or reporting.
- Use "confirm" BEFORE any irreversible action: checkout, purchase, send message, submit form, delete. Put a clear human-readable summary of what is about to happen in "summary" — the user will hear this and say yes or no.
- Use "done" when the task is fully complete and nothing irreversible is pending. Put the final spoken result in "summary".
- Use "needs_input" when a required field has no answer in context — the user will provide it by voice.
- Use "stuck" only as a last resort (CAPTCHA, hard login wall, repeated failures). Always try alternatives first.
- Prefer navigating to pre-built URLs with search params over clicking through menus.
- Use scroll to reveal content before concluding something is absent.
- Do NOT re-extract data you already have. Do NOT re-navigate to a page you just left.
- If approaching the step limit with partial results, use "done" with what you have.
"""

_CU_SYSTEM = """You control a Chrome browser at 1280x900 resolution.
Given a screenshot of the current browser state and a task, decide the SINGLE next action to take.

Return ONLY a JSON object — no markdown, no explanation:

{"action": "click",   "x": 450, "y": 320, "reason": "clicking the First Name field"}
{"action": "type",    "text": "Bhanu Teja Veeramachaneni", "reason": "typing full name"}
{"action": "scroll",  "direction": "down", "amount": 400, "reason": "scrolling to reveal more fields"}
{"action": "key",     "key": "Tab", "reason": "advancing to next field"}
{"action": "confirm", "reason": "all required fields are filled, form is ready to submit"}
{"action": "stuck",   "reason": "CAPTCHA detected — cannot proceed"}
{"action": "needs_input", "field": "Years of experience", "reason": "required field is empty and not in context data"}

Coordinate rules:
- x: pixels from left edge, range 0–1280
- y: pixels from top edge, range 0–900
- Always click the CENTER of the target element
- Estimate coordinates by visual position in the screenshot

Behaviour rules:
- IMPORTANT: If a text field already contains data (a name, email, phone number, location, etc.), do NOT clear or re-type it. It is pre-filled. Skip it and move to the next empty field or click Next/Continue.
- If ALL visible fields are already filled, click the Next, Continue, or Review button to advance.
- If a required field is empty and the answer is in the Context data, click the field and type the value.
- If a required field is empty and the answer is NOT in the Context data, return "needs_input" with the field name — the user will provide the answer via voice.
- Never click a Submit or Apply button. Return "confirm" when the form is complete.
- After typing into a text field, use key:Tab to move to the next field.
- If a file upload field appears, scroll past it — do not interact with it.
- If a dropdown appears, click it first, then click the correct option.
- Use scroll when the page content is cut off before concluding an element is absent.
- If on a job description page (not yet an application form), click the Apply or Easy Apply button.
"""

_CU_DOM_SYSTEM = """You control a Chrome browser. You receive a DOM snapshot of the current page.
Complete form-filling tasks: fill required fields, advance through form pages, stop before submitting.

Return ONLY a JSON object — no markdown, no explanation:

{"action": "click",       "selector": "#submit-btn",                  "reason": "clicking Next"}
{"action": "click_text",  "text": "Continue",                         "reason": "clicking Continue button"}
{"action": "type",        "selector": "#first-name", "text": "Bhanu", "reason": "filling first name"}
{"action": "scroll",      "direction": "down", "amount": 400,         "reason": "revealing more fields"}
{"action": "key",         "key": "Tab",                               "reason": "advancing to next field"}
{"action": "confirm",     "reason": "all required fields filled, form ready to submit"}
{"action": "needs_input", "field": "Years of experience",             "reason": "required, not in context"}
{"action": "stuck",       "reason": "CAPTCHA detected — cannot proceed"}

Rules:
- Use selectors from the INTERACTIVE list. Prefer #id selectors.
- If a field already has a value, skip it and move to the next empty field.
- If all visible fields are filled, click Next/Continue to advance.
- Never click Submit or Apply — return "confirm" when the form is fully complete.
- Use "needs_input" when a required field is empty and not in Context data.
- Use "stuck" only as last resort (CAPTCHA, hard login wall, repeated failures).
"""

_CU_DOM_GENERAL_SYSTEM = """You control a Chrome browser. You receive a DOM snapshot of the current page.
Complete ANY browser task: research, shopping, messaging, form submission, booking, or anything else.

Return ONLY a JSON object — no markdown, no explanation:

{"action": "navigate",    "url": "https://amazon.com",                    "reason": "go to Amazon"}
{"action": "click",       "selector": "#add-to-cart-button",              "reason": "clicking Add to Cart"}
{"action": "click_text",  "text": "Add to Cart",                          "reason": "clicking by visible text"}
{"action": "type",        "selector": "#search", "text": "AirPods Pro",   "reason": "typing search query"}
{"action": "scroll",      "direction": "down", "amount": 400,             "reason": "revealing more content"}
{"action": "key",         "key": "Enter",                                 "reason": "submitting search"}
{"action": "extract",     "label": "price", "value": "$249",              "reason": "recording price"}
{"action": "confirm",     "summary": "About to add AirPods to cart...",   "reason": "irreversible action"}
{"action": "needs_input", "field": "delivery address",                    "reason": "required, not in context"}
{"action": "done",        "summary": "AirPods Pro added to cart.",        "reason": "task complete"}
{"action": "stuck",       "reason": "Login required, no session"}

Rules:
- Use selectors from the INTERACTIVE list. Prefer #id selectors.
- Use "navigate" to jump directly to URLs — faster than clicking menus.
- Use "extract" to record any data worth reporting to the user.
- Use "confirm" BEFORE any irreversible action (checkout, purchase, send, submit, delete).
- Use "done" when task is fully complete with a clear spoken summary.
- Use "needs_input" when a required field is missing from context.
- Use "stuck" only as last resort. Always try alternatives first.
- Do NOT re-extract data you already have. Do NOT re-navigate to a page you just left.
- If approaching step limit with partial results, use "done" with what you have.
"""


def _dom_decide(
    snapshot: str,
    goal: str,
    context_data: dict,
    step: int,
    max_steps: int,
    history: list[dict] | None = None,
) -> dict:
    """
    Groq text model decision for form-fill loop (DOM-first path).
    Never raises — returns {"action": "stuck"} on any error.
    """
    user_text = (
        f"Step {step} of {max_steps}.\n\n"
        f"Goal: {goal}\n\n"
        f"Context (use this data to fill fields):\n{json.dumps(context_data, indent=2)}\n\n"
        f"Current page DOM snapshot:\n{snapshot}"
    )
    if history:
        recent = history[-5:]
        lines = []
        for h in recent:
            desc = f"  Step {h['step']}: {h['action']}"
            if h.get("selector"): desc += f" selector={h['selector']!r}"
            if h.get("text"):     desc += f" text={h['text']!r}"
            if h.get("key"):      desc += f" key={h['key']}"
            if h.get("reason"):   desc += f" — {h['reason']}"
            lines.append(desc)
        user_text += (
            "\n\nPrevious actions already executed (do NOT repeat):\n"
            + "\n".join(lines)
        )
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_DOM_TEXT_MODEL,
            messages=[
                {"role": "system", "content": _CU_DOM_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = _extract_first_json(raw)
        parsed = json.loads(raw)
        if parsed.get("action") not in _VALID_DOM_FORM_ACTIONS:
            raise ValueError(f"Unknown action: {parsed.get('action')!r}")
        logger.info(
            "DOM step %d/%d  action=%r  selector=%s  text=%r  reason=%r",
            step, max_steps,
            parsed.get("action"),
            parsed.get("selector", "-"),
            parsed.get("text", ""),
            parsed.get("reason", ""),
        )
        return parsed
    except Exception as exc:
        logger.error("_dom_decide failed (step %d): %s", step, exc)
        return {"action": "stuck", "reason": f"LLM error: {exc}"}


def _dom_research_decide(
    snapshot: str,
    goal: str,
    step: int,
    max_steps: int,
    history: list[dict],
    collected_data: list[dict],
) -> dict:
    """
    Groq text model decision for research loop (DOM-first path).
    Never raises — returns {"action": "stuck"} on any error.
    """
    user_text = (
        f"Step {step} of {max_steps}.\n\n"
        f"Goal: {goal}\n\n"
        f"Data collected so far ({len(collected_data)} items):\n"
        f"{json.dumps(collected_data, indent=2)}\n\n"
        f"Current page DOM snapshot:\n{snapshot}"
    )
    if history:
        recent = history[-6:]
        lines = []
        for h in recent:
            desc = f"  Step {h['step']}: {h['action']}"
            if h.get("url"):      desc += f" url={h['url']!r}"
            if h.get("selector"): desc += f" selector={h['selector']!r}"
            if h.get("label"):    desc += f" label={h['label']!r} value={h.get('value', '')!r}"
            if h.get("text"):     desc += f" text={h['text']!r}"
            if h.get("key"):      desc += f" key={h['key']}"
            if h.get("reason"):   desc += f" — {h['reason']}"
            lines.append(desc)
        user_text += (
            "\n\nPrevious actions (do NOT repeat):\n" + "\n".join(lines)
            + "\n\nDo NOT re-extract data you already have. "
              "If you have all needed data, return 'done' with a complete summary."
        )
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_DOM_TEXT_MODEL,
            messages=[
                {"role": "system", "content": _CU_DOM_GENERAL_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = _extract_first_json(raw)
        parsed = json.loads(raw)
        if parsed.get("action") not in _VALID_DOM_RESEARCH_ACTIONS:
            raise ValueError(f"Unknown action: {parsed.get('action')!r}")
        logger.info(
            "DOM research step %d/%d  action=%r  selector=%s  reason=%r",
            step, max_steps,
            parsed.get("action"),
            parsed.get("selector", "-"),
            parsed.get("reason", ""),
        )
        return parsed
    except Exception as exc:
        logger.error("_dom_research_decide failed (step %d): %s", step, exc)
        return {"action": "stuck", "reason": f"LLM error: {exc}"}


def _human_sleep(base: float = _ACTION_SLEEP) -> None:
    """Sleep for base ± 40% to mimic human reaction time."""
    time.sleep(base * random.uniform(0.6, 1.4))


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def take_screenshot() -> str | None:
    """
    Screenshot the visible browser via the worker thread → base64.
    Keeps native 1280×900 so model coordinates map 1-to-1 to Playwright clicks.
    Returns None on any error.
    """
    def _do(page):
        img_bytes = page.screenshot(full_page=False)
        Path(_SCREENSHOT_PATH).write_bytes(img_bytes)
        return base64.b64encode(img_bytes).decode("utf-8")

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.error("take_screenshot failed: %s", exc)
        return None


def _extract_first_json(text: str) -> str:
    """Return the first complete {...} JSON object from text, ignoring trailing content."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return text  # fallback — let json.loads report the error


def decide(
    b64: str,
    goal: str,
    context_data: dict,
    step: int,
    max_steps: int,
    history: list[dict] | None = None,
) -> dict:
    """
    Single Groq Llama-4-Scout vision call → parsed action dict.
    Never raises — returns {"action": "stuck"} on any error.
    """
    user_text = (
        f"Step {step} of {max_steps}.\n\n"
        f"Goal: {goal}\n\n"
        f"Context (use this data to fill fields):\n{json.dumps(context_data, indent=2)}"
    )
    if history:
        recent = history[-5:]
        lines = []
        for h in recent:
            desc = f"  Step {h['step']}: {h['action']}"
            if h.get("text"):
                desc += f" text={h['text']!r}"
            if h.get("x") is not None:
                desc += f" at ({h['x']},{h['y']})"
            if h.get("key"):
                desc += f" key={h['key']}"
            if h.get("reason"):
                desc += f" — {h['reason']}"
            lines.append(desc)
        user_text += (
            "\n\nPrevious actions already executed (do NOT repeat):\n"
            + "\n".join(lines)
            + "\n\nIf you already typed a value into a field, do NOT type it again. "
            "Move to the next field with Tab or click a different element."
        )
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {"role": "system", "content": _CU_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        # Extract only the first complete JSON object — model sometimes appends extra text
        raw = _extract_first_json(raw)
        parsed = json.loads(raw)

        if parsed.get("action") not in _VALID_ACTIONS:
            raise ValueError(f"Unknown action: {parsed.get('action')!r}")

        logger.info(
            "Step %d/%d  action=%r  x=%s y=%s text=%r key=%r  reason=%r",
            step, max_steps,
            parsed.get("action"),
            parsed.get("x", "-"), parsed.get("y", "-"),
            parsed.get("text", ""),
            parsed.get("key", ""),
            parsed.get("reason", ""),
        )
        return parsed

    except Exception as exc:
        logger.error("decide failed (step %d): %s", step, exc)
        return {"action": "stuck", "reason": f"LLM error: {exc}"}


def _research_decide(
    b64: str,
    goal: str,
    step: int,
    max_steps: int,
    history: list[dict],
    collected_data: list[dict],
) -> dict:
    """
    Groq Llama-4-Scout vision call for research loop.
    Never raises — returns {"action": "stuck"} on error.
    """
    user_text = (
        f"Step {step} of {max_steps}.\n\n"
        f"Goal: {goal}\n\n"
        f"Data collected so far ({len(collected_data)} items):\n"
        f"{json.dumps(collected_data, indent=2)}"
    )
    if history:
        recent = history[-6:]
        lines = []
        for h in recent:
            desc = f"  Step {h['step']}: {h['action']}"
            if h.get("url"):      desc += f" url={h['url']!r}"
            if h.get("label"):    desc += f" label={h['label']!r} value={h.get('value','')!r}"
            if h.get("text"):     desc += f" text={h['text']!r}"
            if h.get("x") is not None: desc += f" at ({h['x']},{h['y']})"
            if h.get("key"):      desc += f" key={h['key']}"
            if h.get("reason"):   desc += f" — {h['reason']}"
            lines.append(desc)
        user_text += (
            "\n\nPrevious actions (do NOT repeat):\n" + "\n".join(lines)
            + "\n\nDo NOT re-extract data you already have. "
              "Do NOT re-navigate to a page you just left. "
              "If you have all the data needed, return 'done' with a complete summary."
        )
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {"role": "system", "content": _CU_GENERAL_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = _extract_first_json(raw)
        parsed = json.loads(raw)
        if parsed.get("action") not in _VALID_GENERAL_ACTIONS:
            raise ValueError(f"Unknown research action: {parsed.get('action')!r}")
        logger.info(
            "Groq research step %d/%d  action=%r  url=%s  label=%s  reason=%r",
            step, max_steps, parsed.get("action"),
            parsed.get("url", "-"), parsed.get("label", "-"), parsed.get("reason", ""),
        )
        return parsed
    except Exception as exc:
        logger.error("_research_decide failed (step %d): %s", step, exc)
        return {"action": "stuck", "reason": f"Groq error: {exc}"}


_CLAUDE_CLIENT = None


def _get_claude_client():
    global _CLAUDE_CLIENT
    if _CLAUDE_CLIENT is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — Claude fallback unavailable")
        import anthropic as _anthropic
        _CLAUDE_CLIENT = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _CLAUDE_CLIENT


def _claude_research_decide(
    b64: str,
    goal: str,
    step: int,
    max_steps: int,
    history: list[dict],
    collected_data: list[dict],
) -> dict:
    """
    Claude claude-sonnet-4-6 vision call for research fallback.
    Identical JSON contract to _research_decide().
    Never raises — returns {"action": "stuck"} on error.
    """
    user_text = (
        f"Step {step} of {max_steps}.\n\n"
        f"Goal: {goal}\n\n"
        f"Data collected so far ({len(collected_data)} items):\n"
        f"{json.dumps(collected_data, indent=2)}"
    )
    if history:
        recent = history[-6:]
        lines = []
        for h in recent:
            desc = f"  Step {h['step']}: {h['action']}"
            if h.get("url"):      desc += f" url={h['url']!r}"
            if h.get("label"):    desc += f" label={h['label']!r} value={h.get('value','')!r}"
            if h.get("text"):     desc += f" text={h['text']!r}"
            if h.get("x") is not None: desc += f" at ({h['x']},{h['y']})"
            if h.get("key"):      desc += f" key={h['key']}"
            if h.get("reason"):   desc += f" — {h['reason']}"
            lines.append(desc)
        user_text += (
            "\n\nPrevious actions (do NOT repeat):\n" + "\n".join(lines)
            + "\n\nIf you have all the data needed, return 'done' with a complete summary."
        )
    try:
        client = _get_claude_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=_CU_GENERAL_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                ],
            }],
        )
        raw = next(b.text for b in response.content if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = _extract_first_json(raw)
        parsed = json.loads(raw)
        if parsed.get("action") not in _VALID_GENERAL_ACTIONS:
            raise ValueError(f"Unknown action: {parsed.get('action')!r}")
        logger.info(
            "Claude research step %d/%d  action=%r  url=%s  label=%s  reason=%r",
            step, max_steps, parsed.get("action"),
            parsed.get("url", "-"), parsed.get("label", "-"), parsed.get("reason", ""),
        )
        return parsed
    except Exception as exc:
        logger.error("_claude_research_decide failed (step %d): %s", step, exc)
        return {"action": "stuck", "reason": f"Claude error: {exc}"}


def execute(action: dict) -> None:
    """
    Execute one action on the page via the browser worker thread.
    All Playwright calls happen inside agent_browser.run() on the correct thread.
    """
    act = action.get("action", "")

    def _do(page):
        if act == "click":
            if "selector" in action:
                page.locator(action["selector"]).first.click(timeout=3000)
            else:
                page.mouse.click(int(action.get("x", 0)), int(action.get("y", 0)))
        elif act == "click_text":
            _ct = action.get("text", "").replace('"', '\\"')
            page.locator(f'text="{_ct}"').first.click(timeout=3000)
        elif act == "type":
            if "selector" in action:
                page.locator(action["selector"]).first.fill(
                    action.get("text", ""), timeout=3000
                )
            else:
                # Type at cursor position — do NOT select-all first, as that
                # destroys pre-filled values in LinkedIn Easy Apply fields.
                # Per-character delay to mimic human keystroke timing.
                for char in action.get("text", ""):
                    page.keyboard.type(char)
                    time.sleep(random.uniform(0.03, 0.12))
        elif act == "scroll":
            direction = action.get("direction", "down")
            amount = int(action.get("amount", 400))
            page.mouse.wheel(0, amount if direction == "down" else -amount)
        elif act == "key":
            page.keyboard.press(action.get("key", "Tab"))
        elif act in ("confirm", "stuck", "navigate", "extract", "done", "needs_input"):
            pass  # caller handles these
        else:
            raise ValueError(f"Unknown action type: {act!r}")

    agent_browser.run(_do)


def _format_progress(action: dict) -> str:
    """Convert an action dict to a short human-readable progress string."""
    act = action.get("action", "")
    reason = action.get("reason", "")
    text = action.get("text", "")
    url = action.get("url", "")

    if act == "navigate" and url:
        parts = url.split("/")
        domain = parts[2] if len(parts) > 2 else url
        return f"Navigating to {domain}"
    if act == "type" and text:
        short = text[:30] + "..." if len(text) > 30 else text
        return f"Typing: {short}"
    if act == "click":
        return "Clicking" + (f": {reason[:40]}" if reason else "")
    if act == "scroll":
        return f"Scrolling {action.get('direction', 'down')}"
    if act == "search" and text:
        return f"Searching for {text[:40]}"
    if act == "extract":
        label = action.get("label", "data")
        return f"Extracting {label}"
    if act == "key":
        key = action.get("key", "key")
        return f"Pressing {key}"
    if act == "click_text":
        text = action.get("text", "")
        short = text[:30] + "..." if len(text) > 30 else text
        return f"Clicking: {short}"
    if reason:
        words = reason.split()[:10]
        return " ".join(words)
    return act.capitalize() if act else "Working..."


def run_loop(
    goal: str,
    context_data: dict,
    max_steps: int = 30,
    start_step: int = 1,
    progress_fn=None,
) -> tuple[str, dict | None]:
    """
    Run the see→think→act loop until done.

    Returns a tuple of (status, data):
        ("confirm",     None)           — form complete, ready for submission
        ("stuck",       None)           — cannot proceed
        ("needs_input", {"field": ...}) — empty field needs user voice input
        ("max_steps",   None)           — hit the step limit
    """
    history: list[dict] = []
    _recent: list[tuple] = []
    _recovered_once = False

    for step in range(start_step, max_steps + 1):
        snapshot, interactive_count = dom_browser.get_dom_snapshot()
        _vision_mode = interactive_count < 5

        if _vision_mode:
            b64 = take_screenshot()
            if b64 is None:
                logger.error("Screenshot failed at step %d — aborting loop", step)
                return ("stuck", None)
            dom_browser.save_debug_screenshot(f"step_{step:02d}")
            action = decide(b64, goal, context_data, step, max_steps, history=history)
        else:
            b64 = None
            action = _dom_decide(snapshot, goal, context_data, step, max_steps, history=history)

        if action["action"] == "confirm":
            return ("confirm", None)
        if action["action"] == "stuck":
            logger.warning("Loop stuck at step %d: %s", step, action.get("reason"))
            return ("stuck", None)
        if action["action"] == "needs_input":
            logger.info("Loop needs_input at step %d: field=%r", step, action.get("field"))
            return ("needs_input", {"field": action.get("field", "this field"), "step": step})

        # Detect repeated identical actions
        act_key = (
            action["action"],
            action.get("x"), action.get("y"),
            action.get("text"), action.get("key"),
        )
        _recent.append(act_key)
        if len(_recent) > 3:
            _recent.pop(0)
        if len(_recent) == 3 and len(set(_recent)) == 1:
            if not _recovered_once:
                # Try recovery: Tab to move focus, then continue
                logger.warning(
                    "Loop repeating %s — attempting recovery (Tab + scroll)", act_key
                )
                try:
                    execute({"action": "key", "key": "Tab"})
                    time.sleep(0.5)
                    execute({"action": "scroll", "direction": "down", "amount": 200})
                except Exception as exc:
                    logger.warning("Recovery action failed: %s", exc)
                _recent.clear()
                _recovered_once = True
                history.append({
                    "step": step, "action": "recovery",
                    "reason": "auto Tab+scroll to break repetition loop",
                })
                _human_sleep()
                continue
            else:
                logger.warning(
                    "Loop stuck after recovery: same action repeated 3 times %s — aborting", act_key
                )
                return ("stuck", None)

        # Record in history for the model to see next step
        history.append({
            "step": step,
            "action": action["action"],
            "selector": action.get("selector"),
            "x": action.get("x"),
            "y": action.get("y"),
            "text": action.get("text"),
            "key": action.get("key"),
            "reason": action.get("reason", ""),
        })

        if progress_fn is not None:
            try:
                progress_fn(_format_progress(action))
            except Exception:
                pass

        try:
            execute(action)
        except Exception as exc:
            logger.warning("Step %d execute error: %s — continuing", step, exc)

        _human_sleep()

        # Change detection — warn model if action had no visible effect
        if _vision_mode:
            post_b64 = take_screenshot()
            if post_b64 is not None:
                pre_hash = hashlib.md5(b64.encode()).hexdigest()
                post_hash = hashlib.md5(post_b64.encode()).hexdigest()
                if pre_hash == post_hash:
                    history.append({
                        "step": step,
                        "action": "no_change",
                        "reason": "page did not visibly change after last action — try a different approach",
                    })
        else:
            post_snapshot, _ = dom_browser.get_dom_snapshot()
            pre_interactive = snapshot.split("PAGE TEXT:")[0]
            post_interactive = post_snapshot.split("PAGE TEXT:")[0]
            if pre_interactive == post_interactive:
                history.append({
                    "step": step,
                    "action": "no_change",
                    "reason": "page did not change after last action — try a different approach",
                })

    logger.warning("run_loop hit max_steps=%d without confirm or stuck", max_steps)
    return ("max_steps", None)


_CLAUDE_MAX_STEPS = 20   # hard cap on Claude API steps per research task (cost control)


def research_loop(
    goal: str,
    context_data: dict | None = None,
    max_steps: int = 80,
    confirm_fn=None,
    input_fn=None,
    progress_fn=None,
) -> str:
    """
    General-purpose browser task loop — research, shopping, forms, messaging, anything.

    confirm_fn(summary: str) -> bool   called before irreversible actions; return True to proceed
    input_fn(field: str) -> str | None called when a required field is missing; return value or None

    Uses Groq Llama-4-Scout by default (free, fast).
    If Groq gets stuck twice in a row with zero collected data,
    switches to Claude claude-sonnet-4-6 for up to _CLAUDE_MAX_STEPS steps.
    Returns a plain spoken-answer string. Never raises.
    """
    collected_data: list[dict] = []
    history: list[dict] = []
    _recent: list[tuple] = []
    _recovered_once = False
    _groq_consecutive_stucks = 0
    _use_claude = False
    _claude_steps_used = 0

    for step in range(1, max_steps + 1):
        snapshot, interactive_count = dom_browser.get_dom_snapshot()
        _vision_mode = interactive_count < 5

        if _vision_mode or _use_claude:
            b64 = take_screenshot()
            if b64 is None:
                logger.error("research_loop screenshot failed at step %d", step)
                return "I ran into a browser error while researching that."
            dom_browser.save_debug_screenshot(f"research_step_{step:02d}")

        if _use_claude:
            if _claude_steps_used >= _CLAUDE_MAX_STEPS:
                logger.warning("research_loop hit Claude step cap (%d)", _CLAUDE_MAX_STEPS)
                if collected_data:
                    partial = "; ".join(f"{d['label']}: {d['value']}" for d in collected_data)
                    return f"I used my full research budget. Here's what I found: {partial}"
                return "I couldn't complete the research within the step budget."
            action = _claude_research_decide(b64, goal, step, max_steps, history, collected_data)
            _claude_steps_used += 1
        elif _vision_mode:
            action = _research_decide(b64, goal, step, max_steps, history, collected_data)
        else:
            action = _dom_research_decide(snapshot, goal, step, max_steps, history, collected_data)

        if action["action"] == "done":
            summary = action.get("summary", "").strip()
            if not summary:
                if collected_data:
                    summary = "Here's what I found: " + "; ".join(
                        f"{d['label']}: {d['value']}" for d in collected_data
                    )
                else:
                    summary = "I finished but didn't find anything to report."
            logger.info("research_loop done at step %d (claude=%s)", step, _use_claude)
            return summary

        if action["action"] == "confirm":
            summary = action.get("summary", "").strip() or "Ready to proceed. Should I continue?"
            logger.info("research_loop confirm at step %d: %s", step, summary)
            if confirm_fn is not None:
                approved = confirm_fn(summary)
                if approved:
                    history.append({"step": step, "action": "confirm_approved", "reason": "user said yes"})
                    continue
                else:
                    return "Got it, I stopped there. Nothing irreversible was done."
            # No confirm_fn — treat as done (non-interactive mode)
            return summary

        if action["action"] == "needs_input":
            field = action.get("field", "a required field")
            logger.info("research_loop needs_input at step %d: field=%r", step, field)
            if input_fn is not None:
                value = input_fn(field)
                if value:
                    history.append({"step": step, "action": "input_provided",
                                    "reason": f"user provided {field!r}: {value!r}"})
                    collected_data.append({"label": field, "value": value})
                    _human_sleep()
                    continue
            return f"I need {field} to continue but couldn't get it. Task paused."

        if action["action"] == "stuck":
            reason = action.get("reason", "unknown")
            logger.warning("research_loop stuck at step %d (claude=%s): %s", step, _use_claude, reason)

            if not _use_claude:
                _groq_consecutive_stucks += 1
                if _groq_consecutive_stucks >= 2 and not collected_data:
                    if config.ANTHROPIC_API_KEY:
                        logger.info("Switching to Claude fallback after %d Groq stucks", _groq_consecutive_stucks)
                        _use_claude = True
                        _groq_consecutive_stucks = 0
                        continue
                    else:
                        logger.warning("ANTHROPIC_API_KEY not set — cannot activate Claude fallback")
                elif collected_data:
                    partial = "; ".join(f"{d['label']}: {d['value']}" for d in collected_data)
                    return f"I got stuck partway through. Here's what I found: {partial}"
            else:
                if collected_data:
                    partial = "; ".join(f"{d['label']}: {d['value']}" for d in collected_data)
                    return f"I got stuck partway through. Here's what I found: {partial}"
                return f"I couldn't complete that research. The browser got stuck: {reason}"

            history.append({"step": step, "action": "stuck_skipped", "reason": reason})
            _human_sleep()
            continue

        _groq_consecutive_stucks = 0

        act_key = (
            action["action"],
            action.get("x"), action.get("y"),
            action.get("text"), action.get("key"),
            action.get("url"), action.get("label"),
        )
        _recent.append(act_key)
        if len(_recent) > 3:
            _recent.pop(0)
        if len(_recent) == 3 and len(set(_recent)) == 1:
            if not _recovered_once:
                logger.warning("research_loop repeating — scrolling to recover")
                try:
                    execute({"action": "scroll", "direction": "down", "amount": 300})
                except Exception as exc:
                    logger.warning("Recovery scroll failed: %s", exc)
                _recent.clear()
                _recovered_once = True
                history.append({"step": step, "action": "recovery",
                                 "reason": "auto-scroll to break repetition"})
                _human_sleep()
                continue
            else:
                logger.warning("research_loop stuck after recovery — aborting")
                if collected_data:
                    partial = "; ".join(f"{d['label']}: {d['value']}" for d in collected_data)
                    return f"I got stuck in a loop. Here's what I found before that: {partial}"
                return "I got stuck in a loop and couldn't complete the research."

        h_entry = {"step": step, "action": action["action"], "reason": action.get("reason", "")}
        if action["action"] == "navigate":     h_entry["url"] = action.get("url", "")
        elif action["action"] == "extract":    h_entry["label"] = action.get("label", ""); h_entry["value"] = action.get("value", "")
        elif action["action"] == "click":      h_entry["selector"] = action.get("selector"); h_entry["x"] = action.get("x"); h_entry["y"] = action.get("y")
        elif action["action"] == "click_text": h_entry["text"] = action.get("text", "")
        elif action["action"] == "type":       h_entry["selector"] = action.get("selector"); h_entry["text"] = action.get("text", "")
        elif action["action"] == "key":        h_entry["key"] = action.get("key", "")
        history.append(h_entry)

        if action["action"] == "navigate":
            if progress_fn is not None:
                try:
                    progress_fn(_format_progress(action))
                except Exception:
                    pass
            url = action.get("url", "").strip()
            if not url or not url.startswith(("http://", "https://")):
                logger.warning("research_loop navigate: invalid URL %r — skipping", url)
                history.append({"step": step, "action": "navigate_failed",
                                 "reason": f"invalid URL {url!r} — try a different approach"})
                _human_sleep()
                continue
            try:
                agent_browser.navigate(url, settle_secs=3.0)
            except Exception as exc:
                logger.warning("research_loop navigate %r failed: %s", url, exc)
                history.append({"step": step, "action": "navigate_failed",
                                 "reason": f"navigation error: {exc}"})
            _human_sleep()
            continue

        elif action["action"] == "extract":
            label = action.get("label", f"item_{len(collected_data)+1}").strip()
            value = str(action.get("value", "")).strip()
            if label and value:
                collected_data.append({"label": label, "value": value})
                logger.info("Extracted: %r = %r", label, value)
            else:
                logger.warning("extract action had empty label or value — skipping")
            _human_sleep()
            continue

        else:
            if progress_fn is not None:
                try:
                    progress_fn(_format_progress(action))
                except Exception:
                    pass
            try:
                execute(action)
            except Exception as exc:
                logger.warning("Step %d execute error: %s — continuing", step, exc)
            _human_sleep()

            if _vision_mode or _use_claude:
                post_b64 = take_screenshot()
                if post_b64 is not None:
                    pre_hash = hashlib.md5(b64.encode()).hexdigest()
                    post_hash = hashlib.md5(post_b64.encode()).hexdigest()
                    if pre_hash == post_hash:
                        history.append({
                            "step": step, "action": "no_change",
                            "reason": "page did not visibly change — try a different approach",
                        })
            else:
                post_snapshot, _ = dom_browser.get_dom_snapshot()
                pre_interactive = snapshot.split("PAGE TEXT:")[0]
                post_interactive = post_snapshot.split("PAGE TEXT:")[0]
                if pre_interactive == post_interactive:
                    history.append({
                        "step": step, "action": "no_change",
                        "reason": "page did not change after last action — try a different approach",
                    })

    logger.warning("research_loop hit max_steps=%d", max_steps)
    if collected_data:
        partial = "; ".join(f"{d['label']}: {d['value']}" for d in collected_data)
        return f"I ran out of steps before finishing. Here's what I found: {partial}"
    return "I ran out of steps and couldn't complete the research. Try a more specific request."
