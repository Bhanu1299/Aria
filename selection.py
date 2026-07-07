"""
selection.py — read the text the user currently has highlighted.

get_selected_text() -> str   ("" when nothing is selected)

Two layers:
  Layer 1: Accessibility API — AXSelectedText from the focused UI element.
           Exact and instant, but some apps (Chrome pre-flag, Electron) don't expose it.
  Layer 2: clipboard trick — save clipboard, synthesize cmd+C, read pbpaste,
           restore the original clipboard. Works almost everywhere text is selectable.

Never raises — all error paths return "".
Requires the Accessibility permission Aria already needs for pynput.
"""

from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

_MAX_CHARS = 8000          # cap what we feed into the vision/LLM prompt
_COPY_SETTLE_SECS = 0.15   # time for the frontmost app to service cmd+C


def _ax_selected_text() -> str:
    """Layer 1 — AXSelectedText from the system-wide focused element."""
    import ApplicationServices as AS  # lazy: pyobjc import is slow

    system = AS.AXUIElementCreateSystemWide()
    err, focused = AS.AXUIElementCopyAttributeValue(
        system, "AXFocusedUIElement", None
    )
    if err != 0 or focused is None:
        return ""
    err, text = AS.AXUIElementCopyAttributeValue(focused, "AXSelectedText", None)
    if err != 0 or not text:
        return ""
    return str(text)


def _clipboard_selected_text() -> str:
    """Layer 2 — cmd+C into the clipboard, then restore what was there."""
    original = subprocess.run(
        ["pbpaste"], capture_output=True, timeout=5
    ).stdout
    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "c" using command down'],
        capture_output=True, timeout=5,
    )
    time.sleep(_COPY_SETTLE_SECS)
    captured = subprocess.run(
        ["pbpaste"], capture_output=True, timeout=5
    ).stdout
    # put the user's clipboard back exactly as it was
    subprocess.run(["pbcopy"], input=original, timeout=5)
    if captured == original:
        return ""  # cmd+C copied nothing new — nothing was selected
    return captured.decode("utf-8", errors="replace")


def get_selected_text() -> str:
    """Best-effort read of the currently highlighted text. Never raises."""
    try:
        text = _ax_selected_text()
    except Exception as exc:
        logger.debug("AX selection read failed: %s", exc)
        text = ""
    if not text:
        try:
            text = _clipboard_selected_text()
        except Exception as exc:
            logger.debug("Clipboard selection read failed: %s", exc)
            text = ""
    text = (text or "").strip()
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS]
    return text
