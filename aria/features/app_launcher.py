"""
app_launcher.py — Aria app intent executor

open_app(app_name, contact=None) → spoken result string

Uses macOS `open -a` to launch applications by name.
Never raises — all errors are captured and returned as spoken strings.
"""

from __future__ import annotations

import subprocess
import logging

logger = logging.getLogger(__name__)

_MESSAGING_APPS = {"whatsapp", "signal", "telegram", "messages", "imessage"}


def open_app(app_name: str, contact: str | None = None) -> str:
    """
    Open a macOS application by name.

    Args:
        app_name: Application name as it appears in /Applications
                  (e.g. "Safari", "FaceTime", "Spotify").
        contact:  Optional contact name — only meaningful for FaceTime/Messages.

    Returns:
        A spoken response string. Never raises.
    """
    if not app_name or not app_name.strip():
        return "I'm not sure which app to open. Could you say the app name again?"

    app_name = app_name.strip()

    if app_name.lower() == "facetime" and contact:
        error = _launch_app(app_name)
        if error:
            return error
        return f"Opening FaceTime for {contact}. You will need to place the call manually."

    if app_name.lower() in _MESSAGING_APPS and contact:
        error = _launch_app(app_name)
        if error:
            return error
        return f"Opening {app_name} for {contact}. You will need to send the message manually."

    error = _launch_app(app_name)
    if error:
        return error
    return f"Opening {app_name}."


def _launch_app(app_name: str) -> str | None:
    """
    Run `open -a app_name`.

    Returns:
        None on success, or a spoken error string on failure.
    """
    try:
        subprocess.run(
            ["open", "-a", app_name],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Opened app: %s", app_name)
        return None
    except FileNotFoundError:
        logger.error("'open' command not found — not running on macOS?")
        return f"I couldn't open {app_name} — this feature requires macOS."
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        logger.error("App not found or failed to open: %s — %s", app_name, stderr)
        return f"I couldn't find {app_name} on your Mac."
    except Exception as exc:
        logger.error("Unexpected error opening %s: %s", app_name, exc)
        return f"Something went wrong while trying to open {app_name}."
