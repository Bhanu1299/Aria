"""
config.py — Aria configuration loader.

Reads .env via python-dotenv, exposes typed constants, auto-detects location
via ip-api.com on startup, and provides check_permissions().
"""

from __future__ import annotations

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hotkey
# ---------------------------------------------------------------------------

HOTKEY_KEY: str = os.getenv("HOTKEY_KEY", "space")

_raw_mods: str = os.getenv("HOTKEY_MODS", "alt")
HOTKEY_MODS: list[str] = [m.strip() for m in _raw_mods.split(",") if m.strip()]

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

BROWSER_TIMEOUT: int = int(os.getenv("BROWSER_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Anthropic (Claude fallback for complex browser research)
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Location — auto-detected at startup, never hardcoded
# ---------------------------------------------------------------------------

def _detect_location() -> tuple[str, str]:
    """Call ip-api.com to detect current city/region and timezone.

    Returns (location_str, timezone_str).
    Falls back to ("Unknown Location", "UTC") on any error without crashing.
    """
    try:
        import requests
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        data = resp.json()
        if data.get("status") == "success":
            city = data.get("city", "")
            region = data.get("regionName", "")
            timezone = data.get("timezone", "UTC")
            if city and region:
                location = f"{city}, {region}"
            elif city:
                location = city
            elif region:
                location = region
            else:
                location = "Unknown Location"
            return location, timezone
    except Exception as exc:
        logger.warning("Location detection failed: %s", exc)
    return "Unknown Location", "UTC"


CURRENT_LOCATION, CURRENT_TIMEZONE = _detect_location()
print(f"Aria detected location: {CURRENT_LOCATION}")


# ---------------------------------------------------------------------------
# Permission checks
# ---------------------------------------------------------------------------

def check_permissions() -> bool:
    """
    Check that Aria has the macOS permissions it needs to run.

    Checks:
      1. Accessibility — required by pynput to listen for global hotkeys.
      2. Microphone   — required by sounddevice to capture audio.

    Prints actionable guidance for any missing permission.
    Returns True only when both checks pass.
    """
    all_ok = True

    # ------------------------------------------------------------------
    # 1. Accessibility (pynput global key listener)
    # ------------------------------------------------------------------
    try:
        from pynput import keyboard as _kb  # noqa: F401

        _listener = _kb.Listener(on_press=None)
        _listener.start()
        _listener.stop()
    except Exception as exc:
        err = str(exc).lower()
        if "accessibility" in err or "axobserver" in err or "permission" in err:
            print(
                "\n[Aria] Accessibility permission required.\n"
                "  Go to: System Settings → Privacy & Security → Accessibility\n"
                "         → click '+' and add Terminal (or your terminal app).\n"
                f"  (underlying error: {exc})\n"
            )
        all_ok = False

    # ------------------------------------------------------------------
    # 2. Microphone (sounddevice device enumeration)
    # ------------------------------------------------------------------
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if not input_devices:
            raise RuntimeError("No input audio devices found.")
    except Exception as exc:
        err = str(exc).lower()
        if "permission" in err or "no input" in err or "coreaudio" in err:
            print(
                "\n[Aria] Microphone permission required.\n"
                "  Go to: System Settings → Privacy & Security → Microphone\n"
                "         → click '+' and add Terminal (or your terminal app).\n"
                f"  (underlying error: {exc})\n"
            )
        all_ok = False

    if all_ok:
        print("[Aria] All permissions OK.")

    return all_ok


if __name__ == "__main__":
    print(f"HOTKEY_KEY      = {HOTKEY_KEY!r}")
    print(f"HOTKEY_MODS     = {HOTKEY_MODS!r}")
    print(f"BROWSER_TIMEOUT = {BROWSER_TIMEOUT}")
    print(f"CURRENT_LOCATION= {CURRENT_LOCATION!r}")
    print(f"CURRENT_TIMEZONE= {CURRENT_TIMEZONE!r}")
    print()
    check_permissions()
