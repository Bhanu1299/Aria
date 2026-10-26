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
# GateLLM — every LLM call in Aria goes through it (~/GateLLM, OpenAI-compatible).
# Provider API keys live in GateLLM's .env, not here.
# ---------------------------------------------------------------------------

GATEWAY_URL: str = os.getenv("ARIA_GATEWAY_URL", "http://127.0.0.1:8000")
# Aria starts GateLLM on the first LLM call that finds it down, and stops it
# after GATEWAY_IDLE_SECONDS without calls (aria/llm/autostart.py).
GATEWAY_AUTOSTART: bool = os.getenv("ARIA_GATEWAY_AUTOSTART", "1") != "0"
GATEWAY_DIR: str = os.path.expanduser(os.getenv("ARIA_GATEWAY_DIR", "~/GateLLM"))
GATEWAY_IDLE_SECONDS: int = int(os.getenv("ARIA_GATEWAY_IDLE_SECONDS", "600"))

# ---------------------------------------------------------------------------
# Google + job sheet. Two accounts (aria/integrations/google_auth.py):
# GOOGLE_ACCOUNT is the personal assistant account, GOOGLE_JOBS_ACCOUNT the job
# pipeline's (sheet, outreach drafts). Setup pre-selects and verifies each.
# Tab names are configurable because the sheet has other tabs Aria must not touch.
# ---------------------------------------------------------------------------

GOOGLE_ACCOUNT: str = os.getenv("GOOGLE_ACCOUNT", "")
GOOGLE_JOBS_ACCOUNT: str = os.getenv("GOOGLE_JOBS_ACCOUNT", "")
MUSE_SHEET_ID: str = os.getenv("MUSE_SHEET_ID", "")
MUSE_SHEET_TAB: str = os.getenv("MUSE_SHEET_TAB", "Sheet1")
MUSE_APPLIED_TAB: str = os.getenv("MUSE_APPLIED_TAB", "Applied")
MUSE_CONTACTS_TAB: str = os.getenv("MUSE_CONTACTS_TAB", "Contacts")

# ---------------------------------------------------------------------------
# Coldstart — the standalone email-finding service, reached over HTTP only
# (POST {COLDSTART_URL}/find). Aria never imports it for the Contacts flow.
# A local COLDSTART_URL that's down is started from COLDSTART_DIR's venv
# (`python -m coldstart serve`); COLDSTART_AUTOSTART=0 disables that.
# COLDSTART_MIN_INTERVAL: seconds between calls (rate limit).
# OUTREACH_FROM: the only address Contacts drafts may be created in
# (defaults to GOOGLE_JOBS_ACCOUNT; with neither set, no drafts are made).
# ---------------------------------------------------------------------------

COLDSTART_URL: str = os.getenv("COLDSTART_URL", "http://127.0.0.1:8765").rstrip("/")
COLDSTART_DIR: str = os.path.expanduser(os.getenv("COLDSTART_DIR", "~/Documents/trae_projects/coldstart"))
COLDSTART_AUTOSTART: bool = os.getenv("COLDSTART_AUTOSTART", "1") != "0"
COLDSTART_TIMEOUT: float = float(os.getenv("COLDSTART_TIMEOUT", "90"))
COLDSTART_MIN_INTERVAL: float = float(os.getenv("COLDSTART_MIN_INTERVAL", "2"))
CONTACTS_PER_COMPANY: int = int(os.getenv("CONTACTS_PER_COMPANY", "3"))
OUTREACH_FROM: str = os.getenv("OUTREACH_FROM", "") or GOOGLE_JOBS_ACCOUNT
# The PDF attached to every Contacts draft (blank = resume_path from the answer bank).
OUTREACH_RESUME: str = os.path.expanduser(os.getenv("OUTREACH_RESUME", ""))

# ---------------------------------------------------------------------------
# Coder — agentic code execution
# ---------------------------------------------------------------------------

PROJECTS_HOME: str = os.path.expanduser(os.getenv("PROJECTS_HOME", "~/Documents/trae_projects"))
CODER_BASH_TIMEOUT: int = int(os.getenv("CODER_BASH_TIMEOUT", "60"))
CODER_MAX_TOOL_CALLS: int = int(os.getenv("CODER_MAX_TOOL_CALLS", "50"))

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

    # pynput doesn't fail without Accessibility — the hotkey just never
    # fires. Ask macOS directly; with prompt=True it shows the system
    # dialog naming the responsible app (Aria.app or the terminal).
    try:
        from HIServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
        if not AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}):
            print(
                "\n[Aria] Accessibility not granted yet — the ⌥ Space hotkey won't work "
                "until it is (wake word still works).\n"
                "  System Settings → Privacy & Security → Accessibility → enable Aria "
                "(or your terminal app), then restart Aria.\n"
            )
    except Exception:
        pass

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
