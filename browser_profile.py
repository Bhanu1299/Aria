"""
browser_profile.py — Persistent Playwright browser profile for Aria

Stores an authenticated Chromium profile at ~/.aria/browser_profile so
logged-in sessions (Gmail, LinkedIn, etc.) survive between runs.

Public API:
  get_persistent_context(headless=True) → BrowserContext
  login_session(url)                    → interactive login flow
"""

from __future__ import annotations

import logging
from pathlib import Path

from patchright.sync_api import sync_playwright, BrowserContext

logger = logging.getLogger(__name__)

PROFILE_DIR = str(Path.home() / ".aria" / "browser_profile")


def get_persistent_context(headless: bool = True) -> BrowserContext:
    """
    Launch Chromium with a persistent profile stored on disk.

    Args:
        headless: True for background operation, False for interactive login.

    Returns:
        A Playwright BrowserContext backed by PROFILE_DIR.
        Caller is responsible for closing the context when done.
    """
    Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)

    pw = sync_playwright().start()
    context = pw.chromium.launch_persistent_context(
        PROFILE_DIR,
        channel="chrome",
        headless=headless,
        args=[
            "--no-first-run",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
        ],
        viewport={"width": 1280, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    # Stash pw on the context so callers can stop it after closing
    context._aria_pw = pw  # type: ignore[attr-defined]
    return context


def close_persistent_context(context: BrowserContext) -> None:
    """Close a persistent context and its underlying Playwright instance."""
    try:
        context.close()
    except Exception as exc:
        logger.warning("Error closing persistent context: %s", exc)
    pw = getattr(context, "_aria_pw", None)
    if pw is not None:
        try:
            pw.stop()
        except Exception as exc:
            logger.warning("Error stopping Playwright: %s", exc)


def login_session(url: str) -> None:
    """
    Open a visible browser window so the user can log in manually.

    The persistent profile saves cookies/session to disk automatically.
    After the user presses Enter, the browser closes and future headless
    calls to get_persistent_context() reuse the saved session.

    Args:
        url: The login page URL (e.g. "https://mail.google.com").
    """
    print(f"Log in to {url} in the browser window, then press Enter here when done.")
    context = get_persistent_context(headless=False)
    page = context.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    except Exception as exc:
        logger.error("Failed to load %s: %s", url, exc)
        print(f"Warning: page load issue — {exc}")

    input("Press Enter after you have logged in...")
    print("Session saved. Aria will reuse your login automatically.")
    close_persistent_context(context)
