"""
browser_profile.py — Persistent Playwright browser profile for Aria

Stores an authenticated Chromium profile at ~/.aria/browser_profile so
logged-in sessions (Gmail, LinkedIn, etc.) survive between runs.

Public API:
  get_persistent_context(headless=True) → BrowserContext
  login_session(url)                    → interactive login flow
  launch_chrome(port)                   → start Aria's visible Chrome, detached, with a
                                          local debug port (it outlives Aria)
  connect_cdp(port)                     → (browser, context) over that port
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from patchright.sync_api import sync_playwright, BrowserContext

logger = logging.getLogger(__name__)

PROFILE_DIR = str(Path.home() / ".aria" / "browser_profile")
CDP_PORT = 9333
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
_CHROME_LOG = Path.home() / ".aria" / "logs" / "chrome.log"
_ARGS = [
    "--no-first-run",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--no-default-browser-check",
    "--disable-blink-features=AutomationControlled",
]


def cdp_up(port: int = CDP_PORT) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/json/version", timeout=1.0).status_code == 200
    except Exception:
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def chrome_pid() -> Optional[int]:
    """Pid of the Chrome holding Aria's profile (from its SingletonLock), or None."""
    try:
        pid = int(os.readlink(Path(PROFILE_DIR) / "SingletonLock").rsplit("-", 1)[-1])
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _clear_stale_lock(profile: Path) -> None:
    """Chrome's SingletonLock is a symlink to "<host>-<pid>". After a crash or power loss
    it outlives Chrome; remove it (and its siblings) when that pid is gone."""
    lock = profile / "SingletonLock"
    try:
        pid = int(os.readlink(lock).rsplit("-", 1)[-1])
    except (OSError, ValueError):
        return
    if _pid_alive(pid):
        return
    logger.info("Removing a stale Chrome profile lock (pid %d is gone)", pid)
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        try:
            os.unlink(profile / name)
        except OSError:
            pass


def launch_chrome(port: int = CDP_PORT, wait_secs: float = 15.0) -> None:
    """Start Chrome on Aria's profile in its own session (so quitting Aria leaves it and its
    tabs open), debug port bound to 127.0.0.1. No-op if it's already up. Raises RuntimeError."""
    if cdp_up(port):
        return
    profile = Path(PROFILE_DIR)
    _clear_stale_lock(profile)
    if os.path.lexists(profile / "SingletonLock"):
        # Chrome holds the profile but answers no debug port: an old Playwright-launched
        # window, or a crash leftover. Give it a moment in case it's still starting.
        deadline = time.time() + wait_secs
        while time.time() < deadline:
            if cdp_up(port):
                return
            time.sleep(0.2)
        raise RuntimeError("Chrome is already open on Aria's profile without remote debugging — "
                           "quit that Chrome window and try again.")
    profile.mkdir(parents=True, exist_ok=True)
    _CHROME_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _CHROME_LOG.open("ab") as log:
        subprocess.Popen(
            [CHROME_BIN, f"--user-data-dir={profile}", f"--remote-debugging-port={port}",
             "--remote-debugging-address=127.0.0.1", "--window-size=1280,900", *_ARGS],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
        )
    deadline = time.time() + wait_secs
    while time.time() < deadline:
        if cdp_up(port):
            logger.info("Chrome up on debug port %d", port)
            return
        time.sleep(0.3)
    raise RuntimeError(f"Chrome didn't open its debug port within {wait_secs:.0f} s.")


def connect_cdp(port: int = CDP_PORT):
    """(browser, context) for the running Chrome. Closing `browser` only disconnects."""
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    except Exception:
        pw.stop()
        raise
    browser._aria_pw = pw  # type: ignore[attr-defined]
    context = browser.contexts[0] if browser.contexts else browser.new_context()
    return browser, context


def disconnect_cdp(browser) -> None:
    """Drop the connection; Chrome and every tab stay open."""
    try:
        browser.close()
    except Exception as exc:
        logger.warning("Error disconnecting from Chrome: %s", exc)
    pw = getattr(browser, "_aria_pw", None)
    if pw is not None:
        try:
            pw.stop()
        except Exception as exc:
            logger.warning("Error stopping Playwright: %s", exc)


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
