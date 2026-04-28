# Phase 2B-2 — generic headless fetcher + Google-first URL extractor + goto()
# Executor pattern: future executors must implement fetch(url) -> str
"""
browser.py — Aria BrowserExecutor

Headless Chromium that never steals focus. Supports two operations:
  fetch(url)         → navigate + extract readable page text (str)
  extract_links(url) → navigate + return top external URLs from the page (list[str])

Module-level helper:
  goto(url)          → open url in the user's default browser via macOS `open`

Thread safety: all Playwright calls run on _browser_thread.
fetch() and extract_links() are safe to call from any thread.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
import logging
from pathlib import Path

from playwright.sync_api import (
    sync_playwright,
    Page,
    BrowserContext,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)

import config

logger = logging.getLogger(__name__)

PROFILE_DIR = str(Path(__file__).parent / ".aria_profile")


# ---------------------------------------------------------------------------
# Garbage detection — identifies content not worth summarizing
# ---------------------------------------------------------------------------

_GARBAGE_SIGNALS = frozenset({
    "sign in",
    "log in",
    "login",
    "enable javascript",
    "javascript is required",
    "access denied",
    "403 forbidden",
    "404 not found",
    "page not found",
    "error 403",
    "error 404",
})


def is_garbage(text: str) -> bool:
    """
    Return True when fetched content is too thin or auth-walled to be useful.

    Criteria:
      - Empty or whitespace only
      - Shorter than 100 characters after stripping
      - Short page (<300 chars) that contains auth-wall / error signals
        (long pages like a logged-in Gmail inbox may mention "sign in"
         incidentally — that does not mean the page is an auth wall)
    """
    stripped = text.strip()
    if not stripped or len(stripped) < 100:
        return True
    # Only reject on auth-wall signals when the page is thin.
    # A real auth wall is a short redirect/challenge page, not a
    # content-rich page that happens to contain "sign in" somewhere.
    if len(stripped) < 300:
        lower = stripped.lower()
        if any(signal in lower for signal in _GARBAGE_SIGNALS):
            return True
    return False


# ---------------------------------------------------------------------------
# Module-level helper — open a URL in the default browser (no headless)
# ---------------------------------------------------------------------------

def goto(url: str) -> None:
    """Open url in the user's default browser using macOS `open`.

    This does NOT use the headless Playwright browser — it delegates to the
    OS so the user's real browser opens the page.

    Args:
        url: Fully-qualified URL to open (must start with http/https).

    Raises:
        RuntimeError: If the `open` command fails (e.g. url is malformed).
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"goto() requires an http/https URL, got: {url!r}")
    try:
        subprocess.run(["open", url], check=True, capture_output=True, text=True)
        logger.info("Opened in default browser: %s", url)
    except subprocess.CalledProcessError as exc:
        logger.error("open failed for %s: %s", url, exc.stderr.strip())
        raise RuntimeError(f"Failed to open {url}: {exc.stderr.strip()}") from exc
    except FileNotFoundError as exc:
        logger.error("'open' command not found — not running on macOS?")
        raise RuntimeError("'open' command not found — macOS required.") from exc

# ---------------------------------------------------------------------------
# JavaScript — extract readable body text, strip boilerplate
# ---------------------------------------------------------------------------

_EXTRACT_TEXT_JS = """
() => {
    const noisy = [
        'script', 'style', 'nav', 'footer', 'header', 'aside',
        'noscript', 'iframe', '[aria-hidden="true"]',
        '.ad', '.ads', '.advertisement', '#cookie-banner', '#cookie-notice'
    ];
    noisy.forEach(sel => {
        try { document.querySelectorAll(sel).forEach(el => el.remove()); } catch (e) {}
    });
    const main = document.querySelector('article')
                 || document.querySelector('main')
                 || document.querySelector('[role="main"]')
                 || document.body;
    if (!main) return '';
    return main.innerText.replace(/\\s{2,}/g, ' ').trim().substring(0, 8000);
}
"""

# ---------------------------------------------------------------------------
# JavaScript — extract top external URLs from a page (Google-result-aware)
# ---------------------------------------------------------------------------

_EXTRACT_LINKS_JS = """
() => {
    const excluded = [
        'google.', 'gstatic.', 'googleapis.', 'webcache.googleusercontent',
        'accounts.google', 'support.google', 'maps.google',
        'translate.google', 'policies.google', 'chrome.google'
    ];
    const seen = new Set();
    const urls = [];

    for (const a of document.querySelectorAll('a[href]')) {
        let href = a.href;
        if (!href) continue;

        // Decode Google redirect URLs (/url?q=https://target.com&...)
        if (href.includes('/url?') || href.includes('google.com/url')) {
            try {
                const target = new URL(href).searchParams.get('q');
                if (target && target.startsWith('http')) {
                    href = target;
                } else {
                    continue;
                }
            } catch (e) { continue; }
        }

        if (!href.startsWith('http')) continue;
        if (excluded.some(e => href.includes(e))) continue;
        if (href.includes('/search?') && href.includes('google')) continue;
        if (seen.has(href)) continue;

        seen.add(href);
        urls.push(href);
        if (urls.length >= 5) break;
    }
    return urls;
}
"""

# ---------------------------------------------------------------------------
# Authenticated fetch — uses persistent browser profile
# ---------------------------------------------------------------------------

def fetch_authenticated(url: str, query: str) -> str | None:
    """
    Fetch a page using the persistent Playwright profile (with saved logins).

    Uses the same text extraction logic as BrowserExecutor._fetch_impl.
    Falls back to None if content is garbage (auth-walled, too thin, etc.).

    Args:
        url:   Page URL to fetch.
        query: The user's query (used for logging only here).

    Returns:
        Extracted page text, or None if garbage/error.
    """
    import time as _time
    from browser_profile import get_persistent_context, close_persistent_context

    context = None
    try:
        context = get_persistent_context(headless=True)
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=config.BROWSER_TIMEOUT * 1_000)
        # Wait for main content area to appear (Gmail, etc. are heavy SPAs)
        try:
            page.wait_for_selector('[role="main"]', timeout=5_000)
        except Exception:
            pass  # selector may not exist on all pages — fall through
        _time.sleep(3.0)  # extra time for JS to finish rendering

        text = page.evaluate(_EXTRACT_TEXT_JS)
        text = text or ""
        logger.info("fetch_authenticated: %d chars from %s", len(text), url)

        if is_garbage(text):
            logger.warning("fetch_authenticated: garbage content from %s", url)
            return None
        # Large login pages pass is_garbage (they're >300 chars) but are still auth walls.
        # If the page has BOTH a sign-in prompt AND a join/register prompt, the persistent
        # profile session isn't active — fall through to vision.
        _lower = text.lower()
        if (any(s in _lower for s in ("sign in", "log in", "login")) and
                any(s in _lower for s in ("join now", "sign up", "create account", "register", "get started"))):
            logger.warning("fetch_authenticated: session not active for %s — falling back to vision", url)
            return None
        return text

    except Exception as exc:
        logger.error("fetch_authenticated failed for %s: %s", url, exc)
        return None
    finally:
        if context is not None:
            close_persistent_context(context)


# Sentinel used to signal the browser thread to shut down.
_STOP = object()

# Queue item operations
_OP_FETCH = "fetch"
_OP_LINKS = "links"


class BrowserExecutor:
    """Headless Chromium browser for fetching web page text and extracting links.

    All Playwright operations run on _browser_thread.
    fetch() and extract_links() are safe to call from any thread.
    """

    def __init__(self) -> None:
        self._cmd_queue: queue.Queue = queue.Queue()
        self._browser_thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._start_error: Exception | None = None

        # Touched only from _browser_thread:
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------
    # Public API — safe to call from any thread
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the browser thread and wait until Chromium is ready."""
        self._browser_thread = threading.Thread(
            target=self._thread_main, daemon=True, name="aria-browser"
        )
        self._browser_thread.start()
        self._ready_event.wait(timeout=60)
        if self._start_error is not None:
            raise self._start_error

    def fetch(self, url: str) -> str | None:
        """Thread-safe: navigate to url and return extracted readable page text.

        Returns:
            Page text (str) on success, or None when content is garbage / auth-walled.
        """
        result_q: queue.Queue = queue.Queue()
        self._cmd_queue.put((_OP_FETCH, url, result_q))
        try:
            return result_q.get(timeout=config.BROWSER_TIMEOUT + 10)
        except queue.Empty:
            logger.error("Browser fetch timed out for: %s", url)
            return None

    def extract_links(self, url: str) -> list[str]:
        """Thread-safe: navigate to url and return top external URLs found on the page."""
        result_q: queue.Queue = queue.Queue()
        self._cmd_queue.put((_OP_LINKS, url, result_q))
        try:
            result = result_q.get(timeout=config.BROWSER_TIMEOUT + 10)
            return result if isinstance(result, list) else []
        except queue.Empty:
            logger.error("Browser extract_links timed out for: %s", url)
            return []

    def stop(self) -> None:
        """Signal the browser thread to shut down and wait for it."""
        self._cmd_queue.put(_STOP)
        if self._browser_thread is not None:
            self._browser_thread.join(timeout=15)

    # ------------------------------------------------------------------
    # Browser thread — all Playwright calls live here
    # ------------------------------------------------------------------

    def _thread_main(self) -> None:
        try:
            self._start_browser()
            self._ready_event.set()
        except Exception as exc:
            self._start_error = exc
            self._ready_event.set()
            return

        while True:
            try:
                item = self._cmd_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            op, url, result_q = item
            try:
                if op == _OP_FETCH:
                    result = self._fetch_impl(url)
                elif op == _OP_LINKS:
                    result = self._extract_links_impl(url)
                else:
                    logger.error("Unknown browser op: %s", op)
                    result = None if op == _OP_FETCH else []
            except Exception as exc:
                logger.error("Unexpected browser error (%s %s): %s", op, url, exc)
                result = None if op == _OP_FETCH else []
            result_q.put(result)

        self._stop_browser()

    def _start_browser(self) -> None:
        logger.info("Starting headless Chromium (profile: %s)...", PROFILE_DIR)
        self._playwright = sync_playwright().start()

        self._context = self._playwright.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=True,
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

        self._page = self._context.new_page()
        logger.info("Browser ready.")

    def _stop_browser(self) -> None:
        try:
            if self._context:
                self._context.close()
        except Exception as exc:
            logger.warning("Error closing browser context: %s", exc)
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception as exc:
            logger.warning("Error stopping Playwright: %s", exc)
        finally:
            self._context = None
            self._page = None
            self._playwright = None

    # ------------------------------------------------------------------
    # Core operations (run on browser thread only)
    # ------------------------------------------------------------------

    def _navigate(self, url: str, settle_secs: float = 1.5) -> bool:
        """Navigate to url and wait for DOM. Returns True on success."""
        if self._page is None:
            raise RuntimeError("Browser not started.")
        try:
            self._page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=config.BROWSER_TIMEOUT * 1_000,
            )
            time.sleep(settle_secs)
            return True
        except PlaywrightTimeoutError:
            logger.error("Page load timed out: %s", url)
            return False
        except Exception as exc:
            logger.error("Page load failed for %s: %s", url, exc)
            return False

    def _fetch_impl(self, url: str) -> str | None:
        """Navigate to url and return extracted readable text, or None if garbage."""
        if not self._navigate(url, settle_secs=1.5):
            return None
        try:
            text = self._page.evaluate(_EXTRACT_TEXT_JS)  # type: ignore[union-attr]
            text = text or ""
            logger.info("Fetched %d chars from %s", len(text), url)
            if is_garbage(text):
                logger.warning("Garbage content detected for %s — triggering fallback", url)
                return None
            return text
        except Exception as exc:
            logger.error("Text extraction failed for %s: %s", url, exc)
            return None

    def _extract_links_impl(self, url: str) -> list[str]:
        """Navigate to url and return top external links (Google-result-aware)."""
        if not self._navigate(url, settle_secs=2.0):
            return []
        try:
            links = self._page.evaluate(_EXTRACT_LINKS_JS)  # type: ignore[union-attr]
            logger.info("Extracted %d links from %s", len(links or []), url)
            return links or []
        except Exception as exc:
            logger.error("Link extraction failed for %s: %s", url, exc)
            return []
