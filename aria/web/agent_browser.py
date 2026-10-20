"""
agent_browser.py — Phase 3C: Singleton visible browser for Aria

One persistent Chromium window owned by a dedicated worker thread.
Playwright sync_api requires all calls to happen on the thread that created
the browser — so every operation is submitted to the worker via a queue.

Public API:
  navigate(url, settle_secs, tab=None)  navigate to url on the browser thread
  run(fn, tab=None)                     execute fn(page) on the browser thread, return result
  close()                               shut down the worker thread and close the browser

tab: None is the shared page every tool uses. A name (e.g. "apply") gets its own
tab in the same window, so a form left open for review survives other browsing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from aria.web.browser_profile import get_persistent_context, close_persistent_context

logger = logging.getLogger(__name__)

try:
    from playwright_stealth import stealth_sync as _stealth_sync
    _STEALTH = True
except ImportError:
    _STEALTH = False
    logger.warning("playwright-stealth not installed — bot detection reduction disabled")

# ---------------------------------------------------------------------------
# Internal state — owned exclusively by the worker thread
# ---------------------------------------------------------------------------
_work_queue: queue.Queue = queue.Queue()
_thread: threading.Thread | None = None
_thread_lock = threading.Lock()

# Playwright objects live only on the worker thread
_context = None
_page = None
_tabs: dict = {}      # name → Page, for work that must not be navigated away


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def _open_browser():
    """Open the visible browser. Must be called on the worker thread."""
    global _context, _page
    if _context is not None and _page is not None:
        try:
            _ = _page.url   # liveness check
            return _page
        except Exception:
            logger.warning("Page is dead — reopening browser")
            try:
                _context.close()
            except Exception:
                pass
            _context = None
            _page = None
            _tabs.clear()

    logger.info("Opening visible browser window...")
    _context = get_persistent_context(headless=False)
    _page = _context.new_page()
    if _STEALTH:
        _stealth_sync(_page)
        logger.debug("Stealth patches applied to new page.")
    logger.info("Visible browser ready.")
    return _page


def _get_page(tab):
    """The shared page (tab=None) or a named tab, opened on first use. Worker thread only."""
    page = _open_browser()
    if tab is None:
        return page
    named = _tabs.get(tab)
    if named is not None:
        try:
            if not named.is_closed():
                return named
        except Exception:
            pass
    named = _context.new_page()
    if _STEALTH:
        _stealth_sync(named)
    _tabs[tab] = named
    return named


def _worker():
    """Persistent thread that owns all Playwright state."""
    global _context, _page
    while True:
        item = _work_queue.get()
        if item is None:
            # Shutdown signal
            if _context is not None:
                try:
                    close_persistent_context(_context)
                except Exception as exc:
                    logger.warning("Error closing browser on shutdown: %s", exc)
            _context = None
            _page = None
            _tabs.clear()
            break

        fn, result_queue = item
        try:
            result = fn()
            result_queue.put(("ok", result))
        except Exception as exc:
            result_queue.put(("err", exc))


def _ensure_thread() -> None:
    """Start the worker thread if it isn't running."""
    global _thread
    with _thread_lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(
                target=_worker, daemon=True, name="aria-browser-worker"
            )
            _thread.start()
            logger.debug("Browser worker thread started.")


def _submit(fn):
    """Submit fn to the worker thread and block until it returns. Raises on error."""
    _ensure_thread()
    result_queue: queue.Queue = queue.Queue()
    _work_queue.put((fn, result_queue))
    status, value = result_queue.get()
    if status == "err":
        raise value
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(fn, tab=None):
    """
    Execute fn(page) on the browser worker thread and return the result.
    The browser is opened automatically if not already open.
    fn receives the Playwright Page (shared, or the named tab) as its only argument.
    """
    def _do():
        return fn(_get_page(tab))
    return _submit(_do)


def navigate(url: str, settle_secs: float = 3.0, wait_until: str = "domcontentloaded", tab=None) -> None:
    """
    Navigate the visible browser (or a named tab) to url. Blocks until the page
    loads and the settle delay elapses. Opens the browser if not already open.
    """
    logger.info("Navigating%s to: %s", f" [{tab}]" if tab else "", url)

    def _do():
        page = _get_page(tab)
        try:
            page.goto(url, wait_until=wait_until, timeout=30_000)
        except Exception as exc:
            logger.warning("Navigation to %s raised: %s (continuing)", url, exc)
        time.sleep(settle_secs)

    _submit(_do)


def close() -> None:
    """Shut down the browser worker thread and close the browser window."""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            logger.info("Closing visible browser.")
            _work_queue.put(None)   # shutdown signal
            _thread.join(timeout=5)
        _thread = None
