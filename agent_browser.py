"""
agent_browser.py — Phase 3C: Singleton visible browser for Aria

One persistent Chromium window owned by a dedicated worker thread.
Playwright sync_api requires all calls to happen on the thread that created
the browser — so every operation is submitted to the worker via a queue.

Public API:
  navigate(url, settle_secs)  navigate to url on the browser thread
  run(fn)                     execute fn(page) on the browser thread, return result
  close()                     shut down the worker thread and close the browser
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from browser_profile import get_persistent_context, close_persistent_context

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

    logger.info("Opening visible browser window...")
    _context = get_persistent_context(headless=False)
    _page = _context.new_page()
    if _STEALTH:
        _stealth_sync(_page)
        logger.debug("Stealth patches applied to new page.")
    logger.info("Visible browser ready.")
    return _page


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

def run(fn):
    """
    Execute fn(page) on the browser worker thread and return the result.
    The browser is opened automatically if not already open.
    fn receives the active Playwright Page as its only argument.
    """
    def _do():
        page = _open_browser()
        return fn(page)
    return _submit(_do)


def navigate(url: str, settle_secs: float = 3.0, wait_until: str = "domcontentloaded") -> None:
    """
    Navigate the visible browser to url. Blocks until the page loads and
    the settle delay elapses. Opens the browser if not already open.
    """
    logger.info("Navigating to: %s", url)

    def _do():
        page = _open_browser()
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
