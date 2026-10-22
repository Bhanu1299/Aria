"""
agent_browser.py — Phase 3C: Singleton visible browser for Aria

One visible Chrome window, driven over a local debug port by a dedicated worker
thread. Playwright sync_api requires all calls to happen on the thread that created
the connection — so every operation is submitted to the worker via a queue.

Chrome is launched detached (browser_profile.launch_chrome), so quitting Aria only
disconnects: the window and every tab stay open, and the next start re-attaches.

Public API:
  navigate(url, settle_secs, tab=None)  navigate to url on the browser thread
  run(fn, tab=None)                     execute fn(page) on the browser thread, return result
  use_tab(name, gate=None)              context manager: this thread's tab=None calls go to
                                        `name`; gate() runs before each of them (pause/stop)
  attach({name: url})                   bind names to already-open tabs by host + path
  open_in_tab(url)                      "open X": a new background tab, returns its name
  has_tab / close_tab / front_tabs(names)
  close()                               disconnect (Chrome keeps running) and stop the worker

tab: None is the shared page every tool uses. A name (e.g. "apply-15") gets its own
tab in the same window, so a form left open for review survives other browsing.

Aria's Chrome never takes focus: new tabs open in the background, nothing raises the
window, and macOS never jumps out of a full-screen app — even for "open X" requests.
The window is there to look at whenever the user clicks it.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import queue
import re
import subprocess
import threading
import time
from typing import Callable, Iterator, Optional
from urllib.parse import urlparse

from aria.web.browser_profile import connect_cdp, disconnect_cdp, launch_chrome

logger = logging.getLogger(__name__)

try:
    from playwright_stealth import stealth_sync as _stealth_sync
    _STEALTH = True
except ImportError:
    _STEALTH = False
    logger.warning("playwright-stealth not installed — bot detection reduction disabled")

VIEWPORT = {"width": 1280, "height": 900}

# ---------------------------------------------------------------------------
# Internal state — owned exclusively by the worker thread
# ---------------------------------------------------------------------------
_work_queue: queue.Queue = queue.Queue()
_thread: Optional[threading.Thread] = None
_thread_lock = threading.Lock()

# Playwright objects live only on the worker thread
_browser = None
_context = None
_page = None
_tabs: dict = {}      # name → Page, for work that must not be navigated away
_link_seq = itertools.count(1)   # open_in_tab names: link-1, link-2, …

_local = threading.local()   # caller-thread defaults set by use_tab()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def _prepare(page):
    if _STEALTH:
        _stealth_sync(page)
    try:
        page.set_viewport_size(VIEWPORT)   # vision coordinates assume 1280×900
    except Exception as exc:
        logger.debug("viewport not set: %s", exc)
    return page


def _connected() -> bool:
    try:
        return _browser is not None and _context is not None and _browser.is_connected()
    except Exception:
        return False


def _connect() -> None:
    """Connect to (launching if needed) the visible Chrome. Worker thread only. A dead
    connection is dropped and rebuilt; the caller re-binds tabs (session.reattach)."""
    global _browser, _context, _page
    if _connected():
        return
    if _browser is not None:
        logger.warning("Browser connection is dead — reconnecting")
        disconnect_cdp(_browser)
    _browser = _context = _page = None
    _tabs.clear()
    logger.info("Connecting to the visible Chrome window...")
    launch_chrome()
    _browser, _context = connect_cdp()
    logger.info("Visible browser ready.")


def _open_browser():
    """The shared page, (re)opened on demand — closing it never drops the named tabs."""
    global _page
    _connect()
    try:
        if _page is not None and not _page.is_closed():
            return _page
    except Exception:
        pass
    blank = next((pg for pg in _context.pages if pg.url == "about:blank" and not pg.is_closed()
                  and all(pg is not t for t in _tabs.values())), None)
    _page = _prepare(blank or _new_tab())
    return _page


def _get_page(tab):
    """The shared page (tab=None) or a named tab, opened on first use. Worker thread only."""
    if tab is None:
        return _open_browser()
    _connect()
    named = _tabs.get(tab)
    if named is not None:
        try:
            if not named.is_closed():
                return named
        except Exception:
            pass
    named = _prepare(_new_tab())
    _tabs[tab] = named
    return named


def _new_tab():
    """A tab opened in the background: Chrome doesn't switch to it or raise its window
    (Playwright's new_page() does both). Falls back to new_page() if CDP refuses."""
    try:
        cdp = _browser.new_browser_cdp_session()
        try:
            with _context.expect_page(timeout=5000) as info:
                cdp.send("Target.createTarget", {"url": "about:blank", "background": True})
            return info.value
        finally:
            cdp.detach()
    except Exception as exc:
        logger.debug("background tab failed (%s); opening a normal one", exc)
        return _context.new_page()


def _live(name: str):
    page = _tabs.get(name)
    try:
        return page if page is not None and not page.is_closed() else None
    except Exception:
        return None


def _worker():
    """Persistent thread that owns all Playwright state."""
    global _browser, _context, _page
    while True:
        item = _work_queue.get()
        if item is None:
            # Shutdown signal: disconnect only — Chrome and its tabs stay open.
            if _browser is not None:
                disconnect_cdp(_browser)
            _browser = _context = _page = None
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
    gate = getattr(_local, "gate", None)
    if gate is not None:
        gate()
    _ensure_thread()
    result_queue: queue.Queue = queue.Queue()
    _work_queue.put((fn, result_queue))
    status, value = result_queue.get()
    if status == "err":
        raise value
    return value


def _tab(tab):
    return tab if tab is not None else getattr(_local, "tab", None)


def _same_page(a: str, b: str) -> bool:
    x, y = urlparse(a or ""), urlparse(b or "")
    return bool(x.hostname) and (x.hostname, x.path.rstrip("/")) == (y.hostname, y.path.rstrip("/"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(fn, tab=None):
    """
    Execute fn(page) on the browser worker thread and return the result.
    The browser is opened automatically if not already open.
    fn receives the Playwright Page (shared, or the named tab) as its only argument.
    """
    name = _tab(tab)

    def _do():
        return fn(_get_page(name))
    return _submit(_do)


def navigate(url: str, settle_secs: float = 3.0, wait_until: str = "domcontentloaded", tab=None) -> None:
    """
    Navigate the visible browser (or a named tab) to url. Blocks until the page
    loads and the settle delay elapses. Opens the browser if not already open.
    """
    name = _tab(tab)
    logger.info("Navigating%s to: %s", f" [{name}]" if name else "", url)

    def _do():
        page = _get_page(name)
        try:
            page.goto(url, wait_until=wait_until, timeout=30_000)
        except Exception as exc:
            logger.warning("Navigation to %s raised: %s (continuing)", url, exc)

    _submit(_do)
    if settle_secs:
        time.sleep(settle_secs)   # off the worker, so other tabs aren't held up


@contextlib.contextmanager
def use_tab(name: Optional[str], gate: Optional[Callable[[], None]] = None) -> Iterator[None]:
    """On this thread, tab=None calls go to `name`, and gate() runs before each call."""
    prev = (getattr(_local, "tab", None), getattr(_local, "gate", None))
    _local.tab, _local.gate = name, gate
    try:
        yield
    finally:
        _local.tab, _local.gate = prev


def current_gate() -> Optional[Callable[[], None]]:
    """This thread's gate (from an enclosing use_tab), to keep it in a nested use_tab."""
    return getattr(_local, "gate", None)


def attach(wanted: dict) -> set:
    """Bind each name to an open tab showing the same host + path. Returns the names bound."""
    def _do():
        _connect()
        bound = set()
        taken = {id(p) for p in _tabs.values()} | ({id(_page)} if _page is not None else set())
        for name, url in wanted.items():
            if _live(name) is not None:
                bound.add(name)
                continue
            for page in list(_context.pages):
                if id(page) in taken:
                    continue
                try:
                    if not page.is_closed() and _same_page(page.url, url):
                        _tabs[name] = page
                        taken.add(id(page))
                        bound.add(name)
                        break
                except Exception:
                    continue
        return bound
    return _submit(_do)


def has_tab(name: str) -> bool:
    return _submit(lambda: _live(name) is not None)


def close_tab(name: str) -> bool:
    """Close one named tab (only on the user's words). False if it wasn't open."""
    def _do():
        page = _live(name)
        _tabs.pop(name, None)
        if page is None:
            return False
        page.close()
        return True
    return _submit(_do)


def open_in_tab(url: str, settle_secs: float = 0.0) -> str:
    """The user asked to open a page: a new background tab in Aria's Chrome (never their
    default browser, never in front). Returns the tab's name."""
    name = f"link-{next(_link_seq)}"
    navigate(url, settle_secs=settle_secs, tab=name)
    return name


def strip_window_title(title: str) -> str:
    """macOS window title → the active tab's page title ("<title> - Google Chrome[ - profile]")."""
    return re.sub(r" - Google Chrome(?: - .*)?$", "", (title or "").strip())


def _front_window_title() -> Optional[str]:
    """Title of the frontmost window of Aria's Chrome (by its pid, so the user's own Chrome
    is never read). None if it can't be read (no Accessibility permission, no window)."""
    from aria.web.browser_profile import chrome_pid
    pid = chrome_pid()
    if pid is None:
        return None
    try:
        out = subprocess.run(
            ["osascript", "-e", f'tell application "System Events" to get name of front window of '
                                f'(first process whose unix id is {pid})'],
            capture_output=True, text=True, timeout=5)
        return strip_window_title(out.stdout) if out.returncode == 0 and out.stdout.strip() else None
    except Exception as exc:
        logger.debug("front window title unreadable: %s", exc)
        return None


def front_tabs(names: list) -> list:
    """The names (of those given) whose tab is the active tab of Chrome's front window.
    document.visibilityState can't tell: Playwright's focus emulation makes every tab
    'visible'. The window title is the active tab's title; ties and unreadable → []."""
    title = _front_window_title()
    if not title:
        return []

    def _do():
        out = []
        for name in names:
            page = _live(name)
            try:
                if page is not None and page.title().strip() == title:
                    out.append(name)
            except Exception:
                continue
        return out
    return _submit(_do)


def close() -> None:
    """Disconnect from Chrome (it keeps running, tabs included) and stop the worker."""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            logger.info("Disconnecting from the visible browser.")
            _work_queue.put(None)   # shutdown signal
            _thread.join(timeout=5)
        _thread = None
