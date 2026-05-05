"""
prevent_sleep.py — Keep macOS awake during long Aria tasks.

Uses caffeinate -i (idle sleep prevention) with a 5-minute timeout that
restarts every 4 minutes. Reference-counted so multiple callers are safe.
No-op on non-macOS platforms.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

_TIMEOUT_SECS = 300       # caffeinate auto-exits after 5 min
_RESTART_INTERVAL = 240   # restart every 4 min before timeout

_process: subprocess.Popen | None = None
_ref_count: int = 0
_timer: threading.Timer | None = None
_lock = threading.Lock()


def start() -> None:
    """Increment ref count and spawn caffeinate if not already running."""
    global _ref_count
    with _lock:
        _ref_count += 1
        if _ref_count == 1:
            _spawn()


def stop() -> None:
    """Decrement ref count and kill caffeinate when it reaches zero."""
    global _ref_count
    with _lock:
        if _ref_count > 0:
            _ref_count -= 1
        if _ref_count == 0:
            _kill()


def _spawn() -> None:
    global _process, _timer
    if sys.platform != "darwin":
        return
    try:
        _process = subprocess.Popen(
            ["caffeinate", "-i", "-t", str(_TIMEOUT_SECS)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.debug("prevent_sleep: caffeinate started (pid %d)", _process.pid)
        _timer = threading.Timer(_RESTART_INTERVAL, _restart)
        _timer.daemon = True
        _timer.start()
    except Exception as exc:
        logger.warning("prevent_sleep: failed to start caffeinate: %s", exc)


def _restart() -> None:
    with _lock:
        if _ref_count > 0:
            _kill_process()
            _spawn()


def _kill() -> None:
    global _timer
    if _timer is not None:
        _timer.cancel()
        _timer = None
    _kill_process()


def _kill_process() -> None:
    global _process
    if _process is not None:
        try:
            _process.kill()
            logger.debug("prevent_sleep: caffeinate stopped")
        except Exception:
            pass
        _process = None
