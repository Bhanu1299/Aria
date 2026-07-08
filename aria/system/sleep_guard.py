"""
sleep_guard.py — Keep Mac awake during long browser tasks.

Uses macOS `caffeinate -i -t 300` to prevent idle sleep.
Auto-restarts every 4 minutes (before the 5-min timeout expires).
Reference-counted so nested callers are safe: the caffeinate process
is only killed when the last release() fires.

Ported from claude-code-main/src/services/preventSleep.ts.
"""
from __future__ import annotations

import subprocess
import threading
from typing import Optional

# Exposed as a module-level constant so tests can monkeypatch it.
import sys as _sys
_PLATFORM: str = _sys.platform

# Caffeinate timeout passed via -t flag (seconds).  caffeinate auto-exits
# after this so orphaned processes don't persist if the parent is SIGKILL'd.
_CAFFEINATE_TIMEOUT_SECONDS: int = 300

# Restart the caffeinate process before it expires.
# 4 minutes gives comfortable headroom before the 5-minute timeout.
_RESTART_INTERVAL_SECONDS: int = 240  # 4 * 60


class SleepGuard:
    """
    Reference-counted sleep prevention using macOS caffeinate.

    Usage (typical — acquire at task start, release in finally):

        sleep_guard = SleepGuard()
        sleep_guard.acquire()
        try:
            do_long_task()
        finally:
            sleep_guard.release()

    Nested callers are safe: caffeinate is only killed when the outermost
    release() fires (ref count reaches zero).

    On non-darwin platforms, acquire() and release() are silent no-ops
    (ref count is still tracked for correctness).
    """

    def __init__(self) -> None:
        self._ref_count: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Increment ref count.  Spawns caffeinate on the first call."""
        with self._lock:
            self._ref_count += 1
            if self._ref_count == 1:
                self._spawn()
                self._schedule_restart()

    def release(self) -> None:
        """Decrement ref count.  Kills caffeinate when it reaches zero."""
        with self._lock:
            if self._ref_count <= 0:
                return
            self._ref_count -= 1
            if self._ref_count == 0:
                self._cancel_timer()
                self._kill()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn(self) -> None:
        """Spawn `caffeinate -i -t 300`.  No-op on non-darwin platforms."""
        if _PLATFORM != "darwin":
            return
        if self._process is not None:
            return  # already running
        try:
            self._process = subprocess.Popen(
                ["caffeinate", "-i", "-t", str(_CAFFEINATE_TIMEOUT_SECONDS)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            # caffeinate unavailable or spawn failed — degrade gracefully
            print(f"[SleepGuard] WARNING: failed to spawn caffeinate: {exc}")
            self._process = None

    def _kill(self) -> None:
        """Send SIGKILL to caffeinate for immediate termination."""
        proc = self._process
        self._process = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass  # process may have already exited

    def _schedule_restart(self) -> None:
        """Schedule a timer to restart caffeinate before it expires."""
        if _PLATFORM != "darwin":
            return
        if self._timer is not None:
            return
        self._timer = threading.Timer(
            _RESTART_INTERVAL_SECONDS,
            self._restart,
        )
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        """Cancel the pending restart timer."""
        timer = self._timer
        self._timer = None
        if timer is not None:
            timer.cancel()

    def _restart(self) -> None:
        """Kill the current caffeinate process and spawn a fresh one."""
        with self._lock:
            if self._ref_count == 0:
                return  # no longer needed
            self._timer = None   # timer has fired; clear before re-scheduling
            self._kill()
            self._spawn()
            self._schedule_restart()
