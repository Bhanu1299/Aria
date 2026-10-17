"""system/single_instance.py — refuse to start a second Aria.

Two Arias means two hotkey listeners, two wake engines fighting over the
mic, and every answer spoken twice. An flock on ~/.aria/aria.lock is held
for the life of the process; the OS releases it on exit or crash, so a
stale lock can never block startup.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

ALREADY_RUNNING_EXIT = 3  # Aria.app's launcher maps this to "already running"

_LOCK_PATH = Path.home() / ".aria" / "aria.lock"
_held = None  # keep the file object alive — closing it drops the lock


def acquire(path: Path | None = None) -> bool:
    """True if this process now holds the Aria lock, False if another Aria does."""
    global _held
    path = path or _LOCK_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        _held = fh
        return True
    except Exception:
        return True  # lock trouble must never stop Aria from starting
