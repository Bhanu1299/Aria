"""applying/control.py — pause / resume / stop for the application run in progress.

checkpoint() runs before every browser call the run thread makes (agent_browser.use_tab
gate): it blocks while paused and raises Stopped once stopped. `current` is the run's
control while one is active, else None.
"""
from __future__ import annotations

import threading
from typing import Optional


class Stopped(Exception):
    """The user said stop: end the run, leave every tab open."""


class RunControl:
    def __init__(self) -> None:
        self._go = threading.Event()
        self._go.set()
        self._stopped = threading.Event()
        self.active = True

    @property
    def paused(self) -> bool:
        return not self._go.is_set() and not self._stopped.is_set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def pause(self) -> None:
        self._go.clear()

    def resume(self) -> None:
        self._go.set()

    def stop(self) -> None:
        self._stopped.set()
        self._go.set()

    def checkpoint(self) -> None:
        self._go.wait()
        if self._stopped.is_set():
            raise Stopped()

    def wait(self, seconds: float) -> None:
        """Sleep that wakes early on stop (raises Stopped) and holds while paused."""
        if self._stopped.wait(seconds):
            raise Stopped()
        self.checkpoint()


current: Optional[RunControl] = None
