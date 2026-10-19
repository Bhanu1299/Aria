"""plugins/system/timers.py — in-process countdown timers.

Timers live only while Aria runs (threading.Timer, daemon). When one fires,
on_fire(label) is called — the plugin speaks and posts a notification.
"""
from __future__ import annotations

import itertools
import threading
import time
from typing import Callable

from aria.dispatch.parse import humanize_seconds


def _remaining_phrase(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds >= 60:
        seconds = int(round(seconds / 60.0)) * 60
    return humanize_seconds(seconds) if seconds else "a moment"


class TimerManager:
    def __init__(self, on_fire: Callable[[str], None], clock: Callable[[], float] = time.monotonic) -> None:
        self._on_fire = on_fire
        self._clock = clock
        self._lock = threading.Lock()
        self._timers: dict[int, tuple[str, float, threading.Timer]] = {}
        self._ids = itertools.count(1)

    def start(self, seconds: int, label: str = "") -> str:
        label = (label or "").strip()
        timer_id = next(self._ids)
        t = threading.Timer(seconds, self._fire, args=(timer_id,))
        t.daemon = True
        with self._lock:
            self._timers[timer_id] = (label, self._clock() + seconds, t)
        t.start()
        what = f"{label.capitalize()} timer" if label else "Timer"
        return f"{what} set for {humanize_seconds(seconds)}."

    def _fire(self, timer_id: int) -> None:
        with self._lock:
            entry = self._timers.pop(timer_id, None)
        if entry is not None:
            self._on_fire(entry[0])

    def status(self) -> str:
        with self._lock:
            active = sorted(self._timers.values(), key=lambda e: e[1])
        if not active:
            return "You don't have any timers running."
        now = self._clock()
        if len(active) == 1:
            label, end, _ = active[0]
            name = f"your {label} timer" if label else "your timer"
            return f"About {_remaining_phrase(end - now)} left on {name}."
        parts = [f"{label or 'timer'}: {_remaining_phrase(end - now)} left" for label, end, _ in active]
        return "You have {} timers. {}.".format(len(active), ". ".join(p.capitalize() for p in parts))

    def cancel(self, label: str = "") -> str:
        label = (label or "").strip().lower()
        with self._lock:
            if not self._timers:
                return "You don't have any timers running."
            if label and label != "all":
                ids = [i for i, (lbl, _, _) in self._timers.items() if lbl.lower() == label]
                if not ids:
                    return f"I couldn't find a {label} timer."
            else:
                ids = list(self._timers)
            for i in ids:
                self._timers.pop(i)[2].cancel()
        if len(ids) == 1:
            return f"Canceled your {label} timer." if label and label != "all" else "Canceled your timer."
        return f"Canceled {len(ids)} timers."
