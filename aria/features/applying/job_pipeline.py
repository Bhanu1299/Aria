"""applying/job_pipeline.py — "run the job pipeline": the whole automation, one command.

start(say, voice_ask) → spoken line; the work goes on in background threads
  1. bring up   startup.bring_up(): GateLLM → Coldstart → dashboard → Chrome (first failure
                named, nothing starts); the dashboard opens in a background tab on Live run
  2. at once    applications  session.start: new sheet jobs, best fit first, each in its own
                              visible tab, never stops at a wall, never submits
                contacts      contacts_tab.start_background: Coldstart emails for the Contacts
                              tab → jobs-Gmail DRAFTS (never sends; no browser, so it runs
                              alongside the fills)
  3. voice      one line now (what started), one line at the end with both results

Only these words start it ("run the job pipeline"); "start applying" is applications only,
"find emails for my contacts" is contacts only, "open my dashboard" is the dashboard only.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from aria.features.applying import session, startup

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_current: Optional["_Run"] = None


class _Run:
    """Collects the two end lines; speaks them together once both parts are done."""

    def __init__(self, say: Callable[[str], None], parts: set) -> None:
        self.say = say
        self.waiting = set(parts)
        self.lines: dict = {}
        self._lock = threading.Lock()

    def done(self, part: str, line: str) -> None:
        self._end(part, (line or "").strip())

    def drop(self, part: str) -> None:
        """That part never started: don't wait for it (speak now if the other one already ended)."""
        self._end(part, None)

    def _end(self, part: str, line: Optional[str]) -> None:
        with self._lock:
            if part not in self.waiting:
                return
            self.waiting.discard(part)
            if line:
                self.lines[part] = line
            if self.waiting or not self.lines:
                return
            out = " ".join(self.lines[p] for p in ("apply", "contacts") if self.lines.get(p))
        _finish(self)
        print(f"[Aria] {out}")
        try:
            self.say(out)
        except Exception as exc:
            logger.warning("job pipeline: speaking failed: %s", exc)


def _finish(run: _Run) -> None:
    global _current
    with _lock:
        if _current is run:
            _current = None


def running() -> bool:
    return _current is not None


def _start_contacts(done_fn: Callable[[str], None]) -> bool:
    from aria.features.outreach import contacts_tab
    from aria.web import websearch
    return contacts_tab.start_background(done_fn=done_fn,
                                         search_fn=lambda q: websearch.search(q, max_results=8))


def start(say: Callable[[str], None], voice_ask: Optional[Callable[[str], str]] = None) -> str:
    global _current
    with _lock:
        if _current is not None:
            return "The job pipeline is already running. Say stop applications to end the application part."
    ok, message = startup.bring_up()
    if not ok:
        return message
    session.reattach()

    run = _Run(say, {"apply", "contacts"})
    with _lock:
        _current = run

    try:
        contacts_on = _start_contacts(lambda line: run.done("contacts", line))
    except Exception as exc:
        logger.error("job pipeline: contacts didn't start: %s", exc)
        contacts_on = False
    if not contacts_on:
        run.drop("contacts")

    before = session._thread
    try:
        apply_line = session.start(say, voice_ask, finish_fn=lambda line: run.done("apply", line))
    except Exception as exc:
        logger.error("job pipeline: applications didn't start: %s", exc)
        apply_line = "The application run didn't start."
    apply_on = session._thread is not before
    if not apply_on:
        run.drop("apply")

    if not apply_on and not contacts_on:
        _finish(run)
        return f"{apply_line} The Contacts tab is already being worked on, so nothing new started."

    parts = [apply_line if apply_on else f"No applications this time: {apply_line}"]
    parts.append("Finding contact emails and writing drafts in the background, nothing gets sent."
                 if contacts_on else "The Contacts tab is already being worked on.")
    parts.append("I'll tell you once, when it's all done.")
    return " ".join(parts)
