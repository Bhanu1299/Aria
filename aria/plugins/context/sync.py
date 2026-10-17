"""plugins/context/sync.py — keeps the context index fresh in the background.

First sync of a source pulls 30 days; later syncs pull the last 3 days
(email skips messages already indexed). Calendar is re-read whole each
time over a -7d…+14d window so moved/cancelled events don't linger.
Items older than 90 days are pruned.
"""
from __future__ import annotations

import logging
import threading
import time

from aria.plugins.context import sources

logger = logging.getLogger(__name__)

_FIRST_DAYS = 30
_INCREMENTAL_DAYS = 3
_CAL_PAST_DAYS = 7
_RETENTION_DAYS = 90


class ContextSync:
    def __init__(self, store, gmail=None, calendar=None, imessage=None,
                 read_contacts=None, interval_s: float = 1800) -> None:
        self.store = store
        self.gmail = gmail
        self.calendar = calendar
        self.imessage = imessage
        self.read_contacts = read_contacts
        self.interval_s = interval_s
        self._running = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> str:
        """Sync every source once. Returns a one-line summary. Never raises."""
        if not self._running.acquire(blocking=False):
            return "A context sync is already running."
        try:
            return self._run()
        except Exception as exc:
            logger.warning("context sync failed: %s", exc)
            return f"Context sync failed: {exc}"
        finally:
            self._running.release()

    def _run(self) -> str:
        states = self.store.sync_states()

        def window(source: str) -> int:
            st = states.get(source)
            return _INCREMENTAL_DAYS if st and not st["last_error"] else _FIRST_DAYS

        indexed, problems = [], []

        if self.read_contacts is not None:
            n, err = sources.sync_contacts(self.store, self.read_contacts)
            self.store.set_sync_state("contacts", n, err)
            if err:
                problems.append(f"Contacts: {err}")

        if self.gmail is not None:
            known = {i.split(":", 1)[1] for i in self.store.known_ids("email")}
            items, err = sources.gmail_items(self.gmail, window("email"), known)
            self._record("email", items, err, indexed, problems, "new emails")

        if self.calendar is not None:
            items, err = sources.calendar_items(self.calendar, past_days=_CAL_PAST_DAYS)
            if not err:
                self.store.delete_source_since("calendar", time.time() - _CAL_PAST_DAYS * 86400)
            self._record("calendar", items, err, indexed, problems, "calendar events")

        if self.imessage is not None:
            items, err = sources.imessage_items(self.imessage, window("imessage"))
            self._record("imessage", items, err, indexed, problems, "iMessage conversations")

        self.store.prune(time.time() - _RETENTION_DAYS * 86400)
        counts = self.store.counts()
        summary = (f"Indexed {', '.join(indexed) or 'nothing new'}. "
                   f"{counts.get('people', 0)} people linked.")
        if problems:
            summary += " Not synced — " + "; ".join(problems)
        logger.info("context sync: %s", summary)
        return summary

    def _record(self, source, items, err, indexed, problems, label) -> None:
        written = self.store.upsert_items(items) if items else 0
        self.store.set_sync_state(source, written, err)
        if err:
            problems.append(f"{source}: {err}")
        else:
            indexed.append(f"{written} {label}")

    def start(self, initial_delay_s: float = 45) -> None:
        """Background loop: sync after a startup delay, then every interval."""
        if self._thread is not None:
            return

        def loop() -> None:
            if self._stop.wait(initial_delay_s):
                return
            while not self._stop.is_set():
                self.run_once()
                if self._stop.wait(self.interval_s):
                    return

        self._thread = threading.Thread(target=loop, daemon=True, name="context-sync")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
