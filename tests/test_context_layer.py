"""tests/test_context_layer.py — cross-app context index: linking, lookup, sync."""
from __future__ import annotations

import os
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from aria.core.tool import ToolRegistry
from aria.plugins.context import ContextPlugin, lookup
from aria.plugins.context.people import (email_alias, name_alias, parse_address_header,
                                         phone_alias)
from aria.plugins.context.store import ContextStore
from aria.plugins.context.sync import ContextSync

NOW = time.time()
HOUR = 3600
DAY = 86400


@pytest.fixture
def store(tmp_path):
    return ContextStore(tmp_path / "context.db")


class FakeGmail:
    def __init__(self, msgs, me="me@gmail.com"):
        self.msgs = msgs
        self.me = me
        self.calls = []

    def my_address(self):
        return self.me

    def recent_messages(self, days=30, limit=200, skip_ids=None):
        self.calls.append({"days": days, "skip_ids": set(skip_ids or ())})
        return [m for m in self.msgs if m["id"] not in (skip_ids or set())]


class FakeCalendar:
    def __init__(self, events):
        self.events = events

    def list_events(self, time_min, time_max, max_results=20):
        return self.events


class FakeIMessage:
    def __init__(self, msgs):
        self.msgs = msgs

    def recent(self, days=30):
        return self.msgs


def _iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _world():
    """Priya: email (work address), iMessage (phone) and a meeting — linked only
    through her Contacts card. Plus a recruiter thread about Macro."""
    gmail = FakeGmail([
        {"id": "m1", "from": "Priya Sharma <priya@acme.com>", "to": "me@gmail.com", "cc": "",
         "subject": "Re: Next steps", "ts": NOW - 2 * DAY, "snippet": "Can you do Thursday?"},
        {"id": "m2", "from": "Jordan Lee <jordan@macro.com>", "to": "me@gmail.com", "cc": "",
         "subject": "Macro interview loop", "ts": NOW - 1 * DAY,
         "snippet": "Excited to have you meet the team"},
    ])
    cal = FakeCalendar([
        {"id": "e1", "title": "Coffee with Priya", "start": _iso(NOW + 2 * DAY), "end": "",
         "location": "Blue Bottle", "description": "",
         "attendees": [{"email": "priya@acme.com", "name": "", "self": False},
                       {"email": "me@gmail.com", "name": "", "self": True}]},
    ])
    im = FakeIMessage([
        {"handle": "+1 (555) 123-4567", "text": "see you thursday!", "ts": NOW - 3 * HOUR,
         "is_from_me": False},
        {"handle": "+1 (555) 123-4567", "text": "yes!", "ts": NOW - 2 * HOUR,
         "is_from_me": True},
    ])
    contacts = lambda: [{"name": "Priya Sharma", "handles": ["+15551234567", "priya@acme.com"]}]
    return gmail, cal, im, contacts


def _synced(store):
    gmail, cal, im, contacts = _world()
    sync = ContextSync(store, gmail=gmail, calendar=cal, imessage=im, read_contacts=contacts)
    summary = sync.run_once()
    return sync, summary, gmail


# ----------------------------------------------------------------------
# normalization
# ----------------------------------------------------------------------

def test_aliases_normalize_handles():
    assert email_alias(" <Priya@Acme.COM> ") == "email:priya@acme.com"
    assert phone_alias("+1 (555) 123-4567") == phone_alias("5551234567") == "tel:5551234567"
    assert name_alias('"Priya  Sharma"') == "name:priya sharma"
    assert name_alias("priya@acme.com") == ""
    assert parse_address_header("Priya <p@x.com>, b@y.com") == [("Priya", "p@x.com"),
                                                                ("", "b@y.com")]


# ----------------------------------------------------------------------
# entity linking
# ----------------------------------------------------------------------

def test_link_merges_people_that_share_a_handle(store):
    a = store.link("", ["priya@acme.com"])
    b = store.link("", ["+15551234567"])
    assert a != b
    merged = store.link("Priya Sharma", ["priya@acme.com", "+1 555 123 4567"], authoritative=True)
    assert merged == min(a, b)
    pid, name, _, how = store.resolve_person("priya")
    assert pid == merged and name == "Priya Sharma" and how == "words"
    assert set(store.person_handles(pid)) == {"priya@acme.com", "5551234567"}


def test_single_word_names_never_merge_people(store):
    a = store.link("John", ["john@a.com"])
    b = store.link("John", ["john@b.com"])
    assert a != b


# ----------------------------------------------------------------------
# sync + lookup
# ----------------------------------------------------------------------

def test_person_timeline_spans_email_imessage_and_calendar(store):
    _, summary, _ = _synced(store)
    assert "2 new emails" in summary and "1 calendar events" in summary
    out = lookup(store, "Priya")
    assert out.startswith("Priya Sharma")
    assert "[email] Re: Next steps" in out and "(email_id m1)" in out
    assert "[imessage]" in out and "see you thursday" in out
    assert "Coming up:" in out and "Coffee with Priya" in out
    # upcoming section comes before recent
    assert out.index("Coming up:") < out.index("Recent, newest first:")


def test_fuzzy_name_from_voice_transcript(store):
    _synced(store)
    assert lookup(store, "Pria Sharma").startswith("Priya Sharma")


def test_topic_lookup_lists_people_involved(store):
    _synced(store)
    out = lookup(store, "macro interview")
    assert "Macro interview loop" in out and "with Jordan Lee" in out
    assert "Next steps" not in out


def test_topic_beats_fuzzy_person_match(store):
    _synced(store)
    store.link("Marco Rossi", ["marco@x.com"])
    out = lookup(store, "macro")
    assert "Macro interview loop" in out
    assert "close name match: Marco Rossi" in out


def test_self_is_not_indexed_as_a_person(store):
    _synced(store)
    pid, *_ = store.resolve_person("me@gmail.com")
    assert pid is None


def test_incremental_sync_skips_known_emails_and_narrows_window(store):
    sync, _, gmail = _synced(store)
    assert gmail.calls[0]["days"] == 30
    sync.run_once()
    assert gmail.calls[1]["days"] == 3
    assert gmail.calls[1]["skip_ids"] == {"m1", "m2"}


def test_calendar_resync_drops_cancelled_events(store):
    sync, _, _ = _synced(store)
    sync.calendar.events = []
    sync.run_once()
    assert "Coffee with Priya" not in lookup(store, "Priya")


def test_imessage_partial_first_day_is_dropped():
    from aria.plugins.context.sources import imessage_items
    old = FakeIMessage([{"handle": "+15550000000", "text": "edge", "ts": NOW - 3 * DAY + 60,
                         "is_from_me": False}])
    items, err = imessage_items(old, days=3)
    assert err == "" and items == []


def test_unconnected_sources_explain_themselves(store):
    sync = ContextSync(store, gmail=FakeGmail([]), read_contacts=lambda: "needs Full Disk Access")
    sync.gmail.recent_messages = lambda **kw: "Google isn't connected yet."
    summary = sync.run_once()
    assert "Google isn't connected" in summary
    out = lookup(store, "anyone")
    assert out.startswith("Nothing is indexed yet") and "Google isn't connected" in out


def test_concurrent_sync_is_refused(store):
    sync = ContextSync(store)
    sync._running.acquire()
    try:
        assert "already running" in sync.run_once()
    finally:
        sync._running.release()


# ----------------------------------------------------------------------
# plugin wiring
# ----------------------------------------------------------------------

def test_plugin_registers_tools_and_never_raises(store):
    syncer = ContextSync(store)
    registry = ToolRegistry()
    ContextPlugin(store=store, syncer=syncer, autostart=False).register(registry)
    tool = registry.get("context_for")
    assert tool is not None and registry.get("context_sync") is not None
    assert "Tell me which" in tool.execute({})
    store.resolve_person = MagicMock(side_effect=RuntimeError("boom"))
    assert "couldn't search" in tool.execute({"query": "x"})


def test_plugin_is_discovered():
    import aria.plugins as plugins
    assert "ContextPlugin" in {c.__name__ for c in plugins.discover()}


# ----------------------------------------------------------------------
# client additions the index depends on
# ----------------------------------------------------------------------

def _blob(text: str) -> bytes:
    raw = text.encode()
    length = bytes([len(raw)]) if len(raw) < 0x80 else b"\x81" + len(raw).to_bytes(2, "little")
    return b"streamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString\x00\x84\x84" \
           b"\x08NSObject\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+" + length + raw + b"\x86"


def test_decode_attributed_body_short_and_long():
    from aria.plugins.messaging.imessage import decode_attributed_body
    assert decode_attributed_body(_blob("hello there")) == "hello there"
    long = "x" * 300
    assert decode_attributed_body(_blob(long)) == long
    assert decode_attributed_body(None) == "" and decode_attributed_body(b"junk") == ""


def test_imessage_recent_reads_text_and_attributed_body(tmp_path):
    from aria.plugins.messaging.imessage import iMessageClient
    db = str(tmp_path / "chat.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
    conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, is_from_me INTEGER, "
                 "text TEXT, attributedBody BLOB, date INTEGER, handle_id INTEGER)")
    conn.execute("INSERT INTO handle VALUES (1, '+15551234567')")
    mac_ns = lambda ts: int((ts - 978307200) * 1e9)
    conn.execute("INSERT INTO message VALUES (1, 0, 'plain', NULL, ?, 1)", (mac_ns(NOW - HOUR),))
    conn.execute("INSERT INTO message VALUES (2, 1, NULL, ?, ?, 1)",
                 (_blob("from blob"), mac_ns(NOW - 60)))
    conn.execute("INSERT INTO message VALUES (3, 0, 'ancient', NULL, ?, 1)",
                 (mac_ns(NOW - 60 * DAY),))
    conn.commit()
    conn.close()
    with patch("aria.plugins.messaging.imessage._CHAT_DB", db):
        msgs = iMessageClient().recent(days=30)
    assert [m["text"] for m in msgs] == ["plain", "from blob"]
    assert abs(msgs[0]["ts"] - (NOW - HOUR)) < 1 and msgs[1]["is_from_me"] is True


def test_gmail_recent_messages_pages_and_skips_known(tmp_path):
    from aria.plugins.productivity.gmail import GmailClient
    client = GmailClient("personal")
    svc = MagicMock()
    pages = [{"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "t"},
             {"messages": [{"id": "c"}]}]
    svc.users().messages().list().execute.side_effect = pages
    svc.users().messages().get().execute.return_value = {
        "internalDate": "1700000000000", "snippet": "hi",
        "payload": {"headers": [{"name": "From", "value": "A <a@x.com>"},
                                {"name": "Subject", "value": "S"}]}}
    client._service = svc
    with patch.object(GmailClient, "is_configured", return_value=True):
        out = client.recent_messages(days=30, limit=200, skip_ids={"b"})
    assert [m["id"] for m in out] == ["a", "c"]
    assert out[0]["ts"] == 1700000000.0 and out[0]["from"] == "A <a@x.com>"


def test_calendar_events_include_attendees():
    from aria.plugins.productivity.gcal import _format_event
    e = _format_event({"id": "1", "summary": "Sync", "start": {"dateTime": "x"},
                       "attendees": [{"email": "p@x.com", "displayName": "P", "self": True}]})
    assert e["attendees"] == [{"email": "p@x.com", "name": "P", "self": True}]
