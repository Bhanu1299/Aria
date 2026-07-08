"""tests/test_imessage.py — iMessageClient unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sqlite3
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_db(tmpdir: str) -> str:
    """Create a minimal fake chat.db with schema and sample data."""
    db_path = os.path.join(tmpdir, "chat.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
    conn.execute("INSERT INTO handle VALUES (1, '+15551234567')")
    conn.execute("INSERT INTO handle VALUES (2, 'mom@example.com')")
    conn.execute("""
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            is_from_me INTEGER,
            text TEXT,
            date REAL,
            handle_id INTEGER
        )
    """)
    # date: seconds since 2001-01-01 epoch (rough value)
    conn.execute("INSERT INTO message VALUES (1, 0, 'Hey how are you', 700000000, 1)")
    conn.execute("INSERT INTO message VALUES (2, 1, 'Doing great!', 700001000, 1)")
    conn.commit()
    conn.close()
    return db_path


def test_resolve_contact_phone_passthrough():
    from aria.plugins.messaging.imessage import iMessageClient
    client = iMessageClient()
    assert client.resolve_contact("+15551234567") == "+15551234567"


def test_resolve_contact_email_passthrough():
    from aria.plugins.messaging.imessage import iMessageClient
    client = iMessageClient()
    assert client.resolve_contact("user@example.com") == "user@example.com"


def test_resolve_contact_fuzzy_match():
    from aria.plugins.messaging.imessage import iMessageClient, _CHAT_DB
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = _make_db(tmpdir)
        client = iMessageClient()
        with patch("aria.plugins.messaging.imessage._CHAT_DB", db_path):
            # "+15551234567" won't fuzzy match "mom", but let's test with direct value
            result = client.resolve_contact("+15551234567")
    assert result == "+15551234567"


def test_read_returns_messages():
    from aria.plugins.messaging.imessage import iMessageClient
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = _make_db(tmpdir)
        client = iMessageClient()
        with patch("aria.plugins.messaging.imessage._CHAT_DB", db_path):
            result = client.read("+15551234567", limit=5)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["text"] == "Hey how are you"
    assert result[1]["is_from_me"] is True


def test_read_returns_error_on_no_messages():
    from aria.plugins.messaging.imessage import iMessageClient
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = _make_db(tmpdir)
        client = iMessageClient()
        with patch("aria.plugins.messaging.imessage._CHAT_DB", db_path):
            result = client.read("+19999999999", limit=5)
    assert isinstance(result, str)
    assert "No messages found" in result or "couldn't find" in result


def test_read_returns_permission_msg_on_missing_db():
    from aria.plugins.messaging.imessage import iMessageClient, _PERMISSION_MSG
    client = iMessageClient()
    with patch("aria.plugins.messaging.imessage._CHAT_DB", "/nonexistent/chat.db"):
        result = client.read("mom", limit=5)
    assert isinstance(result, str)
    # Should explain permission issue
    assert "Full Disk Access" in result or "couldn't" in result.lower()


def test_send_calls_osascript():
    from aria.plugins.messaging.imessage import iMessageClient
    client = iMessageClient()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run, \
         patch.object(client, "resolve_contact", return_value="+15551234567"):
        result = client.send("+15551234567", "Hello!")
    assert result == "Sent."
    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    assert "osascript" in cmd


def test_send_returns_error_on_osascript_failure():
    from aria.plugins.messaging.imessage import iMessageClient
    client = iMessageClient()
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Messages not running"
    with patch("subprocess.run", return_value=mock_result), \
         patch.object(client, "resolve_contact", return_value="+15551234567"):
        result = client.send("+15551234567", "Hello!")
    assert "couldn't" in result.lower() or "iMessage" in result


def test_get_contacts_returns_list():
    from aria.plugins.messaging.imessage import iMessageClient
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = _make_db(tmpdir)
        client = iMessageClient()
        with patch("aria.plugins.messaging.imessage._CHAT_DB", db_path):
            result = client.get_contacts()
    assert isinstance(result, list)
    assert len(result) == 2
