"""tests/test_whatsapp.py — WhatsAppClient unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

_NOT_SETUP_MSG = "WhatsApp isn't set up yet. Run: python -m aria.setup whatsapp"
_DISCONNECTED_MSG = "WhatsApp lost its connection. I'll try to reconnect."


def _unconfigured_client():
    from aria.plugins.messaging.whatsapp import WhatsAppClient
    with patch("aria.plugins.messaging.whatsapp.WhatsAppClient.is_configured", return_value=False):
        client = WhatsAppClient.__new__(WhatsAppClient)
        import threading
        client._client = None
        client._lock = threading.Lock()
        client._message_ids_seen = set()
        client._ready = False
    return client


def test_read_returns_setup_msg_when_not_configured():
    client = _unconfigured_client()
    with patch.object(client, "is_configured", return_value=False):
        result = client.read("mom")
    assert result == _NOT_SETUP_MSG


def test_send_returns_setup_msg_when_not_configured():
    client = _unconfigured_client()
    with patch.object(client, "is_configured", return_value=False):
        result = client.send("mom", "Hello!")
    assert result == _NOT_SETUP_MSG


def test_get_contacts_returns_setup_msg_when_not_configured():
    client = _unconfigured_client()
    with patch.object(client, "is_configured", return_value=False):
        result = client.get_contacts()
    assert result == _NOT_SETUP_MSG


def test_read_returns_disconnected_when_no_client():
    client = _unconfigured_client()
    with patch.object(client, "is_configured", return_value=True):
        # _get_client returns None when not ready
        result = client.read("mom")
    assert result == _DISCONNECTED_MSG


def test_send_with_mock_client():
    client = _unconfigured_client()
    mock_wa = MagicMock()
    mock_chat = MagicMock()
    mock_msg = MagicMock()
    mock_msg.id = "msg123"
    mock_chat.send_message.return_value = mock_msg
    mock_wa.get_chats.return_value = [mock_chat]
    mock_chat.id = "mom_id"
    mock_chat.name = "mom"
    client._client = mock_wa
    client._ready = True

    with patch.object(client, "is_configured", return_value=True), \
         patch.object(client, "_find_chat", return_value=mock_chat):
        result = client.send("mom", "Hello!")
    assert result == "Sent."
    mock_chat.send_message.assert_called_once_with("Hello!")


def test_read_with_mock_client():
    import time
    client = _unconfigured_client()
    mock_wa = MagicMock()
    mock_chat = MagicMock()
    mock_msg = MagicMock()
    mock_msg.from_me = False
    mock_msg.author = "mom"
    mock_msg.body = "How are you?"
    mock_msg.timestamp = time.time()
    mock_chat.fetch_messages.return_value = [mock_msg]
    client._client = mock_wa
    client._ready = True

    with patch.object(client, "is_configured", return_value=True), \
         patch.object(client, "_find_chat", return_value=mock_chat):
        result = client.read("mom", limit=1)
    assert isinstance(result, list)
    assert result[0]["text"] == "How are you?"
    assert result[0]["is_from_me"] is False


def test_dedup_message_id_tracked_on_send():
    client = _unconfigured_client()
    mock_wa = MagicMock()
    mock_chat = MagicMock()
    mock_msg = MagicMock()
    mock_msg.id = "unique_msg_id"
    mock_chat.send_message.return_value = mock_msg
    client._client = mock_wa
    client._ready = True

    with patch.object(client, "is_configured", return_value=True), \
         patch.object(client, "_find_chat", return_value=mock_chat):
        client.send("mom", "test")
    assert "unique_msg_id" in client._message_ids_seen
