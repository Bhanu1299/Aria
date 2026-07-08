"""tests/test_gmail.py — GmailClient unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

_NOT_SETUP = "Google isn't connected yet. Run: python -m aria.setup gmail"
_AUTH_EXPIRED = "Google auth expired. Run: python -m aria.setup gmail"


def _unconfigured():
    from aria.plugins.productivity.gmail import GmailClient
    client = GmailClient.__new__(GmailClient)
    client._service = None
    return client


def _configured(mock_svc):
    from aria.plugins.productivity.gmail import GmailClient
    client = GmailClient.__new__(GmailClient)
    client._service = mock_svc
    return client


def _make_svc():
    svc = MagicMock()
    return svc


def test_list_returns_not_setup_when_unconfigured():
    client = _unconfigured()
    with patch.object(client, "is_configured", return_value=False):
        result = client.list_messages("is:unread")
    assert result == _NOT_SETUP


def test_send_returns_not_setup_when_unconfigured():
    client = _unconfigured()
    with patch.object(client, "is_configured", return_value=False):
        result = client.send_message("a@b.com", "Hi", "Body")
    assert result == _NOT_SETUP


def test_reply_returns_not_setup_when_unconfigured():
    client = _unconfigured()
    with patch.object(client, "is_configured", return_value=False):
        result = client.reply_message("msg123", "Sure!")
    assert result == _NOT_SETUP


def test_list_messages_returns_formatted_list():
    svc = _make_svc()
    svc.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "abc"}]
    }
    svc.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "abc",
        "snippet": "Hello there",
        "labelIds": ["UNREAD"],
        "threadId": "t1",
        "payload": {
            "headers": [
                {"name": "From", "value": "recruiter@amazon.com"},
                {"name": "Subject", "value": "Opportunity"},
                {"name": "Date", "value": "Mon, 6 May 2026"},
            ]
        }
    }
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.list_messages("is:unread", 5)
    assert isinstance(result, list)
    assert result[0]["from"] == "recruiter@amazon.com"
    assert result[0]["is_unread"] is True


def test_list_messages_returns_empty_list_when_none():
    svc = _make_svc()
    svc.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": []
    }
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.list_messages("from:nobody")
    assert result == []


def test_send_message_returns_sent():
    svc = _make_svc()
    svc.users.return_value.messages.return_value.send.return_value.execute.return_value = {"id": "sent1"}
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.send_message("a@b.com", "Hello", "Body text")
    assert result == "Sent."


def test_reply_message_returns_sent():
    svc = _make_svc()
    svc.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "threadId": "t1",
        "payload": {
            "headers": [
                {"name": "From", "value": "recruiter@amazon.com"},
                {"name": "Subject", "value": "Opportunity"},
                {"name": "Message-ID", "value": "<orig123>"},
                {"name": "To", "value": "me@example.com"},
            ]
        }
    }
    svc.users.return_value.messages.return_value.send.return_value.execute.return_value = {"id": "reply1"}
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.reply_message("msg123", "Very interested!")
    assert result == "Sent."


def test_rate_limit_error_returns_friendly_message():
    svc = _make_svc()
    svc.users.return_value.messages.return_value.list.return_value.execute.side_effect = \
        Exception("quota exceeded 429")
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.list_messages("is:unread")
    assert "rate-limited" in result.lower()
