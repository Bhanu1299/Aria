"""tests/test_gcal.py — CalendarClient unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

_NOT_SETUP = "Google isn't connected yet. Run: python -m aria.setup gmail"


def _configured(mock_svc):
    from plugins.productivity.gcal import CalendarClient
    client = CalendarClient.__new__(CalendarClient)
    client._service = mock_svc
    client._tz = "America/New_York"
    return client


def _unconfigured():
    from plugins.productivity.gcal import CalendarClient
    client = CalendarClient.__new__(CalendarClient)
    client._service = None
    client._tz = "UTC"
    return client


def test_list_returns_not_setup_when_unconfigured():
    client = _unconfigured()
    with patch.object(client, "is_configured", return_value=False):
        result = client.list_events()
    assert result == _NOT_SETUP


def test_create_returns_not_setup_when_unconfigured():
    client = _unconfigured()
    with patch.object(client, "is_configured", return_value=False):
        result = client.create_event("Meeting", "2026-05-07T14:00:00", "2026-05-07T15:00:00")
    assert result == _NOT_SETUP


def test_delete_returns_not_setup_when_unconfigured():
    client = _unconfigured()
    with patch.object(client, "is_configured", return_value=False):
        result = client.delete_event("evt123")
    assert result == _NOT_SETUP


def test_list_events_returns_formatted():
    svc = MagicMock()
    svc.events.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "e1",
                "summary": "Team Standup",
                "start": {"dateTime": "2026-05-07T09:00:00"},
                "end": {"dateTime": "2026-05-07T09:30:00"},
                "location": "",
                "description": "",
            }
        ]
    }
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.list_events("today")
    assert isinstance(result, list)
    assert result[0]["title"] == "Team Standup"
    assert result[0]["id"] == "e1"


def test_list_events_returns_empty_list_when_none():
    svc = MagicMock()
    svc.events.return_value.list.return_value.execute.return_value = {"items": []}
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.list_events("today")
    assert result == []


def test_create_event_returns_dict_with_id():
    svc = MagicMock()
    svc.events.return_value.insert.return_value.execute.return_value = {
        "id": "evt_new", "htmlLink": "https://calendar.google.com/event/evt_new"
    }
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.create_event(
            "Deep Work", "2026-05-07T14:00:00Z", "2026-05-07T15:00:00Z"
        )
    assert isinstance(result, dict)
    assert result["event_id"] == "evt_new"


def test_delete_event_returns_deleted():
    svc = MagicMock()
    svc.events.return_value.delete.return_value.execute.return_value = None
    client = _configured(svc)
    with patch.object(client, "is_configured", return_value=True):
        result = client.delete_event("evt_new")
    assert result == "Deleted."


def test_parse_time_today():
    from plugins.productivity.gcal import _parse_time
    result = _parse_time("today")
    assert "T" in result


def test_parse_time_passthrough_iso():
    from plugins.productivity.gcal import _parse_time
    iso = "2026-05-07T14:00:00Z"
    assert _parse_time(iso) == iso


def test_system_timezone_returns_string():
    from plugins.productivity.gcal import _system_timezone
    tz = _system_timezone()
    assert isinstance(tz, str)
    assert len(tz) > 0
