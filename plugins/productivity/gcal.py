"""plugins/productivity/gcal.py — CalendarClient via Google Calendar API."""
from __future__ import annotations

import logging
import os
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CREDS_PATH = Path.home() / ".aria" / "credentials" / "google.json"
_NOT_SETUP_MSG = "Google isn't connected yet. Run: python -m aria.setup gmail"
_AUTH_EXPIRED_MSG = "Google auth expired. Run: python -m aria.setup gmail"
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
]


def _system_timezone() -> str:
    """Read system timezone from /etc/localtime symlink."""
    try:
        lt = Path("/etc/localtime")
        if lt.is_symlink():
            target = str(lt.resolve())
            if "zoneinfo/" in target:
                return target.split("zoneinfo/")[-1]
    except Exception:
        pass
    return "UTC"


def _parse_time(value: str) -> str:
    """Convert natural time strings to ISO 8601. Returns value unchanged if already ISO."""
    stripped = value.strip()
    lower = stripped.lower()
    today = date.today()
    if lower in ("today", "now"):
        return datetime.combine(today, datetime.min.time()).isoformat() + "Z"
    if lower == "tomorrow":
        return datetime.combine(today + timedelta(days=1), datetime.min.time()).isoformat() + "Z"
    if lower == "this week":
        return datetime.combine(today, datetime.min.time()).isoformat() + "Z"
    if lower == "next week":
        return datetime.combine(today + timedelta(days=7), datetime.min.time()).isoformat() + "Z"
    # Return as-is if it already looks like ISO
    return stripped


class CalendarClient:
    def __init__(self) -> None:
        self._service = None
        self._tz = _system_timezone()

    @staticmethod
    def is_configured() -> bool:
        return _CREDS_PATH.exists()

    def _get_service(self):
        if self._service is not None:
            return self._service
        if not self.is_configured():
            return None
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            creds = Credentials.from_authorized_user_file(str(_CREDS_PATH), _SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                _CREDS_PATH.write_text(creds.to_json())
            self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
            return self._service
        except Exception as exc:
            logger.warning("gcal: auth failed: %s", exc)
            return None

    def list_events(self, time_min: str = "today", time_max: str = "",
                    calendar_id: str = "primary") -> list | str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        svc = self._get_service()
        if svc is None:
            return _AUTH_EXPIRED_MSG
        try:
            t_min = _parse_time(time_min)
            kwargs = {
                "calendarId": calendar_id,
                "timeMin": t_min,
                "singleEvents": True,
                "orderBy": "startTime",
                "maxResults": 20,
            }
            if time_max:
                kwargs["timeMax"] = _parse_time(time_max)
            result = svc.events().list(**kwargs).execute()
            events = result.get("items", [])
            return [_format_event(e) for e in events]
        except Exception as exc:
            return _handle_api_error(exc)

    def create_event(self, title: str, start: str, end: str,
                     description: str = "", location: str = "",
                     calendar_id: str = "primary") -> dict | str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        svc = self._get_service()
        if svc is None:
            return _AUTH_EXPIRED_MSG
        try:
            body = {
                "summary": title,
                "start": {"dateTime": start, "timeZone": self._tz},
                "end": {"dateTime": end, "timeZone": self._tz},
            }
            if description:
                body["description"] = description
            if location:
                body["location"] = location
            event = svc.events().insert(calendarId=calendar_id, body=body).execute()
            return {
                "event_id": event.get("id", ""),
                "url": event.get("htmlLink", ""),
            }
        except Exception as exc:
            return _handle_api_error(exc)

    def delete_event(self, event_id: str, calendar_id: str = "primary") -> str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        svc = self._get_service()
        if svc is None:
            return _AUTH_EXPIRED_MSG
        try:
            svc.events().delete(calendarId=calendar_id, eventId=event_id).execute()
            return "Deleted."
        except Exception as exc:
            return _handle_api_error(exc)


def _format_event(event: dict) -> dict:
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id", ""),
        "title": event.get("summary", "(no title)"),
        "start": start.get("dateTime", start.get("date", "")),
        "end": end.get("dateTime", end.get("date", "")),
        "location": event.get("location", ""),
        "description": event.get("description", ""),
    }


def _handle_api_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "quota" in msg or "rate" in msg or "429" in msg:
        return "Google Calendar is rate-limited. Try again in a minute."
    if "invalid_grant" in msg or "token" in msg or "auth" in msg:
        return _AUTH_EXPIRED_MSG
    return f"Calendar error: {exc}"
