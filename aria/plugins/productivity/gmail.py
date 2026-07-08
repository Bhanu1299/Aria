"""plugins/productivity/gmail.py — GmailClient via Google API."""
from __future__ import annotations

import base64
import email as _email_lib
import logging
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

logger = logging.getLogger(__name__)

_CREDS_PATH = Path.home() / ".aria" / "credentials" / "google.json"
_NOT_SETUP_MSG = "Google isn't connected yet. Run: python -m aria.setup gmail"
_AUTH_EXPIRED_MSG = "Google auth expired. Run: python -m aria.setup gmail"
_RATE_LIMIT_MSG = "Gmail is rate-limited. Try again in a minute."
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]


class GmailClient:
    def __init__(self) -> None:
        self._service = None

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
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            return self._service
        except Exception as exc:
            logger.warning("gmail: auth failed: %s", exc)
            return None

    def list_messages(self, query: str = "", limit: int = 5) -> list | str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        svc = self._get_service()
        if svc is None:
            return _AUTH_EXPIRED_MSG
        limit = min(max(1, limit), 20)
        try:
            resp = svc.users().messages().list(
                userId="me", q=query or "", maxResults=limit
            ).execute()
            msgs = resp.get("messages", [])
            if not msgs:
                return []
            results = []
            for m in msgs:
                meta = svc.users().messages().get(
                    userId="me", id=m["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                ).execute()
                headers = {h["name"]: h["value"]
                           for h in meta.get("payload", {}).get("headers", [])}
                results.append({
                    "id": m["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": meta.get("snippet", ""),
                    "is_unread": "UNREAD" in meta.get("labelIds", []),
                })
            return results
        except Exception as exc:
            return _handle_api_error(exc)

    def read_message(self, email_id: str) -> dict | str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        svc = self._get_service()
        if svc is None:
            return _AUTH_EXPIRED_MSG
        try:
            msg = svc.users().messages().get(
                userId="me", id=email_id, format="full"
            ).execute()
            payload = msg.get("payload", {})
            headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
            body = _extract_body(payload)
            return {
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
                "body_text": body,
            }
        except Exception as exc:
            return _handle_api_error(exc)

    def send_message(self, to: str, subject: str, body: str, cc: str = "") -> str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        svc = self._get_service()
        if svc is None:
            return _AUTH_EXPIRED_MSG
        try:
            msg = MIMEText(body)
            msg["to"] = to
            msg["subject"] = subject
            if cc:
                msg["cc"] = cc
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            svc.users().messages().send(userId="me", body={"raw": raw}).execute()
            return "Sent."
        except Exception as exc:
            return _handle_api_error(exc)

    def reply_message(self, email_id: str, body: str) -> str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        svc = self._get_service()
        if svc is None:
            return _AUTH_EXPIRED_MSG
        try:
            orig = svc.users().messages().get(
                userId="me", id=email_id, format="metadata",
                metadataHeaders=["From", "Subject", "Message-ID", "To"],
            ).execute()
            headers = {h["name"]: h["value"]
                       for h in orig.get("payload", {}).get("headers", [])}
            thread_id = orig.get("threadId", "")
            msg = MIMEText(body)
            msg["to"] = headers.get("From", "")
            msg["subject"] = "Re: " + headers.get("Subject", "")
            if headers.get("Message-ID"):
                msg["In-Reply-To"] = headers["Message-ID"]
                msg["References"] = headers["Message-ID"]
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            svc.users().messages().send(
                userId="me",
                body={"raw": raw, "threadId": thread_id},
            ).execute()
            return "Sent."
        except Exception as exc:
            return _handle_api_error(exc)


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text body from Gmail payload."""
    mime_type = payload.get("mimeType", "")
    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        result = _extract_body(part)
        if result:
            return result
    return ""


def _handle_api_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "quota" in msg or "rate" in msg or "429" in msg:
        return _RATE_LIMIT_MSG
    if "invalid_grant" in msg or "token" in msg or "auth" in msg:
        return _AUTH_EXPIRED_MSG
    return f"Gmail error: {exc}"
