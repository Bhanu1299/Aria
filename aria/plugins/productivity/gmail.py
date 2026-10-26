"""plugins/productivity/gmail.py — GmailClient via Google API."""
from __future__ import annotations

import base64
import email as _email_lib
import logging
import re
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Sequence

from aria.integrations import google_auth

logger = logging.getLogger(__name__)

_RATE_LIMIT_MSG = "Gmail is rate-limited. Try again in a minute."
_SCOPES = [google_auth.GMAIL_MODIFY, google_auth.GMAIL_LABELS, google_auth.GMAIL_COMPOSE]


class GmailClient:
    """Gmail for ONE Google account ("personal" or "jobs") — never the other."""

    def __init__(self, account: str) -> None:
        self.account = account
        self._service = None
        cmd, who = google_auth.setup_cmd(account), google_auth.label(account)
        self._not_setup_msg = f"{who} isn't connected yet. Run: {cmd}"
        self._auth_expired_msg = f"{who} auth expired or is missing permissions. Run: {cmd}"

    def is_configured(self) -> bool:
        return google_auth.is_configured(self.account)

    def _get_service(self):
        if self._service is not None:
            return self._service
        if not self.is_configured():
            return None
        try:
            self._service = google_auth.build_service("gmail", "v1", _SCOPES, account=self.account)
            return self._service
        except Exception as exc:
            logger.warning("gmail: auth failed: %s", exc)
            return None

    def list_messages(self, query: str = "", limit: int = 5) -> list | str:
        if not self.is_configured():
            return self._not_setup_msg
        svc = self._get_service()
        if svc is None:
            return self._auth_expired_msg
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
            return _handle_api_error(exc, self._auth_expired_msg)

    def my_address(self) -> str:
        """The connected account's own address (lowercased), or '' if unknown."""
        svc = self._get_service() if self.is_configured() else None
        if svc is None:
            return ""
        try:
            profile = svc.users().getProfile(userId="me").execute()
            return (profile.get("emailAddress") or "").lower()
        except Exception as exc:
            logger.warning("gmail: getProfile failed: %s", exc)
            return ""

    def recent_messages(self, days: int = 30, limit: int = 200,
                        skip_ids: set | None = None) -> list | str:
        """Bulk metadata for messages newer than `days` (for the context index).

        Pages past list_messages' 20 cap. IDs in skip_ids are not re-fetched.
        Each item: {id, from, to, cc, subject, ts, snippet}; ts is epoch seconds.
        """
        if not self.is_configured():
            return self._not_setup_msg
        svc = self._get_service()
        if svc is None:
            return self._auth_expired_msg
        skip_ids = skip_ids or set()
        try:
            ids: list = []
            page_token = None
            while len(ids) < limit:
                kwargs = {"userId": "me", "q": f"newer_than:{max(1, days)}d",
                          "maxResults": min(100, limit - len(ids))}
                if page_token:
                    kwargs["pageToken"] = page_token
                resp = svc.users().messages().list(**kwargs).execute()
                ids.extend(m["id"] for m in resp.get("messages", []))
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            results = []
            for mid in ids:
                if mid in skip_ids:
                    continue
                meta = svc.users().messages().get(
                    userId="me", id=mid, format="metadata",
                    metadataHeaders=["From", "To", "Cc", "Subject"],
                ).execute()
                headers = {h["name"]: h["value"]
                           for h in meta.get("payload", {}).get("headers", [])}
                results.append({
                    "id": mid,
                    "from": headers.get("From", ""),
                    "to": headers.get("To", ""),
                    "cc": headers.get("Cc", ""),
                    "subject": headers.get("Subject", ""),
                    "ts": int(meta.get("internalDate", 0)) / 1000.0,
                    "snippet": meta.get("snippet", ""),
                })
            return results
        except Exception as exc:
            return _handle_api_error(exc, self._auth_expired_msg)

    def read_message(self, email_id: str) -> dict | str:
        if not self.is_configured():
            return self._not_setup_msg
        svc = self._get_service()
        if svc is None:
            return self._auth_expired_msg
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
            return _handle_api_error(exc, self._auth_expired_msg)

    def message_bodies(self, email_id: str) -> dict | str:
        """{from, subject, date, ts, text, html} — html too, for mail that has no plain part."""
        if not self.is_configured():
            return self._not_setup_msg
        svc = self._get_service()
        if svc is None:
            return self._auth_expired_msg
        try:
            msg = svc.users().messages().get(userId="me", id=email_id, format="full").execute()
            payload = msg.get("payload", {})
            headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
            return {"from": headers.get("From", ""), "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""), "ts": int(msg.get("internalDate") or 0) / 1000.0,
                    "text": _extract_part(payload, "text/plain"), "html": _extract_part(payload, "text/html")}
        except Exception as exc:
            return _handle_api_error(exc, self._auth_expired_msg)

    def send_message(self, to: str, subject: str, body: str, cc: str = "") -> str:
        if not self.is_configured():
            return self._not_setup_msg
        svc = self._get_service()
        if svc is None:
            return self._auth_expired_msg
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
            return _handle_api_error(exc, self._auth_expired_msg)

    def create_draft(self, to: str, subject: str, body: str, cc: str = "", reply_to_id: str = "") -> str:
        """Save a draft in Gmail. Never sends. reply_to_id threads it as a reply to that email."""
        return self.save_draft(to, subject, body, cc, reply_to_id)["message"]

    def save_draft(self, to: str, subject: str, body: str, cc: str = "", reply_to_id: str = "",
                   attachments: Sequence[str] = (), draft_id: str = "") -> dict:
        """create_draft with the result: {"ok", "message", "draft_id", "message_id", "link"}. Never sends.
        link opens the draft in Gmail's web UI, in this account (u/<address>).
        attachments: file paths attached as-is (e.g. the resume PDF).
        draft_id: replace that draft's content instead of making a new one (still never sends)."""
        out = {"ok": False, "message": "", "draft_id": "", "message_id": "", "link": ""}
        if not self.is_configured():
            return {**out, "message": self._not_setup_msg}
        svc = self._get_service()
        if svc is None:
            return {**out, "message": self._auth_expired_msg}
        try:
            msg = MIMEText(body)
            if attachments:
                outer = MIMEMultipart()
                outer.attach(msg)
                for path in attachments:
                    part = MIMEApplication(Path(path).read_bytes(), Name=Path(path).name)
                    part["Content-Disposition"] = f'attachment; filename="{Path(path).name}"'
                    outer.attach(part)
                msg = outer
            message: dict = {}
            if reply_to_id:
                orig = svc.users().messages().get(
                    userId="me", id=reply_to_id, format="metadata",
                    metadataHeaders=["From", "Subject", "Message-ID"],
                ).execute()
                headers = {h["name"]: h["value"] for h in orig.get("payload", {}).get("headers", [])}
                to = to or headers.get("From", "")
                subject = subject or "Re: " + headers.get("Subject", "")
                if headers.get("Message-ID"):
                    msg["In-Reply-To"] = headers["Message-ID"]
                    msg["References"] = headers["Message-ID"]
                if orig.get("threadId"):
                    message["threadId"] = orig["threadId"]
            msg["to"] = to
            msg["subject"] = subject
            if cc:
                msg["cc"] = cc
            message["raw"] = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            if draft_id:
                resp = svc.users().drafts().update(userId="me", id=draft_id,
                                                   body={"id": draft_id, "message": message}).execute()
            else:
                resp = svc.users().drafts().create(userId="me", body={"message": message}).execute()
            addr = google_auth.account_email(self.account)
            where = addr or f"your {self.account} Gmail"
            resp = resp if isinstance(resp, dict) else {}
            msg_id = str((resp.get("message") or {}).get("id") or "")
            link = f"https://mail.google.com/mail/u/{addr or 0}/#drafts?compose={msg_id}" if msg_id else ""
            return {"ok": True, "draft_id": str(resp.get("id") or ""), "message_id": msg_id, "link": link,
                    "message": f"Draft saved for {to}: {subject}. It's in the drafts of {where}; nothing was sent."}
        except Exception as exc:
            return {**out, "message": _handle_api_error(exc, self._auth_expired_msg)}

    def reply_message(self, email_id: str, body: str) -> str:
        if not self.is_configured():
            return self._not_setup_msg
        svc = self._get_service()
        if svc is None:
            return self._auth_expired_msg
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
            return _handle_api_error(exc, self._auth_expired_msg)


def _extract_part(payload: dict, mime: str) -> str:
    """The first body of this MIME type anywhere in the payload, decoded ('' if none)."""
    if payload.get("mimeType", "") == mime:
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        found = _extract_part(part, mime)
        if found:
            return found
    return ""


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


def _handle_api_error(exc: Exception, auth_expired_msg: str) -> str:
    msg = str(exc).lower()
    if "quota" in msg or "rate" in msg or "429" in msg:
        return _RATE_LIMIT_MSG
    if "invalid_grant" in msg or "token" in msg or "auth" in msg:
        return auth_expired_msg
    return f"Gmail error: {exc}"
