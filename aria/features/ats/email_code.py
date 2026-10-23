"""features/ats/email_code.py — a portal's verification email: the code to type, or the link to open.

client_for(email)  → the jobs GmailClient when its connected address IS the application email,
                     else None (then the step goes to the user — never another inbox)
extract(subject, text, html) → ("code", "483920") | ("link", "https://…verify…") | None   (pure)
wait_for(email, senders, since, polls, poll_secs) → the newest match from `senders`
                     received after `since`, polling POLLS times; None → hand it to the user

Code first, no LLM. Only messages from the portal's own domains are read (e.g. myworkday.com
plus the company's domain), and only the code / link leaves this module — never the body.
"""
from __future__ import annotations

import html as _html
import logging
import re
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

POLLS = 4
POLL_SECS = 15.0

# "Your verification code is 483920", "Code: 48 39 20", "passcode - A7K29Q", "483920 is your sign-in code".
# A number that isn't next to the word code (or alone on its line) is never taken.
_CODE_NEAR = re.compile(
    r"(?:verification|security|one[- ]time|confirmation|access|login|sign[- ]in)?\s*(?:code|passcode|pin|otp)\b"
    r"[^\w\n]{0,3}(?:is|:|-)?[^\w\n]{0,3}((?=[A-Z0-9]*\d)[A-Z0-9]{4,8}|\d{2,4}[ -]\d{2,4}(?:[ -]\d{2,4})?)\b", re.I)
_CODE_FIRST = re.compile(r"\b((?=[A-Z0-9]*\d)[A-Z0-9]{4,8}) is your\b[\w\s-]{0,40}?\b(?:code|passcode|pin|otp)\b", re.I)
_CODE_LINE = re.compile(r"^\s*(\d{4,8})\s*$", re.M)
_URL = re.compile(r"""https?://[^\s"'<>]+""")
_VERIFY_URL = re.compile(r"verif|confirm|activat|validat|token=|/email/", re.I)
_TAG = re.compile(r"<[^>]+>")


def _plain(html: str) -> str:
    text = re.sub(r"<(?:br|/p|/div|/tr|/h\d)[^>]*>", "\n", html or "", flags=re.I)
    return _html.unescape(_TAG.sub(" ", text))


def _code(text: str) -> Optional[str]:
    for m in _CODE_NEAR.finditer(text or ""):
        code = re.sub(r"[ -]", "", m.group(1))
        if any(c.isdigit() for c in code) and 4 <= len(code) <= 8:
            return code.upper()
    m = _CODE_FIRST.search(text or "") or _CODE_LINE.search(text or "")
    return m.group(1).upper() if m else None


def extract(subject: str, text: str, html: str = "") -> Optional[tuple]:
    body = (text or "") + "\n" + _plain(html)
    code = _code(subject or "") or _code(body)
    if code:
        return "code", code
    hrefs = [_html.unescape(h) for h in re.findall(r"""href=["']([^"']+)["']""", html or "", re.I)]
    for url in hrefs + _URL.findall(body):
        if _VERIFY_URL.search(url) and not re.search(r"unsubscribe|privacy|terms", url, re.I):
            return "link", url.rstrip(").,")
    return None


def client_for(email: str):
    """The jobs Gmail account, only when its connected address IS `email`; else None.
    The job pipeline never reads the personal inbox (tests/test_gmail_drafts.py guards it)."""
    from aria.integrations import google_auth
    from aria.plugins.productivity.gmail import GmailClient
    want = (email or "").strip().lower()
    if not want or google_auth.account_email(google_auth.JOBS).strip().lower() != want:
        return None
    client = GmailClient(google_auth.JOBS)
    return client if client.is_configured() else None


def query(senders: list) -> str:
    who = " OR ".join(s for s in senders if s)
    return f"newer_than:1h from:({who})" if who else "newer_than:1h"


def wait_for(email: str, senders: list, since: float, polls: Optional[int] = None,
             poll_secs: Optional[float] = None, client=None,
             sleep: Callable[[float], None] = time.sleep) -> Optional[tuple]:
    polls = POLLS if polls is None else polls
    poll_secs = POLL_SECS if poll_secs is None else poll_secs
    client = client or client_for(email)
    if client is None:
        logger.info("email code: no connected Gmail account is %s — handing it to you", email or "the application email")
        return None
    q = query(senders)
    for attempt in range(polls):
        if attempt:
            sleep(poll_secs)
        msgs = client.list_messages(q, limit=5)
        if isinstance(msgs, str):                # not set up / auth expired / rate-limited
            logger.warning("email code: Gmail unavailable: %s", msgs)
            return None
        for m in msgs:                            # newest first
            got = client.message_bodies(m["id"])
            if isinstance(got, str) or got.get("ts", 0) < since - 5:
                continue
            found = extract(got.get("subject", ""), got.get("text", ""), got.get("html", ""))
            if found:
                logger.info("email code: found a %s from %s", found[0], got.get("from", "")[:60])
                return found
    return None
