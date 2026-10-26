"""features/jobprep/blocklist.py — the do-not-apply list (table do_not_apply).

An entry is a company (matched case- and punctuation-insensitively, "Acme, Inc." = "acme")
or a domain (the posting URL's host is that domain or a subdomain of it). A hit skips the
job in batch runs (noted in the end report); a single "apply to X" asks once.

    add(text, reason="", scope=None) → entry dict    scope guessed: "x.com" → domain
    remove(text) → bool
    entries() → [entry]
    hit(job) → entry | None                          job: {"company", "url" / "apply_link" / "post_link"}

Never raises on reads (an unreadable db = no entries); add/remove raise ValueError on empty input.
"""
from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlparse

from aria.features.jobprep import store

logger = logging.getLogger(__name__)

_SUFFIXES = re.compile(r"\b(?:inc|llc|ltd|corp|corporation|co|company|technologies|technology|labs|group|holdings|"
                       r"the|gmbh|plc|ai)\b\.?", re.I)
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?:/.*)?$", re.I)


def company_key(name: str) -> str:
    """'Acme, Inc.' → 'acme'; 'The Trade Desk' → 'tradedesk'."""
    stripped = _SUFFIXES.sub(" ", (name or "").lower())
    key = re.sub(r"[^a-z0-9]", "", stripped)
    return key or re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _domain(text: str) -> str:
    m = _DOMAIN.match((text or "").strip())
    return m.group(1).lower() if m else ""


def add(text: str, reason: str = "", scope: Optional[str] = None) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Say which company or domain.")
    dom = _domain(text) if scope in (None, "domain") else ""
    scope = "domain" if dom and (scope == "domain" or " " not in text) else "company"
    value = dom if scope == "domain" else company_key(text)
    if not value:
        raise ValueError(f"I can't tell which company “{text}” is.")
    with store.conn() as c:
        c.execute("INSERT INTO do_not_apply (value, label, scope, reason) VALUES (?, ?, ?, ?) "
                  "ON CONFLICT(value) DO UPDATE SET label=excluded.label, reason=excluded.reason",
                  (value, text, scope, reason or ""))
    return {"value": value, "label": text, "scope": scope, "reason": reason or ""}


def remove(text: str) -> bool:
    keys = {company_key(text), _domain(text)} - {""}
    if not keys:
        return False
    with store.conn() as c:
        n = c.execute(f"DELETE FROM do_not_apply WHERE value IN ({','.join('?' * len(keys))})", tuple(keys)).rowcount
    return n > 0


def entries() -> list:
    try:
        return store.rows("SELECT value, label, scope, reason, since FROM do_not_apply ORDER BY since DESC")
    except Exception as exc:
        logger.warning("blocklist: read failed: %s", exc)
        return []


def _host(job: dict) -> str:
    for k in ("url", "apply_link", "post_link", "link"):
        u = (job.get(k) or "").strip()
        if u:
            host = (urlparse(u if "://" in u else f"https://{u}").hostname or "").lower()
            if host:
                return host[4:] if host.startswith("www.") else host
    return ""


def hit(job: dict, listed: Optional[list] = None) -> Optional[dict]:
    listed = entries() if listed is None else listed
    if not listed:
        return None
    key = company_key(job.get("company", ""))
    host = _host(job)
    for e in listed:
        if e["scope"] == "company" and key and e["value"] == key:
            return e
        if e["scope"] == "domain" and host and (host == e["value"] or host.endswith("." + e["value"])):
            return e
    return None


def describe(e: dict) -> str:
    return f"{e['label']} is on your do-not-apply list" + (f" ({e['reason']})" if e.get("reason") else "")
