"""features/outreach/channel.py — emails from Coldstart, and which channel to reach someone on.

    find_emails(domain, people, company) → people, each with .lookup = Coldstart's result
    recommend_channel(person) → ("email" | "linkedin", why)       pure code
    choose(people) → (person, channel, why) | None                one contact per company

Coldstart is called through its Python API (pip install -e'd into Aria's venv); it
owns the contact cache, so nothing here stores an address. An email is worth drafting
only when Coldstart verified it, the source published it, or the domain's format is
known with high confidence (catch-all / unverifiable domains). Everything else —
bounced, blocked, role inboxes, guesses — goes to LinkedIn.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

EMAIL = "email"
LINKEDIN = "linkedin"
EMAIL_MIN_CONFIDENCE = 0.8     # unverified (catch-all / unknown) addresses below this → LinkedIn
_NEVER_EMAIL = {"invalid", "blocked", "role_inbox"}


def find_emails(domain: str, people: list, company: str = "", verify: bool = True) -> list:
    """Attach Coldstart's best address to each person (`person.lookup`). Never raises."""
    for p in people:
        p.lookup = {}
    if not domain or not people:
        return people
    try:
        from coldstart import api as coldstart
        results = coldstart.find_emails(domain, [{"name": p.name, "title": p.title} for p in people],
                                        company=company, verify=verify)
    except Exception as exc:
        logger.error("outreach: Coldstart lookup for %s failed: %s", domain, exc)
        return people
    for p, r in zip(people, results or []):
        p.lookup = r or {}
    return people


def recommend_channel(person) -> tuple[str, str]:
    """Email when the address is verified, published, or a high-confidence format guess."""
    r = getattr(person, "lookup", None) or {}
    email, status, conf = r.get("email"), r.get("status") or "unknown", float(r.get("confidence") or 0)
    if not email and getattr(person, "email", ""):
        return EMAIL, "address published in the posting"
    if not email:
        return LINKEDIN, "no email found"
    if status in _NEVER_EMAIL:
        return LINKEDIN, f"email is {status.replace('_', ' ')}"
    if status == "valid":
        return EMAIL, "email verified"
    if r.get("source") in ("site", "github"):
        return EMAIL, f"address published ({r['source']})"
    if conf >= EMAIL_MIN_CONFIDENCE:
        return EMAIL, f"{status.replace('_', ' ')} domain, format confidence {conf:.2f}"
    return LINKEDIN, f"email unconfirmed ({status}, confidence {conf:.2f})"


def email_of(person) -> str:
    r = getattr(person, "lookup", None) or {}
    return r.get("email") or getattr(person, "email", "") or ""


def choose(people: list) -> Optional[tuple]:
    """The one contact for this company: the best-ranked person we can email, else the
    best-ranked person on LinkedIn."""
    if not people:
        return None
    picks = [(p,) + recommend_channel(p) for p in people]
    return next((x for x in picks if x[1] == EMAIL), picks[0])
