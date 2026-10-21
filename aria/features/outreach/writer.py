"""features/outreach/writer.py — the outreach text. The LLM writes the body; code does the rest.

    email(target) → {"to", "subject", "body"} | None        task outreach_email (frontier)
    linkedin_note(target) → str | None                     task outreach_linkedin_note (frontier)

Code owns the subject ("<Role> — <Name>", which the dashboard parses to link the thread
to its job), greeting, sign-off and length limits; the LLM only writes the middle, from
the outreach profile (education, experience, skills, the sheet's fit notes) and the
roles. The profile never includes legal or demographic facts. Output that fails
validation (length, placeholders, a greeting or sign-off of its own) is dropped —
never sent anywhere, and the caller records that nothing was drafted.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from aria.features import answer_bank
from aria.llm import llm_client

logger = logging.getLogger(__name__)

NOTE_LIMIT = 300                 # LinkedIn's connection-note cap
EMAIL_WORDS = (50, 180)
_PROFILE_KEYS = ("preferred_name", "first_name", "last_name", "school", "degree", "grad_date",
                 "years_experience", "current_company", "location", "linkedin", "github")
_IDENTITY_KEYS = ("summary", "skills", "experience", "education", "target_roles", "projects")
_PLACEHOLDER = re.compile(r"\[[^\]]*\]|\{[^}]*\}|<[^>]+>|\bTODO\b|lorem ipsum", re.I)
_GREETING = re.compile(r"(?i)^\s*(hi|hello|hey|dear)\b")
_SIGNOFF = re.compile(r"(?im)^\s*(best|thanks|thank you|regards|cheers|sincerely)[,!.]?\s*$")

_EMAIL_SYSTEM = (
    "You write the body of a short cold email from a job candidate to someone at a company "
    "they want to work at. Plain text, first person, 2 short paragraphs, 70-150 words. Say which "
    "role(s), one or two concrete reasons the candidate fits drawn ONLY from the profile and fit "
    "notes, and one easy ask (a 15-minute chat, or a pointer to the right person). Never invent "
    "projects, employers, numbers or credentials; never mention visa, sponsorship, age or other "
    "personal facts. No greeting, no sign-off, no subject line, no placeholders — the greeting "
    "and signature are added for you."
)
_NOTE_SYSTEM = (
    "You write a LinkedIn connection note from a job candidate to someone at a company. Plain "
    "text, one or two sentences, at most 230 characters. Name the role, one concrete reason the "
    "candidate fits from the profile only, and a light ask. No greeting (it's added for you), no "
    "sign-off, no hashtags, no emoji, no placeholders, never invent facts."
)


def _identity() -> dict:
    from aria.features.ats.greenhouse import _identity as identity
    return identity()


def sender_name(bank: Optional[dict] = None) -> str:
    a = (bank or answer_bank.load())["answers"]
    first = a.get("preferred_name") or a.get("first_name") or ""
    return " ".join(x for x in (first, a.get("last_name") or "") if x).strip()


def outreach_profile(target) -> str:
    """Facts the LLM may use: education/experience basics, identity summary, fit notes."""
    bank = answer_bank.load()
    lines = [f"{k}: {bank['answers'][k]}" for k in _PROFILE_KEYS
             if bank["answers"].get(k) not in (None, "") and not str(bank["answers"][k]).startswith("<")]
    ident = _identity()
    for k in _IDENTITY_KEYS:
        if ident.get(k):
            lines.append(f"{k}: {ident[k] if isinstance(ident[k], str) else json.dumps(ident[k])}")
    for j in target.jobs:
        fit = next((v for k, v in (j.get("fields") or {}).items() if k.lower().startswith("why it's a fit")), "")
        if fit:
            lines.append(f"fit notes for {j['role']}:\n{fit[:1500]}")
    return "\n".join(lines)


def roles_text(target) -> str:
    roles = list(dict.fromkeys(j["role"] for j in target.jobs if j.get("role")))
    if len(roles) <= 1:
        return roles[0] if roles else "a role"
    return ", ".join(roles[:-1]) + " and " + roles[-1]


def _brief(target) -> str:
    p = target.contact
    return (f"Company: {target.company}\nRole(s): {roles_text(target)}\n"
            f"Recipient: {p.name}, {p.title or 'works there'}\n\nCandidate profile:\n{outreach_profile(target)}")


def first_name(name: str) -> str:
    return (name or "").split()[0] if name else "there"


def _clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip().strip('"').strip())


def _valid_body(text: str) -> Optional[str]:
    text = _clean(text)
    words = len(text.split())
    if not EMAIL_WORDS[0] <= words <= EMAIL_WORDS[1] or _PLACEHOLDER.search(text):
        return None
    if _GREETING.match(text) or _SIGNOFF.search(text) or re.match(r"(?i)subject:", text):
        return None
    return text


def _note_validator(prefix: str):
    def check(text: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", _clean(text))
        if not text or _PLACEHOLDER.search(text) or _GREETING.match(text) or "#" in text:
            return None
        note = prefix + text
        return note if len(note) <= NOTE_LIMIT else None
    return check


def email(target, to: str) -> Optional[dict]:
    """Draft for the target's contact. None when the LLM output didn't pass validation."""
    try:
        _resp, body = llm_client.complete_checked(
            "outreach_email", [{"role": "user", "content": _brief(target)}],
            validate=_valid_body, system=_EMAIL_SYSTEM, max_tokens=500)
    except Exception as exc:
        logger.error("outreach: email for %s failed: %s", target.company, exc)
        return None
    if not body:
        return None
    me = sender_name()
    sign = "\n".join(x for x in (me, answer_bank.load()["answers"].get("linkedin") or "") if x)
    full = f"Hi {first_name(target.contact.name)},\n\n{body}\n\nBest,\n{sign}".rstrip()
    return {"to": to, "subject": f"{roles_text(target)} — {me or target.company}", "body": full}


def linkedin_note(target) -> Optional[str]:
    prefix = f"Hi {first_name(target.contact.name)}, "
    try:
        _resp, note = llm_client.complete_checked(
            "outreach_linkedin_note", [{"role": "user", "content": _brief(target)}],
            validate=_note_validator(prefix), system=_NOTE_SYSTEM, max_tokens=200)
    except Exception as exc:
        logger.error("outreach: LinkedIn note for %s failed: %s", target.company, exc)
        return None
    return note or None
