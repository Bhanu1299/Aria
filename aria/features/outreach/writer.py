"""features/outreach/writer.py — the outreach text. The LLM writes the body; code does the rest.

    email(target, to, posting="", fact=None) → {"to", "subject", "body"} | None   task outreach_email
    linkedin_note(target, fact=None) → str | None                  task outreach_linkedin_note

Every email is about the company: `fact` (company_fact.Fact — what they solve, read off
their own pages) is REQUIRED, and the body ties it to the candidate's closest project or
job. No fact → None (the caller holds the draft; a generic email is worse than none).

Code owns the subject ("<Role> — <Name>, <hook>": the dashboard links the thread to its
job by the part before " — "; the LLM writes only the short hook), greeting, sign-off
(name, LinkedIn | GitHub) and the checks; the LLM writes the hook and the middle from
the outreach profile (today's date, education, experience, the resume text, the sheet's
fit notes), the roles and the fact. The profile never includes legal or demographic
facts. Output is dropped — never sent anywhere — when it fails a check: length, a
placeholder, its own greeting or sign-off, a filler opener, no mention of the company's
problem, or a number that isn't in the profile, resume, fact or posting.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from aria.features import answer_bank
from aria.features.outreach.company_fact import words as _words
from aria.llm import llm_client

logger = logging.getLogger(__name__)

NOTE_LIMIT = 300                 # LinkedIn's connection-note cap
EMAIL_WORDS = (60, 125)          # aim ~90: founders skim
_PROFILE_KEYS = ("preferred_name", "first_name", "last_name", "school", "degree", "grad_date",
                 "years_experience", "current_company", "location", "linkedin", "github")
_IDENTITY_KEYS = ("summary", "skills", "experience", "education", "target_roles", "projects")
RESUME_CHARS = 5000             # resume text the writer sees (the facts it may use)
_PLACEHOLDER = re.compile(r"\[[^\]]*\]|\{[^}]*\}|<[^>]+>|\bTODO\b|lorem ipsum", re.I)
_GREETING = re.compile(r"(?i)^\s*(hi|hello|hey|dear)\b")
_SIGNOFF = re.compile(r"(?im)^\s*(best|thanks|thank you|regards|cheers|sincerely)[,!.]?\s*$")

_EMAIL_SYSTEM = (
    "You write a short cold email from a job candidate to someone at a company they want to "
    "work at. Output exactly this:\n"
    "HOOK: <3-6 words on what the candidate brings that fits THIS company; lowercase except "
    "acronyms like LLM>\n"
    "<blank line>\n"
    "<the body>\n\n"
    "The body is plain text, first person, about 90 words (never over 115), three short "
    "paragraphs:\n"
    "1. One sentence: who the candidate is (degree, school, when it ended; use today's date "
    "for tense) and that they're applying for the role. Then one sentence on the problem the company is solving, from "
    "the company fact, and why it caught the candidate's eye.\n"
    "2. The candidate's one or two projects or jobs closest to that problem, from the resume "
    "only, each with how it connects to what the company does. At most one number per item, "
    "copied exactly from the resume.\n"
    "3. One easy ask: 15 minutes this week, or a pointer to the right engineer.\n\n"
    "Never open with filler (\"I am reaching out\", \"express my interest\", \"I hope this "
    "finds you\", \"My name is\"). Never invent projects, employers, numbers, credentials, or "
    "anything about the company beyond the company fact. Never mention visa, sponsorship, age "
    "or other personal facts. No greeting, sign-off, subject line or placeholders — the "
    "greeting and signature are added for you."
)
_NOTE_SYSTEM = (
    "You write a LinkedIn connection note from a job candidate to someone at a company. Plain "
    "text, one or two sentences, at most 230 characters. Name the role, one concrete reason the "
    "candidate fits from the profile only, and a light ask. No greeting (it's added for you), no "
    "sign-off, no hashtags, no emoji, no placeholders, never invent facts."
)


def _identity() -> dict:
    from aria.features.ats.planner import _identity as identity
    return identity()


def _resume() -> str:
    """The resume's text (the same one form answers are written from), trimmed. '' if none."""
    try:
        from aria.features.ats import planner, resume_text
        return resume_text.text(planner.resume_path())[:RESUME_CHARS]
    except Exception as exc:
        logger.info("outreach: no resume text: %s", exc)
        return ""


def sender_name(bank: Optional[dict] = None) -> str:
    a = (bank or answer_bank.load())["answers"]
    first = a.get("preferred_name") or a.get("first_name") or ""
    return " ".join(x for x in (first, a.get("last_name") or "") if x).strip()


def outreach_profile(target) -> str:
    """Facts the LLM may use: today's date, education/experience basics, identity summary,
    the resume, fit notes."""
    bank = answer_bank.load()
    lines = [f"today: {time.strftime('%Y-%m-%d')}"]
    lines += [f"{k}: {bank['answers'][k]}" for k in _PROFILE_KEYS
             if bank["answers"].get(k) not in (None, "") and not str(bank["answers"][k]).startswith("<")]
    ident = _identity()
    for k in _IDENTITY_KEYS:
        if ident.get(k):
            lines.append(f"{k}: {ident[k] if isinstance(ident[k], str) else json.dumps(ident[k])}")
    resume = _resume()
    if resume:
        lines.append(f"resume:\n{resume}")
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


def _brief(target, posting: str = "", fact=None) -> str:
    p = target.contact
    text = (f"Company: {target.company}\nRole(s): {roles_text(target)}\n"
            f"Recipient: {p.name}, {p.title or 'works there'}\n\nCandidate profile:\n{outreach_profile(target)}")
    if fact is not None:
        text += f"\n\nCompany fact (from their own page, {fact.url}):\n{fact.problem}\n{fact.detail}"
    if posting:
        text += f"\n\nJob posting (for what the role needs; quote nothing):\n{posting}"
    return text


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


_FILLER = re.compile(r"(?i)\b(?:i am|i'm) (?:reaching|writing) out|express(?:ing)? my (?:\w+ )?interest"
                     r"|i hope (?:this|you)|\bmy name is\b")
_HOOK = re.compile(r"(?i)^\s*hook:\s*(.+?)\s*$")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ASK = re.compile(r"(?i)\b\d+[- ]?min(?:ute)?s?\b")


def _numbers(text: str) -> set:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(_ASK.sub(" ", text or ""))}


def _email_validator(brief: str, fact, company: str):
    """(hook, body) when the output passes every check, else None."""
    known = _numbers(brief)

    def check(text: str):
        lines = _clean(text).split("\n", 1)
        m = _HOOK.match(lines[0]) if lines else None
        if not m or len(lines) < 2:
            return None
        hook = m.group(1).strip().rstrip(".").strip('"')
        body = _valid_body(lines[1])
        if not body or not (2 <= len(hook.split()) <= 7) or len(hook) > 45 or _PLACEHOLDER.search(hook):
            return None
        if _FILLER.search(body):
            return None
        if any(n not in known and len(n) > 1 for n in _numbers(body)):
            return None                         # a number the candidate never wrote
        if not _mentions(body, fact, company):
            return None
        return hook, body
    return check


def _mentions(body: str, fact, company: str) -> bool:
    """The body names the company's problem: at least two of the fact's content words."""
    want = _words(f"{fact.problem} {fact.detail}") - _words(company)
    return len(want & _words(body)) >= 2


def signature(bank: Optional[dict] = None) -> str:
    """Name, then 'linkedin.com/in/x | github.com/y' (whichever the bank has)."""
    a = (bank or answer_bank.load())["answers"]
    links = " | ".join(re.sub(r"^https?://(www\.)?", "", str(a[k])).rstrip("/")
                       for k in ("linkedin", "github") if a.get(k) and not str(a[k]).startswith("<"))
    return "\n".join(x for x in (sender_name({"answers": a}), links) if x)


def _note_validator(prefix: str):
    def check(text: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", _clean(text))
        if not text or _PLACEHOLDER.search(text) or _GREETING.match(text) or "#" in text:
            return None
        note = prefix + text
        return note if len(note) <= NOTE_LIMIT else None
    return check


def email(target, to: str, posting: str = "", fact=None) -> Optional[dict]:
    """Draft for the target's contact, about `fact` (what the company solves). None when
    there is no fact or the LLM output didn't pass the checks.
    posting: text of the job page, so the email speaks to that role's needs."""
    if fact is None:
        logger.info("outreach: no company fact for %s; no email", target.company)
        return None
    brief = _brief(target, posting, fact)
    try:
        _resp, out = llm_client.complete_checked(
            "outreach_email", [{"role": "user", "content": brief}],
            validate=_email_validator(brief, fact, target.company), system=_EMAIL_SYSTEM, max_tokens=600)
    except Exception as exc:
        logger.error("outreach: email for %s failed: %s", target.company, exc)
        return None
    if not out:
        return None
    hook, body = out
    me = sender_name()
    full = f"Hi {first_name(target.contact.name)},\n\n{body}\n\nBest,\n{signature()}".rstrip()
    return {"to": to, "subject": f"{roles_text(target)} — {me or target.company}, {hook}", "body": full}


def linkedin_note(target, fact=None) -> Optional[str]:
    prefix = f"Hi {first_name(target.contact.name)}, "
    try:
        _resp, note = llm_client.complete_checked(
            "outreach_linkedin_note", [{"role": "user", "content": _brief(target, fact=fact)}],
            validate=_note_validator(prefix), system=_NOTE_SYSTEM, max_tokens=200)
    except Exception as exc:
        logger.error("outreach: LinkedIn note for %s failed: %s", target.company, exc)
        return None
    return note or None
