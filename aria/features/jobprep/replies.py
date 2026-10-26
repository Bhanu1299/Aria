"""features/jobprep/replies.py — employer replies → a suggested sheet status the user approves.

classify(subject, snippet, sender) → (kind, by)
    kind: interview | rejected | offer | noise | None. Rules first (offer, rejected,
    interview, a bare "unfortunately", noise — in that order); only when no rule can tell, one small-tier LLM call
    (task reply_classify) validated to one of the four kinds. None = couldn't tell.
scan(messages, jobs) → [new suggestion]
    messages: the jobs Gmail's recent inbound mail ({id, from, subject, snippet}); each one
    from a job's company (sender domain, else the company named in subject / sender name) is
    classified once and stored in reply_suggestions. Noise is stored with no suggested status
    so it isn't asked about again.
pending() → [suggestion]          interview / rejected / offer, not yet approved or dismissed
approve(id, sheet=None) → str     JobSheet.set_cell(Status) — the ONLY sheet write, and only here
dismiss(id) → None

NOTHING is written to the sheet until the user presses Approve in the dashboard.
"""
from __future__ import annotations

import logging
import re
from email.utils import parseaddr
from typing import Optional

from aria.features.ats import untrusted
from aria.features.jobprep import store
from aria.features.jobprep.blocklist import company_key

logger = logging.getLogger(__name__)

KINDS = ("interview", "rejected", "offer", "noise")
STATUS = {"interview": "Interview", "rejected": "Rejected", "offer": "Offer"}

_OFFER = re.compile(r"\boffer letter\b|\bpleased to (?:extend|offer)\b|\bextend (?:you )?an offer\b|"
                    r"\boffer of employment\b|\bverbal offer\b|\bjob offer\b", re.I)
_REJECTED = re.compile(r"\bnot (?:to )?(?:be )?mov(?:e|ing) forward\b|\bdecided to (?:pursue|move forward with) "
                       r"other\b|\bother candidates\b|\bposition has been filled\b|\bnot (?:been )?selected\b|"
                       r"\bregret to inform\b|\bwill not be (?:moving|proceeding)\b|\bno longer (?:being )?considered\b|"
                       r"\bnot a (?:fit|match) at this time\b", re.I)
_UNFORTUNATELY = re.compile(r"\bunfortunately\b", re.I)      # a rejection only when nothing says interview
_INTERVIEW = re.compile(r"\binterview\w*\b|\bschedul\w+ (?:a |some )?(?:call|time|chat|conversation)\b|"
                        r"\byour availability\b|\bphone screen\b|\bnext steps?\b|\bcalendly\.com\b|\btechnical screen\b|"
                        r"\bonsite\b|\bmeet (?:with )?(?:the|our) team\b|\bhiring manager would like\b", re.I)
_NOISE = re.compile(r"\bthank(?:s| you) for (?:applying|your (?:application|interest))\b|\bapplication (?:received|"
                    r"confirmation|submitted)\b|\bwe(?:'ve| have) received your application\b|\bnewsletter\b|"
                    r"\bunsubscribe\b|\bjob alert\b|\bjobs? (?:you may|for you)\b|\bverify your email\b|"
                    r"\bcomplete your profile\b|\bsecurity code\b|\bverification code\b", re.I)
_AUTO_SENDER = re.compile(r"no-?reply|notifications?@|jobs-?noreply|mailer|alerts?@|talent-?community", re.I)

_SYSTEM = ("You classify an email a job seeker received from a company they applied to. Reply with exactly one "
           "word: interview (they want to talk or schedule), rejected (they're not moving forward), offer (a job "
           "offer), or noise (auto-confirmation, newsletter, marketing, anything else). " + untrusted.RULE)


def classify(subject: str, snippet: str = "", sender: str = "", use_llm: bool = True) -> tuple:
    text = f"{subject}\n{snippet}"
    if _OFFER.search(text):
        return "offer", "rules"
    if _REJECTED.search(text):
        return "rejected", "rules"
    if _INTERVIEW.search(text) and not _NOISE.search(subject or ""):
        return "interview", "rules"
    if _UNFORTUNATELY.search(text):
        return "rejected", "rules"
    if _NOISE.search(text) or _AUTO_SENDER.search(sender or ""):
        return "noise", "rules"
    if not use_llm:
        return None, "rules"
    from aria.llm import llm_client
    block, _flags = untrusted.guard(f"From: {sender}\nSubject: {subject}\n\n{snippet[:1500]}", "Email")

    def validate(raw: str):
        word = (raw or "").strip().lower().strip(".\"' ")
        return word if word in KINDS else None

    try:
        _resp, kind = llm_client.complete_checked("reply_classify", [{"role": "user", "content": block}],
                                                  validate=validate, system=_SYSTEM, max_tokens=5)
    except Exception as exc:
        logger.warning("replies: LLM classify failed: %s", exc)
        return None, "llm"
    return kind, "llm"


def _sender_domain(sender: str) -> str:
    addr = parseaddr(sender or "")[1].lower()
    return addr.rsplit("@", 1)[-1] if "@" in addr else ""


def match_job(msg: dict, jobs: list) -> Optional[dict]:
    """The job this email is about: the sender's domain names the company, else the
    company's name is in the subject or the sender's display name."""
    dom = _sender_domain(msg.get("from", ""))
    label = re.sub(r"[^a-z0-9]", "", dom.split(".")[-2] if dom.count(".") >= 1 else dom)
    name = parseaddr(msg.get("from", ""))[0]
    for j in jobs:
        key = company_key(j.get("company", ""))
        if len(key) < 3:
            continue
        if label and (label == key or (len(label) >= 4 and (key.startswith(label) or label.startswith(key)))):
            return j
    for j in jobs:
        co = (j.get("company") or "").strip()
        if len(company_key(co)) >= 3 and re.search(r"(?<!\w)" + re.escape(co) + r"(?!\w)",
                                                    f"{msg.get('subject', '')} {name}", re.I):
            return j
    return None


def scan(messages: list, jobs: list, use_llm: bool = True) -> list:
    """Classifies messages not seen before; returns the new suggestions (noise included)."""
    try:
        seen = {r["thread_id"] for r in store.rows("SELECT thread_id FROM reply_suggestions")}
    except Exception as exc:
        logger.warning("replies: read failed: %s", exc)
        return []
    new = []
    for m in messages or []:
        mid = str(m.get("thread_id") or m.get("id") or "")
        if not mid or mid in seen:
            continue
        job = match_job(m, jobs)
        if job is None:
            continue
        kind, by = classify(m.get("subject", ""), m.get("snippet", ""), m.get("from", ""), use_llm=use_llm)
        if kind is None:
            continue
        row = {"job_id": str(job.get("job_id") or ""), "company": job.get("company", ""), "role": job.get("role", ""),
               "row": job.get("row"), "thread_id": mid, "subject": (m.get("subject") or "")[:200],
               "sender": (m.get("from") or "")[:200], "kind": kind, "by": by, "suggested_status": STATUS.get(kind, "")}
        try:
            with store.conn() as c:
                c.execute("INSERT OR IGNORE INTO reply_suggestions (job_id, company, role, row, thread_id, subject, "
                          "sender, kind, by, suggested_status) VALUES (:job_id, :company, :role, :row, :thread_id, "
                          ":subject, :sender, :kind, :by, :suggested_status)", row)
        except Exception as exc:
            logger.warning("replies: store failed: %s", exc)
            continue
        seen.add(mid)
        new.append(row)
    return new


def scan_gmail(jobs: list, gmail=None, days: int = 30, use_llm: bool = True) -> str:
    """scan() over the jobs Gmail's recent inbound mail. One spoken / dashboard line."""
    try:
        if gmail is None:
            from aria.integrations import google_auth
            from aria.plugins.productivity.gmail import GmailClient
            gmail = GmailClient(google_auth.JOBS)
        got = gmail.recent_messages(days=days, limit=200)
        if isinstance(got, str):
            return got
        me = gmail.my_address() if hasattr(gmail, "my_address") else ""
        inbound = [m for m in got if not me or me not in (m.get("from") or "").lower()]
        new = [s for s in scan(inbound, jobs, use_llm=use_llm) if s["kind"] != "noise"]
    except Exception as exc:
        logger.warning("replies: Gmail scan failed: %s", exc)
        return f"I couldn't check the jobs Gmail: {type(exc).__name__}."
    if not new:
        return "No new replies to act on."
    return (f"{len(new)} new repl{'ies' if len(new) != 1 else 'y'}: "
            + ", ".join(f"{s['company']} ({s['kind']})" for s in new[:5]) + ". Approve the status in the dashboard.")


def pending() -> list:
    try:
        return store.rows("SELECT * FROM reply_suggestions WHERE suggested_status != '' AND approved_at IS NULL "
                          "AND rejected_at IS NULL ORDER BY created_at DESC")
    except Exception as exc:
        logger.warning("replies: read failed: %s", exc)
        return []


def _get(sid: int) -> Optional[dict]:
    got = store.rows("SELECT * FROM reply_suggestions WHERE id = ?", (int(sid),))
    return got[0] if got else None


def approve(sid: int, sheet=None) -> str:
    """Writes the suggested status to the job's Status cell (row re-found by Job ID / company
    + role), then marks the suggestion approved. Raises ValueError / SheetError on failure
    (the dashboard shows it)."""
    s = _get(sid)
    if s is None or not s["suggested_status"]:
        raise ValueError("That suggestion is gone.")
    if s["approved_at"] or s["rejected_at"]:
        raise ValueError("That suggestion was already handled.")
    from aria.integrations import sheets
    args = (int(s["row"] or 0), sheets.COL_STATUS, s["suggested_status"])
    kw = {"company": s["company"] or None, "role": s["role"] or None, "job_id": s["job_id"] or None}
    if sheet is not None:
        row = sheet.set_cell(*args, **kw)
    else:
        import aria.core.config as config
        try:
            row = sheets.default_sheet().set_cell(*args, **kw)
        except sheets.SheetError:                       # moved to the Applied tab since
            if not config.MUSE_APPLIED_TAB:
                raise
            row = sheets.tab_sheet(config.MUSE_APPLIED_TAB).set_cell(0, *args[1:], **kw)
    with store.conn() as c:
        c.execute("UPDATE reply_suggestions SET approved_at = datetime('now') WHERE id = ?", (int(sid),))
    return f"{s['company']}: Status set to {s['suggested_status']} (row {row})."


def dismiss(sid: int) -> None:
    with store.conn() as c:
        c.execute("UPDATE reply_suggestions SET rejected_at = datetime('now') WHERE id = ?", (int(sid),))
