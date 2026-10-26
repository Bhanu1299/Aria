"""features/jobprep/followups.py — follow-up cadence after an application (table followups).

When a job turns Applied: follow-up 1 due 7 business days later; when it's drafted, follow-up
2 is due 7 business days after that; never more than 2. "Draft my follow-ups" (voice: a
direct route and an agent tool) makes jobs-Gmail DRAFTS only — never sends — to the job's
contact from Muse's Contacts tab when there is one, else as a reply in the application's
email thread. Each draft attaches the job's tailored resume.pdf when Aria prepared one.

    on_applied(job, applied_at=None)      schedules follow-up 1 (idempotent)
    sync(applied_jobs)                    on_applied for every Applied job (dashboard / runs)
    due(today=None) → [row]               follow-ups due now, not yet drafted
    upcoming(limit) → [row]
    draft_due(gmail=None, contact_fn=None, thread_fn=None, today=None) → spoken summary
    mark_done(job_id, n, link="")         drafted → the next one is scheduled (max 2)
    add_business_days(d, n) → date

Never raises out of draft_due / sync (each job's failure is reported in the summary).
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Callable, Optional

from aria.features.jobprep import store
from aria.features.jobprep.blocklist import company_key

logger = logging.getLogger(__name__)

GAP_BUSINESS_DAYS = 7
MAX_FOLLOWUPS = 2


def add_business_days(d: date, n: int) -> date:
    out = d
    while n > 0:
        out += timedelta(days=1)
        if out.weekday() < 5:
            n -= 1
    return out


def job_key(job: dict) -> str:
    jid = str(job.get("job_id") or "").strip()
    return jid.upper() if jid else f"{company_key(job.get('company', ''))}|{(job.get('role') or '').strip().lower()}"


def _day(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def on_applied(job: dict, applied_at=None) -> Optional[dict]:
    key = job_key(job)
    if not key.strip("|"):
        return None
    applied = _day(applied_at) if applied_at else date.today()
    due = add_business_days(applied, GAP_BUSINESS_DAYS)
    try:
        with store.conn() as c:
            c.execute("INSERT OR IGNORE INTO followups (job_id, company, role, n, applied_at, next_due) "
                      "VALUES (?, ?, ?, 1, ?, ?)",
                      (key, job.get("company", ""), job.get("role", ""), applied.isoformat(), due.isoformat()))
    except Exception as exc:
        logger.warning("followups: schedule failed: %s", exc)
        return None
    return {"job_id": key, "n": 1, "next_due": due.isoformat()}


def sync(applied: list) -> int:
    """applied: [(job dict, applied date | None)]. Returns how many were newly scheduled."""
    before = len(_all())
    for job, when in applied or []:
        on_applied(job, when)
    return len(_all()) - before


def _all() -> list:
    try:
        return store.rows("SELECT * FROM followups")
    except Exception as exc:
        logger.warning("followups: read failed: %s", exc)
        return []


def due(today: Optional[date] = None) -> list:
    t = (today or date.today()).isoformat()
    try:
        return store.rows("SELECT * FROM followups WHERE done_at IS NULL AND next_due <= ? ORDER BY next_due", (t,))
    except Exception as exc:
        logger.warning("followups: read failed: %s", exc)
        return []


def upcoming(limit: int = 10, today: Optional[date] = None) -> list:
    t = (today or date.today()).isoformat()
    try:
        return store.rows("SELECT * FROM followups WHERE done_at IS NULL AND next_due > ? ORDER BY next_due LIMIT ?",
                          (t, limit))
    except Exception:
        return []


def mark_done(job_id: str, n: int, link: str = "", today: Optional[date] = None) -> None:
    today = today or date.today()
    with store.conn() as c:
        row = c.execute("SELECT * FROM followups WHERE job_id = ? AND n = ?", (job_id, n)).fetchone()
        if row is None:
            return
        c.execute("UPDATE followups SET done_at = ?, draft_link = ? WHERE job_id = ? AND n = ?",
                  (today.isoformat(), link, job_id, n))
        if n < MAX_FOLLOWUPS:
            c.execute("INSERT OR IGNORE INTO followups (job_id, company, role, n, applied_at, next_due) "
                      "VALUES (?, ?, ?, ?, ?, ?)",
                      (job_id, row["company"], row["role"], n + 1, row["applied_at"],
                       add_business_days(today, GAP_BUSINESS_DAYS).isoformat()))


# ---------------------------------------------------------------------------
# drafting
# ---------------------------------------------------------------------------

def body(row: dict, name: str = "") -> tuple:
    """(subject, body) — written in code, no LLM: short, polite, specific to the role."""
    role, company = row["role"] or "the open role", row["company"] or "your team"
    applied = _day(row["applied_at"]).strftime("%B %-d")
    hello = f"Hi {name.split()[0]}," if name and name.strip() and name.strip() != "—" else "Hello,"
    if int(row["n"]) == 1:
        subject = f"Following up on my application: {role}"
        text = (f"{hello}\n\nI applied for the {role} position at {company} on {applied} and wanted to follow up. "
                f"I'm still very interested in the role and would welcome the chance to talk about how I could "
                f"help the team.\n\nI've attached my resume for convenience. Thank you for your time.\n\nBest regards,")
    else:
        subject = f"Checking in: {role} application"
        text = (f"{hello}\n\nI wanted to check in once more on my application for the {role} position at {company}. "
                f"If the role is still open, I'd be glad to share more about my background or answer any "
                f"questions.\n\nThank you again for your consideration.\n\nBest regards,")
    return subject, text


def _resume_for(row: dict) -> list:
    from aria.features.resume import prepare
    if not prepare.JOBS_DIR.is_dir():
        return []
    jid = "" if "|" in row["job_id"] else row["job_id"]
    slug = prepare.slug(row["company"])
    for folder in prepare.JOBS_DIR.iterdir():
        if folder.name.endswith(f"-{slug}") and (not jid or folder.name.upper().startswith(jid.upper() + "-")):
            pdf = folder / "resume.pdf"
            if pdf.is_file():
                return [str(pdf)]
    return []


def default_contact(row: dict) -> Optional[dict]:
    """{"name", "email"} from Muse's Contacts tab (joined on Job ID, else company), or None."""
    try:
        import aria.core.config as config
        from aria.features.outreach import contacts_tab as ct
        from aria.integrations.header_sheet import HeaderSheet
        if not config.MUSE_CONTACTS_TAB:
            return None
        records = HeaderSheet(config.MUSE_SHEET_ID, config.MUSE_CONTACTS_TAB, writable=()).records()
        want_id = row["job_id"] if "|" not in row["job_id"] else ""
        want_co = company_key(row["company"])
        for r in records:
            email = ct.first_email(r)
            if not email:
                continue
            rid = str(r.get("Job ID") or "").strip().upper()
            if (want_id and rid == want_id) or (not rid and company_key(r.get("Company", "")) == want_co) or \
                    (not want_id and company_key(r.get("Company", "")) == want_co):
                return {"name": str(r.get("Person") or ""), "email": email}
    except Exception as exc:
        logger.warning("followups: contacts lookup failed: %s", exc)
    return None


def default_thread(row: dict, gmail) -> Optional[str]:
    """The newest message about this application in the jobs Gmail (its id), or None."""
    company = (row["company"] or "").strip()
    if not company:
        return None
    try:
        got = gmail.list_messages(f'"{company}" (application OR applying OR applied) newer_than:120d', limit=5)
    except Exception as exc:
        logger.warning("followups: thread search failed: %s", exc)
        return None
    if not isinstance(got, list):
        return None
    for m in got:
        if re.search(re.escape(company), f"{m.get('subject', '')} {m.get('from', '')} {m.get('snippet', '')}", re.I):
            return m.get("id")
    return None


def draft_due(gmail=None, contact_fn: Optional[Callable] = None, thread_fn: Optional[Callable] = None,
              today: Optional[date] = None) -> str:
    todo = due(today)
    if not todo:
        nxt = upcoming(1, today)
        return "No follow-ups are due" + (f"; the next is {nxt[0]['company']} on {nxt[0]['next_due']}." if nxt else ".")
    if gmail is None:
        from aria.integrations import google_auth
        from aria.plugins.productivity.gmail import GmailClient
        gmail = GmailClient(google_auth.JOBS)
    contact_fn = contact_fn or default_contact
    thread_fn = thread_fn or (lambda row: default_thread(row, gmail))
    made, skipped = [], []
    for row in todo:
        try:
            who = contact_fn(row)
            reply_to = "" if who else (thread_fn(row) or "")
            if not who and not reply_to:
                skipped.append(f"{row['company']} (no contact or application email found)")
                continue
            subject, text = body(row, (who or {}).get("name", ""))
            got = gmail.save_draft((who or {}).get("email", ""), "" if reply_to else subject, text,
                                   reply_to_id=reply_to, attachments=_resume_for(row))
            if not got.get("ok"):
                skipped.append(f"{row['company']} ({got.get('message', 'draft failed')[:80]})")
                continue
            mark_done(row["job_id"], int(row["n"]), got.get("link", ""), today)
            made.append(row["company"])
        except Exception as exc:
            logger.warning("followups: draft for %s failed: %s", row.get("company"), exc)
            skipped.append(f"{row.get('company')} ({type(exc).__name__})")
    out = []
    if made:
        out.append(f"I drafted {len(made)} follow-up{'s' if len(made) != 1 else ''} in your jobs Gmail: "
                   f"{', '.join(made[:5])}{' and more' if len(made) > 5 else ''}. Nothing was sent.")
    if skipped:
        out.append(f"Couldn't draft {len(skipped)}: {'; '.join(skipped[:3])}.")
    return " ".join(out)
