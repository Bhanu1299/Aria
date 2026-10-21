"""features/outreach/pipeline.py — sheet rows → people → emails → drafts / LinkedIn notes.

    run_outreach(progress_fn=None) → one or two spoken sentences. Never raises. Never sends.

A row qualifies (needs_outreach) when its fit ("Profile alignment /5") is ≥ MIN_FIT,
its Status isn't closed, the outreach column is empty, and either "Who to contact"
names nobody (Muse writes placeholders like "Via Breezy ATS") or Aria filled it on an
earlier run whose draft didn't get written.

Per company:
  1. contacts.find_people() → 1–3 people → "Who to contact" (placeholder kept after them)
  2. Coldstart find_emails (cache-first by domain) → channel.choose(): one contact + channel
  3. email    → writer.email() → Gmail DRAFT in the jobs account → outreach column
                "Email draft ready — Name (Title), address"
     linkedin → writer.linkedin_note() → outreach column "LinkedIn (Aria): Name (Title) —
                profile\\nNote: …" (the dashboard's Today tab lists it to send by hand)
Every step is in the run log; each company's LLM spend goes to the flight log by row.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from aria.dashboard.model import contact_name, emails_in, is_company_address
from aria.features.outreach import channel, writer
from aria.features.outreach import contacts as finder
from aria.integrations import google_auth, sheets
from aria.llm import context as llm_context
from aria.llm import meter
from aria.observability import flight_recorder, run_log

logger = logging.getLogger(__name__)

MIN_FIT = 3.0
_CLOSED = re.compile(r"(?i)not gonna apply|not applying|skipped|reject|closed|withdrawn")
FOUND_NOTE = "Aria found contacts"
LINKEDIN_PREFIX = "LinkedIn (Aria):"
_WHO_PERSON = re.compile(r"^\s*(?P<name>[^()]+?)\s*(?:\((?P<title>[^)]*)\))?\s*$")


@dataclass
class Target:
    """One outreach unit: a company and the sheet rows it covers."""
    company: str
    jobs: list = field(default_factory=list)
    domain: str = ""
    people: list = field(default_factory=list)
    new_people: bool = False        # found this run (→ written to "Who to contact")
    contact: object = None          # the one Person this company's outreach goes to
    channel: str = ""               # channel.EMAIL | channel.LINKEDIN
    why: str = ""
    outcome: str = ""               # drafted | noted | no contact | not written: …

    @property
    def primary(self) -> dict:
        return self.jobs[0]


def fit_score(job: dict) -> float:
    m = re.search(r"\d+(?:\.\d+)?", job.get("alignment") or "")
    return float(m.group()) if m else 0.0


def _open_high_fit(job: dict) -> bool:
    fields = job.get("fields") or {}
    return (fit_score(job) >= MIN_FIT and not _CLOSED.search(job.get("status") or "")
            and bool(job.get("company")) and not fields.get(sheets.COL_OUTREACH, "").strip())


def needs_contact(job: dict) -> bool:
    """High fit, open, no outreach yet, and nobody named in "Who to contact"."""
    if not _open_high_fit(job):
        return False
    fields = job.get("fields") or {}
    who = fields.get(sheets.COL_WHO, "")
    if contact_name(who):
        return False
    return not any(is_company_address(e) for e in emails_in(who, fields.get("Outreach", "")))


def found_earlier(job: dict) -> bool:
    """Aria named people on an earlier run but no outreach was written."""
    return _open_high_fit(job) and FOUND_NOTE in (job.get("fields") or {}).get(sheets.COL_UPDATES, "")


def people_from_who(cell: str) -> list:
    """'Pat Lee (Recruiter); Ana Diaz (CTO) | Via ATS' → [Person, Person]."""
    out = []
    for part in (cell or "").split(" | ")[0].split(";"):
        m = _WHO_PERSON.match(part)
        if m and finder.looks_like_name(m.group("name")):
            out.append(finder.Person(m.group("name"), (m.group("title") or "").strip(), "sheet"))
    return out


def who_cell(people: list, existing: str) -> str:
    """'Pat Lee (Recruiter); Sam Roe (CTO)', with any placeholder kept after a ' | '."""
    names = "; ".join(p.label() for p in people)
    existing = (existing or "").strip()
    return f"{names} | {existing}" if existing else names


def targets(jobs: list[dict]) -> list[Target]:
    out = []
    for j in jobs:
        if needs_contact(j):
            out.append(Target(j["company"], [j]))
        elif found_earlier(j):
            out.append(Target(j["company"], [j], people=people_from_who((j.get("fields") or {}).get(sheets.COL_WHO, ""))))
    return out


def _say(progress_fn: Optional[Callable[[str], None]], msg: str) -> None:
    if progress_fn:
        try:
            progress_fn(msg)
        except Exception:
            pass


def _plural(n: int) -> str:
    return f"{n} compan{'y' if n == 1 else 'ies'}"


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def find_contacts(t: Target, run: run_log.Run, search_fn=None, client=None) -> None:
    with run.step("find contacts") as s:
        t.domain = finder.company_domain(t.primary, search_fn)
        if t.people:
            s.detail = f"{t.company} ({t.domain or 'no domain'}): from the sheet — " + ", ".join(p.label() for p in t.people)
            return
        urls = [u for j in t.jobs for u in (j.get("post_link"), j.get("apply_link"))]
        t.people = finder.find_people(t.company, t.domain, urls, search_fn=search_fn, client=client)
        t.new_people = True
        s.detail = f"{t.company} ({t.domain or 'no domain'}): " + (
            ", ".join(f"{p.label()} [{p.source}]" for p in t.people) or "nobody found")


def find_email(t: Target, run: run_log.Run) -> None:
    with run.step("find email") as s:
        if t.domain:
            channel.find_emails(t.domain, t.people, company=t.company)
        else:
            s.detail = "no company domain, so no email lookup; "
        pick = channel.choose(t.people)
        if pick:
            t.contact, t.channel, t.why = pick
            addr = channel.email_of(t.contact)
            s.detail += f"{t.contact.label()} via {t.channel}" + (f" <{addr}>" if t.channel == channel.EMAIL else "") + f" — {t.why}"


def write_contacts(t: Target, run: run_log.Run, sheet: sheets.JobSheet) -> str:
    """Names → "Who to contact" on every row of the target. Returns an error string or ""."""
    with run.step("update sheet") as s:
        for j in t.jobs:
            try:
                sheet.set_fields(j["row"], {sheets.COL_WHO: who_cell(t.people, (j.get("fields") or {}).get(sheets.COL_WHO, ""))},
                                 note=FOUND_NOTE, company=j["company"], role=j["role"])
            except sheets.SheetError as exc:
                s.status, s.detail = "error", str(exc)
                return str(exc)
        s.detail = f"Who to contact on row{'s' if len(t.jobs) > 1 else ''} " + ", ".join(str(j["row"]) for j in t.jobs)
    return ""


def outreach_cell(t: Target, note: str = "") -> str:
    p = t.contact
    if t.channel == channel.EMAIL:
        return f"Email draft ready — {p.label()}, {channel.email_of(p)}"
    where = p.url if "linkedin.com" in (p.url or "") else "search LinkedIn for them"
    return f"{LINKEDIN_PREFIX} {p.label()} — {where}\nNote: {note}"


def draft(t: Target, run: run_log.Run, gmail) -> str:
    """Write the email draft or LinkedIn note. Returns the outreach cell text, or "" if nothing was written."""
    with run.step("draft") as s:
        if t.channel == channel.EMAIL:
            msg = writer.email(t, channel.email_of(t.contact))
            if msg is None:
                s.status, s.detail, t.outcome = "error", "the email didn't pass checks; no draft", "not written: email failed checks"
                return ""
            result = gmail.create_draft(msg["to"], msg["subject"], msg["body"])
            if not result.startswith("Draft saved"):
                s.status, s.detail, t.outcome = "error", result, f"not written: {result}"
                return ""
            s.detail, t.outcome = f"Gmail draft (jobs account) to {msg['to']}: {msg['subject']}", "drafted"
            return outreach_cell(t)
        note = writer.linkedin_note(t)
        if note is None:
            s.status, s.detail, t.outcome = "error", "the LinkedIn note didn't pass checks", "not written: note failed checks"
            return ""
        s.detail, t.outcome = f"LinkedIn note for {t.contact.name} ({len(note)} chars)", "noted"
        return outreach_cell(t, note)


def write_outreach(t: Target, cell: str, run: run_log.Run, sheet: sheets.JobSheet) -> str:
    with run.step("update sheet") as s:
        how = "email draft in the jobs Gmail" if t.channel == channel.EMAIL else "LinkedIn note to send"
        try:
            sheet.set_fields(t.primary["row"], {sheets.COL_OUTREACH: cell},
                             note=f"Aria outreach: {how} ({t.contact.name})",
                             company=t.primary["company"], role=t.primary["role"])
        except sheets.SheetError as exc:
            s.status, s.detail = "error", str(exc)
            return str(exc)
        s.detail = f"outreach on row {t.primary['row']}"
    return ""


def process(t: Target, run: run_log.Run, sheet: sheets.JobSheet, gmail, search_fn=None, client=None) -> None:
    """All steps for one company, LLM calls labelled with its primary row. Never raises."""
    j = t.primary
    run.job(t.company, j["role"], j["row"])
    before = meter.snapshot()
    try:
        with llm_context.job(run_log.job_label(t.company, j["role"], j["row"]), run.id):
            find_contacts(t, run, search_fn, client)
            if not t.people:
                t.outcome = "no contact"
                return
            if t.new_people:
                err = write_contacts(t, run, sheet)
                if err:
                    t.outcome = f"not written: {err}"
                    return
            find_email(t, run)
            cell = draft(t, run, gmail)
            if cell:
                err = write_outreach(t, cell, run, sheet)
                if err:
                    t.outcome = f"not written: {err}"
    except Exception as exc:
        logger.error("outreach: %s failed: %s", t.company, exc)
        t.outcome = f"not written: {exc}"
    finally:
        ok = t.outcome in ("drafted", "noted")
        flight_recorder.record_job(j["row"], t.company, j["role"], "outreach", "", meter.since(before), ok,
                                   f"{t.outcome}" + (f" via {t.channel}" if t.channel else ""))


def run_outreach(
    progress_fn: Optional[Callable[[str], None]] = None,
    sheet: Optional[sheets.JobSheet] = None,
    search_fn=None,
    client=None,
    gmail=None,
) -> str:
    """The outreach loop over every qualifying sheet row. Never raises, never sends."""
    if search_fn is None:
        from aria.web import websearch
        search_fn = lambda q: websearch.search(q, max_results=8)
    if gmail is None:
        from aria.plugins.productivity.gmail import GmailClient
        gmail = GmailClient(google_auth.JOBS)
    sheet = sheet or sheets.default_sheet()
    run = run_log.start("outreach")
    try:
        with run.step("read sheet") as s:
            jobs = sheet.jobs()
            todo = targets(jobs)
            s.detail = f"{len(jobs)} rows; {len(todo)} need outreach (fit ≥ {MIN_FIT:g}, no outreach yet)"
    except sheets.SheetError as exc:
        run.finish("error", str(exc))
        return str(exc)
    except Exception as exc:
        logger.error("outreach: reading the sheet failed: %s", exc)
        run.finish("error", f"reading the sheet failed: {exc}")
        return "I couldn't read your job sheet right now."
    if not todo:
        run.finish("no change", "no rows need outreach")
        return "Every job with a fit of 3 or more already has outreach planned."

    _say(progress_fn, f"Working on outreach for {_plural(len(todo))}. This takes a few minutes.")
    for t in todo:
        process(t, run, sheet, gmail, search_fn, client)
    return finish(run, todo)


def finish(run: run_log.Run, todo: list) -> str:
    drafted = [t for t in todo if t.outcome == "drafted"]
    noted = [t for t in todo if t.outcome == "noted"]
    nobody = [t for t in todo if t.outcome == "no contact"]
    failed = [t for t in todo if t.outcome.startswith("not written")]
    parts = []
    if drafted:
        parts.append(f"{len(drafted)} email draft{'s' if len(drafted) != 1 else ''} in your jobs Gmail")
    if noted:
        parts.append(f"{len(noted)} LinkedIn note{'s' if len(noted) != 1 else ''} in the sheet")
    if nobody:
        parts.append(f"nobody found at {_plural(len(nobody))}")
    if failed:
        parts.append(f"{len(failed)} failed ({', '.join(t.company for t in failed[:3])})")
    summary = "Outreach: " + "; ".join(parts) + ". Nothing was sent."
    run.finish("ok" if drafted or noted else ("error" if failed else "no change"), summary)
    return summary
