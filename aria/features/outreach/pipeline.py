"""features/outreach/pipeline.py — sheet rows → people → emails → drafts / LinkedIn notes.

    run_outreach(progress_fn=None) → one or two spoken sentences. Never raises. Never sends.

A row qualifies (needs_outreach) when its fit ("Profile alignment /5") is ≥ MIN_FIT,
its Status isn't closed, the outreach column is empty, and either "Who to contact"
names nobody (Muse writes placeholders like "Via Breezy ATS") or Aria filled it on an
earlier run whose draft didn't get written.

One company gets one outreach (plan() groups rows by dashboard.model.company_slug):
the highest-fit row carries it and names every role; sibling rows get "Outreach via
row N (Role)". A company is skipped when any of its rows — qualifying or not —
already has outreach, an address, or a contact the user named.

Per company:
  1. contacts.find_people() → 1–3 people → "Who to contact" (placeholder kept after them)
  2. Coldstart find_emails (cache-first by domain) → channel.choose(): one contact + channel
  3. email    → needs Muse's research (Company problem + Angle, Contacts tab): without it,
                "needs research" and no draft (writer.email, the user's spec v2)
                "Email draft ready — Name (Title), address"
     linkedin → writer.linkedin_note() → outreach column "LinkedIn (Aria): Name (Title) —
                profile\\nNote: …" (the dashboard's Today tab lists it to send by hand)
Every step is in the run log; each company's LLM spend goes to the flight log by row.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from aria.dashboard.model import company_slug, contact_name, emails_in, is_company_address
from aria.features.outreach import channel, feedback, writer
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
DRY_RUN_DIR = Path.home() / ".aria" / "outreach"
_WHO_PERSON = re.compile(r"^\s*(?P<name>[^()]+?)\s*(?:\((?P<title>[^)]*)\))?\s*$")


NEEDS_RESEARCH = "needs research"     # an email target: no Company problem / Angle (spec v2)


@dataclass
class Target:
    """One outreach unit: a company and the sheet rows it covers."""
    company: str
    jobs: list = field(default_factory=list)
    domain: str = ""
    people: list = field(default_factory=list)
    new_people: bool = False        # found this run, not read back from the sheet
    contact: object = None          # the one Person this company's outreach goes to
    channel: str = ""               # channel.EMAIL | channel.LINKEDIN
    why: str = ""
    outcome: str = ""               # drafted | noted | no contact | needs research | not written: …
    message: dict = field(default_factory=dict)   # the email {to, subject, body} or {"note": …}

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


def contacted(job: dict) -> bool:
    """This row already shows outreach, an address, or a contact the user (not Aria) named."""
    fields = job.get("fields") or {}
    who = fields.get(sheets.COL_WHO, "")
    if fields.get(sheets.COL_OUTREACH, "").strip():
        return True
    if any(is_company_address(e) for e in emails_in(who, fields.get("Outreach", ""))):
        return True
    return bool(contact_name(who)) and FOUND_NOTE not in fields.get(sheets.COL_UPDATES, "")


def plan(jobs: list[dict]) -> tuple[list[Target], list[tuple]]:
    """(targets, skipped) — one Target per company; skipped = [(job, reason)] for logging."""
    groups: dict[str, list] = {}
    for j in jobs:
        if j.get("company"):
            groups.setdefault(company_slug(j["company"]) or j["company"].lower(), []).append(j)
    out, skipped = [], []
    for rows in groups.values():
        todo = [j for j in rows if needs_contact(j) or found_earlier(j)]
        if not todo:
            continue
        done = next((j for j in rows if contacted(j)), None)
        if done is not None:
            skipped += [(j, f"company already contacted on row {done['row']} ({done['role']})") for j in todo]
            continue
        todo.sort(key=lambda j: -fit_score(j))          # stable: equal fits keep sheet order
        earlier = next((j for j in todo if found_earlier(j)), None)
        people = people_from_who((earlier.get("fields") or {}).get(sheets.COL_WHO, "")) if earlier else []
        out.append(Target(todo[0]["company"], todo, people=people))
    return out, skipped


def targets(jobs: list[dict]) -> list[Target]:
    return plan(jobs)[0]


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
    """Names → "Who to contact" on the target's rows that name nobody yet. Returns an error string or ""."""
    rows = [j for j in t.jobs if not contact_name((j.get("fields") or {}).get(sheets.COL_WHO, ""))]
    if not rows:
        return ""
    with run.step("update sheet") as s:
        for j in rows:
            try:
                sheet.set_fields(j["row"], {sheets.COL_WHO: who_cell(t.people, (j.get("fields") or {}).get(sheets.COL_WHO, ""))},
                                 note=FOUND_NOTE, company=j["company"], role=j["role"])
            except sheets.SheetError as exc:
                s.status, s.detail = "error", str(exc)
                return str(exc)
        s.detail = f"Who to contact on row{'s' if len(rows) > 1 else ''} " + ", ".join(str(j["row"]) for j in rows)
    return ""


def outreach_cell(t: Target, note: str = "") -> str:
    p = t.contact
    if t.channel == channel.EMAIL:
        return f"Email draft ready — {p.label()}, {channel.email_of(p)}"
    where = p.url if "linkedin.com" in (p.url or "") else "search LinkedIn for them"
    return f"{LINKEDIN_PREFIX} {p.label()} — {where}\nNote: {note}"


def draft(t: Target, run: run_log.Run, gmail) -> str:
    """Write the LinkedIn note (emails: the Contacts tab). Returns the outreach cell text, or "" if nothing was written."""
    with run.step("draft") as s:
        if t.channel == channel.EMAIL:          # emails need Muse's research, which lives in the Contacts tab
            s.status, s.detail = "skipped", "needs research: emails are drafted from the Contacts tab"
            t.outcome = NEEDS_RESEARCH
            return ""
        note = writer.linkedin_note(t)
        if note is None:
            s.status, s.detail, t.outcome = "error", "the LinkedIn note didn't pass checks", "not written: note failed checks"
            return ""
        s.detail, t.outcome, t.message = f"LinkedIn note for {t.contact.name} ({len(note)} chars)", "noted", {"note": note}
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
        p = t.primary
        for j in t.jobs[1:]:
            try:
                sheet.set_fields(j["row"], {sheets.COL_OUTREACH: f"Outreach via row {p['row']} ({p['role']})"},
                                 company=j["company"], role=j["role"])
            except sheets.SheetError as exc:
                s.status, s.detail = "error", str(exc)
                return str(exc)
        s.detail = f"outreach on row {p['row']}" + (
            f"; rows {', '.join(str(j['row']) for j in t.jobs[1:])} point to it" if len(t.jobs) > 1 else "")
    return ""


def process(t: Target, sheet: sheets.JobSheet, gmail, search_fn=None, client=None, trigger: str = "outreach") -> None:
    """All steps for one company as its own run, LLM calls labelled with its primary row. Never raises."""
    j = t.primary
    run = run_log.start(trigger)
    run.job(t.company, j["role"], j["row"])
    before = meter.snapshot()
    try:
        with llm_context.job(run_log.job_label(t.company, j["role"], j["row"]), run.id):
            find_contacts(t, run, search_fn, client)
            if not t.people:
                t.outcome = "no contact"
                return
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
        how = f"{t.outcome}" + (f" via {t.channel}" if t.channel else "")
        flight_recorder.record_job(j["row"], t.company, j["role"], trigger, "", meter.since(before), ok, how)
        run.finish("ok" if ok else ("no change" if t.outcome in ("no contact", NEEDS_RESEARCH) else "error"), how)


class DrySheet:
    """Reads the real sheet; set_fields only records what it would write."""

    def __init__(self, sheet: sheets.JobSheet) -> None:
        self._sheet, self.writes = sheet, []

    def jobs(self) -> list[dict]:
        return self._sheet.jobs()

    def set_fields(self, row, fields, note="", company=None, role=None) -> int:
        bad = [k for k in fields if k not in sheets.OUTREACH_FIELDS]
        if bad:
            raise sheets.SheetError(f"Aria doesn't write the {bad[0]!r} column.")
        self.writes.append({"row": row, "fields": dict(fields), "note": note})
        return row


class DryGmail:
    """create_draft only records the draft."""

    def __init__(self) -> None:
        self.drafts = []

    def create_draft(self, to, subject, body, cc="", reply_to_id="") -> str:
        self.drafts.append({"to": to, "subject": subject, "body": body})
        return "Draft saved (dry run — not created)."


def dry_run_report(todo: list, sheet: DrySheet, fb: dict, skipped: list, summary: str) -> str:
    """Markdown: per company, who was found, the channel, the message, and the sheet cells."""
    lines = ["# Outreach dry run", "", f"_{time.strftime('%Y-%m-%d %H:%M')} — nothing was drafted, written "
             "or recorded._", "", f"**Result:** {summary}", ""]
    if fb.get("replied") or fb.get("bounced"):
        lines += ["## Coldstart feedback that would be recorded", ""]
        lines += [f"- {e}: {o}" for o in ("replied", "bounced") for e in fb.get(o, [])] + [""]
    for t in todo:
        rows = ", ".join(f"row {j['row']} {j['role']} (fit {j.get('alignment') or '?'})" for j in t.jobs)
        lines += [f"## {t.company}", "", f"- **Rows:** {rows}", f"- **Domain:** {t.domain or 'unknown'}",
                  f"- **Outcome:** {t.outcome or 'nothing'}"]
        for p in t.people:
            lk = p.lookup or {}
            mail = f" — {lk.get('email')} ({lk.get('status')}, {lk.get('confidence')})" if lk.get("email") else ""
            lines.append(f"  - {p.label()} [{p.source}] {p.url}{mail}".rstrip())
        if t.contact is not None:
            lines.append(f"- **Contact:** {t.contact.label()} via {t.channel} — {t.why}")
        if t.message.get("body"):
            lines += ["", f"**To:** {t.message['to']}  ", f"**Subject:** {t.message['subject']}", "", "```",
                      t.message["body"], "```"]
        elif t.message.get("note"):
            lines += ["", f"**LinkedIn note ({len(t.message['note'])} chars):**", "", f"> {t.message['note']}"]
        cells = [w for w in sheet.writes if any(w["row"] == j["row"] for j in t.jobs)]
        if cells:
            lines += ["", "**Sheet cells that would be written:**", ""]
            lines += [f"- row {w['row']} {k}: {v!r}" for w in cells for k, v in w["fields"].items()]
        lines.append("")
    if skipped:
        lines += ["## Skipped", ""] + [f"- {j['company']} — {j['role']} (row {j['row']}): {why}" for j, why in skipped]
    return "\n".join(lines) + "\n"


def run_outreach(
    progress_fn: Optional[Callable[[str], None]] = None,
    sheet: Optional[sheets.JobSheet] = None,
    search_fn=None,
    client=None,
    gmail=None,
    gmail_data: Optional[dict] = None,
    dry_run: bool = False,
    report_dir: Optional[Path] = None,
) -> str:
    """The outreach loop over every qualifying sheet row. Never raises, never sends.
    gmail_data: the jobs Gmail threads for feedback (default: the dashboard's cached load).
    dry_run: run everything but write nothing (drafts, sheet, Coldstart feedback); save a report."""
    if search_fn is None:
        from aria.web import websearch
        search_fn = lambda q: websearch.search(q, max_results=8)
    if dry_run:
        gmail = DryGmail()
    elif gmail is None:
        from aria.plugins.productivity.gmail import GmailClient
        gmail = GmailClient(google_auth.JOBS)
    sheet = sheet or sheets.default_sheet()
    if dry_run:
        sheet = DrySheet(sheet)
    trigger = "outreach (dry run)" if dry_run else "outreach"
    run = run_log.start(trigger)
    with run.step("record feedback") as s:
        fb = feedback.sync(gmail_data, record_fn=(lambda *a, **k: None) if dry_run else None)
        s.detail = fb["error"] or f"{len(fb['replied'])} replied, {len(fb['bounced'])} bounced → Coldstart"
        s.status = "error" if fb["error"] else "ok"
    try:
        with run.step("read sheet") as s:
            jobs = sheet.jobs()
            todo, skipped = plan(jobs)
            rows = sum(len(t.jobs) for t in todo)
            s.detail = (f"{len(jobs)} rows; {rows} need outreach (fit ≥ {MIN_FIT:g}, no outreach yet) "
                        f"at {_plural(len(todo))}; {len(skipped)} skipped (company already contacted)")
        for j, why in skipped:
            run.decision(j["company"], j["role"], j["row"], False, [why])
    except sheets.SheetError as exc:
        run.finish("error", str(exc))
        return str(exc)
    except Exception as exc:
        logger.error("outreach: reading the sheet failed: %s", exc)
        run.finish("error", f"reading the sheet failed: {exc}")
        return "I couldn't read your job sheet right now."
    heard = feedback.spoken(fb)
    if dry_run and heard:
        heard = heard.replace("Told Coldstart about", "Would tell Coldstart about")
    if not todo:
        run.finish("no change", "no rows need outreach")
        return " ".join(x for x in ("Every job with a fit of 3 or more already has outreach planned.", heard) if x)

    _say(progress_fn, f"Working on outreach for {_plural(len(todo))}{' as a dry run' if dry_run else ''}. "
                      "This takes a few minutes.")
    for t in todo:
        process(t, sheet, gmail, search_fn, client, trigger)
    summary = finish(run, todo, dry_run)
    if dry_run:
        path = save_report(dry_run_report(todo, sheet, fb, skipped, summary), report_dir)
        summary += f" The report is in {path.parent.name}/{path.name} in your Aria folder." if path else \
            " I couldn't save the report."
    return " ".join(x for x in (summary, heard) if x)


def save_report(text: str, report_dir: Optional[Path] = None) -> Optional[Path]:
    try:
        d = report_dir or DRY_RUN_DIR
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"dry-run-{time.strftime('%Y%m%d-%H%M%S')}.md"
        path.write_text(text)
        return path
    except OSError as exc:
        logger.error("outreach: couldn't save the dry-run report: %s", exc)
        return None


def finish(run: run_log.Run, todo: list, dry_run: bool = False) -> str:
    drafted = [t for t in todo if t.outcome == "drafted"]
    noted = [t for t in todo if t.outcome == "noted"]
    nobody = [t for t in todo if t.outcome == "no contact"]
    research = [t for t in todo if t.outcome == NEEDS_RESEARCH]
    failed = [t for t in todo if t.outcome.startswith("not written")]
    parts = []
    if drafted:
        parts.append(f"{len(drafted)} email draft{'s' if len(drafted) != 1 else ''} "
                     + ("written, not saved" if dry_run else "in your jobs Gmail"))
    if noted:
        parts.append(f"{len(noted)} LinkedIn note{'s' if len(noted) != 1 else ''} "
                     + ("written, not saved" if dry_run else "in the sheet"))
    if nobody:
        parts.append(f"nobody found at {_plural(len(nobody))}")
    if research:
        parts.append(f"{_plural(len(research))} need{'s' if len(research) == 1 else ''} research for an email "
                     f"({', '.join(t.company for t in research[:3])}; emails come from the Contacts tab)")
    if failed:
        parts.append(f"{len(failed)} failed ({', '.join(t.company for t in failed[:3])})")
    summary = ("Outreach dry run: " if dry_run else "Outreach: ") + "; ".join(parts) + (
        ". Nothing was written." if dry_run else ". Nothing was sent.")
    run.finish("ok" if drafted or noted else ("error" if failed else "no change"), summary)
    return summary
