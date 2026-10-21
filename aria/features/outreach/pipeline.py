"""features/outreach/pipeline.py — sheet rows → people to contact → sheet.

    run_outreach(progress_fn=None) → one or two spoken sentences. Never raises.

A row qualifies (needs_contact) when its fit ("Profile alignment /5") is ≥ MIN_FIT,
its Status isn't closed, "Who to contact" names nobody (Muse writes placeholders like
"Via Breezy ATS") and the outreach column is empty. For each, contacts.find_people()
picks 1–3 people and their names go to "Who to contact"; the placeholder is kept
after them so nothing the user wrote is lost.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from aria.dashboard.model import contact_name, emails_in, is_company_address
from aria.features.outreach import contacts as finder
from aria.integrations import sheets
from aria.observability import run_log

logger = logging.getLogger(__name__)

MIN_FIT = 3.0
_CLOSED = re.compile(r"(?i)not gonna apply|not applying|skipped|reject|closed|withdrawn")


@dataclass
class Target:
    """One outreach unit: a company and the sheet rows it covers."""
    company: str
    jobs: list = field(default_factory=list)
    domain: str = ""
    people: list = field(default_factory=list)

    @property
    def primary(self) -> dict:
        return self.jobs[0]


def fit_score(job: dict) -> float:
    m = re.search(r"\d+(?:\.\d+)?", job.get("alignment") or "")
    return float(m.group()) if m else 0.0


def needs_contact(job: dict) -> bool:
    fields = job.get("fields") or {}
    who = fields.get(sheets.COL_WHO, "")
    outreach = fields.get(sheets.COL_OUTREACH, "")
    if fit_score(job) < MIN_FIT or _CLOSED.search(job.get("status") or "") or not job.get("company"):
        return False
    if outreach.strip() or contact_name(who):
        return False
    return not any(is_company_address(e) for e in emails_in(who, fields.get("Outreach", "")))


def who_cell(people: list, existing: str) -> str:
    """'Pat Lee (Recruiter); Sam Roe (CTO)', with any placeholder kept after a ' | '."""
    names = "; ".join(p.label() for p in people)
    existing = (existing or "").strip()
    return f"{names} | {existing}" if existing else names


def targets(jobs: list[dict]) -> list[Target]:
    return [Target(j["company"], [j]) for j in jobs if needs_contact(j)]


def _say(progress_fn: Optional[Callable[[str], None]], msg: str) -> None:
    if progress_fn:
        try:
            progress_fn(msg)
        except Exception:
            pass


def find_contacts(t: Target, run: run_log.Run, search_fn=None, client=None) -> None:
    with run.step("find contacts") as s:
        t.domain = finder.company_domain(t.primary, search_fn)
        urls = [u for j in t.jobs for u in (j.get("post_link"), j.get("apply_link"))]
        t.people = finder.find_people(t.company, t.domain, urls, search_fn=search_fn, client=client)
        s.detail = f"{t.company} ({t.domain or 'no domain'}): " + (
            ", ".join(f"{p.label()} [{p.source}]" for p in t.people) or "nobody found")


def write_contacts(t: Target, run: run_log.Run, sheet: sheets.JobSheet) -> str:
    """Names → "Who to contact" on every row of the target. Returns an error string or ""."""
    with run.step("update sheet") as s:
        for j in t.jobs:
            try:
                sheet.set_fields(j["row"], {sheets.COL_WHO: who_cell(t.people, (j.get("fields") or {}).get(sheets.COL_WHO, ""))},
                                 company=j["company"], role=j["role"])
            except sheets.SheetError as exc:
                s.status, s.detail = "error", str(exc)
                return str(exc)
        s.detail = f"Who to contact on row{'s' if len(t.jobs) > 1 else ''} " + ", ".join(str(j["row"]) for j in t.jobs)
    return ""


def run_outreach(
    progress_fn: Optional[Callable[[str], None]] = None,
    sheet: Optional[sheets.JobSheet] = None,
    search_fn=None,
    client=None,
) -> str:
    """Find people for every qualifying sheet row and write them to the sheet. Never raises."""
    if search_fn is None:
        from aria.web import websearch
        search_fn = lambda q: websearch.search(q, max_results=8)
    sheet = sheet or sheets.default_sheet()
    run = run_log.start("outreach")
    try:
        with run.step("read sheet") as s:
            jobs = sheet.jobs()
            todo = targets(jobs)
            s.detail = f"{len(jobs)} rows; {len(todo)} need a contact (fit ≥ {MIN_FIT:g}, nobody named yet)"
    except sheets.SheetError as exc:
        run.finish("error", str(exc))
        return str(exc)
    except Exception as exc:
        logger.error("outreach: reading the sheet failed: %s", exc)
        run.finish("error", f"reading the sheet failed: {exc}")
        return "I couldn't read your job sheet right now."
    if not todo:
        run.finish("no change", "no rows need a contact")
        return "Every job with a fit of 3 or more already has someone to contact."

    _say(progress_fn, f"Looking for people at {len(todo)} compan{'y' if len(todo) == 1 else 'ies'}. This takes a minute.")
    found, errors = 0, []
    for t in todo:
        run.job(t.company, t.primary["role"], t.primary["row"])
        try:
            find_contacts(t, run, search_fn, client)
        except Exception as exc:
            logger.error("outreach: %s failed: %s", t.company, exc)
            errors.append(t.company)
            continue
        if not t.people:
            continue
        err = write_contacts(t, run, sheet)
        if err:
            errors.append(t.company)
            continue
        found += 1
    summary = f"Found people at {found} of {len(todo)} compan{'y' if len(todo) == 1 else 'ies'}"
    summary += f"; {len(errors)} failed ({', '.join(errors[:3])})." if errors else "."
    run.finish("ok" if found else "no change", summary)
    return summary + (" Their names are in Who to contact." if found else "")
