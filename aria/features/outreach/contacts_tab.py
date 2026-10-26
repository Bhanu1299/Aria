"""features/outreach/contacts_tab.py — Muse's Contacts + Companies tabs: Coldstart emails → Gmail drafts.

    run(limit=None, step_fn=None) → summary sentence. Never raises. Never sends.
    start_background(...) → bool      the same run on a daemon thread (False if one is running)

Columns are read and written by header name (integrations/header_sheet.py). Muse keeps one
row per job on Companies and one row per person on Contacts, joined by Job ID:
  Companies (Muse's)   Job ID, Company, Role, Job URL, Domain, Email pattern, Company problem,
                       Signal, Signal source, Angle, Blocker, Outreach status
  Contacts (Muse's)    Job ID, Company, Person, Title, LinkedIn URL, Wave, Person type (optional)
  Contacts (Aria's)    Email, Source, Confidence, Find status, Draft status, Gmail draft link,
                       Wave, Updated, Notes
Each Contacts row is read with its Companies row's cells (join). On Companies Aria writes only
Domain and Email pattern, and only into blank cells. Source keeps Muse's text when it has
some (Coldstart's source then goes to Notes). Only Outreach status "active" (or blank, with no
Blocker) is looked up or drafted. Draft status "approved", "sent" and "replied" are the
user's: Aria only ever writes none / drafted / no reply. A sheet without a Companies tab
(every column on Contacts) works the same way.

One run, all code until the email body:
  1. read      the tab
  2. stop      a company with a "replied" row: every other row there → "no reply", no drafts
  3. pending   Company + a real Person, Email blank, Find status blank/pending;
               deduped by LinkedIn URL, else Company + Person (against every row)
  4. find      per pending row: Domain (cell → Job URL → the company's other rows → web
               search), then Coldstart (rate-limited, pattern cached per domain) →
               Email, Source, Confidence, Email pattern, Find status done|error, Updated.
               "not found" is done with a Notes line; an error is "error" with a Notes
               line. Neither is retried — set Find status back to "pending" to retry.
  5. draft     rows with Confidence verified|pattern — a person, or a generic inbox
               (Person "—", one per company) — at most CONTACTS_PER_COMPANY per company,
               numbered by Wave in sheet order (a person listed twice: the company address
               over a personal one). A wave is drafted only once every earlier wave at that
               company is sent; a row with a Gmail draft link is never drafted again.
               Muse's research columns feed the email (writer.email, the user's spec v2):
               Company problem + Angle required, else Notes "needs research" and no draft;
               Signal optional; Person type (else from Person/Title). Which work the email
               is about: writer.plan (Muse's project only when tied to a concrete company
               fact, else the role's; "angle: role override" in Notes). The LLM writes the
               subject and three lines; code checks them and adds the rest. Gmail DRAFT in
               the jobs account (only if it is OUTREACH_FROM), resume PDF attached →
               Draft status drafted, Gmail draft link, Updated; "pattern guess" in Notes
               when Confidence is pattern.
Every step goes to step_fn (one line each) and the run log.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlparse

import aria.core.config as config
from aria.dashboard.model import company_slug, emails_in, is_company_address, norm_text
from aria.features.outreach import contacts as finder
from aria.features.outreach import writer
from aria.integrations import coldstart_client as cs
from aria.integrations import google_auth
from aria.integrations.header_sheet import ROW, HeaderSheet, get
from aria.integrations.sheets import SheetError
from aria.observability import run_log

logger = logging.getLogger(__name__)

COMPANY, ROLE, PERSON, TITLE, JOB_ID = "Company", "Role", "Person", "Title", "Job ID"
LINKEDIN, JOB_URL, DOMAIN = "LinkedIn URL", "Job URL", "Domain"
EMAIL, SOURCE, CONFIDENCE, PATTERN, FIND_STATUS = "Email", "Source", "Confidence", "Email pattern", "Find status"
DRAFT_STATUS, DRAFT_LINK, WAVE, UPDATED, NOTES = "Draft status", "Gmail draft link", "Wave", "Updated", "Notes"
PROBLEM, SIGNAL, SIGNAL_SOURCE, ANGLE, PERSON_TYPE = "Company problem", "Signal", "Signal source", "Angle", "Person type"
BLOCKER, OUTREACH = "Blocker", "Outreach status"
WRITABLE = (DOMAIN, EMAIL, SOURCE, CONFIDENCE, PATTERN, FIND_STATUS, DRAFT_STATUS, DRAFT_LINK, WAVE, UPDATED, NOTES)
COMPANY_WRITABLE = (DOMAIN, PATTERN)                     # on Companies, blank cells only
COMPANY_FIELDS = (ROLE, JOB_URL, DOMAIN, PATTERN, PROBLEM, SIGNAL, SIGNAL_SOURCE, ANGLE, BLOCKER, OUTREACH)
COMPANY_ROW = "_company_row"                            # the joined Companies row, on a Contacts record

FIND_PENDING, FIND_DONE, FIND_ERROR = "pending", "done", "error"
NONE, DRAFTED, APPROVED, SENT, REPLIED, NO_REPLY = "none", "drafted", "approved", "sent", "replied", "no reply"
ARIA_DRAFT_STATUSES = {NONE, DRAFTED, NO_REPLY}          # everything else is the user's to set
_IN_FLIGHT = {DRAFTED, APPROVED, SENT, REPLIED}          # counts toward the per-company cap
_WAVE_CLEARED = {SENT, NO_REPLY}                        # a later wave may start after these
_NOT_A_PERSON = {"", "—", "–", "-", "n/a", "na", "none", "?"}
_GENERIC_INBOX = {"—", "–", "-"}
NEEDS_RESEARCH, PATTERN_GUESS, ROLE_OVERRIDE = "needs research", "pattern guess", "angle: role override"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def status(rec: dict, name: str) -> str:
    return get(rec, name).strip().lower()


def is_person(name: str) -> bool:
    """A name Coldstart can look up: not "—" and not a placeholder like "(hiring contact)"."""
    name = (name or "").strip()
    return name.lower() not in _NOT_A_PERSON and not name.startswith("(")


def is_generic_inbox(name: str) -> bool:
    """Person "—": the row is a company inbox (team@, contact@), written to as "Hi <company> team,"."""
    return (name or "").strip() in _GENERIC_INBOX


def company_key(company: str) -> str:
    """'PointOne (YC W24)' and 'PointOne' are the same company."""
    bare = re.sub(r"\(.*?\)", " ", company or "")
    return company_slug(bare) or norm_text(bare)


def linkedin_key(url: str) -> str:
    u = urlparse((url or "").strip().lower())
    host = (u.hostname or "").removeprefix("www.")
    return f"{host}{u.path.rstrip('/')}" if "linkedin.com" in host and u.path.strip("/") else ""


def dedupe_key(rec: dict) -> str:
    return linkedin_key(get(rec, LINKEDIN)) or f"{company_key(get(rec, COMPANY))}|{norm_text(get(rec, PERSON))}"


def first_email(rec: dict) -> str:
    """'james@acme.com (alt: j@acme.io)' → 'james@acme.com'."""
    found = emails_in(get(rec, EMAIL))
    return found[0] if found else ""


def match_of(rec: dict) -> dict:
    """Cells that identify the row if the sheet moves under us."""
    m = {COMPANY: get(rec, COMPANY), PERSON: get(rec, PERSON)}
    if get(rec, JOB_ID).strip():
        m[JOB_ID] = get(rec, JOB_ID)
    return m


def join(contacts: list[dict], companies: list[dict]) -> list[dict]:
    """Each Contacts record gains its Companies row's cells (COMPANY_FIELDS, where its own
    are blank) and COMPANY_ROW. Matched by Job ID, else by company name. In place."""
    by_id = {get(c, JOB_ID).strip().upper(): c for c in companies if get(c, JOB_ID).strip()}
    by_name: dict = {}
    for c in companies:
        by_name.setdefault(company_key(get(c, COMPANY)), c)
    for r in contacts:
        c = by_id.get(get(r, JOB_ID).strip().upper()) or by_name.get(company_key(get(r, COMPANY)))
        if c is None:
            continue
        r[COMPANY_ROW] = c[ROW]
        for name in COMPANY_FIELDS:
            if not get(r, name).strip() and get(c, name).strip():
                r[name] = get(c, name)
    return contacts


def hold_reason(rec: dict) -> str:
    """'' when Muse's Companies row lets this contact be looked up and drafted, else why not.
    Outreach status decides; with none, a Blocker holds the row."""
    st = status(rec, OUTREACH)
    if st and st != "active":
        blocker = get(rec, BLOCKER).strip()
        return f"Outreach status {st}" + (f" ({blocker})" if blocker else "")
    if not st and get(rec, BLOCKER).strip():
        return f"Blocker: {get(rec, BLOCKER).strip()}"
    return ""


# ---------------------------------------------------------------------------
# selection (pure)
# ---------------------------------------------------------------------------

def is_pending(rec: dict) -> bool:
    return (bool(get(rec, COMPANY).strip()) and is_person(get(rec, PERSON)) and not get(rec, EMAIL).strip()
            and status(rec, FIND_STATUS) in ("", FIND_PENDING))


def pending(records: list[dict]) -> tuple[list[dict], list[tuple]]:
    """(rows to look up, [(row, why skipped)]). A pending row that repeats any earlier row
    — or any row already looked up — is skipped."""
    done = {dedupe_key(r) for r in records if not is_pending(r) and is_person(get(r, PERSON))}
    todo, skipped, seen = [], [], set()
    for r in records:
        if not is_pending(r):
            continue
        held = hold_reason(r)
        if held:
            skipped.append((r, held))
            continue
        key = dedupe_key(r)
        if key in done or key in seen:
            skipped.append((r, "duplicate of another row"))
            continue
        seen.add(key)
        todo.append(r)
    return todo, skipped


def by_company(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        if get(r, COMPANY).strip():
            out.setdefault(company_key(get(r, COMPANY)), []).append(r)
    return out


def replied_companies(records: list[dict]) -> set[str]:
    return {k for k, rows in by_company(records).items() if any(status(r, DRAFT_STATUS) == REPLIED for r in rows)}


def stop_updates(records: list[dict]) -> list[tuple[dict, dict]]:
    """Rows to close because someone at their company replied: [(row, fields)]."""
    stop = replied_companies(records)
    out = []
    for key, rows in by_company(records).items():
        if key not in stop:
            continue
        who = next(get(r, PERSON) for r in rows if status(r, DRAFT_STATUS) == REPLIED)
        for r in rows:
            if status(r, DRAFT_STATUS) not in (REPLIED, NO_REPLY):
                out.append((r, {DRAFT_STATUS: NO_REPLY, UPDATED: _now(),
                                NOTES: add_note(get(r, NOTES), f"stopped: {who} replied")}))
    return out


def draft_queue(records: list[dict], cap: Optional[int] = None) -> tuple[list[dict], dict]:
    """(rows to draft now, {row number: wave to write}). Pure.

    Draftable: a real person or a generic inbox, an address, Confidence verified|pattern, no
    reply yet at the company, no Gmail draft link yet; a person listed twice counts once. Waves 1..cap go to the company's draftable rows in sheet order (rows already in
    flight keep theirs and count toward the cap). The active wave is the lowest one not yet
    cleared (sent / no reply); only its undrafted rows are drafted."""
    cap = config.CONTACTS_PER_COMPANY if cap is None else cap
    stop = replied_companies(records)
    queue, waves = [], {}
    for key, rows in by_company(records).items():
        if key in stop:
            continue
        ok: list = []
        for r in rows:                          # one row per person: the first, or the one in flight
            who = get(r, PERSON)
            if not ((is_person(who) or is_generic_inbox(who)) and first_email(r) and status(r, DRAFT_STATUS) != NO_REPLY
                    and status(r, CONFIDENCE) in (cs.VERIFIED, cs.PATTERN) and not hold_reason(r)):
                continue
            key = dedupe_key(r)
            have = next((x for x in ok if dedupe_key(x) == key), None)
            if have is None:
                ok.append(r)
            elif _better_row(r, have):
                ok[ok.index(have)] = r
        wave_of = {r[ROW]: int(get(r, WAVE)) for r in ok if get(r, WAVE).strip().isdigit()}
        slotted = [r for r in ok if r[ROW] in wave_of or status(r, DRAFT_STATUS) in _IN_FLIGHT]
        nxt = max(wave_of.values(), default=0)
        for r in ok:
            if len(slotted) >= cap:
                break
            if r in slotted:
                continue
            nxt += 1
            wave_of[r[ROW]] = waves[r[ROW]] = nxt
            slotted.append(r)
        for r in slotted:                       # in flight but no Wave written yet
            if r[ROW] not in wave_of:
                nxt += 1
                wave_of[r[ROW]] = waves[r[ROW]] = nxt
        open_waves = [wave_of[r[ROW]] for r in slotted if status(r, DRAFT_STATUS) not in _WAVE_CLEARED]
        if not open_waves:
            continue
        active = min(open_waves)
        queue += [r for r in slotted if wave_of[r[ROW]] == active and status(r, DRAFT_STATUS) in ("", NONE)
                  and not get(r, DRAFT_LINK).strip()]
    queue.sort(key=lambda r: r[ROW])
    return queue, waves


def _better_row(new: dict, have: dict) -> bool:
    """Which of one person's two rows to draft: the one in flight, else the company
    address over a personal one (gmail.com etc.), else verified over pattern."""
    rank = lambda r: (status(r, DRAFT_STATUS) in _IN_FLIGHT, is_company_address(first_email(r)),
                      status(r, CONFIDENCE) == cs.VERIFIED)
    return rank(new) > rank(have)


def add_note(existing: str, line: str) -> str:
    entry = f"{time.strftime('%Y-%m-%d')}: {line}"
    return f"{existing.rstrip()}\n{entry}" if (existing or "").strip() else entry


# ---------------------------------------------------------------------------
# domain + Coldstart
# ---------------------------------------------------------------------------

def _clean_domain(text: str) -> str:
    text = (text or "").strip().lower()
    host = urlparse(text if "//" in text else f"//{text}").hostname or ""
    return finder.registrable(host.split("@")[-1])


def resolve_domain(rec: dict, records: list[dict], search_fn=None) -> tuple[str, str]:
    """(domain, where it came from). ('', why not) when unknown."""
    own = _clean_domain(get(rec, DOMAIN))
    if own:
        return own, "Domain column"
    job = finder.registrable(urlparse(get(rec, JOB_URL)).hostname or "")
    if job and not finder.is_job_host(job):
        return job, "Job URL"
    key = company_key(get(rec, COMPANY))
    same = [r for r in records if r is not rec and company_key(get(r, COMPANY)) == key]
    for r in same:
        d = _clean_domain(get(r, DOMAIN))
        if d:
            return d, f"Domain on row {r[ROW]}"
    for r in same:
        for e in emails_in(get(r, EMAIL)):
            dom = e.split("@")[1]
            if is_company_address(e) and not finder.is_job_host(dom):
                return finder.registrable(dom), f"email on row {r[ROW]}"
    if search_fn is not None:
        company = re.sub(r"\(.*?\)", " ", get(rec, COMPANY)).strip()
        d = finder.company_domain({"company": company}, search_fn)
        if d:
            return d, "web search"
    return "", "no domain in the row, the Job URL, the company's other rows or a web search"


def seed_patterns(client: cs.ColdstartClient, records: list[dict]) -> None:
    """Patterns the sheet already knows, per domain."""
    for r in records:
        pat = get(r, PATTERN)
        if not pat:
            continue
        dom = _clean_domain(get(r, DOMAIN)) or (first_email(r).split("@")[1] if first_email(r) else "")
        client.remember(dom, pat)


def find_fields(lk: cs.Lookup, notes: str, source: str = "") -> dict:
    """Coldstart's answer → the row's cells. source: the row's Source cell now — Muse's
    text there is kept and Coldstart's source goes to Notes instead."""
    if not lk.ok:
        return {FIND_STATUS: FIND_ERROR, UPDATED: _now(), NOTES: add_note(notes, f"Coldstart error: {lk.error}")}
    if lk.confidence == cs.NOT_FOUND:
        fields = {EMAIL: "not found", SOURCE: lk.source, CONFIDENCE: cs.NOT_FOUND, PATTERN: lk.pattern,
                  FIND_STATUS: FIND_DONE, UPDATED: _now(),
                  NOTES: add_note(notes, "Coldstart found no address; no draft")}
    else:
        fields = {EMAIL: lk.email, SOURCE: lk.source, CONFIDENCE: lk.confidence, PATTERN: lk.pattern,
                  FIND_STATUS: FIND_DONE, UPDATED: _now()}
    if source.strip():
        del fields[SOURCE]
        if lk.source:
            fields[NOTES] = add_note(fields.get(NOTES, notes), f"email source: {lk.source}")
    if not fields[PATTERN]:
        del fields[PATTERN]
    return fields


# ---------------------------------------------------------------------------
# drafting
# ---------------------------------------------------------------------------

def sender_problem(gmail) -> str:
    """'' when the jobs Gmail is OUTREACH_FROM, else why no draft may be made there."""
    want = (config.OUTREACH_FROM or "").strip().lower()
    if not want:
        return "OUTREACH_FROM (or GOOGLE_JOBS_ACCOUNT) isn't set in .env"
    have = (google_auth.account_email(getattr(gmail, "account", google_auth.JOBS)) or "").strip().lower()
    if have != want:
        return f"the jobs Gmail is {have or 'not connected'}, not {want}"
    return ""


def brief_of(rec: dict) -> writer.Brief:
    """The email's inputs from one row: Muse's research columns + who it's to."""
    who = get(rec, PERSON)
    return writer.Brief(company=get(rec, COMPANY), role=get(rec, ROLE), problem=get(rec, PROBLEM),
                        angle=get(rec, ANGLE), signal=get(rec, SIGNAL), signal_source=get(rec, SIGNAL_SOURCE),
                        person="" if is_generic_inbox(who) else who, title=get(rec, TITLE),
                        person_type=get(rec, PERSON_TYPE))


def compose(rec: dict, p: Optional[writer.Plan] = None) -> Optional[dict]:
    """{to, subject, body} from the gateway LLM, or None if its output failed checks."""
    return writer.email(brief_of(rec), first_email(rec), p)


def resume_pdf() -> str:
    """The resume attached to every draft: OUTREACH_RESUME, else the answer bank's resume_path."""
    from pathlib import Path
    from aria.features.ats import planner
    path = config.OUTREACH_RESUME or planner.resume_path() or ""
    return str(Path(path).expanduser()) if path and Path(path).expanduser().is_file() else ""


def _note_once(rec: dict, line: str) -> str:
    """Notes with `line` added, unless it's already there."""
    notes = get(rec, NOTES)
    return notes if line in notes else add_note(notes, line)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

@dataclass
class Report:
    found: list = field(default_factory=list)       # (row, confidence, email)
    errors: list = field(default_factory=list)      # (row, why) lookups
    draft_errors: list = field(default_factory=list)  # (row, why)
    drafted: list = field(default_factory=list)     # (row, link)
    held: list = field(default_factory=list)        # rows that need research (no problem / angle)
    stopped: list = field(default_factory=list)     # rows set to "no reply"
    skipped: list = field(default_factory=list)     # (row, why)

    def summary(self) -> str:
        n = lambda k: sum(1 for x in self.found if x[1] == k)
        parts = []
        if self.found:
            parts.append(f"{n(cs.VERIFIED)} verified, {n(cs.PATTERN)} pattern, {n(cs.NOT_FOUND)} not found")
        if self.errors:
            parts.append(f"{len(self.errors)} lookup error{'s' if len(self.errors) != 1 else ''}")
        if self.draft_errors:
            k = len(self.draft_errors)
            parts.append(f"{k} draft{'s' if k != 1 else ''} not created (see Notes)")
        if self.held:
            k = len(self.held)
            parts.append(f"{k} need{'s' if k == 1 else ''} research (Company problem / Angle)")
        if self.drafted:
            parts.append(f"{len(self.drafted)} draft{'s' if len(self.drafted) != 1 else ''} in the jobs Gmail")
        if self.stopped:
            parts.append(f"{len(self.stopped)} row{'s' if len(self.stopped) != 1 else ''} closed after a reply")
        return ("Contacts: " + "; ".join(parts) + ". Nothing was sent.") if parts else \
            "Contacts: nothing to look up or draft."


class _Writer:
    """sheet.update that also refreshes the in-memory row and refuses user-owned statuses.
    With a Companies tab, Domain / Email pattern go to the contact's Companies row, and
    only into cells that are blank there."""

    def __init__(self, sheet: HeaderSheet, step: Callable[[str], None],
                 companies: Optional[HeaderSheet] = None, company_rows: Optional[list[dict]] = None) -> None:
        self.sheet, self.step, self.companies = sheet, step, companies
        self.company_rows = {c[ROW]: c for c in company_rows or []}

    def __call__(self, rec: dict, fields: dict) -> None:
        ds = fields.get(DRAFT_STATUS)
        if ds is not None and ds not in ARIA_DRAFT_STATUSES:
            raise SheetError(f"Aria doesn't set Draft status to {ds!r}.")
        fields = dict(fields)
        if self.companies is not None:
            self._company(rec, {k: fields.pop(k) for k in COMPANY_WRITABLE if k in fields})
        if not fields:
            return
        rec[ROW] = self.sheet.update(rec[ROW], fields, match=match_of(rec))
        rec.update(fields)
        self._said(self.sheet.tab, rec[ROW], fields)

    def _company(self, rec: dict, fields: dict) -> None:
        c = self.company_rows.get(rec.get(COMPANY_ROW))
        if c is None:
            return
        fields = {k: v for k, v in fields.items() if str(v).strip() and not get(c, k).strip()}
        rec.update(fields)
        if not fields:
            return
        match = {JOB_ID: get(c, JOB_ID)} if get(c, JOB_ID).strip() else {COMPANY: get(c, COMPANY)}
        c[ROW] = self.companies.update(c[ROW], fields, match=match)
        c.update(fields)
        self._said(self.companies.tab, c[ROW], fields)

    def _said(self, tab: str, row: int, fields: dict) -> None:
        shown = ", ".join(f"{k}={v!r}" for k, v in fields.items() if k != NOTES)
        self.step(f"  wrote {tab} row {row}: {shown}" + (" (+ Notes)" if NOTES in fields else ""))


def run(
    sheet: Optional[HeaderSheet] = None,
    client: Optional[cs.ColdstartClient] = None,
    companies: Optional[HeaderSheet] = None,
    gmail=None,
    search_fn=None,
    resume: Optional[str] = None,
    limit: Optional[int] = None,
    draft: bool = True,
    step_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """One pass over the Contacts tab (joined with `companies`, Muse's Companies tab; by
    default both tabs of the Muse sheet). resume: the PDF to attach (default resume_pdf()).
    limit: look up at most this many pending rows, and
    draft only rows looked up in this run — or, when none were pending, the first `limit`
    rows ready to draft (the thin slice: limit=1). Never raises."""
    say = _safe(step_fn)
    if sheet is None:
        sheet = HeaderSheet(config.MUSE_SHEET_ID, config.MUSE_CONTACTS_TAB, writable=WRITABLE)
        if companies is None and config.MUSE_COMPANIES_TAB:
            companies = HeaderSheet(config.MUSE_SHEET_ID, config.MUSE_COMPANIES_TAB, writable=COMPANY_WRITABLE)
    client = client or cs.ColdstartClient()
    rep = Report()
    log = run_log.start("contacts")
    try:
        with log.step("read sheet") as s:
            records = sheet.records()
            company_rows = companies.records() if companies is not None else []
            join(records, company_rows)
            s.detail = f"{len(records)} rows in {sheet.tab}, {len(company_rows)} in {companies.tab if companies else '-'}"
        write = _Writer(sheet, say, companies, company_rows)
        say(f"1. read: {len(records)} rows in '{sheet.tab}'"
            + (f", joined with {len(company_rows)} in '{companies.tab}'" if companies is not None else ""))

        with log.step("stop on reply") as s:
            for rec, fields in stop_updates(records):
                write(rec, fields)
                rep.stopped.append(rec[ROW])
            s.detail = f"{len(rep.stopped)} rows → no reply"
        say(f"2. stop on reply: {len(rep.stopped)} row(s) set to 'no reply'")

        todo, skipped = pending(records)
        rep.skipped += [(r[ROW], why) for r, why in skipped]
        if limit is not None:
            todo = todo[:limit]
        dupes = sum(1 for _, why in skipped if why == "duplicate of another row")
        say(f"3. pending: {len(todo)} row(s) to look up" + (f" ({', '.join(str(r[ROW]) for r in todo)})" if todo else "")
            + (f"; {dupes} duplicate(s) skipped" if dupes else "")
            + (f"; {len(skipped) - dupes} held by Companies (Outreach status / Blocker)" if len(skipped) > dupes else ""))

        seed_patterns(client, records)
        if todo:
            down = client.ensure_running()
            if down:                            # nothing written: the rows stay pending for the next run
                say(f"  {down}")
                log.finish("error", down)
                return f"{down}. Nothing was looked up or sent."
        with log.step("find emails") as s:
            for rec in todo:
                say(f"4. find row {rec[ROW]}: {get(rec, PERSON)} at {get(rec, COMPANY)}")
                dom, how = resolve_domain(rec, records, search_fn)
                if not dom:
                    say(f"  domain: none — {how}")
                    write(rec, {FIND_STATUS: FIND_ERROR, UPDATED: _now(),
                                NOTES: add_note(get(rec, NOTES), f"no domain: {how}")})
                    rep.errors.append((rec[ROW], "no domain"))
                    continue
                say(f"  domain: {dom} (from {how})")
                if not get(rec, DOMAIN).strip():
                    write(rec, {DOMAIN: dom})
                cached = dom.lower() in client.patterns
                lk = client.find(get(rec, PERSON), get(rec, COMPANY), dom)
                say(f"  coldstart{' (cached pattern, no call)' if cached and lk.ok else ''}: "
                    + (f"{lk.confidence} {lk.email}".rstrip() + (f", pattern {lk.pattern}" if lk.pattern else "")
                       if lk.ok else f"error — {lk.error}"))
                write(rec, find_fields(lk, get(rec, NOTES), get(rec, SOURCE)))
                if lk.ok:
                    rep.found.append((rec[ROW], lk.confidence, lk.email))
                else:
                    rep.errors.append((rec[ROW], lk.error))
            s.detail = f"{len(rep.found)} found, {len(rep.errors)} errors; {client.calls} Coldstart calls"

        if draft:
            _draft_all(records, todo if limit is not None else None, limit, gmail, resume, write, rep, say, log)
        summary = rep.summary()
        log.finish("ok" if rep.found or rep.drafted or rep.stopped
                   else ("error" if rep.errors or rep.draft_errors or rep.held else "no change"), summary)
        say(f"done: {summary}")
        return summary
    except SheetError as exc:
        log.finish("error", str(exc))
        say(f"stopped: {exc}")
        return str(exc)
    except Exception as exc:
        logger.exception("contacts: run failed")
        log.finish("error", str(exc))
        say(f"stopped: {exc}")
        return "The Contacts run failed; nothing was sent."


def _draft_all(records, only, limit, gmail, resume, write, rep, say, log) -> None:
    queue, waves = draft_queue(records)
    if only is not None:
        rows = {r[ROW] for r in only} or {r[ROW] for r in queue[:limit]}     # nothing looked up: the next draft(s)
        queue = [r for r in queue if r[ROW] in rows]
        waves = {k: v for k, v in waves.items() if k in rows}
    for rec in records:
        if rec[ROW] in waves and rec not in queue:
            write(rec, {WAVE: str(waves[rec[ROW]])})
    say(f"5. draft: {len(queue)} row(s) ready" + (f" ({', '.join(str(r[ROW]) for r in queue)})" if queue else ""))
    if not queue:
        return
    if gmail is None:
        from aria.plugins.productivity.gmail import GmailClient
        gmail = GmailClient(google_auth.JOBS)
    problem = sender_problem(gmail)
    resume = resume if resume is not None else resume_pdf()
    if not problem and not resume:
        problem = "no resume PDF to attach (set resume_path in My answers, or OUTREACH_RESUME)"
    if problem:
        say(f"  no drafts: {problem}")
        log.skip("draft", problem)
        return
    with log.step("draft") as s:
        for rec in queue:
            wave = waves.get(rec[ROW]) or get(rec, WAVE)
            brief = brief_of(rec)
            say(f"  row {rec[ROW]}: {get(rec, PERSON)} <{first_email(rec)}> at {get(rec, COMPANY)}, "
                f"wave {wave}, {brief.kind}")
            gaps = writer.missing(brief)
            if gaps:
                say(f"  needs research: {' and '.join(gaps)} blank or unknown; no draft")
                write(rec, {WAVE: str(wave), UPDATED: _now(), NOTES: _note_once(rec, NEEDS_RESEARCH)})
                rep.held.append(rec[ROW])
                continue
            p = writer.plan(brief)
            say(f"  about: {p.work['label'] if p.work else 'no history'} — {p.why}")
            msg = compose(rec, p)
            if msg is None:
                say("  the LLM draft failed (error or checks; see the log); no draft")
                write(rec, {WAVE: str(wave), UPDATED: _now(), NOTES: add_note(get(rec, NOTES), "LLM draft failed; not created")})
                rep.draft_errors.append((rec[ROW], "LLM draft failed"))
                continue
            say(f"  subject: {msg['subject']}")
            res = gmail.save_draft(msg["to"], msg["subject"], msg["body"], attachments=[resume])
            if not res.get("ok"):
                say(f"  gmail: {res.get('message')}")
                write(rec, {WAVE: str(wave), UPDATED: _now(),
                            NOTES: add_note(get(rec, NOTES), f"draft not created: {res.get('message')}")})
                rep.draft_errors.append((rec[ROW], res.get("message", "")))
                continue
            say(f"  gmail draft: {res.get('link') or res.get('draft_id')}")
            fields = {DRAFT_STATUS: DRAFTED, DRAFT_LINK: res.get("link", ""), WAVE: str(wave), UPDATED: _now()}
            if status(rec, CONFIDENCE) == cs.PATTERN:
                fields[NOTES] = _note_once(rec, PATTERN_GUESS)
            if p.override:
                rec[NOTES] = fields.get(NOTES, get(rec, NOTES))
                fields[NOTES] = _note_once(rec, ROLE_OVERRIDE)
            write(rec, fields)
            rep.drafted.append((rec[ROW], res.get("link", "")))
        s.detail = f"{len(rep.drafted)} drafted"


def _safe(fn):
    def say(msg: str) -> None:
        logger.info("contacts: %s", msg)
        if fn:
            try:
                fn(msg)
            except Exception:
                pass
    return say


# ---------------------------------------------------------------------------
# background
# ---------------------------------------------------------------------------

_bg_lock = threading.Lock()
_bg_thread: Optional[threading.Thread] = None


def start_background(done_fn: Optional[Callable[[str], None]] = None, **kwargs) -> bool:
    """run(**kwargs) on a daemon thread; done_fn(summary) when it ends. False if one is running.
    Nothing here touches the browser, so visible form-filling carries on alongside it."""
    global _bg_thread
    with _bg_lock:
        if _bg_thread is not None and _bg_thread.is_alive():
            return False

        def work() -> None:
            summary = run(**kwargs)
            if done_fn:
                try:
                    done_fn(summary)
                except Exception:
                    pass

        _bg_thread = threading.Thread(target=work, name="coldstart-contacts", daemon=True)
        _bg_thread.start()
        return True


def running() -> bool:
    return _bg_thread is not None and _bg_thread.is_alive()
