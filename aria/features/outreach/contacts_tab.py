"""features/outreach/contacts_tab.py — the Muse "Contacts" tab: Coldstart emails → Gmail drafts.

    run(limit=None, step_fn=None) → summary sentence. Never raises. Never sends.
    start_background(...) → bool      the same run on a daemon thread (False if one is running)

Columns are read and written by header name (integrations/header_sheet.py):
  the user's      Company, Tracked role, Person, Title, LinkedIn URL, Job URL, Domain
  Coldstart's     Email, Source, Confidence, Email pattern, Find status
  Aria's          Draft status, Gmail draft link, Wave, Updated, Notes
Aria writes Domain only when it was blank. Draft status "approved", "sent" and "replied"
are the user's: Aria only ever writes none / drafted / no reply.

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
  5. draft     rows with Confidence verified|pattern, at most CONTACTS_PER_COMPANY per
               company, numbered by Wave in sheet order (a person listed twice: the
               company address over a personal one). A wave is drafted only once every
               earlier wave at that company is sent. The LLM (gateway, task outreach_email)
               writes the body from Tracked role + the Job URL page; code does the rest.
               Gmail DRAFT in the jobs account, only if it is OUTREACH_FROM →
               Draft status drafted, Gmail draft link, Updated.
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

COMPANY, ROLE, PERSON, TITLE = "Company", "Tracked role", "Person", "Title"
LINKEDIN, JOB_URL, DOMAIN = "LinkedIn URL", "Job URL", "Domain"
EMAIL, SOURCE, CONFIDENCE, PATTERN, FIND_STATUS = "Email", "Source", "Confidence", "Email pattern", "Find status"
DRAFT_STATUS, DRAFT_LINK, WAVE, UPDATED, NOTES = "Draft status", "Gmail draft link", "Wave", "Updated", "Notes"
WRITABLE = (DOMAIN, EMAIL, SOURCE, CONFIDENCE, PATTERN, FIND_STATUS, DRAFT_STATUS, DRAFT_LINK, WAVE, UPDATED, NOTES)

FIND_PENDING, FIND_DONE, FIND_ERROR = "pending", "done", "error"
NONE, DRAFTED, APPROVED, SENT, REPLIED, NO_REPLY = "none", "drafted", "approved", "sent", "replied", "no reply"
ARIA_DRAFT_STATUSES = {NONE, DRAFTED, NO_REPLY}          # everything else is the user's to set
_IN_FLIGHT = {DRAFTED, APPROVED, SENT, REPLIED}          # counts toward the per-company cap
_WAVE_CLEARED = {SENT, NO_REPLY}                        # a later wave may start after these
_NOT_A_PERSON = {"", "—", "–", "-", "n/a", "na", "none", "?"}
POSTING_CHARS = 3000


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def status(rec: dict, name: str) -> str:
    return get(rec, name).strip().lower()


def is_person(name: str) -> bool:
    """A name Coldstart can look up: not "—" and not a placeholder like "(hiring contact)"."""
    name = (name or "").strip()
    return name.lower() not in _NOT_A_PERSON and not name.startswith("(")


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
    return {COMPANY: get(rec, COMPANY), PERSON: get(rec, PERSON)}


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

    Draftable: a real person, an address, Confidence verified|pattern, no reply yet at the
    company; a person listed twice counts once. Waves 1..cap go to the company's draftable rows in sheet order (rows already in
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
            if not (is_person(get(r, PERSON)) and first_email(r) and status(r, DRAFT_STATUS) != NO_REPLY
                    and status(r, CONFIDENCE) in (cs.VERIFIED, cs.PATTERN)):
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
        queue += [r for r in slotted if wave_of[r[ROW]] == active and status(r, DRAFT_STATUS) in ("", NONE)]
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


def find_fields(lk: cs.Lookup, notes: str) -> dict:
    """Coldstart's answer → the row's cells."""
    if not lk.ok:
        return {FIND_STATUS: FIND_ERROR, UPDATED: _now(), NOTES: add_note(notes, f"Coldstart error: {lk.error}")}
    if lk.confidence == cs.NOT_FOUND:
        return {EMAIL: "not found", SOURCE: lk.source, CONFIDENCE: cs.NOT_FOUND, PATTERN: lk.pattern,
                FIND_STATUS: FIND_DONE, UPDATED: _now(),
                NOTES: add_note(notes, "Coldstart found no address; no draft")}
    return {EMAIL: lk.email, SOURCE: lk.source, CONFIDENCE: lk.confidence, PATTERN: lk.pattern,
            FIND_STATUS: FIND_DONE, UPDATED: _now()}


# ---------------------------------------------------------------------------
# drafting
# ---------------------------------------------------------------------------

@dataclass
class _Target:
    """What writer.email needs, from one Contacts row."""
    company: str
    jobs: list
    contact: object


def posting_text(url: str, client=None) -> str:
    """Visible text of the job page, trimmed. '' when it can't be fetched."""
    if not (url or "").startswith("http"):
        return ""
    import httpx
    own = client is None
    http = client or httpx.Client()
    try:
        r = http.get(url, headers={"User-Agent": finder._UA}, timeout=10.0, follow_redirects=True)
        if r.status_code != 200:
            return ""
        return " ".join(finder.page_lines(r.text))[:POSTING_CHARS]
    except Exception as exc:
        logger.info("contacts: couldn't read the posting %s: %s", url, exc)
        return ""
    finally:
        if own:
            http.close()


def sender_problem(gmail) -> str:
    """'' when the jobs Gmail is OUTREACH_FROM, else why no draft may be made there."""
    want = (config.OUTREACH_FROM or "").strip().lower()
    if not want:
        return "OUTREACH_FROM (or GOOGLE_JOBS_ACCOUNT) isn't set in .env"
    have = (google_auth.account_email(getattr(gmail, "account", google_auth.JOBS)) or "").strip().lower()
    if have != want:
        return f"the jobs Gmail is {have or 'not connected'}, not {want}"
    return ""


def compose(rec: dict, posting: str) -> Optional[dict]:
    """{to, subject, body} from the gateway LLM, or None if its output failed checks."""
    t = _Target(company=get(rec, COMPANY), jobs=[{"role": get(rec, ROLE), "fields": {}}],
                contact=finder.Person(get(rec, PERSON), get(rec, TITLE)))
    return writer.email(t, first_email(rec), posting=posting)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

@dataclass
class Report:
    found: list = field(default_factory=list)       # (row, confidence, email)
    errors: list = field(default_factory=list)      # (row, why) lookups
    draft_errors: list = field(default_factory=list)  # (row, why)
    drafted: list = field(default_factory=list)     # (row, link)
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
        if self.drafted:
            parts.append(f"{len(self.drafted)} draft{'s' if len(self.drafted) != 1 else ''} in the jobs Gmail")
        if self.stopped:
            parts.append(f"{len(self.stopped)} row{'s' if len(self.stopped) != 1 else ''} closed after a reply")
        return ("Contacts: " + "; ".join(parts) + ". Nothing was sent.") if parts else \
            "Contacts: nothing to look up or draft."


class _Writer:
    """sheet.update that also refreshes the in-memory row and refuses user-owned statuses."""

    def __init__(self, sheet: HeaderSheet, step: Callable[[str], None]) -> None:
        self.sheet, self.step = sheet, step

    def __call__(self, rec: dict, fields: dict) -> None:
        ds = fields.get(DRAFT_STATUS)
        if ds is not None and ds not in ARIA_DRAFT_STATUSES:
            raise SheetError(f"Aria doesn't set Draft status to {ds!r}.")
        rec[ROW] = self.sheet.update(rec[ROW], fields, match=match_of(rec))
        rec.update(fields)
        shown = ", ".join(f"{k}={v!r}" for k, v in fields.items() if k != NOTES)
        self.step(f"  wrote row {rec[ROW]}: {shown}" + (" (+ Notes)" if NOTES in fields else ""))


def run(
    sheet: Optional[HeaderSheet] = None,
    client: Optional[cs.ColdstartClient] = None,
    gmail=None,
    search_fn=None,
    posting_fn: Optional[Callable[[str], str]] = None,
    limit: Optional[int] = None,
    draft: bool = True,
    step_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """One pass over the Contacts tab. limit: look up at most this many pending rows, and
    draft only rows looked up in this run — or, when none were pending, the first `limit`
    rows ready to draft (the thin slice: limit=1). Never raises."""
    say = _safe(step_fn)
    sheet = sheet or HeaderSheet(config.MUSE_SHEET_ID, config.MUSE_CONTACTS_TAB, writable=WRITABLE)
    client = client or cs.ColdstartClient()
    posting_fn = posting_fn or posting_text
    write = _Writer(sheet, say)
    rep = Report()
    log = run_log.start("contacts")
    try:
        with log.step("read sheet") as s:
            records = sheet.records()
            s.detail = f"{len(records)} rows in {sheet.tab}"
        say(f"1. read: {len(records)} rows in '{sheet.tab}'")

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
        say(f"3. pending: {len(todo)} row(s) to look up" + (f" ({', '.join(str(r[ROW]) for r in todo)})" if todo else "")
            + (f"; {len(skipped)} duplicate(s) skipped" if skipped else ""))

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
                write(rec, find_fields(lk, get(rec, NOTES)))
                if lk.ok:
                    rep.found.append((rec[ROW], lk.confidence, lk.email))
                else:
                    rep.errors.append((rec[ROW], lk.error))
            s.detail = f"{len(rep.found)} found, {len(rep.errors)} errors; {client.calls} Coldstart calls"

        if draft:
            _draft_all(records, todo if limit is not None else None, limit, gmail, posting_fn, write, rep, say, log)
        summary = rep.summary()
        log.finish("ok" if rep.found or rep.drafted or rep.stopped
                   else ("error" if rep.errors or rep.draft_errors else "no change"), summary)
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


def _draft_all(records, only, limit, gmail, posting_fn, write, rep, say, log) -> None:
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
    if problem:
        say(f"  no drafts: {problem}")
        log.skip("draft", problem)
        return
    with log.step("draft") as s:
        for rec in queue:
            wave = waves.get(rec[ROW]) or get(rec, WAVE)
            posting = posting_fn(get(rec, JOB_URL))
            say(f"  row {rec[ROW]}: {get(rec, PERSON)} <{first_email(rec)}>, wave {wave}, "
                f"posting {'read (' + str(len(posting)) + ' chars)' if posting else 'unavailable'}")
            msg = compose(rec, posting)
            if msg is None:
                say("  the LLM draft failed (error or checks; see the log); no draft")
                write(rec, {WAVE: str(wave), UPDATED: _now(),
                            NOTES: add_note(get(rec, NOTES), "LLM draft failed; not created")})
                rep.draft_errors.append((rec[ROW], "LLM draft failed"))
                continue
            say(f"  subject: {msg['subject']}")
            res = gmail.save_draft(msg["to"], msg["subject"], msg["body"])
            if not res.get("ok"):
                say(f"  gmail: {res.get('message')}")
                write(rec, {WAVE: str(wave), UPDATED: _now(),
                            NOTES: add_note(get(rec, NOTES), f"draft not created: {res.get('message')}")})
                rep.draft_errors.append((rec[ROW], res.get("message", "")))
                continue
            say(f"  gmail draft: {res.get('link') or res.get('draft_id')}")
            write(rec, {DRAFT_STATUS: DRAFTED, DRAFT_LINK: res.get("link", ""), WAVE: str(wave), UPDATED: _now()})
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
