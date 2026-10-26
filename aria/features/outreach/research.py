"""features/outreach/research.py — Aria's own research when Muse left an applied job blank.

Muse researches each applied job into the Companies + Contacts tabs. Aria steps in only
when a job it prepared (~/.aria/jobs/<ID>-…/posting.txt) is Applied and still has no
Companies row GRACE_HOURS after Aria first saw the gap (user, 2026-10-26: Muse first;
"if Muse did not enter anything, Aria can do the research"):

    sources   the posting (cleaned), the company's home + about page (domain from the sheet's
              links, a search, else <name>.com/.ai/.io/.co whose title names the company), one
              news search; no site and no news → nothing is written (left for Muse)
    LLM       task outreach_research: Company problem, Signal + its URL, one company fact
    code      every claim is checked against the sources — the problem shares 3+ content
              words with the company's own site / news, not just the posting (else nothing
              is written); a Signal names one of the source
              URLs, its numbers are in that source and it shares 4+ words with it (else
              "unknown"); the fact shares 2+ words with the site / news that the posting doesn't use (else the Angle is the
              role's work alone, which the writer treats as a role override)
    blocker   code: citizenship / clearance / no sponsorship in the posting → "review"
    people    contacts.find_people → up to 3 Contacts rows (Wave 1, Find status pending,
              Added by Aria); nobody found → one "—" team-inbox row

Rows are only appended: a Job ID already on the Companies tab is never touched. The
Companies row's Notes say "researched by Aria"; the normal Contacts run then looks the
emails up and drafts (drafts only, never sent).

    gaps(companies, jobs, now) → [job]          Applied, prepared, no Companies row, past grace
    research(job, ...) → Research | None
    fill_gaps(companies, contacts, ...) → [line] what was added (never raises)
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx

from aria.features.ats import untrusted
from aria.features.outreach import contacts as finder
from aria.features.outreach import writer
from aria.integrations.header_sheet import get
from aria.llm import llm_client

logger = logging.getLogger(__name__)

GRACE_HOURS = 24.0                  # Muse's head start, from when Aria first sees the gap
MAX_PER_RUN = 5                     # companies researched in one run
SEEN_PATH = Path.home() / ".aria" / "outreach" / "research_seen.json"
SOURCE_CHARS = 3000                 # per source the LLM sees
ABOUT_PATHS = ("/about", "/about-us", "/company")
ARIA = "Aria"
REVIEW, ACTIVE = "review", "active"
UNKNOWN = "unknown"
COMPANY_COLUMNS = ("Job ID", "Company", "Role", "Job URL", "Domain", "Company problem", "Signal",
                   "Signal source", "Angle", "Blocker", "Outreach status", "Notes")
CONTACT_COLUMNS = ("Job ID", "Company", "Person", "Title", "LinkedIn URL", "Find status", "Wave",
                   "Added by", "Notes")

_BLOCKERS = (
    (re.compile(r"(?i)\b(?:u\.?s\.? citizen(?:ship)?|united states citizen(?:ship)?)\b[^.\n]{0,40}\brequired\b|"
                r"\bmust be (?:a )?(?:u\.?s\.?|united states) citizen\b|\bcitizenship (?:is )?required\b"),
     "U.S. citizenship required"),
    (re.compile(r"(?i)\b(?:security clearance|clearance (?:is )?required|active (?:secret|ts/sci|top secret))\b"),
     "security clearance"),
    (re.compile(r"(?i)\b(?:no|not|unable to|won't|will not|cannot|can't|does not|do not)\b[^.\n]{0,40}"
                r"\bsponsor(?:ship)?\b"),
     "posting says no visa sponsorship"),
)

_SYSTEM = (
    "You research one company for a job applicant's short cold email. Use ONLY the sources "
    "given; never add what they don't say. Reply with JSON only:\n"
    '{"problem": "<one plain sentence, under 35 words: who the company helps and how — what it is '
    'solving>", "signal": "<one recent concrete event from a source (a launch, funding, a hire, '
    'growth), close to its wording, or unknown>", "signal_source": "<that source\'s URL, or '
    'unknown>", "fact": "<one concrete thing the company builds, ships or runs, from its own site or '
    'news (never the job posting), that touches the candidate\'s work below; start with the company '
    'name, under 25 words, or unknown>"}\n' + untrusted.RULE)


@dataclass
class Source:
    url: str
    text: str


@dataclass
class Research:
    job_id: str
    company: str
    role: str
    job_url: str
    domain: str
    problem: str
    signal: str = UNKNOWN
    signal_source: str = UNKNOWN
    angle: str = ""
    blocker: str = ""
    people: list = field(default_factory=list)      # [finder.Person]

    @property
    def status(self) -> str:
        return REVIEW if self.blocker else ACTIVE


# ---------------------------------------------------------------------------
# which jobs
# ---------------------------------------------------------------------------

def _load_seen() -> dict:
    try:
        return json.loads(SEEN_PATH.read_text())
    except Exception:
        return {}


def _save_seen(seen: dict) -> None:
    try:
        SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        SEEN_PATH.write_text(json.dumps(seen, indent=1, sort_keys=True))
    except Exception as exc:
        logger.warning("outreach research: couldn't save %s: %s", SEEN_PATH, exc)


def job_folder(job_id: str) -> Optional[Path]:
    from aria.features.resume.prepare import JOBS_DIR
    jid = (job_id or "").strip()
    if not jid or not JOBS_DIR.is_dir():
        return None
    return next((d for d in sorted(JOBS_DIR.iterdir()) if d.is_dir() and d.name.split("-")[0] == jid
                 and (d / "posting.txt").is_file()), None)


def gaps(company_rows: list, jobs: list, now: Optional[float] = None, seen: Optional[dict] = None) -> list:
    """Applied jobs Aria prepared that have no Companies row, GRACE_HOURS after first seen.
    `seen` ({job_id: first-seen epoch}) is updated in place: new gaps are stamped, filled ones dropped."""
    now = time.time() if now is None else now
    seen = _load_seen() if seen is None else seen
    have = {get(r, "Job ID").strip().upper() for r in company_rows if get(r, "Job ID").strip()}
    out, open_ids = [], set()
    for job in jobs:
        jid = (job.get("job_id") or "").strip().upper()
        if not jid or jid in have or (job.get("status") or "").strip().lower() != "applied":
            continue
        if job_folder(jid) is None:
            continue
        open_ids.add(jid)
        first = seen.setdefault(jid, now)
        if now - float(first) >= GRACE_HOURS * 3600:
            out.append(job)
    for jid in [k for k in seen if k not in open_ids]:
        del seen[jid]
    return out


# ---------------------------------------------------------------------------
# research
# ---------------------------------------------------------------------------

def blocker(posting: str) -> str:
    return next((why for rx, why in _BLOCKERS if rx.search(posting or "")), "")


_META = re.compile(r"""(?is)<meta[^>]+(?:name|property)=["'](?:description|og:description|og:site_name)["'][^>]*>""")
_CONTENT = re.compile(r"""(?is)content=["']([^"']*)["']""")
_PARKED = re.compile(r"(?i)domain (?:is )?for sale|buy this domain|parked (?:free|domain)|this domain may be for sale")


def meta_text(html: str) -> str:
    """The page's description / og:description / og:site_name (what a JS-built site still serves)."""
    out = []
    for tag in _META.findall(html or ""):
        m = _CONTENT.search(tag)
        if m and m.group(1).strip() not in out:
            out.append(m.group(1).strip())
    return ". ".join(out)


def _page(client: httpx.Client, url: str) -> str:
    r = finder._get(client, url)
    if r is None or "html" not in r.headers.get("content-type", "html"):
        return ""
    return f"{meta_text(r.text)} {' '.join(finder.page_lines(r.text))}".strip()[:SOURCE_CHARS]


def sources(posting: str, domain: str, company: str, search_fn: Optional[Callable[[str], list]],
            client: httpx.Client) -> list:
    out = [Source("the job posting", posting[:SOURCE_CHARS])] if posting.strip() else []
    if domain:
        robots = finder._robots(client, domain)
        for path in ("/",) + ABOUT_PATHS:
            url = f"https://{domain}{path}"
            if robots is not None and not robots.can_fetch(finder._UA, url):
                continue
            text = _page(client, url)
            if text:
                out.append(Source(url, text))
            if path != "/" and text:
                break
    if search_fn is not None:
        for r in (search_fn(f"{company} news 2026") or [])[:5]:
            if r.get("url") and (r.get("snippet") or r.get("title")):
                out.append(Source(r["url"], f"{r.get('title', '')}. {r.get('snippet', '')}"))
    return out


def guess_domain(company: str, client: httpx.Client) -> str:
    """The company's site without a search (DDG throttles): <name>.com / .ai / .io / .co, kept
    only when the home page's <title>, og:site_name or © line names the company and it isn't
    a parked / for-sale page. '' when none does."""
    name = re.sub(r"(?i)\b(?:inc|llc|ltd|corp(?:oration)?|co|company|technologies|group|holdings)\b\.?", "",
                  writer.bare_company(company))
    slug = re.sub(r"[^a-z0-9]", "", name.lower())
    if len(slug) < 3:
        return ""
    flat = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    for tld in (".com", ".ai", ".io", ".co"):
        r = finder._get(client, f"https://{slug}{tld}/")
        page = r.text if r is not None else ""
        if not page or _PARKED.search(page):
            continue
        m = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
        site = re.search(r"""(?is)<meta[^>]+property=["']og:site_name["'][^>]*content=["']([^"']*)""", page)
        text = " ".join(finder.page_lines(page))       # tags out: "©<span>2026</span> Virtusa Corp."
        owner = re.search(r"(?i)(?:©|\bcopyright\b)\s*(?:\d{4}\s*[-–]?\s*)*([^|.\n]{2,60})", text)
        if any(m_ and slug in flat(m_.group(1)) for m_ in (m, site, owner)):
            return finder.registrable(httpx.URL(str(r.url)).host or f"{slug}{tld}")
    return ""


def _first_point(text: str) -> str:
    """The first bullet / sentence of a work description, without the bullet mark."""
    first = next((l for l in (text or "").splitlines() if l.strip(" •-*\t")), "")
    return re.sub(r"\s+", " ", first.strip(" •-*\t"))[:220]


def _work_name(work: dict) -> str:
    label = work.get("label") or ""
    return label.split(" at ", 1)[1] if " at " in label else label


def _prompt(company: str, role: str, work: Optional[dict], srcs: list) -> str:
    text = f"Company: {company}\nRole: {role}\n"
    if work:
        text += f"Candidate's work for this role: {work['label']}: {work['text'][:600]}\n"
    for s in srcs:
        text += "\n" + untrusted.wrap(s.text, f"Source {s.url}") + "\n"
    return text


def _parse(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    return {k: str(d.get(k) or "").strip() for k in ("problem", "signal", "signal_source", "fact")}


def grounded(d: dict, srcs: list) -> Optional[dict]:
    """The parsed answer with every unsupported claim blanked; None when the problem isn't
    supported by the sources (then nothing is written)."""
    # The company's own words (site / news), not the posting: a problem or fact that only the
    # posting supports describes the job, not the company.
    site = set().union(*(writer.words(s.text) for s in srcs if s.url.startswith("http"))) or set()
    problem = re.sub(r"\s+", " ", d.get("problem", ""))
    if writer._blank(problem) or len(problem.split()) > 45 or len(writer.words(problem) & site) < 3:
        return None
    signal, url = d.get("signal", ""), d.get("signal_source", "")
    src = next((s for s in srcs if s.url == url and url.startswith("http")), None)
    if writer._blank(signal) or src is None or len(writer.words(signal) & writer.words(src.text)) < 4 \
            or not writer._numbers(signal) <= writer._numbers(src.text):
        signal, url = UNKNOWN, UNKNOWN
    fact = d.get("fact", "")
    posting = set().union(*(writer.words(s.text) for s in srcs if not s.url.startswith("http"))) or set()
    if writer._blank(fact) or len(writer.words(fact) & (site - posting)) < 2 or writer._numbers(fact) - writer._numbers(
            " ".join(s.text for s in srcs)):
        fact = ""
    return {"problem": problem, "signal": signal, "signal_source": url, "fact": fact}


def research(job: dict, search_fn: Optional[Callable[[str], list]] = None,
             client: Optional[httpx.Client] = None) -> Optional[Research]:
    """One job's Companies row + people. None when the sources don't support a problem
    statement or the LLM failed (nothing is written then)."""
    jid, company, role = job.get("job_id", ""), job.get("company", ""), job.get("role", "")
    folder = job_folder(jid)
    posting = (folder / "posting.txt").read_text(errors="replace") if folder else ""
    own = client is None
    client = client or httpx.Client()
    try:
        domain = finder.company_domain(job, search_fn) or guess_domain(company, client)
        srcs = sources(posting, domain, company, search_fn, client)
        if not any(s.url.startswith("http") for s in srcs):
            logger.warning("outreach research: %s — only the posting to go on (no site, no news)", company)
            return None
        work = writer.role_work(writer.Brief(company=company, role=role, problem="", angle=""))
        check = lambda text: grounded(_parse(text) or {}, srcs) if _parse(text) else None
        _resp, d = llm_client.complete_checked(
            "outreach_research", [{"role": "user", "content": _prompt(company, role, work, srcs)}],
            validate=check, system=_SYSTEM, max_tokens=600)
        if d is None:
            logger.warning("outreach research: %s — nothing the sources support", company)
            return None
        name = _work_name(work) if work else ""
        lead = (writer.bare_company(company).split() or [company])[0].lower()
        fact = d["fact"] if lead in d["fact"].lower() else f"{company} {d['fact']}"
        angle = f"{name}: {fact}" if name and d["fact"] else \
            (f"{name}: {_first_point(work['text'])}" if work else "")
        people = finder.find_people(company, domain, [u for u in (job.get("post_link"), job.get("apply_link")) if u],
                                    search_fn=search_fn, client=client)
        return Research(job_id=jid, company=company, role=role, job_url=job.get("url") or job.get("post_link", ""),
                        domain=domain or UNKNOWN, problem=d["problem"], signal=d["signal"],
                        signal_source=d["signal_source"], angle=angle, blocker=blocker(posting), people=people)
    except Exception as exc:
        logger.error("outreach research: %s failed: %s", company, exc)
        return None
    finally:
        if own:
            client.close()


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def company_fields(r: Research, today: str) -> dict:
    return {"Job ID": r.job_id, "Company": r.company, "Role": r.role, "Job URL": r.job_url, "Domain": r.domain,
            "Company problem": r.problem, "Signal": r.signal, "Signal source": r.signal_source, "Angle": r.angle,
            "Blocker": r.blocker, "Outreach status": r.status, "Notes": f"{today}: researched by Aria (Muse left it blank)"}


def contact_rows(r: Research) -> list:
    base = {"Job ID": r.job_id, "Company": r.company, "Find status": "pending", "Wave": "1", "Added by": ARIA}
    if not r.people:
        return [dict(base, Person="—", Notes="no named contact found; team inbox")]
    return [dict(base, Person=p.name, Title=p.title,
                 **({"LinkedIn URL": p.url} if "linkedin.com" in (p.url or "") else {}),
                 Notes=f"found by Aria ({p.source})") for p in r.people]


def fill_gaps(companies, contacts, jobs_fn: Optional[Callable[[], list]] = None,
              search_fn: Optional[Callable[[str], list]] = None, say: Callable[[str], None] = lambda s: None,
              now: Optional[float] = None) -> list:
    """Research every gap (up to MAX_PER_RUN) and append its rows. Returns a line per company
    added. Never raises."""
    from datetime import date
    added = []
    try:
        jobs = (jobs_fn or _applied_jobs)()
        seen = _load_seen()
        todo = gaps(companies.records(), jobs, now=now, seen=seen)
        _save_seen(seen)
        if not todo:
            return []
        say(f"0. research: {len(todo)} applied job(s) Muse left blank — Aria researches "
            f"{min(len(todo), MAX_PER_RUN)}")
        for job in todo[:MAX_PER_RUN]:
            r = research(job, search_fn=search_fn)
            if r is None:
                say(f"  {job.get('company')}: nothing the sources support; left for Muse")
                continue
            companies.append(company_fields(r, date.today().isoformat()))
            for row in contact_rows(r):
                contacts.append(row)
            line = (f"{r.company} ({r.job_id}): {len(r.people) or 'team inbox'} contact(s)"
                    + (f", held for review — {r.blocker}" if r.blocker else ""))
            say(f"  {line}")
            added.append(line)
    except Exception as exc:
        logger.error("outreach research: run failed: %s", exc)
        say(f"  research stopped: {exc}")
    return added


def _applied_jobs() -> list:
    """Jobs on the Applied tab, plus Jobs-tab rows whose Status is Applied."""
    import aria.core.config as config
    from aria.integrations import sheets
    out = []
    for sh in (sheets.default_sheet(), sheets.tab_sheet(config.MUSE_APPLIED_TAB)):
        try:
            for j in sh.jobs():
                if sh.tab == config.MUSE_APPLIED_TAB:
                    j = dict(j, status="Applied")
                out.append(j)
        except Exception as exc:
            logger.warning("outreach research: couldn't read %s: %s", sh.tab, exc)
    return out
