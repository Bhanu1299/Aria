"""features/outreach/contacts.py — find 1–3 people to contact at a company. Pure code, no LLM.

Sources, most trusted first:
  posting    people named in the job posting ("reach out to Pat Lee, our recruiter")
  team       the company's own team / about pages (name line followed by a title line)
  search     web search for LinkedIn profiles: "Name - Title - Company | LinkedIn" titles
  github     public members of the company's GitHub org (title from their bio)

Every person gets a title score from TITLE_SCORES (recruiters first, then engineering
leaders, founders, engineers); people with no relevant title are dropped. rank() keeps
the top `limit`, ties broken by source trust.

    company_domain(job, search_fn) → "acme.com" | ""
    find_people(company, domain, posting_urls, ...) → [Person], best first

Network calls use one httpx.Client (injectable, so tests use MockTransport) and the
DDG search in aria.web.websearch. Every source fails soft: an error means no people
from that source, never an exception.
"""
from __future__ import annotations

import html as _html
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from aria.dashboard.model import company_slug, domain_matches_company, emails_in, is_company_address, norm_text

logger = logging.getLogger(__name__)

SOURCES = ("posting", "team", "search", "github")   # trust order
MAX_PEOPLE = 3
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"
_TIMEOUT = 8.0
TEAM_PATHS = ("/team", "/about", "/about-us", "/company", "/people", "/leadership")
GITHUB_API = "https://api.github.com"
_GITHUB_MEMBERS = 8

# (pattern, score) — first match wins, so specific titles come before general ones.
TITLE_SCORES: list[tuple[re.Pattern, int]] = [(re.compile(p, re.I), s) for p, s in [
    (r"\b(university|campus|early.career|new.grad|technical|engineering|ai|ml)\b.{0,20}\brecruit", 105),
    (r"\brecruit|\btalent (acquisition|partner|lead|manager)|\bsourcer\b|\bhead of talent", 100),
    (r"\bhiring manager\b", 95),
    (r"\b(engineering|eng|software|ai|ml|machine learning|data|platform|infrastructure) (manager|lead|director)\b"
     r"|\b(head|director|vp|vice president) of (engineering|ai|ml|machine learning|data|platform|research)\b"
     r"|\bvp,? engineering\b", 85),
    (r"\bcto\b|\bchief technology officer\b|\bco-?founder\b|\bfounder\b|\bceo\b|\bchief executive\b", 75),
    (r"\b(head of people|people (ops|operations|partner)|hr (manager|business partner)|human resources)\b", 60),
    (r"\b(staff|principal|senior|lead|sr\.?) (software |ml |machine learning |ai |research |data )?"
     r"(engineer|scientist|developer)\b", 45),
    (r"\b(engineer|scientist|developer|swe)\b", 35),
]]
NAMED_IN_POSTING = 90     # a person the posting names, even without a title
_SKIP_TITLE = re.compile(r"(?i)\b(sales|marketing|account executive|legal|counsel|finance|intern)\b")

_NAME_TOKEN = r"[A-Z][a-zA-Z'À-ſ-]+"
_NAME = re.compile(rf"^{_NAME_TOKEN}(?: (?:[A-Z]\.|{_NAME_TOKEN})){{1,3}}$")
_NOT_NAME_WORDS = {
    "about", "team", "our", "the", "and", "join", "us", "careers", "contact", "company", "people",
    "leadership", "board", "home", "news", "blog", "privacy", "terms", "learn", "more", "read", "view",
    "open", "roles", "jobs", "apply", "now", "meet", "advisors", "investors", "product", "platform",
    "engineering", "inc", "llc", "linkedin", "github", "twitter", "san", "francisco", "new", "york",
}
# "Pat Lee - Technical Recruiter - Acme | LinkedIn" / "Pat Lee – Recruiter at Acme | LinkedIn"
_LINKEDIN_TITLE = re.compile(r"^(?P<name>[^|–—-]{3,60}?)\s+[-–—]\s+(?P<title>[^|]{2,120}?)\s*(?:\|\s*LinkedIn.*)?$")
_POSTING_PERSON = re.compile(
    rf"(?i:reach out to|contact|e-?mail|questions\??\s*(?:to|for)|send (?:it|your \w+|them) to|"
    rf"hiring manager(?: is|:)?|recruiter(?: is|:)?|report(?:s|ing)? to)\s+(?P<name>{_NAME_TOKEN}(?: {_NAME_TOKEN}){{1,2}})"
    rf"(?:\s*\((?P<paren>[^)]{{2,60}})\)|,\s*(?:our |the )?(?P<title>[A-Za-z /&-]{{3,50}}?)(?:[,.;]|\s+at\b))?")
_BLOCK_TAGS = re.compile(r"(?is)<(script|style|noscript|svg)\b.*?</\1>")
_BREAK_TAGS = re.compile(r"(?i)<\s*(br|/p|/div|/h\d|/li|/td|/tr|/section|/article|/span|/a|/figcaption)\b[^>]*>")
_TAG = re.compile(r"<[^>]+>")

# Hosts that carry postings for many companies: never the company's own domain.
JOB_HOSTS = {
    "greenhouse.io", "lever.co", "ashbyhq.com", "myworkdayjobs.com", "workday.com", "linkedin.com",
    "indeed.com", "breezy.hr", "smartrecruiters.com", "jobvite.com", "icims.com", "joinhandshake.com",
    "wellfound.com", "angel.co", "ycombinator.com", "workatastartup.com", "builtin.com", "glassdoor.com",
    "workable.com", "bamboohr.com", "rippling.com", "dover.com", "notion.site", "google.com", "gem.com",
    "recruitee.com", "teamtailor.com", "personio.de", "jazzhr.com", "applytojob.com", "paylocity.com",
    "ultipro.com", "taleo.net", "successfactors.com", "simplyhired.com", "ziprecruiter.com", "dice.com",
    "otta.com", "welcometothejungle.com", "hired.com", "github.com", "wikipedia.org", "crunchbase.com",
    "facebook.com", "twitter.com", "x.com", "youtube.com", "instagram.com", "bloomberg.com", "medium.com",
    "pitchbook.com", "zoominfo.com", "rocketreach.co", "apollo.io", "levels.fyi", "theorg.com",
}
_SECOND_LEVEL = {"co", "com", "ac", "org", "net", "gov", "edu"}


@dataclass
class Person:
    name: str
    title: str = ""
    source: str = ""          # one of SOURCES
    url: str = ""             # where they were found (LinkedIn profile, page, GitHub user)
    email: str = ""           # only when the source published it
    score: int = 0

    def label(self) -> str:
        return f"{self.name} ({self.title})" if self.title else self.name


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def title_score(title: str) -> int:
    """How useful a person with this title is to contact about a role. 0 = don't."""
    if not title or _SKIP_TITLE.search(title):
        return 0
    return next((s for p, s in TITLE_SCORES if p.search(title)), 0)


def name_key(name: str) -> str:
    return norm_text(name)


def rank(people: list[Person], limit: int = MAX_PEOPLE) -> list[Person]:
    """Best `limit` distinct people: title score, then source trust. Unscored people are dropped."""
    best: dict[str, Person] = {}
    for p in people:
        p.score = max(title_score(p.title), NAMED_IN_POSTING if p.source == "posting" else 0)
        if p.score <= 0 or not looks_like_name(p.name):
            continue
        key = name_key(p.name)
        have = best.get(key)
        if have is None or (p.score, -SOURCES.index(p.source)) > (have.score, -SOURCES.index(have.source)):
            if have is not None:   # keep what the other source knew
                p.email, p.url = p.email or have.email, p.url or have.url
            best[key] = p
        else:
            have.email, have.url = have.email or p.email, have.url or p.url
    ordered = sorted(best.values(), key=lambda p: (-p.score, SOURCES.index(p.source)))
    return ordered[:limit]


def looks_like_name(text: str) -> bool:
    text = (text or "").strip()
    if not _NAME.match(text):
        return False
    words = [w.lower().strip(".") for w in text.split()]
    return not any(w in _NOT_NAME_WORDS for w in words) and not title_score(text)


# ---------------------------------------------------------------------------
# domain
# ---------------------------------------------------------------------------

def registrable(host: str) -> str:
    """'jobs.eu.acme.co.uk' → 'acme.co.uk'; 'www.acme.com' → 'acme.com'."""
    labels = [l for l in (host or "").lower().strip(".").split(".") if l]
    if len(labels) < 2:
        return ""
    n = 3 if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL and len(labels[-1]) == 2 else 2
    return ".".join(labels[-n:])


def is_job_host(domain: str) -> bool:
    d = registrable(domain)
    return d in JOB_HOSTS or any(d.endswith("." + h) for h in JOB_HOSTS)


def company_domain(job: dict, search_fn: Optional[Callable[[str], list]] = None) -> str:
    """The company's own email domain: an address in the row, else a non-job-board link host,
    else the first search result whose host matches the company name. '' if unknown."""
    fields = job.get("fields") or {}
    for e in emails_in(*fields.values()):
        if is_company_address(e) and not is_job_host(e.split("@")[1]):
            return registrable(e.split("@")[1])
    for link in (job.get("post_link"), job.get("apply_link")):
        host = urlparse(link or "").hostname or ""
        dom = registrable(host)
        if dom and not is_job_host(dom):
            return dom
    company = job.get("company") or ""
    if search_fn is None or not company_slug(company):
        return ""
    for r in search_fn(f"{company} official website") or []:
        dom = registrable(urlparse(r.get("url", "")).hostname or "")
        if dom and not is_job_host(dom) and domain_matches_company(dom, company):
            return dom
    return ""


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def page_lines(html: str) -> list[str]:
    """Visible text of a page, one block element per line."""
    text = _BREAK_TAGS.sub("\n", _BLOCK_TAGS.sub(" ", html or ""))
    text = _html.unescape(_TAG.sub(" ", text))
    return [re.sub(r"\s+", " ", l).strip() for l in text.split("\n") if l.strip()]


def from_posting(text: str, domain: str = "") -> list[Person]:
    """People a posting names, plus first.last@domain addresses it publishes."""
    out = []
    flat = re.sub(r"\s+", " ", text or "")
    for m in _POSTING_PERSON.finditer(flat):
        title = (m.group("paren") or m.group("title") or "").strip()
        out.append(Person(m.group("name").strip(), title if title_score(title) else "", "posting"))
    for e in emails_in(flat):
        local, _, dom = e.partition("@")
        if domain and registrable(dom) != domain:
            continue
        parts = [p for p in re.split(r"[._-]", local) if p.isalpha()]
        if len(parts) == 2 and all(len(p) > 1 for p in parts):
            name = " ".join(p.capitalize() for p in parts)
            hit = next((p for p in out if name_key(p.name) == name_key(name)), None)
            if hit:
                hit.email = e
            else:
                out.append(Person(name, "", "posting", email=e))
    return out


def from_team_page(html: str, url: str = "") -> list[Person]:
    """Name line followed by a title line, or "Name — Title" / "Name, Title" on one line."""
    lines, out = page_lines(html), []
    for i, line in enumerate(lines):
        one = re.match(r"^(.{3,50}?)\s*(?:[—–|,]|\s-\s)\s*(.{2,80})$", line)
        if one and looks_like_name(one.group(1)) and title_score(one.group(2)):
            out.append(Person(one.group(1), one.group(2), "team", url))
        elif looks_like_name(line) and i + 1 < len(lines) and len(lines[i + 1]) <= 80 and title_score(lines[i + 1]):
            out.append(Person(line, lines[i + 1], "team", url))
    return out


def from_search_results(results: list, company: str) -> list[Person]:
    """LinkedIn profile results whose title or snippet names this company."""
    out, want = [], norm_text(company)
    for r in results or []:
        if "linkedin.com/in/" not in (r.get("url") or ""):
            continue
        m = _LINKEDIN_TITLE.match((r.get("title") or "").strip())
        if not m:
            continue
        title = m.group("title")
        if want and want not in norm_text(f"{title} {r.get('snippet', '')}"):
            continue
        title = re.split(r"\s+[-–—]\s+|\s+at\s+", title)[0].strip()
        out.append(Person(m.group("name").strip(), title, "search", r["url"]))
    return out


def search_people(company: str, search_fn: Callable[[str], list]) -> list[Person]:
    queries = [
        f'site:linkedin.com/in "{company}" (recruiter OR "talent acquisition" OR "university recruiter")',
        f'site:linkedin.com/in "{company}" ("engineering manager" OR "head of engineering" OR CTO OR founder)',
    ]
    out: list[Person] = []
    for q in queries:
        try:
            out += from_search_results(search_fn(q) or [], company)
        except Exception as exc:
            logger.warning("outreach: search failed for %r: %s", q, exc)
    return out


def _get(client: httpx.Client, url: str) -> Optional[httpx.Response]:
    try:
        r = client.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT, follow_redirects=True)
        return r if r.status_code == 200 else None
    except Exception as exc:
        logger.info("outreach: GET %s failed: %s", url, exc)
        return None


def _robots(client: httpx.Client, domain: str) -> Optional[RobotFileParser]:
    r = _get(client, f"https://{domain}/robots.txt")
    if r is None:
        return None
    rp = RobotFileParser()
    rp.parse(r.text.splitlines())
    return rp


def team_people(domain: str, client: httpx.Client) -> list[Person]:
    if not domain:
        return []
    robots, out = _robots(client, domain), []
    for path in TEAM_PATHS:
        url = f"https://{domain}{path}"
        if robots is not None and not robots.can_fetch(_UA, url):
            continue
        r = _get(client, url)
        if r is not None and "html" in r.headers.get("content-type", "html"):
            out += from_team_page(r.text, str(r.url))
    return out


def posting_people(urls: list[str], domain: str, client: httpx.Client) -> list[Person]:
    out = []
    for url in dict.fromkeys(u for u in urls if u and u.startswith("http")):
        r = _get(client, url)
        if r is not None:
            out += from_posting(" ".join(page_lines(r.text)), domain)
    return out


def github_people(company: str, domain: str, client: httpx.Client) -> list[Person]:
    """Public members of the company's GitHub org; the org must point at the company's domain
    (its blog URL) or carry the company's name."""
    slug = company_slug(company)
    if not slug:
        return []
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "aria-outreach"}
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

    def api(path: str):
        try:
            r = client.get(f"{GITHUB_API}{path}", headers=headers, timeout=_TIMEOUT)
            return r.json() if r.status_code == 200 else None
        except Exception as exc:
            logger.info("outreach: GitHub %s failed: %s", path, exc)
            return None

    org = None
    for login in dict.fromkeys([slug, domain.split(".")[0] if domain else ""]):
        info = api(f"/orgs/{login}") if login else None
        if not info:
            continue
        blog = registrable(urlparse(info.get("blog") or "").hostname or (info.get("blog") or ""))
        if (domain and blog == domain) or company_slug(info.get("name") or "") == slug:
            org = info["login"]
            break
    if not org:
        return []
    out = []
    for m in (api(f"/orgs/{org}/public_members?per_page={_GITHUB_MEMBERS}") or [])[:_GITHUB_MEMBERS]:
        user = api(f"/users/{m.get('login')}") or {}
        name = (user.get("name") or "").strip()
        if not name:
            continue
        bio = user.get("bio") or ""
        title = next((t.strip() for t in re.split(r"[|•·@,.]| at ", bio) if title_score(t)), "Engineer")
        out.append(Person(name, title, "github", user.get("html_url") or ""))
    return out


def find_people(
    company: str,
    domain: str,
    posting_urls: list[str],
    search_fn: Optional[Callable[[str], list]] = None,
    client: Optional[httpx.Client] = None,
    limit: int = MAX_PEOPLE,
) -> list[Person]:
    """1–`limit` people at the company, best first. [] when no source finds anyone relevant."""
    own = client is None
    client = client or httpx.Client()
    try:
        found = posting_people(posting_urls, domain, client)
        found += team_people(domain, client)
        if search_fn is not None:
            found += search_people(company, search_fn)
        if len(rank(list(found), limit)) < limit:
            found += github_people(company, domain, client)
        return rank(found, limit)
    except Exception as exc:
        logger.error("outreach: finding people at %s failed: %s", company, exc)
        return []
    finally:
        if own:
            client.close()
