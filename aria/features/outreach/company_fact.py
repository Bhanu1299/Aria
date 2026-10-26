"""features/outreach/company_fact.py — what a company is solving, read off its own pages.

    find(company, domain, source="", job_url="") → Fact | None      never raises

Every outreach email must name something real about the company (a cold email that
could go to anyone gets skimmed). Code picks and reads the pages; the LLM only picks
the fact out of the text:

  1. pages   URLs in the row's Source cell on the company's domain or a sibling
             (aurelian.com for aurelian.io) — where Coldstart
             found the address, e.g. "Published on aurelian.com/studies/macc-911" —
             then the homepage, then the job posting. At most MAX_PAGES that load.
  2. pick    task outreach_company_fact: {"problem", "detail", "url"} — the problem
             they solve for whom, and one concrete detail (a product, a customer, a
             result) — or {"none": true}.
  3. check   the url is one of the pages read, and the problem and detail are each
             grounded in that text (most of their content words appear there).

None when no page loads or nothing on them passes the check: the caller holds the
draft rather than write a generic one.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse

from aria.features.outreach import contacts as finder
from aria.llm import llm_client

logger = logging.getLogger(__name__)

MAX_PAGES = 3
PAGE_CHARS = 2500
_URLISH = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s,;)\"'<>]*)?", re.I)
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]{3,}")
_STOP = {"that", "with", "from", "they", "their", "them", "this", "these", "into", "about", "while",
         "which", "where", "when", "have", "help", "helps", "more", "than", "over", "every", "each",
         "built", "build", "builds", "company", "platform", "teams", "using", "used"}

_SYSTEM = (
    "You read a company's own web pages and report, in plain words, the problem the company "
    "solves and for whom, plus ONE concrete detail from the pages (a named product, a named "
    "customer, a number, a result). Use only what the pages say; never infer or embellish. "
    'Reply with JSON only: {"problem": "<one sentence, under 30 words>", "detail": "<one '
    'sentence, under 25 words>", "url": "<the page the detail is from, exactly as given>"}. '
    'If the pages say nothing specific about what they build, reply {"none": true}.'
)


@dataclass
class Fact:
    problem: str      # what they solve, for whom
    detail: str       # one concrete thing from their pages
    url: str          # where it came from

    def text(self) -> str:
        return f"{self.problem} {self.detail} (from {self.url})"


def candidate_urls(domain: str, source: str = "", job_url: str = "") -> list[str]:
    """Pages to read, in order: Source URLs on the company's domain, the homepage, the posting."""
    domain = (domain or "").lower()
    out = []
    for m in _URLISH.finditer(source or ""):
        raw = m.group(0).rstrip(".")
        url = raw if raw.lower().startswith("http") else f"https://{raw}"
        host = finder.registrable(urlparse(url).hostname or "")
        if domain and _same_company(host, domain) and not finder.is_job_host(host):
            out.append(url)
    if domain:
        out.append(f"https://{domain}")
    if (job_url or "").startswith("http"):
        out.append(job_url)
    return list(dict.fromkeys(out))


def _same_company(host: str, domain: str) -> bool:
    """aurelian.com and aurelian.io are one company's sites."""
    return host == domain or (bool(host) and host.split(".")[0] == domain.split(".")[0])


def read_page(url: str, client=None) -> str:
    """Visible text of a page, trimmed. '' when it can't be fetched."""
    import httpx
    own = client is None
    http = client or httpx.Client()
    try:
        r = http.get(url, headers={"User-Agent": finder._UA}, timeout=10.0, follow_redirects=True)
        return " ".join(finder.page_lines(r.text))[:PAGE_CHARS] if r.status_code == 200 else ""
    except Exception as exc:
        logger.info("company fact: couldn't read %s: %s", url, exc)
        return ""
    finally:
        if own:
            http.close()


def words(text: str) -> set:
    """Content words (4+ letters, common filler dropped), lowercased."""
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def grounded(claim: str, page: str, company: str = "", share: float = 0.6) -> bool:
    """Most of the claim's content words (the company's own name aside) appear on the page."""
    claimed = words(claim) - words(company)
    if len(claimed) < 2:
        return False
    have = words(page)
    return sum(1 for w in claimed if w in have) / len(claimed) >= share


def _validator(pages: dict, company: str) -> Callable[[str], Optional[Fact]]:
    def check(text: str) -> Optional[Fact]:
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("none"):
            return None
        problem, detail = str(data.get("problem") or "").strip(), str(data.get("detail") or "").strip()
        url = str(data.get("url") or "").strip()
        if url not in pages or not (10 <= len(problem) <= 260) or not (5 <= len(detail) <= 220):
            return None
        everything = " ".join(pages.values())
        if not (grounded(problem, everything, company) and grounded(detail, pages[url], company)):
            return None
        return Fact(problem, detail, url)
    return check


def find(company: str, domain: str, source: str = "", job_url: str = "",
         fetch: Optional[Callable[[str], str]] = None) -> Optional[Fact]:
    """One fact about what `company` solves, from its own pages. None if there's nothing solid."""
    fetch = fetch or read_page
    pages: dict = {}
    for url in candidate_urls(domain, source, job_url):
        text = fetch(url)
        if text:
            pages[url] = text
        if len(pages) >= MAX_PAGES:
            break
    if not pages:
        return None
    brief = f"Company: {company}\n\n" + "\n\n".join(f"Page: {u}\n{t}" for u, t in pages.items())
    try:
        _resp, fact = llm_client.complete_checked(
            "outreach_company_fact", [{"role": "user", "content": brief}],
            validate=_validator(pages, company), system=_SYSTEM, max_tokens=300)
    except Exception as exc:
        logger.error("company fact: %s failed: %s", company, exc)
        return None
    logger.info("company fact: %s → %s", company, fact.text() if fact else "nothing solid")
    return fact
