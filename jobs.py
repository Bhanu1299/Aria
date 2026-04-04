"""
jobs.py — Phase 3C: Voice-driven job search for Aria

Pipeline:
  1. Groq parses role + location from voice query
  2. agent_browser opens a VISIBLE LinkedIn Jobs search page
  3. computer_use.take_screenshot() captures the results
  4. Groq vision extracts job listings from the screenshot
  5. DOM query extracts real LinkedIn job URLs in one pass
  6. Returns top 5 with real URLs — browser stays open for apply flow

Indeed is supported as a fallback (when LinkedIn returns 0 results)
or when the user explicitly mentions "indeed" in their query.
"""

from __future__ import annotations

import json
import logging
import re
import time
from urllib.parse import urlencode

from groq import Groq

import agent_browser
import computer_use
import config

logger = logging.getLogger(__name__)

_CLIENT: Groq | None = None
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_SETTLE_JOBS = 5.0     # seconds to wait after LinkedIn Jobs loads (SPA-heavy)


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def _parse_salary_filter(query: str) -> str:
    """
    Extract LinkedIn salary filter value (f_SB2) from a voice query.

    LinkedIn salary tiers (f_SB2):
      1=$40k+  2=$60k+  3=$80k+  4=$100k+  5=$120k+  6=$140k+

    Returns the string filter value, or '' if no salary mentioned.
    """
    q = query.lower()

    # "six figures" / "six-figure" → $100k+
    if "six figure" in q:
        return "4"

    # Pattern: "100k", "120,000", "$140k", "80 thousand"
    m = re.search(r'\$?(\d{2,3})[\s,]?(?:k|,000|thousand)', q)
    if m:
        amount = int(m.group(1))
        if amount >= 140:
            return "6"
        if amount >= 120:
            return "5"
        if amount >= 100:
            return "4"
        if amount >= 80:
            return "3"
        if amount >= 60:
            return "2"
        if amount >= 40:
            return "1"

    return ""


def _parse_filters(query: str) -> dict:
    """Extract job filter params from voice query for LinkedIn URL."""
    filters = {}
    q = query.lower()
    if "remote" in q:
        filters["f_WT"] = "2"
    elif "hybrid" in q:
        filters["f_WT"] = "3"
    elif "on-site" in q or "onsite" in q or "in person" in q or "in office" in q:
        filters["f_WT"] = "1"
    if "today" in q or "past day" in q or "last 24" in q:
        filters["f_TPR"] = "r86400"
    elif "this week" in q or "past week" in q or "last week" in q:
        filters["f_TPR"] = "r604800"
    elif "this month" in q or "past month" in q:
        filters["f_TPR"] = "r2592000"
    salary_tier = _parse_salary_filter(query)
    if salary_tier:
        filters["f_SB2"] = salary_tier
    return filters


def _strip_filter_words(text: str) -> str:
    """Remove filter keywords from search text to clean up the role query."""
    remove = ["remote", "hybrid", "on-site", "onsite", "in person", "in office",
              "today", "past day", "last 24", "this week", "past week", "last week",
              "this month", "past month", "posted"]
    words = text.split()
    return " ".join(w for w in words if w.lower() not in remove)


def _parse_query(query: str) -> tuple[str, str]:
    """
    Extract job role and location from a voice query via Groq.
    Returns (role, location). Falls back to (query, "") on error.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract job search intent. Return JSON only with 'role' "
                        "(job title/type) and 'location' (city, 'remote', or empty string). "
                        "No markdown, no extras. "
                        'Example: {"role": "software engineer", "location": "New York"}'
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        role = str(parsed.get("role") or query).strip()
        location = str(parsed.get("location") or "").strip()
        return role, location
    except Exception as exc:
        logger.error("_parse_query failed (%s) — using raw query", exc)
        return query, ""


def _build_linkedin_jobs_url(role: str, location: str) -> str:
    """Build a LinkedIn Jobs search URL."""
    params: dict[str, str] = {"keywords": role}
    if location:
        params["location"] = location
    return "https://www.linkedin.com/jobs/search/?" + urlencode(params)


def _vision_ask(b64: str, prompt: str) -> str:
    """Send a screenshot (base64) + text prompt to Groq Llama-4-Scout. Returns response text, or '' on error."""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("_vision_ask failed: %s", exc)
        return ""


def _extract_listings_from_screenshot(b64: str) -> list[dict]:
    """
    Ask vision to extract job listings from a LinkedIn Jobs screenshot.
    Returns up to 5 dicts: title, company, location, posted, url, platform.
    """
    prompt = (
        "This is a LinkedIn Jobs search results page. "
        "Extract up to 5 job listings visible in the results list. "
        "Return a valid JSON array only — no markdown, no explanation. "
        'Each object must have exactly these keys: "title" (string), '
        '"company" (string), "location" (string), '
        '"posted" (string, e.g. "2 days ago" or "today"), '
        '"url" (empty string — URLs will be filled separately), '
        '"platform" (always "LinkedIn"). '
        'Use empty string for any field that is not visible. '
        'Return [] if no job listings are visible.'
    )
    raw = _vision_ask(b64, prompt)
    logger.info("Vision raw for LinkedIn Jobs: %r", raw[:400])

    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        listings = json.loads(raw)
        if not isinstance(listings, list):
            logger.error("Vision returned non-list: %r", raw[:200])
            return []
        results = []
        for item in listings:
            if not isinstance(item, dict):
                continue
            results.append({
                "title":    str(item.get("title") or "").strip(),
                "company":  str(item.get("company") or "").strip(),
                "location": str(item.get("location") or "").strip(),
                "posted":   str(item.get("posted") or "").strip(),
                "url":      "",
                "platform": "LinkedIn",
            })
        return results
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse listings JSON: %s — raw: %r", exc, raw[:300])
        return []


def _get_job_urls() -> list[str]:
    """
    Extract LinkedIn job view URLs from the DOM via the browser worker thread.
    LinkedIn renders <a href="/jobs/view/JOBID/"> for every visible card.
    Returns a deduplicated list of full URLs.
    """
    def _do(page):
        return page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href*="/jobs/view/"]'))
                .map(a => {
                    try {
                        const url = new URL(a.href);
                        return url.origin + url.pathname;
                    } catch(e) {
                        return a.href.split('?')[0];
                    }
                })
                .filter((v, i, a) => v && a.indexOf(v) === i)
        """)
    try:
        links = agent_browser.run(_do)
        logger.info("DOM extracted %d LinkedIn job URLs", len(links))
        return links
    except Exception as exc:
        logger.warning("_get_job_urls DOM query failed: %s", exc)
        return []


def _extract_indeed_listings_from_screenshot(b64: str) -> list[dict]:
    """
    Ask vision to extract job listings from an Indeed search screenshot.
    Returns up to 5 dicts: title, company, location, posted, url, platform.
    """
    prompt = (
        "This is an Indeed job search results page. "
        "Extract up to 5 job listings visible in the results list. "
        "Return a valid JSON array only — no markdown, no explanation. "
        'Each object must have exactly these keys: "title" (string), '
        '"company" (string), "location" (string), '
        '"posted" (string, e.g. "2 days ago" or "today"), '
        '"url" (empty string — URLs will be filled separately), '
        '"platform" (always "Indeed"). '
        'Use empty string for any field that is not visible. '
        'Return [] if no job listings are visible.'
    )
    raw = _vision_ask(b64, prompt)
    logger.info("Vision raw for Indeed Jobs: %r", raw[:400])

    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        listings = json.loads(raw)
        if not isinstance(listings, list):
            logger.error("Vision returned non-list for Indeed: %r", raw[:200])
            return []
        results = []
        for item in listings:
            if not isinstance(item, dict):
                continue
            results.append({
                "title":    str(item.get("title") or "").strip(),
                "company":  str(item.get("company") or "").strip(),
                "location": str(item.get("location") or "").strip(),
                "posted":   str(item.get("posted") or "").strip(),
                "url":      "",
                "platform": "Indeed",
            })
        return results
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse Indeed listings JSON: %s — raw: %r", exc, raw[:300])
        return []


def _search_indeed(role: str, location: str) -> list[dict]:
    """Search Indeed for jobs and extract listings via DOM, with vision fallback."""
    import urllib.parse
    url = f"https://www.indeed.com/jobs?q={urllib.parse.quote(role)}&l={urllib.parse.quote(location)}"
    logger.info("Indeed Jobs URL: %s", url)
    agent_browser.navigate(url, settle_secs=3.0)

    # DOM extraction via page.evaluate()
    def _extract(page):
        return page.evaluate("""
            (() => {
                const cards = document.querySelectorAll('[data-jk], .job_seen_beacon, .resultContent');
                const results = [];
                for (const card of cards) {
                    const titleEl = card.querySelector('h2 a, .jobTitle a, [data-jk] a');
                    const companyEl = card.querySelector('[data-testid="company-name"], .companyName, .company');
                    const locationEl = card.querySelector('[data-testid="text-location"], .companyLocation, .location');
                    const dateEl = card.querySelector('.date, .result-footer .date');
                    if (!titleEl) continue;
                    const title = titleEl.innerText.trim();
                    const href = titleEl.href || '';
                    results.push({
                        title: title,
                        company: companyEl ? companyEl.innerText.trim() : '',
                        location: locationEl ? locationEl.innerText.trim() : '',
                        posted: dateEl ? dateEl.innerText.trim() : '',
                        url: href.startsWith('http') ? href : 'https://www.indeed.com' + href,
                        platform: 'Indeed'
                    });
                    if (results.length >= 5) break;
                }
                return results;
            })()
        """)

    try:
        listings = agent_browser.run(_extract)
        if listings:
            logger.info("Indeed DOM extracted %d listings", len(listings))
            return listings[:5]
    except Exception as exc:
        logger.warning("Indeed DOM extraction failed: %s", exc)

    # Fallback: vision-based extraction
    logger.info("Indeed DOM extraction empty — trying vision fallback")
    b64 = computer_use.take_screenshot()
    if b64 is None:
        logger.error("Screenshot failed during Indeed job search")
        return []

    listings = _extract_indeed_listings_from_screenshot(b64)
    logger.info("Indeed vision extracted %d listings", len(listings))
    return listings[:5]


def _wants_indeed(query: str) -> bool:
    """Return True if the user explicitly asked for Indeed results."""
    q = query.lower()
    return "indeed" in q or "on indeed" in q


def _search_linkedin(query: str, role: str, location: str) -> list[dict]:
    """
    Search LinkedIn Jobs: navigate, screenshot, vision extract, DOM URLs.
    Returns up to 5 deduplicated listings.
    """
    url = _build_linkedin_jobs_url(role, location)
    filters = _parse_filters(query)
    if filters:
        url += "&" + urlencode(filters)
    logger.info("LinkedIn Jobs URL: %s", url)

    agent_browser.navigate(url, settle_secs=_SETTLE_JOBS)

    b64 = computer_use.take_screenshot()
    if b64 is None:
        logger.error("Screenshot failed during LinkedIn job search")
        return []

    listings = _extract_listings_from_screenshot(b64)
    logger.info("LinkedIn vision extracted %d listings", len(listings))

    job_urls = _get_job_urls()
    for i, listing in enumerate(listings):
        if i < len(job_urls):
            listing["url"] = job_urls[i]

    return listings


def _dedupe_listings(listings: list[dict]) -> list[dict]:
    """Deduplicate listings by (title, company) and assign index numbers."""
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in listings:
        key = (r["title"].lower().strip(), r["company"].lower().strip())
        if key not in seen and (r["title"] or r["company"]):
            seen.add(key)
            deduped.append(r)
    top5 = deduped[:5]
    for i, r in enumerate(top5):
        r["index"] = i + 1
    return top5


def search_jobs(query: str) -> list[dict]:
    """
    Full job search pipeline via visible browser.

    1. Parse role + location from voice query.
    2. If user explicitly asks for Indeed, search Indeed directly.
    3. Otherwise search LinkedIn first; if 0 results, fall back to Indeed.
    4. Deduplicate and return top 5.

    Browser stays open after this call — apply flow reuses it.
    """
    role, location = _parse_query(query)
    role = _strip_filter_words(role)
    logger.info("Job search: role=%r location=%r", role, location)

    use_indeed_first = _wants_indeed(query)

    if use_indeed_first:
        # User explicitly requested Indeed
        logger.info("User requested Indeed — searching Indeed directly")
        listings = _search_indeed(role, location)
        if not listings:
            logger.info("Indeed returned 0 — falling back to LinkedIn")
            listings = _search_linkedin(query, role, location)
    else:
        # Default: LinkedIn first, Indeed as fallback
        listings = _search_linkedin(query, role, location)
        if not listings:
            logger.info("LinkedIn returned 0 — falling back to Indeed")
            listings = _search_indeed(role, location)

    results = _dedupe_listings(listings)
    url_count = sum(1 for r in results if r.get("url"))
    logger.info("Returning %d jobs (%d with URLs)", len(results), url_count)
    return results


def format_spoken_results(results: list[dict]) -> str:
    """Format job results as a natural spoken string for TTS."""
    if not results:
        return "I couldn't find any job listings. Try rephrasing your search."
    ordinals = ["First", "Second", "Third", "Fourth", "Fifth"]
    parts = []
    for r in results:
        idx = r.get("index", 1)
        ordinal = ordinals[idx - 1] if 1 <= idx <= 5 else f"Number {idx}"
        loc_part = f", {r['location']}" if r.get("location") else ""
        posted_part = f", posted {r['posted']}" if r.get("posted") else ""
        parts.append(f"{ordinal} — {r['title']} at {r['company']}{loc_part}{posted_part}.")
    count = len(results)
    return f"Found {count} job{'s' if count != 1 else ''}. " + " ".join(parts)
