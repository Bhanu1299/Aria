"""
jobs.py — Phase 3A: Voice-driven job search for Aria

Pipeline per platform (LinkedIn, Indeed):
  1. Groq parses role + location from voice query
  2. Playwright headless: Google site: search → screenshot → vision extracts job search URL
  3. Playwright headless: navigate to job search URL → screenshot → vision returns JSON listings
  4. Deduplicate by (title, company), assign index 1-5, return top 5
"""

from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote_plus

from groq import Groq

import config
from browser_profile import get_persistent_context, close_persistent_context

logger = logging.getLogger(__name__)

_CLIENT: Groq | None = None
_SCREENSHOT_PATH = "/tmp/aria_jobs.jpg"
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_SETTLE_GOOGLE = 3.0   # seconds to wait after Google loads
_SETTLE_JOBS = 5.0     # seconds to wait after job page loads (SPA-heavy)


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def _parse_query(query: str) -> tuple[str, str]:
    """
    Extract job role and location from a voice query via Groq.

    Returns:
        (role, location) — location is empty string if not specified.
        Falls back to (query, "") on any error.
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


def _screenshot_page(url: str, settle_secs: float) -> str | None:
    """
    Navigate to url headlessly via Playwright persistent context, take a
    full-viewport screenshot, resize to 1920px max dimension with sips,
    and return the image as a base64 string.

    Uses a separate Playwright context per call (open → screenshot → close).
    Returns None on any error.
    """
    context = None
    try:
        context = get_persistent_context(headless=True)
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=config.BROWSER_TIMEOUT * 1_000)
        time.sleep(settle_secs)
        img_bytes = page.screenshot(full_page=False)
        Path(_SCREENSHOT_PATH).write_bytes(img_bytes)
        subprocess.run(["sips", "-Z", "1920", _SCREENSHOT_PATH], capture_output=True)
        resized = Path(_SCREENSHOT_PATH).read_bytes()
        logger.info("Screenshot of %s: %d bytes", url, len(resized))
        return base64.b64encode(resized).decode("utf-8")
    except Exception as exc:
        logger.error("_screenshot_page failed for %s: %s", url, exc)
        return None
    finally:
        if context is not None:
            close_persistent_context(context)


def _vision_ask(b64: str, prompt: str) -> str:
    """Send a screenshot (base64) + text prompt to Groq Llama-4-Scout. Returns response text, or "" on error."""
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


def _get_search_url(platform: str, role: str, location: str) -> str | None:
    """
    Find the job search URL for a platform by screenshotting a Google
    site: search result and asking the vision model to extract the URL.

    Returns the URL string or None if not found / on error.
    """
    domain_map = {"LinkedIn": "linkedin.com/jobs", "Indeed": "indeed.com/jobs"}
    domain = domain_map.get(platform, platform.lower())
    search_terms = f"site:{domain} {role} {location}".strip()
    google_url = f"https://www.google.com/search?q={quote_plus(search_terms)}"
    logger.info("Discovering %s URL via Google: %s", platform, google_url)

    b64 = _screenshot_page(google_url, _SETTLE_GOOGLE)
    if b64 is None:
        return None

    top_domain = domain.split("/")[0]
    prompt = (
        f"This is a Google search results page. "
        f"Find the first search result URL that starts with https://{top_domain}. "
        "Return only the full URL starting with https://, nothing else. "
        "If no such URL is visible, return an empty string."
    )
    raw = _vision_ask(b64, prompt).strip()

    if raw.startswith("https://") and top_domain in raw.lower():
        logger.info("%s search URL: %s", platform, raw)
        return raw

    logger.warning("Vision did not return a valid %s URL: %r", platform, raw)
    return None


def _extract_jobs_from_page(url: str, platform: str) -> list[dict]:
    """
    Navigate to a job search results page, screenshot it, and ask the
    vision model to return job listings as a JSON array.

    Returns up to 5 dicts with keys: title, company, location, posted, url, platform.
    Returns [] on any error or if vision returns no valid JSON.
    """
    logger.info("Extracting jobs from %s (%s)", url, platform)
    b64 = _screenshot_page(url, _SETTLE_JOBS)
    if b64 is None:
        return []

    prompt = (
        f"This is a {platform} job search results page. "
        "Extract the top 5 job listings that are visible. "
        "Return a valid JSON array only — no markdown, no explanation. "
        'Each object must have exactly these keys: "title" (string), '
        '"company" (string), "location" (string), '
        '"posted" (string, e.g. "2 days ago" or "today"), '
        '"url" (string, full URL to the job listing or empty string). '
        'Use empty string for any field that is not visible. '
        'Return [] if no job listings are visible.'
    )
    raw = _vision_ask(b64, prompt)

    # Strip markdown fences if model wrapped the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        listings = json.loads(raw)
        if not isinstance(listings, list):
            logger.error("Vision returned non-list for %s: %r", platform, raw[:200])
            return []
        results = []
        for item in listings:
            if not isinstance(item, dict):
                continue
            results.append({
                "title": str(item.get("title") or "").strip(),
                "company": str(item.get("company") or "").strip(),
                "location": str(item.get("location") or "").strip(),
                "posted": str(item.get("posted") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "platform": platform,
            })
        return results
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse job listings JSON from %s: %s — raw: %r",
                     platform, exc, raw[:300])
        return []


def search_jobs(query: str) -> list[dict]:
    """
    Full job search pipeline.

    1. Parse role + location from voice query.
    2. Discover LinkedIn and Indeed search URLs via Google + vision.
    3. Extract job listings from each URL via screenshot + vision.
    4. Deduplicate by (title, company), assign index 1–5.
    5. Return top 5 results.
    """
    role, location = _parse_query(query)
    logger.info("Job search: role=%r location=%r", role, location)

    results: list[dict] = []
    for platform in ("LinkedIn", "Indeed"):
        url = _get_search_url(platform, role, location)
        if url:
            results += _extract_jobs_from_page(url, platform)

    # Deduplicate: first occurrence of (title, company) wins
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in results:
        key = (r["title"].lower().strip(), r["company"].lower().strip())
        if key not in seen and (r["title"] or r["company"]):
            seen.add(key)
            deduped.append(r)

    top5 = deduped[:5]
    for i, r in enumerate(top5):
        r["index"] = i + 1
    return top5


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
