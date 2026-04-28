"""
briefing.py — Aria Phase 2E: On-demand morning briefing

Four data fetchers (weather, calendar, gmail, news) run concurrently via
ThreadPoolExecutor, then a Groq call assembles them into a natural spoken briefing.

Public API:
  build_briefing() -> str   — returns a spoken briefing string for speaker.py
"""

from __future__ import annotations

import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from groq import Groq

import config

logger = logging.getLogger(__name__)

_CLIENT: Groq | None = None


def _time_of_day() -> str:
    h = datetime.now().hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


# ---------------------------------------------------------------------------
# 1. Weather
# ---------------------------------------------------------------------------

def get_weather() -> str:
    """Fetch current weather from wttr.in using auto-detected city from config."""
    try:
        city = config.CURRENT_LOCATION.split(",")[0].strip()
        if not city or city == "Unknown Location":
            return "Weather unavailable"
        resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()
        if text:
            return text
        return "Weather unavailable"
    except Exception as exc:
        logger.error("[briefing] Weather fetch failed: %s", exc)
        return "Weather unavailable"


# ---------------------------------------------------------------------------
# 2. Apple Calendar — today's events via AppleScript
# ---------------------------------------------------------------------------

def get_calendar_events() -> str:
    """Read today's calendar events via AppleScript."""
    script = '''set todayStart to current date
set (time of todayStart) to 0
set todayEnd to todayStart + 1 * days - 1
set output to ""
tell application "Calendar"
    repeat with c in calendars
        try
            repeat with e in (events of c whose start date >= todayStart and start date <= todayEnd)
                set output to output & (summary of e) & " at " & (time string of (start date of e)) & ", "
            end repeat
        end try
    end repeat
end tell
return output'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            logger.debug("[briefing] Calendar AppleScript error: %s", err)
            return "Calendar unavailable"

        raw = result.stdout.strip().rstrip(",").strip()
        logger.info("[briefing] Calendar raw: %r", raw)
        if not raw:
            return "No events today"

        # Parse "Meeting at 10:00:00 AM, Lunch at 12:00:00 PM" into readable list
        events = [e.strip() for e in raw.split(",") if e.strip()]
        if not events:
            return "No events today"
        return ". ".join(events) + "."

    except subprocess.TimeoutExpired:
        logger.error("[briefing] Calendar AppleScript timed out")
        return "Calendar unavailable"
    except Exception as exc:
        logger.error("[briefing] Calendar fetch failed: %s", exc)
        return "Calendar unavailable"


# ---------------------------------------------------------------------------
# 3. Gmail unread
# ---------------------------------------------------------------------------

def get_gmail_unread() -> str:
    """
    Fetch unread email count and subjects by loading Gmail in a headless browser.

    Uses the persistent Playwright profile (cookie-based auth). Navigates to
    the Gmail inbox, checks for login redirect, then scrapes unread rows via
    Gmail's stable 'tr.zA' selector (unread thread rows).
    """
    import time as _time
    from browser_profile import get_persistent_context, close_persistent_context

    context = None
    try:
        context = get_persistent_context(headless=True)
        page = context.new_page()

        page.goto(
            "https://mail.google.com/mail/u/0/#inbox",
            wait_until="domcontentloaded",
            timeout=config.BROWSER_TIMEOUT * 1_000,
        )

        # Allow JS to render the inbox
        _time.sleep(3.0)

        # --- Auth check ---
        current_url = page.url
        title = page.title()
        logger.info("[briefing] Gmail page title: %r  url: %r", title, current_url)

        if "accounts.google.com" in current_url or "signin" in current_url.lower():
            return "Gmail not logged in — run: python main.py --login gmail"
        if "sign in" in title.lower() or "sign-in" in title.lower():
            return "Gmail not logged in — run: python main.py --login gmail"

        # --- Unread count from page title ---
        # Gmail sets title to "(N) Inbox - user@gmail.com - Gmail" when N > 0
        count_match = re.match(r'^\((\d+)\)', title.strip())
        unread_count = int(count_match.group(1)) if count_match else 0

        # --- Subject lines from unread rows ---
        # tr.zA = Gmail's stable CSS class for unread thread rows
        subjects: list[str] = []
        try:
            rows = page.query_selector_all("tr.zA")
            logger.info("[briefing] Gmail unread rows found: %d", len(rows))
            for row in rows[:3]:
                # Subject span: .bog (bold sender), .bqe/.b8 (subject snippet)
                subj_el = (
                    row.query_selector("span.bog")
                    or row.query_selector("span.bqe")
                    or row.query_selector("span[data-thread-id]")
                )
                sender_el = row.query_selector("span.zF")
                subj = subj_el.inner_text().strip() if subj_el else ""
                sender = sender_el.inner_text().strip() if sender_el else ""
                if subj:
                    subjects.append(f"{subj} from {sender}" if sender else subj)
        except Exception as exc:
            logger.warning("[briefing] Gmail row scrape failed: %s", exc)

        # If title had a count but scrape got nothing, still report the count
        if unread_count == 0 and not subjects:
            return "No unread emails"

        display_count = unread_count if unread_count > 0 else len(subjects)
        header = f"{display_count} unread email{'s' if display_count != 1 else ''}."
        if subjects:
            return header + " " + ". ".join(subjects) + "."
        return header

    except Exception as exc:
        logger.error("[briefing] Gmail fetch failed: %s", exc)
        return "Gmail unavailable"
    finally:
        if context is not None:
            close_persistent_context(context)


# ---------------------------------------------------------------------------
# 4. News headlines
# ---------------------------------------------------------------------------

_RSS_FEEDS = [
    ("BBC",      "https://feeds.bbci.co.uk/news/rss.xml"),
    ("NPR",      "https://feeds.npr.org/1001/rss.xml"),
    ("Guardian", "https://www.theguardian.com/world/rss"),
]


def get_news() -> str:
    """Fetch top headlines from BBC, NPR, and Guardian RSS — up to 2 per source."""
    seen: set[str] = set()
    parts: list[str] = []

    for source, url in _RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = root.findall(".//item/title")
            count = 0
            for item in items:
                if count >= 2:
                    break
                text = (item.text or "").strip()
                if not text:
                    continue
                # Deduplicate by normalised lowercase key
                key = re.sub(r'[^a-z0-9 ]', '', text.lower())
                if key in seen:
                    continue
                seen.add(key)
                parts.append(f"{source}: {text}")
                count += 1
        except Exception as exc:
            logger.debug("[briefing] %s RSS failed: %s", source, exc)

    if not parts:
        return "News unavailable"

    return ". ".join(parts[:5]) + "."


# ---------------------------------------------------------------------------
# 5. Briefing assembler
# ---------------------------------------------------------------------------

_BRIEFING_PROMPT = """You are Aria — Bhanu's personal AI, built to be fast, sharp, and occasionally hilarious. You have the competence of Jarvis and the personality of a witty best friend with dark humor.

Deliver a natural, spoken {time_of_day} briefing using the data below. Be concise — 60 seconds spoken max. No markdown, no bullet points, no headers — plain prose only. Start with weather, then calendar, then email, then news. Don't say "bullet point" or "here are" — just deliver it like Jarvis would. Light commentary is fine. Never invent data.
IMPORTANT: If any data source says "unavailable" or "not logged in", say so — but make it quick and move on.

Weather: {weather}
Calendar: {calendar}
Gmail: {gmail}
News: {news}"""


def build_briefing() -> str:
    """
    Run all four fetchers concurrently, then assemble into a spoken briefing via Groq.

    Returns a plain-prose spoken briefing string. Never raises.
    """
    results = {
        "weather": "Weather unavailable",
        "calendar": "Calendar unavailable",
        "gmail": "Gmail unavailable",
        "news": "News unavailable",
    }

    fetchers = {
        "weather": get_weather,
        "calendar": get_calendar_events,
        "gmail": get_gmail_unread,
        "news": get_news,
    }

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn): name for name, fn in fetchers.items()}
            for future in as_completed(futures, timeout=45):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.error("[briefing] %s fetcher failed: %s", name, exc)
    except Exception as exc:
        logger.error("[briefing] ThreadPoolExecutor error: %s", exc)

    # Assemble via Groq
    try:
        client = _get_client()
        prompt = _BRIEFING_PROMPT.format(**results, time_of_day=_time_of_day())
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Give me my briefing."},
            ],
            temperature=0.5,
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
        logger.debug("[briefing] Assembled briefing: %s", answer[:100])
        return answer

    except Exception as exc:
        logger.error("[briefing] Groq assembly failed: %s", exc)
        # Raw fallback — speak the data directly without Groq
        parts = []
        if results["weather"] != "Weather unavailable":
            parts.append(results["weather"])
        if results["calendar"] != "Calendar unavailable":
            parts.append(results["calendar"])
        if results["gmail"] != "Gmail unavailable":
            parts.append(results["gmail"])
        if results["news"] != "News unavailable":
            parts.append(results["news"])
        if parts:
            return " ".join(parts)
        return "I couldn't gather your briefing data right now. Please try again."
