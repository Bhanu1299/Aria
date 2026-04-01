"""
router.py — Aria intent router (Phase 2B-2)

Single Groq call classifies the voice command and extracts clean search terms.
Python then constructs all URLs — the LLM never invents URLs.

Return dict schema:
  {
    "type":         str,   # "knowledge" | "web_search" | "web_direct" | "app" | "media" | "navigate"
    "query":        str,   # clean query, location-injected for web_search if needed
    "url":          str,   # Google URL (web_search) | site URL (web_direct/navigate/media) | "" otherwise
    "instructions": str,   # what to extract from the page for the summarizer
    "app_name":     str,   # only for app intent: macOS application name
    "contact":      str,   # only for app intent with contact disambiguation
    "site_name":    str,   # only for navigate intent: display name of site
  }
"""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote_plus

from groq import Groq

import config

logger = logging.getLogger(__name__)

_CLIENT: Groq | None = None

# ---------------------------------------------------------------------------
# Site URL templates for web_direct queries
# ---------------------------------------------------------------------------

_SITE_TEMPLATES: dict[str, str] = {
    "youtube":     "https://www.youtube.com/results?search_query={}",
    "reddit":      "https://www.reddit.com/search/?q={}",
    "linkedin":    "https://www.linkedin.com/search/results/all/?keywords={}",
    "github":      "https://github.com/search?q={}",
    "hackernews":  "https://hn.algolia.com/?q={}",
    "hn":          "https://hn.algolia.com/?q={}",
    "spotify":     "https://open.spotify.com/search/{}",
}

# ---------------------------------------------------------------------------
# Known sites for navigate intent (open homepage, no search query)
# ---------------------------------------------------------------------------

_KNOWN_SITES: dict[str, str] = {
    "claude":    "https://claude.ai",
    "claud":     "https://claude.ai",   # Whisper mishearing
    "youtube":   "https://youtube.com",
    "reddit":    "https://reddit.com",
    "github":    "https://github.com",
    "google":    "https://google.com",
    "linkedin":  "https://linkedin.com",
    "twitter":   "https://x.com",
    "x":         "https://x.com",
    "gmail":     "https://mail.google.com",
    "notion":    "https://notion.so",
    "spotify":   "https://open.spotify.com",
}

# ---------------------------------------------------------------------------
# Contact disambiguation — pre-check before Groq classifier
# ---------------------------------------------------------------------------

_CONTACT_RE = re.compile(
    r"\b(call|text|message|facetime)\s+(\w+(?:\s+\w+)?)\s*[.,!?]*\s*$",
    re.IGNORECASE,
)

# "open WhatsApp and message Amma" — explicit app + contact verb
_APP_CONTACT_RE = re.compile(
    r"\bopen\s+(\w+(?:\s+\w+)?)\s+and\s+(?:call|text|message|facetime)\s+(\w+(?:\s+\w+)?)\s*[.,!?]*\s*$",
    re.IGNORECASE,
)

# Canonical capitalization for apps that .title() gets wrong
_APP_NAME_MAP: dict[str, str] = {
    "whatsapp": "WhatsApp",
    "facetime": "FaceTime",
    "imessage": "iMessage",
    "linkedin": "LinkedIn",
    "github": "GitHub",
    "youtube": "YouTube",
}

# Words that signal the query needs the user's current location appended
_LOCATION_KEYWORDS = {
    "near", "nearby", "local", "here", "showtimes", "showtime",
    "restaurants", "restaurant", "events", "event", "weather",
    "tonight", "today", "now showing", "playing now", "in my area",
    "close to me", "around me", "near me",
}

# ---------------------------------------------------------------------------
# Classifier prompt
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """You are an intent classifier for a voice assistant. Classify the voice command and extract search terms.

Return ONLY a JSON object — no markdown, no explanation.

Schema:
{
  "type": "knowledge" | "web_search" | "web_direct" | "app" | "media" | "navigate" | "app_control" | "briefing" | "jobs",
  "query": "<clean search terms — remove filler words like 'find me', 'search for', 'look up'>",
  "site": "<only for web_direct: youtube | reddit | linkedin | github | hackernews | spotify | other>",
  "site_name": "<only for navigate: short lowercase site name, e.g. 'youtube', 'github', 'claude', 'reddit'>",
  "app_name": "<only for app: application name as it appears in macOS Applications folder, e.g. 'Safari', 'Spotify', 'Calendar'>",
  "location_sensitive": true | false
}

Type rules:
- knowledge: answerable from training data — facts, definitions, explanations, coding, math, history, science, general knowledge
- web_search: needs live or current data — news, weather, stock prices, sports scores, showtimes, product prices, anything with "latest", "current", "today", "now" (NOT job listings — use "jobs" intent instead)
- web_direct: user explicitly names a website or service AND wants to search/find content on it (YouTube, Reddit, LinkedIn, GitHub, Hacker News, Twitter, etc.)
- navigate: user wants to OPEN a website in the browser — "open [site]", "go to [site]", "take me to [site]", "navigate to [site]"
- app: controlling a Mac application — open, close, switch apps, adjust system settings, volume, brightness
- media: music playback and YouTube — play/pause/skip/stop/what's playing for any music app, YouTube audio streaming, YouTube video
- app_control: native Mac app control via AppleScript — Spotify playback (play/pause/skip/what's playing), system volume (set/mute/up/down), Finder (open folder), Calendar (read/add events), Mail (unread count/read latest), Reminders (add reminder), app open/quit/hide, screen reading
- briefing: user wants a morning briefing, daily summary, or asks what's on their day. Trigger phrases: "give me my briefing", "morning briefing", "what's my day look like", "what do I have today", "daily summary", "what's going on today"
- jobs: user wants to search for job listings — "find me jobs", "search for jobs", "any openings at", "job search", "find [role] jobs", "look for [role] positions", "any [role] roles", "are there any [role] openings"

IMPORTANT: Use "media" for ALL music and YouTube commands: "play", "pause", "skip", "next song", "next track", "what's playing", "now playing", "watch", "stop music", "stop playback", "play X on YouTube", "play X on [music app]". Do NOT use "web_direct" or "web_search" for these.
IMPORTANT: If the user says "search YouTube for X" or "find X on Reddit" → type is "web_direct", never "media".
IMPORTANT: If the user says "open Safari" or "open Spotify" (no URL/site intent) → type is "app", never "navigate".
IMPORTANT: If the user says "open YouTube" or "go to Reddit" or "take me to GitHub" → type is "navigate", not "app".
IMPORTANT: Use "jobs" (NOT "web_search") for any job search query — "find me jobs", "search for [role] jobs", "any [role] positions", "any [role] openings at [company]".
IMPORTANT: Use "app_control" (not "app") for: Spotify play/pause/skip/what's playing, volume set/mute/up/down, brightness set/up/down, dark mode on/off/toggle, focus mode on/off (do not disturb, work, sleep, personal, etc.), wifi on/off, bluetooth on/off, low power mode on/off, screenshot/capture screen, calendar events, unread emails, read emails, check my inbox, latest emails, reminders, open/show Downloads/Documents/Desktop in Finder, quit or hide any app, "what does X say", "read what's on screen", "copy what's on screen", "copy the text on screen", "copy all visible text".

location_sensitive = true when query implies local context: near, nearby, local, here, showtimes, restaurants, weather, events, tonight, today, now showing, in my area, jobs near

If ambiguous between knowledge and web_search → choose knowledge.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def _is_location_sensitive(query: str) -> bool:
    """Keyword fallback — catches cases the LLM might miss."""
    lower = query.lower()
    return any(kw in lower for kw in _LOCATION_KEYWORDS)


def _inject_location(query: str) -> str:
    """Append CURRENT_LOCATION to query if it isn't already there."""
    loc = config.CURRENT_LOCATION
    if loc and loc != "Unknown Location" and loc.lower() not in query.lower():
        return f"{query} {loc}"
    return query


def _build_google_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}"


def _build_site_url(site: str, query: str) -> str:
    template = _SITE_TEMPLATES.get(site.lower())
    if template:
        return template.format(quote_plus(query))
    # Unknown named site → Google it
    return _build_google_url(f"{query} {site}")


def _instructions_for(query: str, intent_type: str, site: str = "") -> str:
    if intent_type == "web_direct" and site == "youtube":
        return f"List only the top 3 video titles (no channel names, no extras) for: {query}"
    if intent_type == "web_direct" and site == "linkedin":
        return f"Extract job titles, company names, and locations for: {query}"
    if intent_type == "web_direct" and site in ("github", "hackernews"):
        return f"Extract the top result titles and brief descriptions for: {query}"
    return f"Extract the most relevant information to answer: {query}"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _check_contact_intent(command: str) -> dict | None:
    """
    Pre-check for contact commands before the Groq classifier.

    Matches: "call Ama", "text Sarah Jane", "facetime Dr Yubi", etc.
    Also matches: "open WhatsApp and message Amma" → uses the named app.
    Short names (1–2 words) after a contact verb are treated as contact names,
    not brand names, preventing misroutes like 'Call Ama' → Amazon Music.

    Returns a routing dict or None if this is not a contact command.
    """
    stripped = command.strip()

    # Check "open [app] and [verb] [contact]" first — explicit app takes priority
    m_app = _APP_CONTACT_RE.search(stripped)
    if m_app:
        raw_app = m_app.group(1).strip()
        app_name = _APP_NAME_MAP.get(raw_app.lower(), raw_app.title())
        contact = m_app.group(2).strip()
        logger.debug("App+contact pre-check matched: app=%r contact=%r", app_name, contact)
        return {
            "type": "app",
            "query": stripped,
            "url": "",
            "instructions": "",
            "app_name": app_name,
            "contact": contact,
            "site_name": "",
        }

    m = _CONTACT_RE.search(stripped)
    if not m:
        return None

    verb = m.group(1).lower()
    contact = m.group(2).strip()

    # Reject obvious brand names that look like contact commands
    _BRANDS = {"amazon", "apple", "google", "spotify", "netflix", "youtube",
               "uber", "lyft", "instagram", "facebook", "twitter", "reddit"}
    if contact.lower() in _BRANDS:
        return None

    app_name = "FaceTime" if verb in ("call", "facetime") else "Messages"
    logger.debug("Contact pre-check matched: verb=%r contact=%r app=%r", verb, contact, app_name)
    return {
        "type": "app",
        "query": stripped,
        "url": "",
        "instructions": "",
        "app_name": app_name,
        "contact": contact,
        "site_name": "",
    }


def route(command: str) -> dict:
    """
    Classify a voice command and return a routing dict.

    Runs a contact pre-check before the Groq classifier to prevent
    short names from being misrouted as brand/site names.
    Never raises — falls back to a knowledge query on any error.
    """
    # Contact disambiguation pre-check — runs before the LLM
    contact_intent = _check_contact_intent(command)
    if contact_intent is not None:
        return contact_intent

    try:
        parsed = _classify(command)
        return _build_intent(command, parsed)
    except Exception as exc:
        logger.error("Router failed (%s) — falling back to knowledge query.", exc)
        return {
            "type": "knowledge",
            "query": command.strip(),
            "url": "",
            "instructions": "",
            "app_name": "",
            "contact": "",
            "site_name": "",
        }


def _classify(command: str) -> dict:
    """One Groq call → classification JSON."""
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user", "content": command.strip()},
        ],
        temperature=0.1,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    logger.debug("Classifier raw: %s", raw)

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw.strip())

    parsed = json.loads(raw)

    if "type" not in parsed or parsed["type"] not in ("knowledge", "web_search", "web_direct", "app", "media", "navigate", "app_control", "briefing", "jobs"):
        raise ValueError(f"Invalid type in classifier response: {parsed.get('type')!r}")

    # Normalise query — LLM sometimes returns null or empty
    raw_query = parsed.get("query")
    if not raw_query or not str(raw_query).strip():
        parsed["query"] = command.strip()
    else:
        parsed["query"] = str(raw_query).strip()

    return parsed


def _build_intent(original_command: str, parsed: dict) -> dict:
    """Construct the routing dict from classifier output. All URL logic is here."""
    intent_type = parsed["type"]
    # Safe: _classify guarantees query is a non-empty string
    query = (parsed.get("query") or original_command).strip()
    # parsed.get("site", "") returns None if key exists with null value — use `or`
    site = (parsed.get("site") or "").strip().lower()
    location_sensitive = parsed.get("location_sensitive", False) or _is_location_sensitive(query)

    # Base dict — all keys always present so callers never need .get() with defaults
    _base: dict = {
        "type": intent_type,
        "query": query,
        "url": "",
        "instructions": "",
        "app_name": "",
        "contact": "",
        "site_name": "",
    }

    if intent_type == "knowledge":
        return {**_base}

    if intent_type == "web_search":
        if location_sensitive:
            query = _inject_location(query)
        return {
            **_base,
            "query": query,
            "url": _build_google_url(query),
            "instructions": _instructions_for(query, intent_type),
        }

    if intent_type == "web_direct":
        url = _build_site_url(site, query)
        return {
            **_base,
            "url": url,
            "instructions": _instructions_for(query, intent_type, site),
        }

    if intent_type == "navigate":
        raw_site = (parsed.get("site_name") or site or query).strip().lower()
        # Strip navigation filler words the LLM may leave in site_name
        for prefix in ("open ", "go to ", "take me to ", "navigate to ", "launch "):
            if raw_site.startswith(prefix):
                raw_site = raw_site[len(prefix):].strip()
        url = _KNOWN_SITES.get(raw_site, f"https://{raw_site}.com")
        display_name = raw_site.capitalize()
        return {
            **_base,
            "url": url,
            "site_name": display_name,
        }

    if intent_type == "app":
        app_name = (parsed.get("app_name") or query).strip()
        return {
            **_base,
            "app_name": app_name,
        }

    if intent_type == "media":
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        return {
            **_base,
            "url": url,
            "instructions": f"List only the top 3 video titles for: {query}",
        }

    if intent_type == "app_control":
        return {**_base}

    if intent_type == "briefing":
        return {**_base}

    if intent_type == "jobs":
        return {**_base}

    # Unknown type — fallback to knowledge
    logger.warning("Unknown intent type %r — falling back to knowledge.", intent_type)
    return {**_base, "type": "knowledge"}
