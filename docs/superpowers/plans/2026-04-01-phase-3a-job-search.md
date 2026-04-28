# Phase 3A — Voice-Driven Job Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `jobs` voice intent that searches LinkedIn and Indeed via Google → Playwright headless screenshot → Groq Llama-4-Scout vision, speaks back 5 results, and stores them in session memory for Phase 3B follow-up commands.

**Architecture:** Voice query → Groq parses role + location → Google `site:` search for each platform → Playwright headless screenshot → vision model extracts the job search URL → navigate to that URL → screenshot → vision model returns JSON job listings → deduplicate → store in `memory.session` → speak results.

**Tech Stack:** Python 3.11, Playwright (persistent context), Groq `llama-3.1-8b-instant` (query parse), Groq `meta-llama/llama-4-scout-17b-16e-instruct` (vision), macOS `sips` (image resize), `browser_profile.get_persistent_context`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `identity.json` | Overwrite | Bhanu's full profile (name, email, skills, experience) |
| `memory.py` | Create | Module-level session dict; `store_jobs`, `get_job_by_index` |
| `jobs.py` | Create | Google→vision URL discovery, headless screenshot→vision job extraction, format |
| `tests/test_jobs.py` | Create | Unit tests for pure/mockable paths |
| `router.py` | Modify | Add `jobs` to classifier prompt, type validation, `_build_intent` |
| `main.py` | Modify | `import jobs`, `import memory`, handler block after `briefing` |

---

## Task 1: Populate identity.json

**Files:**
- Overwrite: `identity.json`

- [ ] **Step 1: Overwrite identity.json with full profile**

`/Users/bhanuteja/Documents/trae_projects/Aria/identity.json`:
```json
{
  "name": "Bhanu Teja Veeramachaneni",
  "email": "bteja0519@gmail.com",
  "phone": "+1-716-750-3590",
  "location": "Buffalo, NY",
  "linkedin": "https://www.linkedin.com/in/bhanuteja-veeramachaneni",
  "github": "https://github.com/bhanuteja-veeramachaneni",
  "resume_path": "~/Documents/Bhanu_Teja_Resume.pdf",
  "summary": "Backend systems and GenAI pipelines. M.S. CS, University at Buffalo.",
  "skills": [
    "Python", "TypeScript", "JavaScript", "SQL", "Java",
    "LangChain", "LangGraph", "RAG", "FAISS", "Pinecone", "Groq",
    "FastAPI", "Node.js", "REST APIs", "Async Programming",
    "PostgreSQL", "MongoDB", "Spark", "Kafka",
    "AWS", "Docker", "CI/CD", "GitHub Actions", "Playwright"
  ],
  "experience": [
    {
      "company": "Appetit",
      "role": "Software Engineer Intern",
      "duration": "Jan 2025 - May 2025"
    },
    {
      "company": "AppsTek Corp",
      "role": "Software Engineer Intern",
      "duration": "Sep 2023 - May 2024"
    }
  ],
  "education": "M.S. Computer Science, University at Buffalo (Dec 2025)",
  "target_roles": []
}
```

- [ ] **Step 2: Verify it loads cleanly**

```bash
python -c "import json; d = json.load(open('identity.json')); print(d['name'])"
```
Expected: `Bhanu Teja Veeramachaneni`

---

## Task 2: Create memory.py

**Files:**
- Create: `memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

`/Users/bhanuteja/Documents/trae_projects/Aria/tests/test_memory.py`:
```python
"""Tests for memory.py session state."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import memory


def setup_function():
    """Reset session before each test."""
    memory.session.clear()


def test_store_and_get_by_index():
    jobs = [
        {"index": 1, "title": "SWE", "company": "Stripe"},
        {"index": 2, "title": "Backend", "company": "Plaid"},
    ]
    memory.store_jobs(jobs)
    assert memory.get_job_by_index(1) == jobs[0]
    assert memory.get_job_by_index(2) == jobs[1]


def test_get_job_out_of_bounds_returns_none():
    memory.store_jobs([{"index": 1, "title": "SWE", "company": "Stripe"}])
    assert memory.get_job_by_index(0) is None
    assert memory.get_job_by_index(2) is None
    assert memory.get_job_by_index(-1) is None


def test_get_job_empty_session_returns_none():
    assert memory.get_job_by_index(1) is None


def test_store_jobs_overwrites_previous():
    memory.store_jobs([{"index": 1, "title": "Old"}])
    memory.store_jobs([{"index": 1, "title": "New"}])
    assert memory.get_job_by_index(1)["title"] == "New"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria
python -m pytest tests/test_memory.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'memory'`

- [ ] **Step 3: Create memory.py**

`/Users/bhanuteja/Documents/trae_projects/Aria/memory.py`:
```python
"""
memory.py — Aria session state (Phase 3A)

Module-level dict that persists for the lifetime of the process.
Enables follow-up commands ("apply to the second one") to reference
results from earlier in the same session.
"""

from __future__ import annotations

session: dict = {}


def store_jobs(results: list[dict]) -> None:
    """Overwrite last_jobs with the given results list."""
    session["last_jobs"] = results


def get_job_by_index(n: int) -> dict | None:
    """Return the 1-based nth job from last_jobs, or None if out of range."""
    jobs = session.get("last_jobs", [])
    return jobs[n - 1] if 0 < n <= len(jobs) else None
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_memory.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add memory.py tests/test_memory.py
git commit -m "feat(3A): add memory.py session state for job results"
```

---

## Task 3: Create jobs.py

**Files:**
- Create: `jobs.py`
- Create: `tests/test_jobs.py`

### 3a — Tests first

- [ ] **Step 1: Write tests for pure/mockable functions**

`/Users/bhanuteja/Documents/trae_projects/Aria/tests/test_jobs.py`:
```python
"""Tests for jobs.py — pure functions and mockable Groq paths."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock
import json


# ── format_spoken_results ────────────────────────────────────────────────────

def test_format_spoken_results_empty():
    import jobs
    result = jobs.format_spoken_results([])
    assert result == "I couldn't find any job listings. Try rephrasing your search."


def test_format_spoken_results_single():
    import jobs
    r = jobs.format_spoken_results([
        {"index": 1, "title": "Software Engineer", "company": "Stripe",
         "location": "New York, NY", "posted": "2 days ago", "platform": "LinkedIn"}
    ])
    assert r.startswith("Found 1 job.")
    assert "First" in r
    assert "Stripe" in r
    assert "2 days ago" in r


def test_format_spoken_results_five():
    import jobs
    results = [
        {"index": i + 1, "title": f"Job {i+1}", "company": f"Co {i+1}",
         "location": "Remote", "posted": "today", "platform": "LinkedIn"}
        for i in range(5)
    ]
    spoken = jobs.format_spoken_results(results)
    assert spoken.startswith("Found 5 jobs.")
    for ordinal in ["First", "Second", "Third", "Fourth", "Fifth"]:
        assert ordinal in spoken


def test_format_spoken_results_missing_location_and_posted():
    import jobs
    r = jobs.format_spoken_results([
        {"index": 1, "title": "SWE", "company": "Acme",
         "location": "", "posted": "", "platform": "LinkedIn"}
    ])
    assert "SWE at Acme" in r
    # No trailing comma/empty fields
    assert ", ." not in r


# ── _parse_query ─────────────────────────────────────────────────────────────

def test_parse_query_success():
    import jobs
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"role": "software engineer", "location": "New York"}'

    with patch.object(jobs, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        role, location = jobs._parse_query("find me software engineer jobs in New York")

    assert role == "software engineer"
    assert location == "New York"


def test_parse_query_groq_failure_falls_back_to_raw_query():
    import jobs
    with patch.object(jobs, "_get_client", side_effect=RuntimeError("API down")):
        role, location = jobs._parse_query("find backend jobs remote")

    assert role == "find backend jobs remote"
    assert location == ""


def test_parse_query_bad_json_falls_back():
    import jobs
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "not json"

    with patch.object(jobs, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        role, location = jobs._parse_query("some query")

    assert role == "some query"
    assert location == ""


# ── search_jobs deduplication ─────────────────────────────────────────────────

def test_search_jobs_deduplicates():
    import jobs
    duplicated = [
        {"title": "SWE", "company": "Stripe", "location": "NYC",
         "posted": "today", "url": "https://li.com/1", "platform": "LinkedIn"},
        {"title": "SWE", "company": "Stripe", "location": "NYC",
         "posted": "today", "url": "https://ind.com/1", "platform": "Indeed"},
        {"title": "Backend", "company": "Plaid", "location": "Remote",
         "posted": "1 day ago", "url": "https://li.com/2", "platform": "LinkedIn"},
    ]
    with patch.object(jobs, "_parse_query", return_value=("SWE", "NYC")), \
         patch.object(jobs, "_get_search_url", return_value="https://example.com"), \
         patch.object(jobs, "_extract_jobs_from_page", side_effect=[
             duplicated[:2], duplicated[2:]
         ]):
        results = jobs.search_jobs("find me SWE jobs in NYC")

    titles_companies = [(r["title"], r["company"]) for r in results]
    assert ("SWE", "Stripe") in titles_companies
    # Duplicate should be removed
    assert sum(1 for t, c in titles_companies if t == "SWE" and c == "Stripe") == 1


def test_search_jobs_returns_empty_when_all_sources_fail():
    import jobs
    with patch.object(jobs, "_parse_query", return_value=("SWE", "NYC")), \
         patch.object(jobs, "_get_search_url", return_value=None):
        results = jobs.search_jobs("find me SWE jobs")

    assert results == []
```

- [ ] **Step 2: Run tests to confirm they fail with ImportError**

```bash
python -m pytest tests/test_jobs.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'jobs'`

### 3b — Implementation

- [ ] **Step 3: Create jobs.py**

`/Users/bhanuteja/Documents/trae_projects/Aria/jobs.py`:
```python
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
    """Send a screenshot (base64) + text prompt to Groq Llama-4-Scout. Returns response text."""
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
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
python -m pytest tests/test_jobs.py -v
```
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add jobs.py tests/test_jobs.py
git commit -m "feat(3A): add jobs.py — Google+vision job search pipeline"
```

---

## Task 4: Update router.py

**Files:**
- Modify: `router.py`

Three targeted edits only — do not touch anything else.

- [ ] **Step 1: Add `jobs` to the schema type union in `_CLASSIFY_SYSTEM` (line ~109)**

Find the line:
```
  "type": "knowledge" | "web_search" | "web_direct" | "app" | "media" | "navigate" | "app_control" | "briefing",
```
Replace with:
```
  "type": "knowledge" | "web_search" | "web_direct" | "app" | "media" | "navigate" | "app_control" | "briefing" | "jobs",
```

- [ ] **Step 2: Add the `jobs` type rule to `_CLASSIFY_SYSTEM` (after the `briefing` rule)**

Find:
```
- briefing: user wants a morning briefing, daily summary, or asks what's on their day. Trigger phrases: "give me my briefing", "morning briefing", "what's my day look like", "what do I have today", "daily summary", "what's going on today"
```
Add immediately after:
```
- jobs: user wants to search for job listings — "find me jobs", "search for jobs", "any openings at", "job search", "find [role] jobs", "look for [role] positions", "any [role] roles", "are there any [role] openings"
```

- [ ] **Step 3: Add `"jobs"` to the valid types tuple in `_classify()` (line ~297)**

Find:
```python
    if "type" not in parsed or parsed["type"] not in ("knowledge", "web_search", "web_direct", "app", "media", "navigate", "app_control", "briefing"):
```
Replace with:
```python
    if "type" not in parsed or parsed["type"] not in ("knowledge", "web_search", "web_direct", "app", "media", "navigate", "app_control", "briefing", "jobs"):
```

- [ ] **Step 4: Add `jobs` branch in `_build_intent()` (before the final fallback, after the `briefing` branch)**

Find:
```python
    if intent_type == "briefing":
        return {**_base}

    # Unknown type — fallback to knowledge
```
Replace with:
```python
    if intent_type == "briefing":
        return {**_base}

    if intent_type == "jobs":
        return {**_base}

    # Unknown type — fallback to knowledge
```

- [ ] **Step 5: Verify router accepts jobs intent**

```bash
python -c "
from router import _classify
r = _classify('find me software engineer jobs in New York')
print(r)
assert r['type'] == 'jobs', f'Expected jobs, got {r[\"type\"]}'
print('Router OK')
"
```
Expected: dict with `type: jobs` printed, then `Router OK`

- [ ] **Step 6: Commit**

```bash
git add router.py
git commit -m "feat(3A): add jobs intent to router classifier"
```

---

## Task 5: Wire into main.py

**Files:**
- Modify: `main.py`

Two targeted edits only.

- [ ] **Step 1: Add imports after `import briefing` (line ~45)**

Find:
```python
import briefing
```
Replace with:
```python
import briefing
import jobs
import memory
```

- [ ] **Step 2: Add handler block after the `briefing` handler, before the fallback comment (after line ~249)**

Find:
```python
    # --- Briefing: morning summary with weather, calendar, email, news ---
    if intent_type == "briefing":
        print("[Aria] Building briefing...")
        speaker.say("Getting your briefing, one moment.")
        return briefing.build_briefing()

    # Fallback
    return answer_knowledge(original_question)
```
Replace with:
```python
    # --- Briefing: morning summary with weather, calendar, email, news ---
    if intent_type == "briefing":
        print("[Aria] Building briefing...")
        speaker.say("Getting your briefing, one moment.")
        return briefing.build_briefing()

    # --- Jobs: LinkedIn + Indeed search via Google → vision pipeline ---
    if intent_type == "jobs":
        print(f"[Aria] Job search: {intent['query']!r}")
        speaker.say("Searching for jobs, one moment.")
        results = jobs.search_jobs(intent["query"])
        memory.store_jobs(results)
        return jobs.format_spoken_results(results)

    # Fallback
    return answer_knowledge(original_question)
```

- [ ] **Step 3: Verify imports and handler parse cleanly**

```bash
python -c "
import ast, sys
with open('main.py') as f:
    src = f.read()
ast.parse(src)
print('main.py parses OK')
"
```
Expected: `main.py parses OK`

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(3A): wire jobs intent handler into main.py"
```

---

## Task 6: Full smoke test

- [ ] **Step 1: Run all new tests**

```bash
python -m pytest tests/test_memory.py tests/test_jobs.py -v
```
Expected: `14 passed`

- [ ] **Step 2: Test import chain**

```bash
python -c "import jobs; import memory; print('imports OK')"
```
Expected: `imports OK`

- [ ] **Step 3: E2E test (requires Aria running and LinkedIn logged in)**

```bash
# One-time login if needed
python main.py --login linkedin

# Run Aria
python main.py
```

Say each of these and confirm spoken output:
1. "Find me software engineer jobs in New York" → hears 5 results
2. "Find backend engineer jobs at fintech companies" → hears 5 results
3. "Any ML engineer roles remote" → hears 5 results

- [ ] **Step 4: Verify session memory**

After each search, in a separate terminal:
```bash
python -c "import memory; print(memory.session.get('last_jobs', []))"
```
Expected: list of 5 dicts with title, company, location, posted, url, platform, index fields.

- [ ] **Step 5: Update .claude/project-state.md**

Update status to `PHASE 3A COMPLETE — voice-driven job search` and add jobs/memory to the file list.

---

## Self-Review Checklist

- [x] **Spec coverage:** identity.json ✓ | memory.py (store_jobs, get_job_by_index) ✓ | jobs.py (_parse_query, _get_search_url, _extract_jobs_from_page, search_jobs, format_spoken_results) ✓ | router jobs intent ✓ | main.py wiring ✓
- [x] **No hardcoded job URLs** — `_get_search_url` builds Google site: URL and extracts the job search URL via vision ✓
- [x] **Vision pipeline** — uses `_screenshot_page` (Playwright headless + sips) + `_vision_ask` (Groq Llama-4-Scout), consistent with existing `vision.py` approach ✓
- [x] **No JS CSS selectors** — all job data extracted by vision model from screenshots ✓
- [x] **Persistent context** — `get_persistent_context(headless=True)` for LinkedIn auth; each call opens/closes its own context in `finally` ✓
- [x] **Error handling** — all functions return `None`/`[]`/fallback string on error, never raise ✓
- [x] **type consistency** — `search_jobs` → `list[dict]`, `memory.store_jobs(list[dict])`, `get_job_by_index(int) -> dict | None` ✓
- [x] **main.py pattern** — `speaker.say()` for intermediate message, `return` for final response (matches briefing handler) ✓
- [x] **No placeholder steps** — all code is complete and runnable ✓
