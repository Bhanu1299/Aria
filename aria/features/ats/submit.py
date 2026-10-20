"""features/ats/submit.py — click an application's own submit button and check it went through.

Only aria.features.applicator.submit_pending() calls this, and only after the user
said "submit it". Pure code: no LLM decides whether or what to submit.

submit(tab) → SubmitOutcome(status, missing)
  status: "submitted"  confirmation text / URL seen
          "invalid"    the form flagged fields (missing = their labels); nothing sent
          "no_button"  no submit button on the page (form closed, or a different page)
          "unconfirmed" clicked, but no confirmation within the wait
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import aria.web.agent_browser as agent_browser

logger = logging.getLogger(__name__)

CONFIRMATION = re.compile(
    r"thank(?:s| you) for (?:applying|your application|your interest)|"
    r"application (?:has been |was )?(?:submitted|received)|we(?:'ve| have) received your application|"
    r"your application (?:is|has been) (?:in|submitted|received)",
    re.I,
)
_CONFIRM_URL = re.compile(r"confirmation|thank[-_]?you|/applied\b", re.I)

# Most specific first: Greenhouse new + legacy, LinkedIn Easy Apply, then any submit inside a form.
_SUBMIT_SELECTORS = [
    '#application-form button[type="submit"]',
    '#application_form [type="submit"]',
    'button[aria-label*="Submit application" i]',
    'form button:has-text("Submit application")',
    'form button:has-text("Submit")',
    'form [type="submit"]',
]

_INVALID_JS = """
() => Array.from(document.querySelectorAll('[aria-invalid="true"]')).map(el => {
  const id = el.id;
  const lab = id ? document.querySelector('label[for="' + CSS.escape(id) + '"]') : null;
  return ((lab && lab.innerText) || el.getAttribute('aria-label') || id || '').trim().replace(/\\s*\\*$/, '');
}).filter(Boolean)
"""


@dataclass
class SubmitOutcome:
    status: str
    missing: list = field(default_factory=list)


def is_confirmation(text: str, url: str = "") -> bool:
    return bool(CONFIRMATION.search(text or "") or _CONFIRM_URL.search(url or ""))


def page_confirms(tab: Optional[str]) -> bool:
    """Does the page in this tab show an application confirmation? Never raises."""
    try:
        text, url = agent_browser.run(lambda page: (page.inner_text("body"), page.url), tab=tab)
        return is_confirmation(text, url)
    except Exception:
        return False


def submit(tab: Optional[str], wait_secs: float = 12.0) -> SubmitOutcome:
    def _click(page) -> bool:
        for sel in _SUBMIT_SELECTORS:
            loc = page.locator(sel)
            if loc.count():
                loc.first.click(timeout=5000)
                logger.info("submit: clicked %s on %s", sel, page.url)
                return True
        return False

    try:
        if not agent_browser.run(_click, tab=tab):
            return SubmitOutcome("no_button")
        deadline = time.time() + wait_secs
        while time.time() < deadline:
            time.sleep(1.0)
            text, url, invalid = agent_browser.run(
                lambda page: (page.inner_text("body"), page.url, page.evaluate(_INVALID_JS)), tab=tab)
            if is_confirmation(text, url):
                return SubmitOutcome("submitted")
            if invalid:
                return SubmitOutcome("invalid", list(dict.fromkeys(invalid)))
        return SubmitOutcome("unconfirmed")
    except Exception as exc:
        logger.error("submit failed: %s", exc)
        return SubmitOutcome("unconfirmed")
