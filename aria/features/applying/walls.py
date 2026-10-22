"""applying/walls.py — is the page blocked by something only the user can clear?

detect(text, url, frames) → "captcha" | "login" | "email_code" | "agreement" | None   (pure)
check(tab)                → the same for an open tab; None on any error

A CAPTCHA counts only when its challenge is VISIBLE: Greenhouse and others load an
invisible reCAPTCHA on every form, which is not a wall. Login = sign-in wording plus a
visible password box (WALL_JS appends "[password-field]" to the text when there is one).
Agreement = an "I Accept / I Decline" privacy notice in front of the form — the user's
consent to give, never clicked by Aria.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

NAMES = {"captcha": "a CAPTCHA", "login": "a login page", "email_code": "an email code check",
         "agreement": "a privacy notice only you can accept"}

_CHALLENGE_FRAME = re.compile(r"recaptcha/api2/bframe|hcaptcha\.com/.*challenge|challenges\.cloudflare\.com", re.I)
_HUMAN_TEXT = re.compile(r"verify (?:that )?you(?:'re| are) (?:a )?human|are you a robot|press (?:and|&) hold", re.I)
_LOGIN_TEXT = re.compile(r"\b(?:sign in|log in|login)\b", re.I)
# Buttons' innerText runs together ("I AcceptI Decline"), so no word boundaries; both must show.
_ACCEPT_TEXT = re.compile(r"I\s*accept", re.I)
_DECLINE_TEXT = re.compile(r"I\s*decline", re.I)
_CODE_TEXT = re.compile(r"enter the (?:\d-digit )?code|verification code (?:we )?sent|code (?:we )?sent to your email|"
                        r"check your email for a code", re.I)
PASSWORD_MARK = "[password-field]"

WALL_JS = """() => {
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 30 && r.height > 30 && s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0'; };
  const frames = Array.from(document.querySelectorAll('iframe')).map(f => ({src: f.src || '', visible: vis(f)}));
  const pw = Array.from(document.querySelectorAll('input[type=password]')).some(vis);
  return {text: (document.body ? document.body.innerText : '').slice(0, 4000) + (pw ? '\\n[password-field]' : ''),
          url: location.href, frames: frames};
}"""


def detect(text: str, url: str, frames: list) -> Optional[str]:
    if any(f.get("visible") and _CHALLENGE_FRAME.search(f.get("src") or "") for f in frames or []):
        return "captcha"
    text = text or ""
    if _HUMAN_TEXT.search(text):
        return "captcha"
    if _CODE_TEXT.search(text):
        return "email_code"
    if PASSWORD_MARK in text and _LOGIN_TEXT.search(text):
        return "login"
    if _ACCEPT_TEXT.search(text) and _DECLINE_TEXT.search(text):
        return "agreement"
    return None


def check(tab: Optional[str]) -> Optional[str]:
    import aria.web.agent_browser as agent_browser
    try:
        got = agent_browser.run(lambda page: page.evaluate(WALL_JS), tab=tab)
        return detect(got.get("text", ""), got.get("url", ""), got.get("frames") or [])
    except Exception as exc:
        logger.debug("wall check failed: %s", exc)
        return None
