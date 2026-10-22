"""applying/walls.py — is the page blocked by something only the user can clear?

detect(text, url, frames, agreement=False) → "captcha" | "login" | "email_code" | "agreement" | None   (pure)
check(tab)                → the same for an open tab; None on any error

A CAPTCHA counts only when its challenge is VISIBLE: Greenhouse and others load an
invisible reCAPTCHA on every form, which is not a wall. Login = sign-in wording plus a
visible password box (WALL_JS appends "[password-field]" to the text when there is one).
Agreement = an Accept / Decline privacy notice in front of the form — the user's consent
to give, never clicked by Aria. WALL_JS decides it from the page itself: two visible real
buttons (button / [role=button] / input[type=button|submit] / a), one reading "(I) Accept"
and one "(I) Decline", within 250px of each other. Page text, checkboxes and radios never count.
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
_CODE_TEXT = re.compile(r"enter the (?:\d-digit )?code|verification code (?:we )?sent|code (?:we )?sent to your email|"
                        r"check your email for a code", re.I)
PASSWORD_MARK = "[password-field]"

WALL_JS = """() => {
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 30 && r.height > 30 && s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0'; };
  const frames = Array.from(document.querySelectorAll('iframe')).map(f => ({src: f.src || '', visible: vis(f)}));
  const pw = Array.from(document.querySelectorAll('input[type=password]')).some(vis);
  const btns = Array.from(document.querySelectorAll('button, [role=button], input[type=button], input[type=submit], a'))
    .filter(b => { const r = b.getBoundingClientRect(); const s = getComputedStyle(b);
      return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; })
    .map(b => ({t: ((b.innerText || b.value || '') + '').replace(/\\s+/g, ' ').trim().toLowerCase(), r: b.getBoundingClientRect()}));
  const acc = btns.filter(b => /^(?:i )?accept$/.test(b.t)), dec = btns.filter(b => /^(?:i )?decline$/.test(b.t));
  const near = (a, b) => Math.hypot((a.r.left + a.r.right - b.r.left - b.r.right) / 2,
                                    (a.r.top + a.r.bottom - b.r.top - b.r.bottom) / 2) <= 250;
  const agreement = acc.some(a => dec.some(d => near(a, d)));
  return {text: (document.body ? document.body.innerText : '').slice(0, 4000) + (pw ? '\\n[password-field]' : ''),
          url: location.href, frames: frames, agreement: agreement};
}"""


def detect(text: str, url: str, frames: list, agreement: bool = False) -> Optional[str]:
    if any(f.get("visible") and _CHALLENGE_FRAME.search(f.get("src") or "") for f in frames or []):
        return "captcha"
    text = text or ""
    if _HUMAN_TEXT.search(text):
        return "captcha"
    if _CODE_TEXT.search(text):
        return "email_code"
    if PASSWORD_MARK in text and _LOGIN_TEXT.search(text):
        return "login"
    if agreement:                  # measured on the page by WALL_JS: two real buttons side by side
        return "agreement"
    return None


def check(tab: Optional[str]) -> Optional[str]:
    import aria.web.agent_browser as agent_browser
    try:
        got = agent_browser.run(lambda page: page.evaluate(WALL_JS), tab=tab)
        return detect(got.get("text", ""), got.get("url", ""), got.get("frames") or [], bool(got.get("agreement")))
    except Exception as exc:
        logger.debug("wall check failed: %s", exc)
        return None
