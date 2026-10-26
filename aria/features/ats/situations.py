"""features/ats/situations.py — what is this page, and what do we do about it?

After an Apply link the next page is often NOT the form: a cookie banner over everything, a
"you are leaving this site" notice, a careers list where the job still has to be picked, a
"this job is closed" page, an "Apply with LinkedIn / manually" choice, an assessment, an
email-only posting. resolve_apply_target asks look(page) at every hop and does what it says.

look(snapshot, job) → Situation(kind, click, why)          (pure: code rules first)
  cookie_banner    a cookie / consent notice     → click its Reject / Decline (else Accept)
  leaving_site     "you are leaving…" notice     → click Continue / Proceed
  pick_job         a list of jobs               → click the link naming this job's role
  closed           posting closed / expired / 404             → stop, report
  already_applied  "you have already applied"                 → stop, report
  email_only       apply by sending your resume by email      → stop, report (never sent)
  assessment       an assessment / test to start              → stop, it's yours
  unknown          none of the above (the caller tries the Apply affordance, then ask())

ask(snapshot, job) → Situation | None    one SMALL-tier LLM call (task apply_page_situation)
  for a page the rules can't read: the model picks a kind from the same list, or one
  numbered control to click. Code validates it: a listed control, never submit / send /
  pay / delete / sign-out, never on a page that already has form fields.

SNAPSHOT_JS gathers what both need in one round trip; clickables carry data-aria-apply="<i>"
(the same marking apply_target.click_outcome clicks).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from aria.features.ats import untrusted

logger = logging.getLogger(__name__)

STOPS = ("closed", "already_applied", "email_only", "assessment")
ACTIONS = ("cookie_banner", "leaving_site", "pick_job")
LLM_KINDS = ("click", "closed", "already_applied", "email_only", "assessment", "sign_in", "loading", "other")

# Every clickable, marked like apply_target.AFFORDANCES_JS, plus the page's text.
SNAPSHOT_JS = """() => {
  const vis = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  // a cookie / consent / legal-notice banner (OneTrust, Cookiebot, Workday's legalNotice, …)
  const BANNER = '[id*=cookie i],[class*=cookie i],[id*=consent i],[class*=consent i],[aria-label*=cookie i],' +
    '[id*=onetrust i],[class*=gdpr i],[id*=gdpr i],[data-automation-id*=legalNotice],[id*=CybotCookiebot]';
  const fields = Array.from(document.querySelectorAll('input, textarea, select')).filter(e =>
    !['hidden', 'search', 'submit', 'button', 'image', 'reset'].includes((e.type || '').toLowerCase()) && vis(e)).length;
  const items = Array.from(document.querySelectorAll('a, button, [role=button], input[type=button], input[type=submit]'))
    .map((e, i) => { e.setAttribute('data-aria-apply', String(i));
      return {i: i, tag: e.tagName.toLowerCase(), visible: vis(e),
        text: ((e.innerText || e.value || e.getAttribute('aria-label') || '') + '').replace(/\\s+/g, ' ').trim().slice(0, 80),
        href: e.getAttribute('href') || '',
        banner: !!e.closest(BANNER)}; })
    .filter(c => c.visible && c.text);
  return {url: location.href, title: document.title || '', fields: fields, clickables: items.slice(0, 300),
          text: (document.body ? document.body.innerText : '').replace(/[ \\t]+/g, ' ').slice(0, 6000)};
}"""

# Never clicked by this module, whatever a rule or the model says.
NEVER_CLICK = re.compile(
    r"\b(?:submit|send|finish|complete (?:my |your |the )?application|confirm|pay|purchase|buy|checkout|delete|"
    r"remove|withdraw|unsubscribe|sign ?out|log ?out|report|flag)\b", re.I)

_COOKIE_TEXT = re.compile(r"cookie|consent|privacy preferences|tracking technologies", re.I)
_COOKIE_REJECT = re.compile(r"(?:reject|decline|deny|refuse)(?: all| optional| non-essential)?(?: cookies)?|"
                            r"(?:only |use )?(?:strictly )?(?:necessary|essential)(?: cookies)?(?: only)?|"
                            r"continue without accepting", re.I)
_COOKIE_ACCEPT = re.compile(r"accept(?: all)?(?: cookies)?|allow(?: all)?(?: cookies)?|i accept|i agree|agree|"
                            r"got it|ok(?:ay)?|understood", re.I)
_LEAVING_TEXT = re.compile(r"you (?:are|'re) (?:now )?(?:about to )?leav(?:e|ing)|"
                           r"(?:will be|being) (?:re)?directed|redirecting you|external (?:site|website|link)|"
                           r"continue to (?:the )?(?:company|employer|external)", re.I)
_LEAVING_GO = re.compile(r"continue|proceed|go to (?:site|website)|ok(?:ay)?|yes|i understand|visit (?:site|website)",
                         re.I)
_CLOSED = re.compile(
    r"no longer (?:accepting (?:applications|candidates)|available|open|active|posted)|"
    r"(?:job|position|posting|role|requisition|opening) (?:has been |is )?(?:closed|filled|expired|removed|"
    r"no longer available|not available)|"
    r"(?:this|the) (?:job|position|posting|role) (?:has )?(?:expired|closed)|not accepting applications|"
    r"applications? (?:are |is )?(?:now )?closed|"
    r"(?:job|page|posting) (?:was )?not found|page (?:you(?:'re| are) looking for )?(?:doesn'?t|does not|no longer) exist|"
    r"\b(?:error )?404\b(?: error| not found| page)", re.I)
_APPLIED = re.compile(r"you(?:'ve| have) already applied|already (?:submitted|applied)|"
                      r"application (?:has been |was )?(?:submitted|received)|thank you for (?:applying|your application)",
                      re.I)
_EMAIL_ONLY = re.compile(r"(?:send|email|e-mail|submit) (?:your |a )?(?:resume|cv|application)s? (?:to|at)|"
                         r"apply (?:by|via) e-?mail|email (?:us|hr|recruiting) (?:your|with)", re.I)
_ASSESSMENT = re.compile(r"(?:start|begin|take|complete) (?:the |an |your )?(?:online )?(?:assessment|test|"
                         r"coding challenge|questionnaire)", re.I)


@dataclass
class Situation:
    kind: str                  # see the module docstring
    click: Optional[int] = None  # data-aria-apply index to click (ACTIONS / an LLM "click")
    why: str = ""
    source: str = "rule"       # rule | llm


def _visible_buttons(snap: dict) -> list:
    return [c for c in snap.get("clickables") or [] if c.get("visible") and c.get("text")]


def _first(snap: dict, pattern: re.Pattern, banner_only: bool = False) -> Optional[dict]:
    return next((c for c in _visible_buttons(snap)
                 if (c.get("banner") or not banner_only) and pattern.fullmatch(c["text"].strip())
                 and not NEVER_CLICK.search(c["text"])), None)


def _role_link(snap: dict, job: dict) -> Optional[dict]:
    """On a careers list: the one link that names this job's role (fuzzy, word by word)."""
    from aria.integrations import sheets
    role = [t for t in sheets._tokens(job.get("role", "")) if t not in sheets._FILLER and len(t) > 1]
    if len(role) < 2:
        return None
    best, best_score = None, 0.0
    for c in _visible_buttons(snap):
        if c.get("tag") != "a" or NEVER_CLICK.search(c["text"]):
            continue
        words = sheets._tokens(c["text"])
        if not words:
            continue
        hit = sum(1 for t in role if any(sheets._sim(t, w) >= 0.85 for w in words)) / len(role)
        if hit > best_score:
            best, best_score = c, hit
    return best if best_score >= 0.75 else None


def look(snap: dict, job: dict) -> Situation:
    """The rules. Order matters: an overlay first (it blocks every click), then stops that
    are true whatever else is on the page, then the steps that lead on."""
    text = snap.get("text") or ""
    if _COOKIE_TEXT.search(text):          # only a button inside the banner, never a stray "OK"
        btn = _first(snap, _COOKIE_REJECT, True) or _first(snap, _COOKIE_ACCEPT, True)
        if btn is not None:
            return Situation("cookie_banner", btn["i"], f"cookie notice: “{btn['text']}”")
    if (snap.get("fields") or 0) == 0:
        if _APPLIED.search(text):
            return Situation("already_applied", why=_APPLIED.search(text).group(0))
        if _CLOSED.search(text):
            return Situation("closed", why=_CLOSED.search(text).group(0))
    if _LEAVING_TEXT.search(text):
        btn = _first(snap, _LEAVING_GO)
        if btn is not None:
            return Situation("leaving_site", btn["i"], f"leaving-site notice: “{btn['text']}”")
    if (snap.get("fields") or 0) == 0:
        link = _role_link(snap, job)
        if link is not None and not re.search(r"\bapply\b", link["text"], re.I):
            return Situation("pick_job", link["i"], f"job list: “{link['text']}”")
        if _ASSESSMENT.search(text):
            return Situation("assessment", why=_ASSESSMENT.search(text).group(0))
        mailto = any(str(c.get("href", "")).lower().startswith("mailto:") for c in _visible_buttons(snap))
        if _EMAIL_ONLY.search(text) and mailto:
            return Situation("email_only", why=_EMAIL_ONLY.search(text).group(0))
    return Situation("unknown")


_SYSTEM = (
    "You are looking at a web page Aria reached while trying to open a job application form. "
    "Decide what the page is. Reply with JSON only: "
    '{"kind": "<click|closed|already_applied|email_only|assessment|sign_in|loading|other>", '
    '"click": <number of the control to click, only for kind=click>, "why": "<few words>"}. '
    "Use kind=click only when clicking one listed control clearly leads toward this job's application "
    "form (an Apply button, the job's own link in a list, Continue on a notice). Never pick a control "
    "that submits, sends, pays, deletes or signs out. If unsure, kind=other. " + untrusted.RULE
)


def ask(snap: dict, job: dict) -> Optional[Situation]:
    """One small-tier LLM look at a page the rules can't read. Validated; None if unusable."""
    from aria.llm import llm_client
    if (snap.get("fields") or 0) > 0:
        return None                                  # a page with fields is the engine's, never clicked here
    buttons = [c for c in _visible_buttons(snap) if not NEVER_CLICK.search(c["text"])][:60]
    listing = "\n".join(f"[{c['i']}] {c['tag']}: {c['text']}" for c in buttons)
    allowed = {c["i"] for c in buttons}
    page, _flags = untrusted.guard(f"Page: {snap.get('title') or ''} — {snap.get('url') or ''}\n\n"
                                   f"Page text (start):\n{(snap.get('text') or '')[:2500]}\n\n"
                                   f"Clickable controls:\n{listing or '(none)'}", "Web page")
    prompt = f"Job: {job.get('role') or '?'} at {job.get('company') or '?'}\n\n{page}"

    def validate(raw: str):
        try:
            got = json.loads(re.search(r"\{.*\}", raw or "", re.S).group(0))
        except Exception:
            return None
        kind = str(got.get("kind") or "")
        if kind not in LLM_KINDS:
            return None
        if kind == "click":
            try:
                i = int(got.get("click"))
            except (TypeError, ValueError):
                return None
            if i not in allowed:
                return None
            return Situation("click", i, str(got.get("why") or "")[:120], "llm")
        return Situation(kind, None, str(got.get("why") or "")[:120], "llm")

    try:
        _resp, out = llm_client.complete_checked("apply_page_situation", [{"role": "user", "content": prompt}],
                                                 validate=validate, system=_SYSTEM, max_tokens=120)
    except Exception as exc:
        logger.warning("situations: LLM look failed: %s", exc)
        return None
    return out


def describe(kind: str, why: str = "") -> str:
    """What a stop means, in one spoken / summary sentence."""
    return {
        "closed": "The posting looks closed or gone",
        "already_applied": "It says you've already applied",
        "email_only": "It only takes applications by email — that's yours to send",
        "assessment": "It starts with an assessment, which is yours to take",
        "sign_in": "It needs you to sign in first",
        "loading": "The page never finished loading",
    }.get(kind, "I couldn't tell how to reach the form") + (f" (“{why}”)." if why else ".")
