"""features/ats/apply_target.py — from a job posting to the application form, in code.

resolve_apply_target(tab, job, progress_fn, wait_fn) → ApplyTarget(tab, ats, kind, url, posting_text, stopped)
  Starts from the page open in `tab` (the job URL). Up to MAX_HOPS times:
    a wall (CAPTCHA / login / code / agreement) → wait_fn(kind), else stop there
    what the page is (situations.look): a cookie banner / leaving-site notice / careers list
      → click through it (its own budget); closed / already applied / email only /
      assessment → stop, named in ApplyTarget.stopped
    ≥ INLINE_FIELDS visible fields → this is the form
    an embedded ATS iframe → open its URL in the same tab                      kind "iframe"
    the best Apply affordance (pick_affordance):
      a link → navigate to its href in the same tab                            kind "same_tab"
      a button → clicked ONLY when the page has no fields (it can't send anything),
                 then: a new tab (popup) → the application's tab name moves to it ("new_tab";
                 the posting tab stays open), the URL changed ("same_tab"), a dialog or new
                 fields appeared ("modal"), nothing → logged
    no Apply control on a page with no fields → ONE small-model look (situations.ask, at most
      MAX_LLM_LOOKS per job): a listed control to click, or what kind of stop it is
  Stops early on a Workday / Handshake page (SELF_STARTING): their adapters start the
  application themselves. After each hop the URL settles (unchanged for SETTLE_STABLE_SECS, at most SETTLE_MAX_SECS).
  The ATS is named from the final URL + DOM (detect.identify); unknown → "generic" and a
  correction row with the domain. Never raises: anything unexpected stops resolving where it is.

pick_affordance(candidates, page_url) → the best Apply control, or None   (pure)
  Text: Apply · Apply now · Apply for this job · Easy Apply · Quick apply · Apply manually ·
  Start application · I'm interested. Never: Apply later / with LinkedIn, Indeed… / Applied /
  share / save. A link whose href is an ATS apply URL counts even with other text.

Tab wraps one agent_browser tab: run(fn) → fn(page) on the browser worker, navigate(url),
adopt(page) (worker only) — so adapters never hold a Playwright page across a wait.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

from aria.features.ats import corrections
from aria.features.ats.detect import domain, identify

logger = logging.getLogger(__name__)

MAX_HOPS = 4
MAX_NOTICE_CLICKS = 3        # cookie banners / leaving-site notices / a careers list: not hops
MAX_LLM_LOOKS = 2
# ATSes whose adapter starts the application itself (Workday's Apply → start options →
# sign-in; Handshake's Apply dialog): resolving stops as soon as one is reached
SELF_STARTING = {"workday", "handshake"}
INLINE_FIELDS = 3            # fewer than this and an Apply link is still followed (job-alert boxes)
SETTLE_STABLE_SECS = 1.5
SETTLE_MAX_SECS = 15.0
CLICK_WAIT_SECS = 6.0

FIELDS_JS = """() => Array.from(document.querySelectorAll('input, textarea, select')).filter(e =>
  !['hidden', 'search', 'submit', 'button', 'image', 'reset'].includes((e.type || '').toLowerCase())
  && e.offsetParent !== null).length"""
DIALOGS_JS = """() => Array.from(document.querySelectorAll('[role=dialog], dialog[open], [aria-modal=true]'))
  .filter(d => d.offsetParent !== null || getComputedStyle(d).position === 'fixed').length"""
# An ATS form embedded in a company careers page.
ATS_FRAME_JS = """() => (Array.from(document.querySelectorAll('iframe')).map(f => f.src || '')
  .find(s => /lever\\.co|ashbyhq\\.com|jobvite\\.com|smartrecruiters\\.com|workable\\.com|bamboohr\\.com|icims\\.com|recruitee\\.com|breezy\\.hr|myworkdayjobs\\.com|applytojob\\.com/i.test(s)) || '')"""
# Every clickable, marked so the chosen one can be clicked exactly (data-aria-apply="<i>").
AFFORDANCES_JS = """() => {
  const vis = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  // custom-element buttons count too (UltiPro / UKG's <ukg-button>Apply now</ukg-button>)
  const custom = Array.from(document.querySelectorAll('*')).filter(e => /-BUTTON$/.test(e.tagName) ||
    e.getAttribute('data-tag-name') === 'button');
  const all = Array.from(document.querySelectorAll('a, button, [role=button], input[type=button], input[type=submit]'));
  custom.forEach(e => { if (!all.includes(e) && !all.some(a => e.contains(a))) all.push(e); });
  return all
    .map((e, i) => { e.setAttribute('data-aria-apply', String(i));
      return {i: i, tag: e.tagName.toLowerCase(), visible: vis(e),
        text: ((e.innerText || e.value || e.getAttribute('aria-label') || '') + '').replace(/\\s+/g, ' ').trim().slice(0, 80),
        href: e.getAttribute('href') || ''}; });
}"""

_APPLY_TEXTS = (
    (re.compile(r"apply manually|apply without (?:an )?account", re.I), 6),
    (re.compile(r"apply(?: now| online| here| today)?|apply (?:for|to) (?:this|the) (?:job|position|role|opening)|"
                r"apply for job", re.I), 5),
    (re.compile(r"easy apply|quick apply|1-click apply|one-click apply", re.I), 4),
    (re.compile(r"start (?:your |an |my )?application|begin (?:your )?application|continue to (?:the )?application", re.I), 4),
    (re.compile(r"apply externally|apply on (?:the )?(?:company|employer)(?:'s)? (?:site|website)", re.I), 4),
    (re.compile(r"i'?m interested|i am interested", re.I), 3),
)
_NOT_APPLY = re.compile(r"\blater\b|\bwith (?:linkedin|indeed|seek|google|facebook|xing|glassdoor)\b|\bapplied\b|"
                        r"\bshare\b|\bsave\b|\brefer\b|\balert", re.I)
_ATS_APPLY_HREF = re.compile(
    r"(?:greenhouse\.io|lever\.co|ashbyhq\.com|myworkdayjobs\.com|myworkdaysite\.com|icims\.com|jobvite\.com|"
    r"smartrecruiters\.com|workable\.com|bamboohr\.com|applytojob\.com|joinhandshake\.com|recruitee\.com|breezy\.hr)"
    r"/\S*(?:apply|application)", re.I)
_DEAD_HREF = ("mailto:", "javascript:", "tel:", "#")


@dataclass
class Tab:
    id: Optional[str]          # agent_browser tab name == apply_tabs.json entry id (None = shared page)
    run: Callable              # run(fn) → fn(page) on the browser worker
    navigate: Callable         # navigate(url, settle_secs=...)
    adopt: Callable            # adopt(page) — worker only: this tab now means `page`

    @classmethod
    def of(cls, name: Optional[str]) -> "Tab":
        import aria.web.agent_browser as agent_browser
        return cls(name,
                   lambda fn: agent_browser.run(fn, tab=name),
                   lambda url, settle_secs=2.0: agent_browser.navigate(url, settle_secs=settle_secs, tab=name),
                   lambda page: agent_browser.rebind(name, page))


@dataclass
class ApplyTarget:
    tab: Tab
    ats: str                   # greenhouse | lever | ashby | workday | icims | handshake | … | generic
    kind: str                  # inline | same_tab | new_tab | modal | iframe
    url: str                   # settled URL of the form
    posting_text: str          # the posting's text, for the LLM's open-text answers
    identified: str = ""       # what identify() named, even without an adapter ("" = unknown)
    stopped: str = ""          # why resolving stopped short of a form (situations.STOPS, "sign_in"), "" = form
    stopped_why: str = ""


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------

def _text_score(text: str) -> int:
    text = (text or "").strip()
    if not text or _NOT_APPLY.search(text):
        return 0
    return next((score for pattern, score in _APPLY_TEXTS if pattern.fullmatch(text)), 0)


def _live_href(href: str, page_url: str) -> Optional[str]:
    href = (href or "").strip()
    if not href or href.lower().startswith(_DEAD_HREF):
        return None
    target = urljoin(page_url, href)
    here, there = urlparse(page_url), urlparse(target)
    if (here.netloc, here.path.rstrip("/"), here.query) == (there.netloc, there.path.rstrip("/"), there.query):
        return None              # an in-page anchor to the same posting
    return target


def pick_affordance(candidates: list, page_url: str) -> Optional[dict]:
    """The best visible Apply control: {"i", "text", "action": "follow"|"click", "url"}."""
    best, best_score = None, 0
    for c in candidates:
        if not c.get("visible"):
            continue
        href = _live_href(c.get("href", ""), page_url) if c.get("tag") == "a" else None
        score = _text_score(c.get("text", ""))
        if href and _ATS_APPLY_HREF.search(href) and not _NOT_APPLY.search(c.get("text", "")):
            score = max(score, 2) + (2 if score else 0)
        if score > best_score:
            best_score = score
            best = {"i": c["i"], "text": c.get("text", ""), "action": "follow" if href else "click", "url": href or ""}
    return best


# ---------------------------------------------------------------------------
# Browser steps (each run() call is one short worker job)
# ---------------------------------------------------------------------------

def _href(tab: Tab) -> str:
    """The live URL. page.url is only refreshed while a Playwright call runs, so a redirect
    that happens during a sleep is seen only through a real round trip."""
    def _do(page) -> str:
        try:
            return page.evaluate("location.href")
        except Exception:          # mid-navigation: the old context is gone
            return page.url
    return tab.run(_do)


def settle(tab: Tab, stable_secs: Optional[float] = None, max_secs: Optional[float] = None) -> str:
    """Wait until the URL stops changing (redirect chains, client-side routing). Returns it."""
    stable_secs = SETTLE_STABLE_SECS if stable_secs is None else stable_secs
    max_secs = SETTLE_MAX_SECS if max_secs is None else max_secs
    deadline = time.time() + max_secs
    last, since = _href(tab), time.time()
    while time.time() < deadline:
        time.sleep(0.4)
        now = _href(tab)
        if now != last:
            last, since = now, time.time()
        elif time.time() - since >= stable_secs:
            break
    try:
        tab.run(lambda page: page.wait_for_load_state("domcontentloaded", timeout=10_000))
    except Exception as exc:
        logger.debug("load state wait: %s", exc)
    return last


def click_outcome(tab: Tab, index: int, wait_secs: Optional[float] = None) -> str:
    """Click candidate `index` and say what happened: new_tab | same_tab | modal | nothing.
    A new tab becomes this Tab (adopt) — the posting tab stays open."""
    wait_secs = CLICK_WAIT_SECS if wait_secs is None else wait_secs

    def _do(page) -> str:
        ctx = page.context
        before_pages = list(ctx.pages)
        before_url = page.url
        before_fields, before_dialogs = page.evaluate(FIELDS_JS), page.evaluate(DIALOGS_JS)
        page.locator(f'[data-aria-apply="{index}"]').first.click(timeout=5000)
        steps = max(1, int(wait_secs / 0.3))
        for _ in range(steps):
            page.wait_for_timeout(300)
            fresh = [p for p in ctx.pages if all(p is not b for b in before_pages) and not p.is_closed()]
            if fresh:
                tab.adopt(fresh[-1])
                return "new_tab"
            if page.url != before_url:
                return "same_tab"
            if page.evaluate(FIELDS_JS) > before_fields or page.evaluate(DIALOGS_JS) > before_dialogs:
                return "modal"
        return "nothing"
    return tab.run(_do)


def _wall(tab: Tab) -> Optional[str]:
    from aria.features.applying import walls
    try:
        got = tab.run(lambda page: page.evaluate(walls.WALL_JS))
        return walls.detect(got.get("text", ""), got.get("url", ""), got.get("frames") or [], bool(got.get("agreement")))
    except Exception as exc:
        logger.debug("wall check failed: %s", exc)
        return None


def resolve_apply_target(tab: Tab, job: dict, progress_fn: Optional[Callable[[str], None]] = None,
                         wait_fn: Optional[Callable[[str], bool]] = None) -> ApplyTarget:
    from aria.features.ats import situations
    say = progress_fn or (lambda _m: None)
    kind = "inline"
    posting = ""
    stopped, stopped_why = "", ""
    notices = looks = 0
    try:
        settle(tab)
        posting = tab.run(lambda page: page.inner_text("body"))[:8000]
        hop = steps = 0
        while hop < MAX_HOPS and steps < 3 * MAX_HOPS:      # steps: walls cleared + notices count too
            steps += 1
            if identify(_href(tab)) in SELF_STARTING:
                break
            wall = _wall(tab)
            if wall:
                if wait_fn is not None and wait_fn(wall):
                    settle(tab)
                    continue
                break                      # the engine sees the same wall and reports it
            snap = tab.run(lambda page: page.evaluate(situations.SNAPSHOT_JS))
            sit = situations.look(snap, job)
            if sit.kind in situations.ACTIONS and notices < MAX_NOTICE_CLICKS:
                notices += 1
                say({"cookie_banner": "Closing the cookie notice.",
                     "leaving_site": "Continuing past the leaving-this-site notice.",
                     "pick_job": "Picking this job from the careers list."}[sit.kind])
                if click_outcome(tab, sit.click) == "nothing" and sit.kind != "cookie_banner":
                    corrections.record(job, "apply_click_nothing", "resolve", f"{sit.why}: nothing happened",
                                       ats=identify(snap.get("url", "")) or "", page_url=snap.get("url", ""), run=tab.run)
                    break
                settle(tab)
                continue
            if sit.kind in situations.STOPS:
                stopped, stopped_why = sit.kind, sit.why
                break
            hop += 1
            fields = tab.run(lambda page: page.evaluate(FIELDS_JS))
            if fields >= INLINE_FIELDS:
                break
            frame = tab.run(lambda page: page.evaluate(ATS_FRAME_JS))
            if frame:
                say("The form is embedded from its applicant system. Opening it directly.")
                tab.navigate(frame, settle_secs=1.0)
                kind = "iframe"
                settle(tab)
                continue
            here = tab.run(lambda page: page.url)
            pick = pick_affordance(tab.run(lambda page: page.evaluate(AFFORDANCES_JS)), here)
            if pick is None:
                if fields == 0 and looks < MAX_LLM_LOOKS:
                    looks += 1
                    seen_llm = situations.ask(snap, job)
                    if seen_llm is not None and seen_llm.kind == "click":
                        say("Taking the next step toward the form.")
                        if click_outcome(tab, seen_llm.click) != "nothing":
                            kind = "same_tab"
                            settle(tab)
                            continue
                    elif seen_llm is not None and seen_llm.kind == "loading":
                        settle(tab, max_secs=SETTLE_MAX_SECS)
                        continue
                    elif seen_llm is not None and seen_llm.kind in situations.STOPS + ("sign_in",):
                        stopped, stopped_why = seen_llm.kind, seen_llm.why
                        break
                if fields == 0:
                    corrections.record(job, "apply_not_found", "resolve", "no form and no Apply control",
                                       ats=identify(here) or "", page_url=here, run=tab.run)
                break
            if pick["action"] == "follow":
                say(f"Following “{pick['text'] or 'Apply'}” to the application form.")
                tab.navigate(pick["url"], settle_secs=1.0)
                kind = "same_tab"
                settle(tab)
                continue
            if fields > 0:
                break                      # a button on a page with fields might send it
            say(f"Clicking “{pick['text']}” to open the application form.")
            outcome = click_outcome(tab, pick["i"])
            if outcome == "nothing":
                corrections.record(job, "apply_click_nothing", "resolve", f"clicked “{pick['text']}”, nothing happened",
                                   ats=identify(here) or "", page_url=here, run=tab.run)
                break
            kind = outcome
            settle(tab)
    except Exception as exc:                 # control.Stopped is a BaseException and passes
        logger.warning("resolve_apply_target stopped early: %s", exc)
    url, html = "", ""
    try:
        url, html = tab.run(lambda page: (page.url, page.content()))
    except Exception as exc:
        logger.warning("resolve_apply_target: page unreadable: %s", exc)
    seen = identify(url, html) or ""
    if not seen and url:
        corrections.record(job, "unknown_ats", "resolve", domain(url), page_url=url, run=tab.run)
    return ApplyTarget(tab=tab, ats=seen or "generic", kind=kind, url=url, posting_text=posting, identified=seen,
                       stopped=stopped, stopped_why=stopped_why)
