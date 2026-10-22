"""features/ats — fill a job application form and stop for review. NEVER submits.

fill_application(job, progress_fn, tab) → FillResult   (tab: the job's own agent_browser tab)
  detect() names an ATS with a deterministic adapter (Greenhouse) → that adapter.
  Otherwise → form_engine: scan / classify / fill / re-scan / verify / next page, in code
  (the LLM only writes open-text answers). Only when the engine finds no fields at all does
  the see-think-act vision loop (computer_use.run_loop) try, under the same never-submit rule.

job: a job-sheet row dict — needs "url"; "company"/"role"/"fields" improve answers.
"""
from __future__ import annotations

import html as _html
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urljoin

import aria.features.answer_bank as answer_bank
from aria.features.ats.detect import detect, identify

logger = logging.getLogger(__name__)

NEVER_SUBMIT = (
    "Do NOT click Submit, Apply, Send or any button that sends the application — "
    "return 'confirm' as soon as every field you can fill is filled."
)


_ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.I | re.S)
_HREF = re.compile(r"""\bhref\s*=\s*["']([^"']+)["']""", re.I)
_APPLY_TEXT = re.compile(r"apply(?: now| online| here| for this (?:job|position|role))?|i'?m interested", re.I)
_HAS_FIELDS_JS = """() => Array.from(document.querySelectorAll('input, textarea, select')).filter(e =>
  !['hidden', 'search', 'submit', 'button'].includes((e.type || '').toLowerCase()) && e.offsetParent !== null).length"""
# An ATS form embedded in a company careers page.
_ATS_FRAME_JS = """() => (Array.from(document.querySelectorAll('iframe')).map(f => f.src || '')
  .find(s => /lever\\.co|ashbyhq\\.com|jobvite\\.com|smartrecruiters\\.com|workable\\.com|bamboohr\\.com|icims\\.com|recruitee\\.com|breezy\\.hr|myworkdayjobs\\.com/i.test(s)) || '')"""
# A button that only OPENS the form (the page has no fields yet, so nothing can be sent).
_OPEN_FORM_BUTTON = re.compile(r"^\s*(?:apply(?: now| for this (?:job|position|role))?|i'?m interested)\s*$", re.I)


def apply_form_link(html: str, base_url: str) -> Optional[str]:
    """The URL behind a posting's "Apply" LINK (an <a href>, never a button), or None.
    Following a link only opens the form — it can't send anything."""
    for attrs, inner in _ANCHOR.findall(html or ""):
        text = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", inner))).strip()
        href = _HREF.search(attrs)
        if not href or not _APPLY_TEXT.fullmatch(text):
            continue
        target = _html.unescape(href.group(1)).strip()
        if target.lower().startswith(("mailto:", "javascript:", "#", "tel:")):
            continue
        return urljoin(base_url, target)
    return None


_COUNTRY_ALIASES = answer_bank.COUNTRY_ALIASES
_COUNTRY_LABEL = re.compile(r"\bcountry\b|(?:location|country|place) of residence", re.I)
_SELECTS_JS = """() => Array.from(document.querySelectorAll('select')).map((s, i) => ({
  i: i, empty: !s.value, visible: s.offsetParent !== null,
  label: ((s.labels && s.labels[0] && s.labels[0].innerText) || s.getAttribute('aria-label') || '').trim(),
  options: Array.from(s.options).map(o => o.text.trim())}))"""


def _norm_place(text: str) -> str:
    return re.sub(r"[^a-z]+", " ", (text or "").lower()).strip()


def is_country_label(label: str) -> bool:
    return bool(_COUNTRY_LABEL.search(label or ""))


def match_country(options: list, country: str) -> Optional[str]:
    """The option naming `country` (exactly or by a common alias such as US / USA), else None."""
    want = _norm_place(country)
    if not want:
        return None
    names = {want} | _COUNTRY_ALIASES.get(want, set())
    return next((o for o in options if _norm_place(o) in names), None)


def _preselect_country(agent_browser, country: str, progress_fn) -> None:
    """Code first: set an empty, visible country/residence <select> to the bank's country
    (the LLM loop clicks native selects instead of choosing). Never raises."""
    try:
        for sel in agent_browser.run(lambda page: page.evaluate(_SELECTS_JS)):
            if not (sel["empty"] and sel["visible"] and is_country_label(sel["label"])):
                continue
            option = match_country(sel["options"], country)
            if option:
                agent_browser.run(lambda page: page.locator("select").nth(sel["i"]).select_option(label=option))
                if progress_fn:
                    progress_fn(f"Set {sel['label'].rstrip(':')} to {option}.")
                time.sleep(1.5)        # the rest of the form renders after this choice
    except Exception as exc:
        logger.warning("country preselect failed: %s", exc)


@dataclass
class FillResult:
    ats: str = ""                 # "greenhouse" | "generic"
    identified: str = ""          # what identify() saw (e.g. "workday"), even without an adapter
    form_url: str = ""
    title: str = ""
    company: str = ""
    description: str = ""
    review: list = field(default_factory=list)
    filled: dict = field(default_factory=dict)   # name → {label, kind, value, source}
    tab: Optional[str] = None                    # agent_browser tab the form is open in
    error: str = ""
    pages: int = 1
    stopped_at: str = ""                         # where the engine stopped (form_engine.EngineResult)

    def count(self, source: str) -> int:
        return sum(1 for r in self.review if r["source"] == source)


def _open_form(agent_browser, progress_fn) -> str:
    """From a posting, get to its form: the Apply link, an embedded ATS iframe, or an
    Apply button on a page with no fields yet. Returns the posting's text (for the LLM)."""
    say = progress_fn or (lambda _m: None)
    posting = agent_browser.run(lambda page: page.inner_text("body"))[:8000]
    if agent_browser.run(lambda page: page.evaluate(_HAS_FIELDS_JS)):
        return posting
    html, here = agent_browser.run(lambda page: (page.content(), page.url))
    form = apply_form_link(html, here)
    if not form:
        form = agent_browser.run(lambda page: page.evaluate(_ATS_FRAME_JS)) or None
    if form:
        say("That's the job posting. Opening its application form.")
        agent_browser.navigate(form, settle_secs=3.0)
        return posting
    def click_open(page) -> bool:
        for loc in page.locator("button, [role=button]").all():
            try:
                if loc.is_visible() and _OPEN_FORM_BUTTON.match(loc.inner_text() or ""):
                    loc.click(timeout=4000)
                    return True
            except Exception:
                continue
        return False
    if agent_browser.run(click_open):
        time.sleep(2.5)
    return posting


def _vision_fallback(job: dict, progress_fn, agent_browser) -> tuple:
    import aria.web.computer_use as computer_use
    import aria.web.dom_browser as dom_browser
    context = answer_bank.fillable_answers(answer_bank.load(), job)   # never always-ask answers
    if context.get("country"):
        _preselect_country(agent_browser, str(context["country"]), progress_fn)
    goal = (
        f"Fill out the job application form for {job.get('role', 'this role')} at "
        f"{job.get('company', 'this company')} using the applicant answers in Context. "
        f"Leave any field you have no answer for empty. {NEVER_SUBMIT}"
    )
    with computer_use.no_submit():    # code, not just the prompt, refuses submit
        status, _data = computer_use.run_loop(goal, context, max_steps=30, progress_fn=progress_fn)
    empty = dom_browser.find_empty_required_fields()
    review = [{"name": e.get("selector", ""), "label": e.get("label", ""), "kind": e.get("field_type", ""),
               "category": "", "value": "", "source": "needs_me", "required": True, "note": "still empty"}
              for e in empty]
    return status, review


def _generic(job: dict, progress_fn, tab: Optional[str] = None,
             wait_fn: Optional[Callable[[str], bool]] = None) -> FillResult:
    """The form engine in the job's tab. computer_use / dom_browser call agent_browser with
    no tab, so everything runs inside use_tab(tab), keeping the caller's pause/stop gate."""
    import aria.web.agent_browser as agent_browser
    from aria.features.ats import form_engine
    from aria.features.ats.greenhouse import _gateway_up

    result = FillResult(ats="generic", company=job.get("company", ""), title=job.get("role", ""), tab=tab)
    try:
        with agent_browser.use_tab(tab, gate=agent_browser.current_gate()):
            agent_browser.navigate(job["url"], settle_secs=3.0)
            posting = _open_form(agent_browser, progress_fn)
            llm_up = _gateway_up()
            eng = form_engine.fill(job, posting, progress_fn=progress_fn, llm_available=llm_up, wait_fn=wait_fn)
            result.form_url = eng.form_url or agent_browser.run(lambda page: page.url)
            result.pages, result.stopped_at = eng.pages, eng.stopped_at
            if eng.stopped_at == "no fields" and not eng.error:
                status, result.review = _vision_fallback(job, progress_fn, agent_browser)
                result.stopped_at = "vision fallback"
                if status not in ("confirm", "needs_input"):
                    result.error = f"No form fields found, and the browser agent stopped early ({status}). Check the form."
                return result
        result.review, result.filled, result.error = eng.review, eng.filled, eng.error
        if not llm_up:
            result.error = result.error or "The LLM gateway was down, so open-text answers weren't written."
    except Exception as exc:
        logger.error("generic fill failed: %s", exc)
        result.error = f"The form fill failed: {exc}"
    return result


def fill_application(job: dict, progress_fn: Optional[Callable[[str], None]] = None,
                     tab: Optional[str] = None, wait_fn: Optional[Callable[[str], bool]] = None) -> FillResult:
    """wait_fn(kind) → True once the user cleared a wall (CAPTCHA / login / agreement) that
    showed up mid-form; the form engine then carries on."""
    url = (job or {}).get("url", "")
    if not url:
        return FillResult(error="That job has no application URL.")
    seen = identify(url) or ""
    html = ""
    if not seen:                 # company career page — the ATS may be embedded in it
        try:
            import aria.web.agent_browser as agent_browser
            agent_browser.navigate(url, settle_secs=2.5, tab=tab)
            html = agent_browser.run(lambda page: page.content(), tab=tab)
            seen = identify(url, html) or ""
        except Exception as exc:
            logger.warning("couldn't load %s to detect the ATS: %s", url, exc)
    if detect(url, html) == "greenhouse":
        from aria.features.ats import greenhouse
        gh_tab = tab or greenhouse.TAB
        gh = greenhouse.fill(job, progress_fn, tab=gh_tab)
        return FillResult(ats="greenhouse", identified=seen, form_url=gh.form_url, title=gh.title,
                          company=gh.company, description=gh.description, review=gh.review,
                          filled=gh.filled, tab=gh_tab, error=gh.error)
    if progress_fn and seen:
        progress_fn(f"That's a {seen.title()} form — filling it field by field.")
    result = _generic(job, progress_fn, tab, wait_fn) if wait_fn is not None else _generic(job, progress_fn, tab)
    result.identified = seen
    return result
