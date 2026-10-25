"""features/ats — fill a job application form and stop for review. NEVER submits.

fill_application(job, progress_fn, tab) → FillResult   (tab: the job's own agent_browser tab)
  A Greenhouse job URL → the Greenhouse filler straight away.
  Otherwise → apply_target.resolve_apply_target (posting → Apply link / popup / modal /
  embedded iframe → the settled form, its ATS named) → adapters.pick(ats): login → fill →
  verify. The generic adapter is form_engine: scan / classify / fill / re-scan / verify /
  next page, in code (the LLM only writes open-text answers). Only when the engine finds no
  fields at all does the see-think-act vision loop (computer_use.run_loop) try, under the
  same never-submit rule. Unknown ATS, unmapped and failed fields → corrections rows.

job: a job-sheet row dict — needs "url"; "company"/"role"/"fields" improve answers.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import aria.features.answer_bank as answer_bank
from aria.features.ats.detect import detect, identify

logger = logging.getLogger(__name__)

NEVER_SUBMIT = (
    "Do NOT click Submit, Apply, Send or any button that sends the application — "
    "return 'confirm' as soon as every field you can fill is filled."
)


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


def _from_report(adapter: str, seen: str, report, tab: Optional[str]) -> FillResult:
    return FillResult(ats=adapter, identified=seen, form_url=report.form_url, title=report.title,
                      company=report.company, description=report.description, review=report.rows,
                      filled=report.filled, tab=tab, error=report.error, pages=report.pages,
                      stopped_at=report.stopped_at)


def log_corrections(job: dict, ats_name: str, rows: list, error: str, run: Optional[Callable]) -> None:
    """Unmapped (needs_me) and failed fields, and a fill error, each as a correction row.
    One page snapshot per kind, on its first row. Never raises."""
    from aria.features.ats import corrections
    for kind, source in (("unmapped_field", "needs_me"), ("failed_field", "failed")):
        for n, r in enumerate(x for x in rows if x.get("source") == source):
            detail = (r.get("label") or r.get("name") or "")[:200] + (f" — {r['note']}" if r.get("note") else "")
            corrections.record(job, kind, "fill", detail, ats=ats_name, run=run if n == 0 else None)
    if error:
        corrections.record(job, "fill_error", "fill", error, ats=ats_name, run=run)


BLANK_PAGE = "blank page"


def _landed(t) -> bool:
    """The tab left about:blank after navigate (a page we can read is showing)."""
    try:
        url = t.run(lambda page: page.url)
    except Exception:
        return True                     # can't tell — let the resolver look
    return url is None or str(url).startswith("http")


def _generic(job: dict, progress_fn, tab: Optional[str] = None,
             wait_fn: Optional[Callable[[str], bool]] = None, here: str = "") -> FillResult:
    """posting → resolve_apply_target (link / popup / modal / iframe, redirects settled, ATS
    named) → that ATS's adapter: login → fill → verify. Everything runs inside use_tab(tab),
    keeping the caller's pause/stop gate. here = the page the tab already shows: resolving
    starts there (after a sign-in, the form you're on), not back at the posting."""
    import aria.web.agent_browser as agent_browser
    from aria.features.ats import adapters
    from aria.features.ats.apply_target import Tab, resolve_apply_target

    result = FillResult(ats="generic", company=job.get("company", ""), title=job.get("role", ""), tab=tab)
    try:
        with agent_browser.use_tab(tab, gate=agent_browser.current_gate()):
            t = Tab.of(tab)
            if not here:
                t.navigate(job["url"], settle_secs=3.0)
                if not _landed(t):
                    result.form_url = job["url"]
                    result.error = "The posting never loaded, the tab stayed blank. Say retry and I'll open it again."
                    result.stopped_at = BLANK_PAGE
                    return result
            target = resolve_apply_target(t, job, progress_fn, wait_fn)
            if target.stopped:                  # closed / already applied / email only / … — no form to fill
                from aria.features.ats import situations
                result.identified, result.form_url = target.identified, target.url
                result.error = situations.describe(target.stopped, target.stopped_why)
                result.stopped_at = target.stopped
                return result
            adapter = adapters.pick(target.ats)
            ctx = adapters.ApplyContext(job=job, target=target, progress_fn=progress_fn or (lambda _m: None),
                                        wait_fn=wait_fn)
            if not adapter.login(t, ctx):
                result.identified, result.form_url = target.identified, target.url
                result.error = "I couldn't get past its sign-in page, so the form is still empty."
                return result
            report = adapter.fill(t, ctx)
            readback = adapter.verify(t, ctx)
            if readback.mismatches:
                logger.info("read-back: %d field(s) didn't stick: %s", len(readback.mismatches), readback.mismatches)
            log_corrections(job, target.identified or adapter.name, report.rows, report.error, t.run)
        result = _from_report(adapter.name, target.identified, report, tab)
        result.company = result.company or job.get("company", "")
        result.title = result.title or job.get("role", "")
    except Exception as exc:
        logger.error("generic fill failed: %s", exc)
        result.error = f"The form fill failed: {exc}"
    return result


def fill_application(job: dict, progress_fn: Optional[Callable[[str], None]] = None,
                     tab: Optional[str] = None, wait_fn: Optional[Callable[[str], bool]] = None,
                     here: str = "") -> FillResult:
    """wait_fn(kind) → True once the user cleared a wall (CAPTCHA / login / agreement) that
    showed up mid-form; the form engine then carries on. here = the URL the job's tab already
    shows (a resumed fill): it's filled where it is, never reloaded from the posting."""
    url = (job or {}).get("url", "")
    if not url:
        return FillResult(error="That job has no application URL.")
    seen = identify(here or url) or ""
    html = ""
    if not seen:                 # company career page — the ATS may be embedded in it
        try:
            import aria.web.agent_browser as agent_browser
            if not here:
                agent_browser.navigate(url, settle_secs=2.5, tab=tab)
            html = agent_browser.run(lambda page: page.content(), tab=tab)
            seen = identify(here or url, html) or ""
        except Exception as exc:
            logger.warning("couldn't load %s to detect the ATS: %s", url, exc)
    if detect(here or url, html) == "greenhouse":
        from aria.features.ats import greenhouse
        gh_tab = tab or greenhouse.TAB
        gh = greenhouse.fill(job, progress_fn, tab=gh_tab)
        try:
            import aria.web.agent_browser as agent_browser
            log_corrections(job, "greenhouse", gh.review, gh.error, lambda fn: agent_browser.run(fn, tab=gh_tab))
        except Exception as exc:
            logger.debug("greenhouse corrections not logged: %s", exc)
        return FillResult(ats="greenhouse", identified=seen, form_url=gh.form_url, title=gh.title,
                          company=gh.company, description=gh.description, review=gh.review,
                          filled=gh.filled, tab=gh_tab, error=gh.error)
    if progress_fn and seen:
        progress_fn(f"That's a {seen.title()} form — filling it field by field.")
    resume = {"here": here} if here else {}
    result = (_generic(job, progress_fn, tab, wait_fn, **resume) if wait_fn is not None
              else _generic(job, progress_fn, tab, **resume))
    result.identified = result.identified or seen
    return result
