"""features/ats — fill a job application form and stop for review. NEVER submits.

fill_application(job, progress_fn, tab) → FillResult   (tab: the job's own agent_browser tab)
  detect() names an ATS with a deterministic adapter (Greenhouse) → that adapter.
  Otherwise → the generic see-think-act loop (computer_use.run_loop) under the same
  never-submit rule; its review lists the required fields still empty.

job: a job-sheet row dict — needs "url"; "company"/"role"/"fields" improve answers.
"""
from __future__ import annotations

import html as _html
import logging
import re
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

    def count(self, source: str) -> int:
        return sum(1 for r in self.review if r["source"] == source)


def _generic(job: dict, progress_fn, tab: Optional[str] = None) -> FillResult:
    """computer_use / dom_browser call agent_browser with no tab, so the loop runs inside
    use_tab(tab), keeping the caller's pause/stop gate."""
    import aria.web.agent_browser as agent_browser
    import aria.web.computer_use as computer_use
    import aria.web.dom_browser as dom_browser

    result = FillResult(ats="generic", company=job.get("company", ""), title=job.get("role", ""), tab=tab)
    try:
        with agent_browser.use_tab(tab, gate=agent_browser.current_gate()):
            agent_browser.navigate(job["url"], settle_secs=3.0)
            if not agent_browser.run(lambda page: page.evaluate(_HAS_FIELDS_JS)):
                html, here = agent_browser.run(lambda page: (page.content(), page.url))
                form = apply_form_link(html, here)
                if form:
                    if progress_fn:
                        progress_fn("That's the job posting. Opening its application form.")
                    agent_browser.navigate(form, settle_secs=3.0)
            context = answer_bank.fillable_answers(answer_bank.load(), job)   # never always-ask answers
            goal = (
                f"Fill out the job application form for {job.get('role', 'this role')} at "
                f"{job.get('company', 'this company')} using the applicant answers in Context. "
                f"Leave any field you have no answer for empty. {NEVER_SUBMIT}"
            )
            status, _data = computer_use.run_loop(goal, context, max_steps=30, progress_fn=progress_fn)
            result.form_url = agent_browser.run(lambda page: page.url)
            empty = dom_browser.find_empty_required_fields()
        result.review = [
            {"name": e.get("selector", ""), "label": e.get("label", ""), "kind": e.get("field_type", ""),
             "value": "", "source": "needs_me", "required": True, "note": "still empty"}
            for e in empty
        ]
        if status not in ("confirm", "needs_input"):
            result.error = f"The browser agent stopped early ({status}). Check the form."
    except Exception as exc:
        logger.error("generic fill failed: %s", exc)
        result.error = f"The form fill failed: {exc}"
    return result


def fill_application(job: dict, progress_fn: Optional[Callable[[str], None]] = None,
                     tab: Optional[str] = None) -> FillResult:
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
        progress_fn(f"That's a {seen.title()} form — I don't have a dedicated filler yet, so I'll do it step by step.")
    result = _generic(job, progress_fn, tab)
    result.identified = seen
    return result
