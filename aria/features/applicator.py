"""applicator.py — the one entry point for job applications: fill → stop for review → you decide.

run_application(job, progress_fn, voice_ask_fn) → spoken one-liner
  1. fill     Greenhouse adapter / LinkedIn Easy Apply / generic browser loop — never submits
  2. review   full-page screenshot + markdown summary in ~/.aria/reviews/ (needs-you first);
              the form stays open in its own browser tab
  3. record   sheet Status "In review" (+ note in Updates), aria.db application_runs,
              flight log job entry with the LLM cost of this job, keyed by sheet row

Then, only from the user's own words (handle_followup, a dispatcher pre-handler —
never an agent tool, so no LLM can decide to submit):
  "submit it"         → learn the answers the user corrected in the form, click the form's
                        submit button, verify the confirmation → tracker + sheet "Applied"
  "I submitted it"    → the user submitted by hand → tracker + sheet "Applied"
  "skip this job"     → sheet "Skipped"
Failures → sheet "Needs human" + the reason in Updates.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import aria.features.answer_bank as answer_bank
import aria.features.tracker as tracker
import aria.state.db as db
import aria.state.memory as memory
from aria.dispatch.parse import normalize
from aria.features import ats
from aria.features.ats import greenhouse
from aria.features.ats import submit as ats_submit
from aria.integrations import sheets
from aria.llm import context as llm_context
from aria.llm import meter
from aria.observability import flight_recorder, run_log

logger = logging.getLogger(__name__)

PENDING_KEY = "pending_application"
REVIEWS_DIR = Path.home() / ".aria" / "reviews"

_ATS_NAME = {"greenhouse": "Greenhouse", "linkedin": "LinkedIn Easy Apply", "generic": "Application"}
_SECTIONS = [
    ("needs_me", "Needs you"),
    ("llm_draft", "Drafted by the LLM: check the wording"),
    ("llm_guess", "Guessed by the LLM: check these"),
    ("bank", "From your answer bank"),
    ("skipped", "Left empty (optional)"),
]
# Corrections to these aren't reusable answers (files, company-specific essays, consents).
_NO_LEARN_KINDS = {"file", "textarea", "checkbox"}

_SUBMIT_RE = re.compile(
    r"(?:yes |ok |okay )?(?:go ahead and )?(?:submit|send)(?: it| that| this| the application| my application|"
    r" the form| this application)?(?: now)?"
)
_BY_HAND_RE = re.compile(
    r"i(?: have|'ve)? (?:already |just )?(?:submitted|sent)(?: it| the application| my application)?|"
    r"i applied(?: to it)?|mark (?:it|this|the job) (?:as )?applied"
)
_SKIP_RE = re.compile(
    r"skip (?:this |the )?(?:job|application|role|one)|don'?t apply(?: to (?:it|this(?: one)?))?|"
    r"not applying(?: to (?:it|this(?: one)?))?|pass on (?:it|this(?: one| job)?)"
)


# ---------------------------------------------------------------------------
# Pending review (persisted, so a restart doesn't forget what's open)
# ---------------------------------------------------------------------------

def pending() -> Optional[dict]:
    p = memory.get_persistent(PENDING_KEY)
    return p if isinstance(p, dict) and p.get("job") else None


def _set_pending(value: Optional[dict]) -> None:
    memory.set_persistent(PENDING_KEY, value)


def _names(job: dict, result=None) -> tuple:
    company = job.get("company") or (result.company if result else "") or "this company"
    role = job.get("role") or (result.title if result else "") or "this role"
    return company, role


# ---------------------------------------------------------------------------
# Records: sheet, aria.db, flight log
# ---------------------------------------------------------------------------

# The pipeline run in progress (one at a time): steps and its outcome go to run_log.
_RUN: Optional[run_log.Run] = None
_RUN_STATUS: dict = {}


def _step(name: str):
    return _RUN.step(name) if _RUN is not None else contextlib.nullcontext(run_log._Step())


def _sheet_update(job: dict, status: str, note: str) -> str:
    with _step("update sheet") as s:
        err = _sheet_update_now(job, status, note)
        s.detail = f"Status → {status}" + (f" — not updated: {err}" if err else "")
        s.status = "error" if err else "ok"
    return err


def _sheet_update_now(job: dict, status: str, note: str) -> str:
    """Update the job's sheet row (if it came from the sheet). Returns an error string or ""."""
    if not job.get("row"):
        return ""
    try:
        sheets.set_status(int(job["row"]), status, note,
                          company=job.get("company") or None, role=job.get("role") or None)
        return ""
    except sheets.SheetError as exc:
        logger.warning("applicator: sheet update failed: %s", exc)
        return str(exc)
    except Exception as exc:
        logger.error("applicator: sheet update failed: %s", exc)
        return "I couldn't update the sheet."


def _finish(job: dict, ats_name: str, stage: str, before: dict, ok: bool = True,
            note: str = "", needs_me: int = 0, review_path: str = "") -> None:
    """Cost of this step → flight log (keyed by sheet row) + aria.db application_runs."""
    usage = meter.since(before)
    company, role = _names(job)
    _RUN_STATUS["status"] = stage if ok else "error"
    flight_recorder.record_job(job.get("row"), company, role, stage, ats_name, usage, ok, note)
    conn = None
    try:
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO application_runs (sheet_row, company, role, url, ats, stage, needs_me, "
            "cost_usd, review_path, note) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job.get("row"), company, role, job.get("url", ""), ats_name, stage, needs_me,
             float(usage.get("cost_usd", 0.0)), review_path, note[:500]),
        )
        conn.commit()
    except Exception as exc:
        logger.error("applicator: run record failed: %s", exc)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Review artifacts
# ---------------------------------------------------------------------------

def _slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:limit] or "job"


def build_summary(job: dict, result, company: str, role: str, screenshot: str = "") -> str:
    """Markdown review: job, then every field grouped by source, needs-you first."""
    ats_name = _ATS_NAME.get(result.ats, result.ats or "Application")
    lines = [f"# Review: {role} at {company}", ""]
    lines.append(f"- **Form:** {ats_name}, {result.form_url or job.get('url', '')}")
    if job.get("row"):
        lines.append(f"- **Sheet row:** {job['row']}")
    if screenshot:
        lines.append(f"- **Screenshot:** {screenshot}")
    lines.append("- **Status:** filled, NOT submitted. Say \"submit it\", or submit in the browser and "
                 "say \"I submitted it\". Say \"skip this job\" to skip it.")
    if result.error:
        lines.append(f"- **Problem:** {result.error}")
    lines.append("")
    for source, title in _SECTIONS:
        rows = [r for r in result.review if r["source"] == source]
        if not rows:
            continue
        lines.append(f"## {title} ({len(rows)})")
        lines.append("")
        if source == "llm_draft":
            for r in rows:
                lines.append(f"### {r['label']}")
                lines.append("")
                lines.extend(f"> {ln}" if ln else ">" for ln in (r["value"] or "").splitlines())
                lines.append("")
            continue
        lines.append("| Field | Value | Required | Note |")
        lines.append("|---|---|---|---|")
        for r in rows:
            cell = lambda t: str(t or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {cell(r['label'])} | {cell(r['value'])} | {'yes' if r['required'] else ''} | {cell(r['note'])} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_review(job: dict, result, company: str, role: str) -> tuple:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(company)}_{_slug(role, 30)}_{date.today().isoformat()}"
    png = REVIEWS_DIR / f"{stem}.png"
    md = REVIEWS_DIR / f"{stem}.md"
    shot = ""
    try:
        import aria.web.agent_browser as agent_browser
        agent_browser.run(lambda page: page.screenshot(path=str(png), full_page=True), tab=result.tab)
        shot = str(png)
    except Exception as exc:
        logger.warning("applicator: screenshot failed: %s", exc)
    md.write_text(build_summary(job, result, company, role, shot))
    _capture_unlabeled(result, REVIEWS_DIR / f"{stem}.fields.json", stem)
    return str(md), shot


_MACHINE_NAME = re.compile(r"^[\w-]+\[|^[\w-]*\d{3,}[\w-]*$|^(?:field|input|question)[_-]?\d+$", re.I)

_FIELD_INFO_JS = """e => {
  const box = e.closest('li, fieldset, .application-question, .field, .form-group, label') || e.parentElement;
  const own = (e.value || '') + (e.innerText || '');
  let near = ((box && box.innerText) || '').replace(own, '').replace(/\\s+/g, ' ').trim();
  return {tag: e.tagName.toLowerCase(), type: e.type || '', placeholder: e.placeholder || '',
          aria: e.getAttribute('aria-label') || '', nearby: near.slice(0, 300)};
}"""


def is_unlabeled(label: str) -> bool:
    """A field whose 'label' is a machine name (cards[uuid][field0], q_12345)."""
    return bool(_MACHINE_NAME.match((label or "").strip()))


def _capture_unlabeled(result, out: Path, stem: str) -> None:
    """For needs_me fields with no human label: field type, nearby text and a screenshot
    crop, saved next to the review so the user can tell what the field is. Never raises."""
    rows = [r for r in result.review if r.get("source") == "needs_me" and is_unlabeled(r.get("label") or r.get("name"))]
    if not rows:
        return
    import json as _json

    import aria.web.agent_browser as agent_browser

    def grab(page) -> list:
        found = []
        for i, r in enumerate(rows):
            entry = {"name": r.get("name", ""), "label": r.get("label", ""), "kind": r.get("kind", "")}
            try:
                name = (r.get("name") or "").replace('"', '\\"')
                el = page.query_selector(f'[name="{name}"]') or page.query_selector(r.get("name") or "x")
                if el is not None:
                    entry.update(el.evaluate(_FIELD_INFO_JS))
                    box = el.evaluate_handle("e => e.closest('li, fieldset, .application-question, .field, "
                                             ".form-group, label') || e.parentElement").as_element() or el
                    crop = REVIEWS_DIR / f"{stem}_field{i}.png"
                    box.screenshot(path=str(crop))
                    entry["crop"] = str(crop)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}"
            found.append(entry)
        return found

    try:
        fields = agent_browser.run(grab, tab=result.tab)
        out.write_text(_json.dumps(fields, indent=2))
    except Exception as exc:
        logger.warning("applicator: unlabeled-field capture failed: %s", exc)


def spoken_result(result, company: str, role: str) -> str:
    ats_name = _ATS_NAME.get(result.ats, "Application")
    n_me, drafts = result.count("needs_me"), result.count("llm_draft")
    need = ("Nothing needs you." if n_me == 0
            else "1 field needs you." if n_me == 1 else f"{n_me} fields need you.")
    out = f"{ats_name} form filled for {role} at {company}. {need}"
    if drafts:
        out += f" I drafted {drafts} answer{'s' if drafts != 1 else ''} — check the wording."
    if result.error:
        out += " Something went wrong partway, so check it closely."
    return out + " Say submit it when you're ready."


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

def _is_linkedin(url: str) -> bool:
    return (urlparse(url).hostname or "").endswith("linkedin.com")


def _fill_linkedin(job: dict, voice_ask_fn) -> "ats.FillResult":
    import aria.features.linkedin_applicator as linkedin_applicator
    import aria.web.agent_browser as agent_browser
    import aria.web.dom_browser as dom_browser

    result = ats.FillResult(ats="linkedin", identified="linkedin", form_url=job["url"])
    agent_browser.navigate(job["url"], settle_secs=5.0)
    context = answer_bank.fillable_answers(answer_bank.load(), job)   # never always-ask answers
    context["resume_path"] = greenhouse.resume_path() or ""
    status, data = linkedin_applicator.run_linkedin_application(job, context, voice_ask_fn or (lambda _f: ""))
    if status != "confirm":
        reason = (data or {}).get("reason", status) if isinstance(data, dict) else status
        result.error = f"LinkedIn Easy Apply stopped ({reason})."
        return result
    result.review = [
        {"name": e.get("selector", ""), "label": e.get("label", ""), "kind": e.get("field_type", ""),
         "value": "", "source": "needs_me", "required": True, "note": "still empty"}
        for e in dom_browser.find_empty_required_fields()
    ]
    return result


def _run_logged(run: run_log.Run, job: dict, fn: Callable[[], str]) -> str:
    """Run fn as pipeline run `run` for `job`: steps + outcome logged, LLM calls labelled."""
    global _RUN
    company, role = _names(job)
    run.job(company, role, job.get("row"))
    prev, _RUN = _RUN, run
    _RUN_STATUS.clear()
    try:
        with llm_context.job(run_log.job_label(company, role, job.get("row")), run.id):
            out = fn()
    finally:
        _RUN = prev
    run.finish(_RUN_STATUS.get("status", "no change"), out)
    return out


def run_application(
    job: dict,
    progress_fn: Optional[Callable[[str], None]] = None,
    voice_ask_fn: Optional[Callable[[str], str]] = None,
    run: Optional[run_log.Run] = None,
) -> str:
    """Fill the application for a job (sheet row dict or {"url": ...}) and stop for review.
    Returns one spoken sentence or two. Never submits, never raises. `run` continues a
    pipeline run the caller started (e.g. after reading the sheet)."""
    job = dict(job or {})
    run = run or run_log.start("apply_job")
    return _run_logged(run, job, lambda: _run_application(job, progress_fn, voice_ask_fn))


def _run_application(
    job: dict,
    progress_fn: Optional[Callable[[str], None]],
    voice_ask_fn: Optional[Callable[[str], str]],
) -> str:
    say = progress_fn or (lambda _msg: None)
    before = meter.snapshot()
    job = dict(job or {})
    url = (job.get("url") or "").strip()
    company, role = _names(job)
    try:
        prior = pending()
        if prior and prior["job"].get("url") != url:
            pc, pr = _names(prior["job"])
            say(f"Closing the review for {pr} at {pc} — it stays In review in the sheet.")

        if not url:
            note = f"no application URL ({job.get('apply_link') or 'empty'})"
            _sheet_update(job, sheets.STATUS_NEEDS_HUMAN, note)
            _finish(job, "", "error", before, ok=False, note=note)
            return f"{role} at {company} has no application link I can open, so I marked it Needs human."

        with _step("fill form") as step:
            result = _fill_linkedin(job, voice_ask_fn) if _is_linkedin(url) else ats.fill_application(job, say)
            n_need = result.count("needs_me") if result.review else 0
            step.detail = (f"{_ATS_NAME.get(result.ats, result.ats or 'form')}: {len(result.filled or {})} filled, "
                           f"{n_need} need you" + (f" — {result.error}" if result.error else ""))
            step.status = "error" if result.error and not result.review else "ok"
        if not job.get("company") and result.company:
            job["company"] = result.company
        if not job.get("role") and result.title:
            job["role"] = result.title
        company, role = _names(job, result)
        if _RUN is not None and (company, role) != _names({}):
            _RUN.job(company, role, job.get("row"))

        if result.error and not result.review:
            _sheet_update(job, sheets.STATUS_NEEDS_HUMAN, result.error)
            _finish(job, result.ats, "error", before, ok=False, note=result.error)
            return f"I couldn't fill the application for {role} at {company}. {result.error}"

        say("Saving a screenshot and the review summary.")
        md, shot = _write_review(job, result, company, role)
        _set_pending({
            "job": {k: job.get(k) for k in ("row", "company", "role", "url", "location")},
            "ats": result.ats, "tab": result.tab, "form_url": result.form_url,
            "filled": result.filled, "review_path": md, "screenshot": shot, "started": time.time(),
        })
        n_me = result.count("needs_me")
        sheet_err = _sheet_update(
            job, sheets.STATUS_IN_REVIEW,
            f"Aria filled the {_ATS_NAME.get(result.ats, 'application')} form, "
            f"{n_me} field{'s' if n_me != 1 else ''} need you; not submitted",
        )
        _finish(job, result.ats, "filled", before, ok=not result.error,
                note=result.error or sheet_err, needs_me=n_me, review_path=md)
        spoken = spoken_result(result, company, role)
        return spoken + (f" (Sheet not updated: {sheet_err})" if sheet_err else "")
    except Exception as exc:
        logger.error("run_application failed: %s", exc)
        _sheet_update(job, sheets.STATUS_NEEDS_HUMAN, f"Aria error: {exc}")
        _finish(job, "", "error", before, ok=False, note=str(exc))
        return f"Something went wrong applying to {role} at {company}. I marked it Needs human."


# ---------------------------------------------------------------------------
# After review — only reachable from the user's own words (handle_followup)
# ---------------------------------------------------------------------------

def _same_form(current: str, expected: str) -> bool:
    a, b = urlparse(current or ""), urlparse(expected or "")
    return bool(a.hostname) and (a.hostname, a.path.rstrip("/")) == (b.hostname, b.path.rstrip("/"))


def learn_corrections(filled: dict, now: dict) -> int:
    """Answers the user changed in the form since Aria filled it → answer bank (source "review")."""
    count = 0
    for name, f in (filled or {}).items():
        if f.get("kind") in _NO_LEARN_KINDS:
            continue
        current = (now.get(name) or "").strip()
        if not current or current == (f.get("value") or "").strip():
            continue
        if re.fullmatch(r"\+\d{1,4}", current):        # country picker shows only the dial code
            continue
        if answer_bank.learn(f.get("label", ""), current, source="review"):
            count += 1
    return count


def _logged_followup(trigger: str):
    """Log a review follow-up (submit / submitted by hand / skip) as its own short run."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper() -> str:
            p = pending()
            if not p:
                return fn()
            return _run_logged(run_log.start(trigger), dict(p["job"]), fn)
        return wrapper
    return deco


@_logged_followup("submit")
def submit_pending() -> str:
    p = pending()
    if not p:
        return "There's no application waiting for review."
    job, tab, ats_name = p["job"], p.get("tab"), p.get("ats", "")
    company, role = _names(job)
    before = meter.snapshot()
    try:
        import aria.web.agent_browser as agent_browser
        current = agent_browser.run(lambda page: page.url, tab=tab)
        if not _same_form(current, p.get("form_url", "")):
            return ("That form isn't open anymore, so I won't submit anything. If you submitted it "
                    "yourself, say I submitted it.")

        learned = 0
        if ats_name == "greenhouse":
            try:
                learned = learn_corrections(p.get("filled") or {}, greenhouse.read_values(list(p.get("filled") or {})))
            except Exception as exc:
                logger.warning("applicator: couldn't read corrections: %s", exc)

        outcome = ats_submit.submit(tab)
        saved = f" I saved {learned} answer{'s' if learned != 1 else ''} you changed." if learned else ""
        if outcome.status == "submitted":
            tracker.log_application(company, role, ats_name, job.get("url") or p.get("form_url", ""), "applied")
            sheet_err = _sheet_update(job, sheets.STATUS_APPLIED, "submitted by Aria after your review")
            _finish(job, ats_name, "submitted", before, review_path=p.get("review_path", ""))
            _set_pending(None)
            msg = f"Submitted {role} at {company}. Logged it and marked the sheet Applied.{saved}"
            return msg + (f" (Sheet not updated: {sheet_err})" if sheet_err else "")
        if outcome.status == "invalid":
            fields = ", ".join(outcome.missing[:4])
            return f"The form didn't accept it yet. Still needed: {fields}.{saved}"
        if outcome.status == "no_button":
            return ("I can't find the submit button, so nothing was sent. If you submitted it yourself, "
                    "say I submitted it.")
        note = "clicked submit but saw no confirmation, check your email"
        _sheet_update(job, sheets.STATUS_NEEDS_HUMAN, note)
        _finish(job, ats_name, "submit_unconfirmed", before, ok=False, note=note)
        return ("I clicked submit but didn't see a confirmation. I marked it Needs human, so check "
                f"the browser and your email.{saved}")
    except Exception as exc:
        logger.error("submit_pending failed: %s", exc)
        return "Something went wrong submitting. Check the browser before trying again."


@_logged_followup("submitted by hand")
def mark_submitted_by_hand() -> str:
    p = pending()
    if not p:
        return "There's no application waiting for review."
    job, ats_name = p["job"], p.get("ats", "")
    company, role = _names(job)
    before = meter.snapshot()
    seen = ats_submit.page_confirms(p.get("tab"))
    tracker.log_application(company, role, ats_name, job.get("url") or p.get("form_url", ""), "applied")
    sheet_err = _sheet_update(job, sheets.STATUS_APPLIED,
                              "you submitted it" + (", confirmation seen" if seen else ""))
    _finish(job, ats_name, "submitted", before, note="by hand", review_path=p.get("review_path", ""))
    _set_pending(None)
    return f"Marked {role} at {company} as applied." + (f" (Sheet not updated: {sheet_err})" if sheet_err else "")


@_logged_followup("skip")
def skip_pending() -> str:
    p = pending()
    if not p:
        return "There's no application waiting for review."
    job = p["job"]
    company, role = _names(job)
    before = meter.snapshot()
    sheet_err = _sheet_update(job, sheets.STATUS_SKIPPED, "skipped after review")
    _finish(job, p.get("ats", ""), "skipped", before, review_path=p.get("review_path", ""))
    _set_pending(None)
    return f"Skipped {role} at {company}." + (f" (Sheet not updated: {sheet_err})" if sheet_err else "")


def handle_followup(text: str) -> Optional[str]:
    """Dispatcher pre-handler: "submit it" / "I submitted it" / "skip this job" while a
    review is open. Anything else (or nothing pending) → None, so routing continues."""
    if not pending():
        return None
    q = normalize(text)
    if _SUBMIT_RE.fullmatch(q):
        return submit_pending()
    if _BY_HAND_RE.fullmatch(q):
        return mark_submitted_by_hand()
    if _SKIP_RE.fullmatch(q):
        return skip_pending()
    return None
