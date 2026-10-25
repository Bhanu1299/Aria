"""applicator.py — the one entry point for job applications: fill → stop for review → you decide.

run_application(job, progress_fn, voice_ask_fn, entry=None, gate=None) → spoken one-liner
  0. tab      every application gets its own visible tab (applying.tabs entry; its id is
              the agent_browser tab name). Aria never closes it.
  1. fill     Greenhouse adapter / LinkedIn Easy Apply / the generic form engine — never submits
  2. review   full-page screenshot + markdown summary in ~/.aria/reviews/ (summary: filled
              count, UNKNOWN list, FAILED list) + <stem>.log.json — every field with its
              category, value, source and status
  3. record   tab entry → filled; sheet Status "In review" (+ note in Updates), aria.db
              application_runs, flight log job entry with this job's LLM cost, by sheet row

Then, only from the user's own words (applying.commands, a dispatcher pre-handler —
never an agent tool, so no LLM can decide to submit), on one tab entry:
  submit_entry           learn the answers the user corrected, click the form's submit,
                         verify the confirmation → tab "submitted" (by aria) + tracker
  mark_submitted_by_hand the user submitted → tab "submitted" (by user) + tracker
  mark_applied           sheet "Applied" + dated note — only on "applications are done"
  skip_entry             sheet "Skipped"; the tab stays open
Fill failures → sheet "Needs human" + the reason in Updates.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import re
import threading
from datetime import date
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import aria.features.answer_bank as answer_bank
import aria.features.tracker as tracker
import aria.state.db as db
from aria.features import ats
from aria.features.applying import control, tabs
from aria.features.ats import greenhouse
from aria.features.ats import submit as ats_submit
from aria.integrations import sheets
from aria.llm import context as llm_context
from aria.llm import meter
from aria.observability import flight_recorder, run_log

logger = logging.getLogger(__name__)

REVIEWS_DIR = Path.home() / ".aria" / "reviews"

_ATS_NAME = {"greenhouse": "Greenhouse", "linkedin": "LinkedIn Easy Apply", "generic": "Application"}
_SECTIONS = [
    ("needs_me", "Unknown — missing from your answer bank"),
    ("failed", "Failed — Aria couldn't set these"),
    ("llm_draft", "Written by the LLM from your resume"),
    ("reused", "Reused LLM answers"),
    ("llm_guess", "Picked by the LLM from your resume"),
    ("bank", "From your answer bank"),
    ("skipped", "Left empty (optional)"),
]
_FILLED = ("bank", "llm_draft", "reused", "llm_guess")
# Corrections to these aren't reusable answers (files, company-specific essays, consents).
_NO_LEARN_KINDS = {"file", "textarea", "checkbox"}

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def _names(job: dict, result=None) -> tuple:
    company = job.get("company") or (result.company if result else "") or "this company"
    role = job.get("role") or (result.title if result else "") or "this role"
    return company, role


# ---------------------------------------------------------------------------
# Records: sheet, aria.db, flight log
# ---------------------------------------------------------------------------

# The pipeline run in progress ON THIS THREAD: the application run and the user's voice
# follow-ups (submit / skip) run on different threads at once, so each keeps its own.
_local = threading.local()


def _current_run() -> Optional[run_log.Run]:
    return getattr(_local, "run", None)


def _set_run(run: Optional[run_log.Run]) -> None:
    _local.run = run


def _run_status() -> dict:
    if not hasattr(_local, "status"):
        _local.status = {}
    return _local.status


def _step(name: str):
    run = _current_run()
    return run.step(name) if run is not None else contextlib.nullcontext(run_log._Step())


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
        sheets.set_status(int(job["row"]), status, note, company=job.get("company") or None,
                          role=job.get("role") or None, job_id=job.get("job_id") or None)
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
    _run_status()["status"] = stage if ok else "error"
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
    filled = sum(result.count(src) for src in _FILLED)
    unknown = [r["label"] for r in result.review if r["source"] == "needs_me"]
    failed = [r["label"] for r in result.review if r["source"] == "failed"]
    lines.append(f"- **Summary:** {filled} filled · {len(unknown)} unknown · {len(failed)} failed")
    lines.append("")
    agreements = [r for r in result.review if r["source"] == "needs_me" and r.get("category") == "consent"]
    if agreements:
        lines.append(f"## Agreements — yours to tick before you submit; Aria never ticks them ({len(agreements)})")
        lines.append("")
        lines.extend(f"- {r['label']}" for r in agreements)
        lines.append("")
    for source, title in _SECTIONS:
        rows = [r for r in result.review if r["source"] == source
                and not (source == "needs_me" and r.get("category") == "consent")]
        if not rows:
            continue
        lines.append(f"## {title} ({len(rows)})")
        lines.append("")
        if source in ("llm_draft", "reused"):
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
    write_log(REVIEWS_DIR / f"{stem}.log.json", job, result, company, role)
    _capture_unlabeled(result, REVIEWS_DIR / f"{stem}.fields.json", stem)
    return str(md), shot


_STATUS = {"needs_me": "unknown", "failed": "failed", "consent": "consent", "skipped": "skipped"}


def build_log(job: dict, result, company: str, role: str) -> dict:
    """The per-application log: every field with its type, value, source and status."""
    return {
        "company": company, "role": role, "job_url": job.get("url", ""), "form_url": result.form_url,
        "sheet_row": job.get("row"), "ats": result.ats, "identified": getattr(result, "identified", ""),
        "date": date.today().isoformat(), "pages": getattr(result, "pages", 1),
        "stopped_at": getattr(result, "stopped_at", ""), "error": result.error,
        "summary": {"filled": sum(result.count(s) for s in _FILLED), "unknown": result.count("needs_me"),
                    "failed": result.count("failed"), "consent": result.count("consent")},
        "fields": [{"label": r.get("label", ""), "type": r.get("category") or "", "kind": r.get("kind", ""),
                    "value": r.get("value", ""), "source": r.get("source", ""),
                    "status": _STATUS.get(r.get("source", ""), "filled"), "required": bool(r.get("required")),
                    "note": r.get("note", ""), "page": r.get("page", 1)} for r in result.review],
    }


def write_log(path: Path, job: dict, result, company: str, role: str) -> None:
    import json as _json
    try:
        path.write_text(_json.dumps(build_log(job, result, company, role), indent=1))
    except Exception as exc:
        logger.warning("applicator: log write failed: %s", exc)


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


def _short_label(label: str) -> str:
    """A form label cut to what's worth saying: the first clause, at most 12 words."""
    text = re.sub(r"\s+", " ", label or "").strip(" *:")
    text = re.split(r"(?<=[?.])\s", text)[0].rstrip("?.")
    words = text.split()
    return " ".join(words[:12]) + (" …" if len(words) > 12 else "")


def _open_questions(result) -> str:
    """The questions left for the user, named (2026-10-25: "3 facts are missing" gave the agent
    nothing to answer "pull them out" with, so it re-ran the fill)."""
    labels = [_short_label(r.get("label") or r.get("name", "")) for r in result.review
              if r.get("source") == "needs_me"]
    labels = [l for l in labels if l] or ["one field"]
    head = "1 question is yours: " if len(labels) == 1 else f"{len(labels)} questions are yours: "
    shown = labels[:4]
    more = f", and {len(labels) - 4} more" if len(labels) > 4 else ""
    return head + "; ".join(shown) + more + "."


def spoken_result(result, company: str, role: str) -> str:
    ats_name = _ATS_NAME.get(result.ats, "Application")
    n_me, n_failed = result.count("needs_me"), result.count("failed")
    drafts = result.count("llm_draft") + result.count("reused")
    need = "Nothing needs you." if n_me == 0 else _open_questions(result)
    out = f"{ats_name} form filled for {role} at {company}. {need}"
    if n_failed:
        out += f" {n_failed} field{'s' if n_failed != 1 else ''} wouldn't take a value."
    if drafts:
        out += f" I wrote {drafts} answer{'s' if drafts != 1 else ''} from your resume."
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
    result.stopped_at = "LinkedIn Easy Apply review"     # reached the review step (no field list)
    result.review = [
        {"name": e.get("selector", ""), "label": e.get("label", ""), "kind": e.get("field_type", ""),
         "value": "", "source": "needs_me", "required": True, "note": "still empty"}
        for e in dom_browser.find_empty_required_fields()
    ]
    return result


def _run_logged(run: run_log.Run, job: dict, fn: Callable[[], str], label: bool = True) -> str:
    """Run fn as pipeline run `run` for `job`: steps + outcome logged. label=True also tags
    LLM calls with this job (the label is process-wide, so follow-ups that make no LLM
    calls pass False and never relabel a fill running on another thread)."""
    company, role = _names(job)
    run.job(company, role, job.get("row"))
    prev = _current_run()
    _set_run(run)
    _run_status().clear()
    scope = (llm_context.job(run_log.job_label(company, role, job.get("row")), run.id) if label
             else contextlib.nullcontext())
    try:
        with scope:
            out = fn()
    except control.Stopped:
        run.finish("stopped", "you said stop; the tab stays open")
        raise
    finally:
        _set_run(prev)
    run.finish(_run_status().get("status", "no change"), out)
    return out


def _tab_page(tab: str) -> str:
    """The page a job's tab already shows ("" = a fresh tab, or it's gone): the fill starts
    there instead of reloading the posting."""
    import aria.web.agent_browser as agent_browser
    try:
        if not agent_browser.has_tab(tab):
            return ""
        url = agent_browser.run(lambda page: page.url, tab=tab) or ""
    except Exception:
        return ""
    return url if url.startswith("http") else ""


def run_application(
    job: dict,
    progress_fn: Optional[Callable[[str], None]] = None,
    voice_ask_fn: Optional[Callable[[str], str]] = None,
    run: Optional[run_log.Run] = None,
    entry: Optional[dict] = None,
    gate: Optional[Callable[[], None]] = None,
    wait_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """Fill the application for a job (sheet row dict or {"url": ...}) in its own tab and
    stop for review. Returns one spoken sentence or two. Never submits, never raises —
    except control.Stopped (from `gate`, the run's pause/stop checkpoint), which leaves the
    tab open and propagates so the run ends. `entry` is the tab entry the caller already
    opened; else one is created. `run` continues a pipeline run the caller started.
    `wait_fn(kind)` waits out a wall that appears mid-form (True = cleared)."""
    job = dict(job or {})
    run = run or run_log.start("apply_job")
    return _run_logged(run, job, lambda: _run_application(job, progress_fn, voice_ask_fn, entry, gate, wait_fn))


def _run_application(
    job: dict,
    progress_fn: Optional[Callable[[str], None]],
    voice_ask_fn: Optional[Callable[[str], str]],
    entry: Optional[dict] = None,
    gate: Optional[Callable[[], None]] = None,
    wait_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    import aria.web.agent_browser as agent_browser
    say = progress_fn or (lambda _msg: None)
    before = meter.snapshot()
    job = dict(job or {})
    url = (job.get("url") or "").strip()
    company, role = _names(job)
    tab = ""
    try:
        if not url:
            note = f"no application URL ({job.get('apply_link') or 'empty'})"
            _sheet_update(job, sheets.STATUS_NEEDS_HUMAN, note)
            _finish(job, "", "error", before, ok=False, note=note)
            return f"{role} at {company} has no application link I can open, so I marked it Needs human."

        # A job that already has its tab carries on THERE — where you signed in — never in a new
        # tab back at the posting (user, 2026-10-24: a new tab asked for the login again).
        entry = entry or tabs.open_for(job) or tabs.add(job)
        tab = entry["id"]
        if entry.get("state") != tabs.FILLING:
            tabs.update(tab, state=tabs.FILLING)
        here = _tab_page(tab)
        resume = {"here": here} if here else {}
        with _step("fill form") as step, agent_browser.use_tab(tab, gate=gate):
            if _is_linkedin(url):
                result = _fill_linkedin(job, voice_ask_fn)
            elif wait_fn is not None:
                result = ats.fill_application(job, say, tab=tab, wait_fn=wait_fn, **resume)
            else:
                result = ats.fill_application(job, say, tab=tab, **resume)
            n_need = result.count("needs_me") if result.review else 0
            step.detail = (f"{_ATS_NAME.get(result.ats, result.ats or 'form')}: {len(result.filled or {})} filled, "
                           f"{n_need} need you" + (f" — {result.error}" if result.error else ""))
            step.status = "error" if result.error and not result.review else "ok"
        if not job.get("company") and result.company:
            job["company"] = result.company
        if not job.get("role") and result.title:
            job["role"] = result.title
        company, role = _names(job, result)
        if _current_run() is not None and (company, role) != _names({}):
            _current_run().job(company, role, job.get("row"))

        result.tab = tab
        tabs.update(tab, company=company, role=role, ats=result.ats, form_url=result.form_url)
        if not result.error and not result.filled and not result.review and \
                result.stopped_at in ("no fields", "", "vision fallback"):
            # Nothing filled and nothing to review: the form was never reached (a sign-in step,
            # a page still loading, a vision pass that did nothing). Never report that as filled.
            result.error = "I couldn't find the form on that page, so nothing was filled."
        if result.error and not result.review and result.stopped_at == ats.BLANK_PAGE:
            # The page never loaded: nothing is wrong with the job, so the sheet keeps its status.
            tabs.update(tab, state=tabs.OPEN, note=result.error)
            _finish(job, result.ats, "error", before, ok=False, note=result.error)
            return f"I couldn't open the application for {role} at {company}. {result.error}"
        if result.error and not result.review:
            tabs.update(tab, state=tabs.OPEN, note=result.error)
            _sheet_update(job, sheets.STATUS_NEEDS_HUMAN, result.error)
            _finish(job, result.ats, "error", before, ok=False, note=result.error)
            return f"I couldn't fill the application for {role} at {company}. {result.error} The tab stays open."

        say("Saving a screenshot and the review summary.")
        md, shot = _write_review(job, result, company, role)
        tabs.update(tab, state=tabs.FILLED, filled=result.filled, review_path=md, screenshot=shot,
                    needs_me=[r.get("label") or r.get("name", "") for r in result.review if r["source"] == "needs_me"],
                    note=result.error)
        n_me = result.count("needs_me")
        sheet_err = _sheet_update(
            job, sheets.STATUS_IN_REVIEW,
            f"Aria filled the {_ATS_NAME.get(result.ats, 'application')} form, "
            f"{n_me} unknown, {result.count('failed')} failed; not submitted",
        )
        _finish(job, result.ats, "filled", before, ok=not result.error,
                note=result.error or sheet_err, needs_me=n_me, review_path=md)
        spoken = spoken_result(result, company, role)
        return spoken + (f" (Sheet not updated: {sheet_err})" if sheet_err else "")
    except control.Stopped:
        if tab:
            tabs.update(tab, state=tabs.OPEN, note="stopped mid-fill")
        _finish(job, "", "stopped", before, ok=False, note="stopped mid-fill")
        raise
    except Exception as exc:
        logger.error("run_application failed: %s", exc)
        if tab:
            tabs.update(tab, state=tabs.OPEN, note=f"Aria error: {exc}")
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
        def wrapper(entry: dict, *args):
            return _run_logged(run_log.start(trigger), _job_of(entry), lambda: fn(entry, *args), label=False)
        return wrapper
    return deco


def _job_of(entry: dict) -> dict:
    return {k: entry.get(k) for k in ("row", "job_id", "company", "role", "url", "location")}


@_logged_followup("submit")
def submit_entry(entry: dict) -> str:
    """Click this tab's submit button (the user said "submit it"). Confirmed → the tab is
    submitted; the sheet says Applied once the user says "applications are done"."""
    job, tab, ats_name = _job_of(entry), entry["id"], entry.get("ats", "")
    company, role = _names(job)
    before = meter.snapshot()
    try:
        import aria.web.agent_browser as agent_browser
        current = agent_browser.run(lambda page: page.url, tab=tab)
        if not _same_form(current, entry.get("form_url", "")):
            return (f"The {company} form isn't open anymore, so I won't submit anything. If you submitted "
                    "it yourself, say I submitted it.")

        learned = 0
        if ats_name == "greenhouse":
            try:
                filled = entry.get("filled") or {}
                learned = learn_corrections(filled, greenhouse.read_values(list(filled), tab=tab))
            except Exception as exc:
                logger.warning("applicator: couldn't read corrections: %s", exc)

        outcome = ats_submit.submit(tab)
        saved = f" I saved {learned} answer{'s' if learned != 1 else ''} you changed." if learned else ""
        if outcome.status == "submitted":
            tracker.log_application(company, role, ats_name, job.get("url") or entry.get("form_url", ""), "applied")
            tabs.update(tab, state=tabs.SUBMITTED, submitted_by="aria", note="confirmation seen")
            _finish(job, ats_name, "submitted", before, review_path=entry.get("review_path", ""))
            return (f"Submitted {role} at {company}. I saw the confirmation; the sheet gets Applied when you "
                    f"say applications are done.{saved}")
        if outcome.status == "invalid":
            fields = ", ".join(outcome.missing[:4])
            return f"The {company} form didn't accept it yet. Still needed: {fields}.{saved}"
        if outcome.status == "no_button":
            return (f"I can't find the submit button on the {company} form, so nothing was sent. If you "
                    "submitted it yourself, say I submitted it.")
        tabs.update(tab, note="clicked submit, no confirmation seen")
        _finish(job, ats_name, "submit_unconfirmed", before, ok=False, note="clicked submit, no confirmation")
        return (f"I clicked submit on {company} but didn't see a confirmation. Check the tab and your email; "
                f"I'll ask you about it when you say applications are done.{saved}")
    except Exception as exc:
        logger.error("submit_entry failed: %s", exc)
        return "Something went wrong submitting. Check the browser before trying again."


@_logged_followup("submitted by hand")
def mark_submitted_by_hand(entry: dict) -> str:
    job, ats_name = _job_of(entry), entry.get("ats", "")
    company, role = _names(job)
    before = meter.snapshot()
    seen = ats_submit.page_confirms(entry["id"])
    tracker.log_application(company, role, ats_name, job.get("url") or entry.get("form_url", ""), "applied")
    tabs.update(entry["id"], state=tabs.SUBMITTED, submitted_by="user",
                note="you submitted it" + (", confirmation seen" if seen else ""))
    _finish(job, ats_name, "submitted", before, note="by hand", review_path=entry.get("review_path", ""))
    return f"Got it: {role} at {company} is submitted."


def mark_applied(entry: dict, note: str) -> str:
    """Sheet Status → Applied with a dated note (only on "applications are done"). Returns
    an error string, or "" when written (or the entry has no sheet row)."""
    err = _sheet_update_now(_job_of(entry), sheets.STATUS_APPLIED, note)
    if not err:
        tabs.update(entry["id"], sheet_marked=True)
    return err


@_logged_followup("skip")
def skip_entry(entry: dict) -> str:
    job = _job_of(entry)
    company, role = _names(job)
    before = meter.snapshot()
    sheet_err = _sheet_update(job, sheets.STATUS_SKIPPED, "skipped after review")
    tabs.update(entry["id"], skipped=True)
    _finish(job, entry.get("ats", ""), "skipped", before, review_path=entry.get("review_path", ""))
    return (f"Skipped {role} at {company}. The tab stays open."
            + (f" (Sheet not updated: {sheet_err})" if sheet_err else ""))
