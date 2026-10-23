"""tests/test_applicator.py — fill → review → submit/skip, sheet + aria.db + flight-log records."""
from __future__ import annotations

import json

import pytest

import aria.features.answer_bank as ab
import aria.observability.flight_recorder as fr
import aria.state.db as db
from aria.core import paths
from aria.features import applicator
from aria.features.ats import FillResult
from aria.features.applying import control, tabs
from aria.features.ats import submit as ats_submit
from aria.integrations import sheets
from aria.llm import meter

FORM = "https://job-boards.greenhouse.io/acme/jobs/1"
JOB = {"row": 7, "company": "Acme", "role": "ML Engineer", "url": FORM, "location": "NYC", "fields": {}}

REVIEW = [
    {"name": "first_name", "label": "First Name", "kind": "text", "value": "Alex Jordan", "source": "bank", "required": True, "note": ""},
    {"name": "q1", "label": "Do you require sponsorship?", "kind": "select", "value": "", "source": "needs_me", "required": True, "note": "no answer in your bank yet"},
    {"name": "q2", "label": "Why Acme?", "kind": "textarea", "value": "Because agents.\nAnd evals.", "source": "llm_draft", "required": True, "note": ""},
    {"name": "q3", "label": "Highest degree", "kind": "select", "value": "Master's", "source": "llm_guess", "required": True, "note": ""},
    {"name": "web", "label": "Website", "kind": "text", "value": "", "source": "skipped", "required": False, "note": ""},
]
FILLED = {r["name"]: {"label": r["label"], "kind": r["kind"], "value": r["value"], "source": r["source"]} for r in REVIEW}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(fr, "_LOG_PATH", str(tmp_path / "flight.jsonl"))
    monkeypatch.setattr(applicator, "REVIEWS_DIR", tmp_path / "reviews")
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "identity.json")
    monkeypatch.setattr(ab, "BANK_PATH", tmp_path / "answer_bank.json")
    monkeypatch.setattr(tabs, "REGISTRY_PATH", tmp_path / "apply_tabs.json")
    sheet_calls = []
    monkeypatch.setattr(sheets, "set_status", lambda row, status, note="", company=None, role=None, job_id=None:
                        sheet_calls.append((row, status, note)) or row)
    applied = []
    monkeypatch.setattr(applicator.tracker, "log_application", lambda *a: applied.append(a))
    browser = {"url": FORM, "screens": []}

    def fake_run(fn, tab=None):
        class Page:
            def is_closed(self):
                return False

            url = browser["url"]

            def screenshot(self, path, full_page):
                browser["screens"].append(path)
                open(path, "wb").write(b"png")
        return fn(Page())
    import aria.web.agent_browser as agent_browser
    monkeypatch.setattr(agent_browser, "run", fake_run)
    return {"sheet": sheet_calls, "applied": applied, "browser": browser, "tmp": tmp_path}


def _fake_fill(monkeypatch, review=REVIEW, error="", cost=0.0123):
    seen_tabs = []

    def fill(job, progress_fn=None, tab=None):
        seen_tabs.append(tab)
        meter.add({"cost_usd": cost, "tokens_in": 100, "tokens_out": 20})
        return FillResult(ats="greenhouse", form_url=FORM, title="ML Engineer", company="Acme",
                          review=list(review), filled=dict(FILLED), tab=tab, error=error)
    monkeypatch.setattr(applicator.ats, "fill_application", fill)
    return seen_tabs


def _runs():
    conn = db.get_connection()
    rows = [dict(r) for r in conn.execute("SELECT * FROM application_runs ORDER BY id")]
    conn.close()
    return rows


# ── review artifacts ────────────────────────────────────────────────────────

def test_summary_puts_needs_me_first_and_quotes_drafts():
    result = FillResult(ats="greenhouse", form_url=FORM, review=REVIEW)
    md = applicator.build_summary(JOB, result, "Acme", "ML Engineer", "/r/acme.png")
    assert md.startswith("# Review: ML Engineer at Acme")
    assert md.index("## Unknown — missing from your answer bank (1)") < md.index("## Written by the LLM") \
        < md.index("## Picked by the LLM") < md.index("## From your answer bank") < md.index("## Left empty")
    assert "- **Summary:** " in md and " unknown · 0 failed" in md
    assert "> Because agents.\n> And evals." in md
    assert "NOT submitted" in md and "Sheet row:** 7" in md


def test_spoken_result():
    result = FillResult(ats="greenhouse", review=REVIEW)
    s = applicator.spoken_result(result, "Acme", "ML Engineer")
    assert s.startswith("Greenhouse form filled for ML Engineer at Acme. 1 fact is missing from your answer bank.")
    assert "wrote 1 answer from your resume" in s and "submit it" in s


# ── run_application ─────────────────────────────────────────────────────────

def test_run_fills_into_its_own_tab_and_records(monkeypatch, env):
    seen_tabs = _fake_fill(monkeypatch)
    said = []
    out = applicator.run_application(dict(JOB), progress_fn=said.append)
    assert out.startswith("Greenhouse form filled for ML Engineer at Acme. 1 fact is missing from your answer bank.")

    [entry] = tabs.live()
    assert seen_tabs == [entry["id"]]                                  # filled in the job's own tab
    assert entry["state"] == tabs.FILLED and entry["row"] == 7 and entry["form_url"] == FORM
    assert entry["needs_me"] == ["Do you require sponsorship?"] and entry["ats"] == "greenhouse"
    md = entry["review_path"]
    assert md.endswith(".md") and "Unknown — missing from your answer bank (1)" in open(md).read()
    assert env["browser"]["screens"] and entry["screenshot"].endswith(".png")

    row, status, note = env["sheet"][-1]
    assert (row, status) == (7, sheets.STATUS_IN_REVIEW) and "1 unknown, 0 failed" in note and "not submitted" in note

    run = _runs()[-1]
    assert (run["sheet_row"], run["stage"], run["needs_me"], run["ats"]) == (7, "filled", 1, "greenhouse")
    assert run["cost_usd"] == pytest.approx(0.0123)
    job_entry = json.loads(open(fr._LOG_PATH).read().splitlines()[-1])
    assert (job_entry["kind"], job_entry["row"], job_entry["stage"]) == ("job", 7, "filled")
    assert job_entry["usage"]["cost_usd"] == pytest.approx(0.0123)
    assert env["applied"] == []                                        # nothing submitted


def test_run_uses_the_entry_it_is_given(monkeypatch, env):
    seen_tabs = _fake_fill(monkeypatch)
    entry = tabs.add(dict(JOB))
    applicator.run_application(dict(JOB), entry=entry)
    assert seen_tabs == [entry["id"]] and [e["id"] for e in tabs.live()] == [entry["id"]]


def test_run_without_url_marks_needs_human(env):
    out = applicator.run_application({"row": 3, "company": "GovStar", "role": "Dev", "url": "",
                                      "apply_link": "Email resume to x@govstar.us"})
    assert "Needs human" in out
    assert env["sheet"][-1][:2] == (3, sheets.STATUS_NEEDS_HUMAN) and "Email resume" in env["sheet"][-1][2]
    assert _runs()[-1]["stage"] == "error"
    assert tabs.live() == []                                           # no tab for a job with no link


def test_run_fill_error_marks_needs_human_tab_stays_open(monkeypatch, env):
    _fake_fill(monkeypatch, review=[], error="I couldn't find any form fields on that page.")
    out = applicator.run_application(dict(JOB))
    assert "couldn't fill" in out
    assert env["sheet"][-1][:2] == (7, sheets.STATUS_NEEDS_HUMAN)
    [entry] = tabs.live()
    assert entry["state"] == tabs.OPEN and "form fields" in entry["note"]


def test_url_only_job_never_touches_the_sheet(monkeypatch, env):
    _fake_fill(monkeypatch)
    out = applicator.run_application({"url": FORM, "company": "", "role": "", "row": None})
    assert "ML Engineer at Acme" in out
    assert env["sheet"] == []


def test_run_never_raises(monkeypatch, env):
    def boom(job, progress_fn=None, tab=None):
        raise RuntimeError("browser died")
    monkeypatch.setattr(applicator.ats, "fill_application", boom)
    assert "Needs human" in applicator.run_application(dict(JOB))


def test_stop_mid_fill_leaves_tab_open_and_propagates(monkeypatch, env):
    def stopped(job, progress_fn=None, tab=None):
        raise control.Stopped()
    monkeypatch.setattr(applicator.ats, "fill_application", stopped)
    with pytest.raises(control.Stopped):
        applicator.run_application(dict(JOB))
    [entry] = tabs.live()
    assert entry["state"] == tabs.OPEN and "stopped" in entry["note"]
    assert env["sheet"] == []                                          # a stop is not Needs human


# ── after review: only the user's words reach these ─────────────────────────

def _entry(env, **kw):
    e = tabs.add(dict(JOB), ats="greenhouse")
    fields = dict(state=tabs.FILLED, form_url=FORM, filled=dict(FILLED), review_path="/r/x.md")
    fields.update(kw)
    return tabs.update(e["id"], **fields)


def test_submit_entry_learns_corrections_and_sets_submitted_without_sheet_write(monkeypatch, env):
    e = _entry(env)
    monkeypatch.setattr(applicator.greenhouse, "read_values",
                        lambda names, tab=None: {"first_name": "Alex Jordan", "q1": "Yes", "q2": "Edited essay",
                                                 "q3": "Master's"})
    learned = []
    monkeypatch.setattr(applicator.answer_bank, "learn", lambda q, a, source="user": learned.append((q, a, source)) or True)
    clicked = []
    monkeypatch.setattr(ats_submit, "submit", lambda tab: clicked.append(tab) or ats_submit.SubmitOutcome("submitted"))
    out = applicator.submit_entry(e)
    assert out.startswith("Submitted ML Engineer at Acme.") and "saved 1 answer" in out
    assert clicked == [e["id"]]
    assert learned == [("Do you require sponsorship?", "Yes", "review")]     # essay (textarea) not learned
    assert env["applied"] == [("Acme", "ML Engineer", "greenhouse", FORM, "applied")]
    assert env["sheet"] == []                                                # Applied waits for "done"
    got = tabs.get(e["id"])
    assert (got["state"], got["submitted_by"]) == (tabs.SUBMITTED, "aria")
    assert _runs()[-1]["stage"] == "submitted"


def test_submit_refuses_when_form_is_not_open(monkeypatch, env):
    e = _entry(env)
    env["browser"]["url"] = "about:blank"
    called = []
    monkeypatch.setattr(ats_submit, "submit", lambda tab: called.append(tab))
    assert "isn't open anymore" in applicator.submit_entry(e)
    assert called == [] and env["applied"] == []


def test_submit_invalid_keeps_tab_filled(monkeypatch, env):
    e = _entry(env)
    monkeypatch.setattr(applicator.greenhouse, "read_values", lambda names, tab=None: {})
    monkeypatch.setattr(ats_submit, "submit", lambda tab: ats_submit.SubmitOutcome("invalid", ["Resume/CV", "Sponsorship"]))
    out = applicator.submit_entry(e)
    assert "Still needed: Resume/CV, Sponsorship" in out
    assert tabs.get(e["id"])["state"] == tabs.FILLED and env["sheet"] == [] and env["applied"] == []


def test_submit_unconfirmed_stays_filled_for_done_to_ask(monkeypatch, env):
    e = _entry(env)
    monkeypatch.setattr(applicator.greenhouse, "read_values", lambda names, tab=None: {})
    monkeypatch.setattr(ats_submit, "submit", lambda tab: ats_submit.SubmitOutcome("unconfirmed"))
    assert "didn't see a confirmation" in applicator.submit_entry(e)
    got = tabs.get(e["id"])
    assert got["state"] == tabs.FILLED and "no confirmation" in got["note"]
    assert env["sheet"] == [] and env["applied"] == []


def test_by_hand_sets_submitted_user(monkeypatch, env):
    e = _entry(env)
    monkeypatch.setattr(ats_submit, "page_confirms", lambda tab: True)
    assert applicator.mark_submitted_by_hand(e) == "Got it: ML Engineer at Acme is submitted."
    got = tabs.get(e["id"])
    assert (got["state"], got["submitted_by"]) == (tabs.SUBMITTED, "user") and "confirmation seen" in got["note"]
    assert env["sheet"] == [] and len(env["applied"]) == 1


def test_mark_applied_writes_status_and_note(env):
    e = _entry(env, state=tabs.SUBMITTED, submitted_by="aria")
    assert applicator.mark_applied(e, "submitted by Aria, confirmation seen") == ""
    assert env["sheet"][-1] == (7, sheets.STATUS_APPLIED, "submitted by Aria, confirmation seen")
    assert tabs.get(e["id"])["sheet_marked"] is True


def test_skip_writes_skipped_and_keeps_tab(env):
    e = _entry(env)
    assert applicator.skip_entry(e) == "Skipped ML Engineer at Acme. The tab stays open."
    assert env["sheet"][-1][:2] == (7, sheets.STATUS_SKIPPED)
    got = tabs.get(e["id"])
    assert got["skipped"] is True and got["state"] == tabs.FILLED and env["applied"] == []


def test_learn_corrections_ignores_unchanged_files_essays_and_dial_codes(monkeypatch):
    learned = []
    monkeypatch.setattr(applicator.answer_bank, "learn", lambda q, a, source="user": learned.append(q) or True)
    filled = {
        "a": {"label": "Phone", "kind": "text", "value": "+1-716"},
        "b": {"label": "Resume", "kind": "file", "value": ""},
        "c": {"label": "Why us", "kind": "textarea", "value": "x"},
        "d": {"label": "Country", "kind": "select", "value": "United States +1"},
        "e": {"label": "Start date", "kind": "text", "value": ""},
    }
    now = {"a": "+1-716", "b": "cv.pdf", "c": "y", "d": "+1", "e": "June 2026"}
    assert applicator.learn_corrections(filled, now) == 1 and learned == ["Start date"]


# ── submit helpers ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,url,ok", [
    ("Thank you for applying to Acme!", "", True),
    ("Your application has been submitted.", "", True),
    ("We've received your application", "", True),
    ("Apply for this job", "https://job-boards.greenhouse.io/acme/jobs/1/confirmation", True),
    ("Apply for this job  Submit application", FORM, False),
])
def test_is_confirmation(text, url, ok):
    assert ats_submit.is_confirmation(text, url) is ok


# ── flight log keeps jobs apart from voice commands ─────────────────────────

def test_job_entries_stay_out_of_command_reports(env):
    fr.record("what time is it", "3pm", 0.2, usage={"cost_usd": 0.0})
    fr.record_job(7, "Acme", "ML Engineer", "filled", "greenhouse", {"cost_usd": 0.01, "calls": 2})
    fr.record_job(7, "Acme", "ML Engineer", "submitted", "greenhouse", {"cost_usd": 0.002})
    fr.record_job(9, "Beta", "SWE", "filled", "greenhouse", {"cost_usd": 0.03})
    assert [e.get("transcript") for e in fr.read_recent(1)] == ["what time is it"]
    s = fr.job_summary(1)
    assert (s["runs"], s["submitted"]) == (2, 1)
    assert s["by_row"] == {"7": 0.012, "9": 0.03} and s["cost_usd"] == pytest.approx(0.042)


# ── apply_job tool ──────────────────────────────────────────────────────────

def _tool():
    from aria.core.tool import ToolRegistry
    from aria.plugins.apply import ApplyPlugin
    reg = ToolRegistry()
    ApplyPlugin().register(reg)
    return reg.get("apply_job")


def test_apply_job_next_row_url_and_errors(monkeypatch, env):
    seen = []
    monkeypatch.setattr(applicator, "run_application", lambda job, progress_fn=None, voice_ask_fn=None, run=None,
                        **kw: seen.append(job) or "filled")
    monkeypatch.setattr(sheets, "get_next_job", lambda: dict(JOB))
    monkeypatch.setattr(sheets, "get_job", lambda row: dict(JOB, row=row) if row == 12 else None)
    monkeypatch.setattr(sheets, "default_sheet", lambda: type("S", (), {"jobs": lambda self: [dict(JOB)]})())
    tool = _tool()
    assert tool.execute({}) == "filled" and seen[-1]["row"] == 7
    assert tool.execute({"row": 12}) == "filled" and seen[-1]["row"] == 12
    assert tool.execute({"row": 99}) == "Row 99 isn't a job in your sheet."
    assert tool.execute({"url": FORM}) == "filled" and seen[-1]["url"] == FORM and seen[-1]["row"] is None
    monkeypatch.setattr(sheets, "get_next_job", lambda: None)
    assert "no new jobs" in tool.execute({})

    def broken():
        raise sheets.SheetError("Google isn't connected yet. Run: python -m aria.setup google")
    monkeypatch.setattr(sheets, "get_next_job", broken)
    assert "aria.setup google" in tool.execute({})


def test_submit_is_not_an_agent_tool():
    from aria.core.tool import ToolRegistry
    from aria.plugins.apply import ApplyPlugin
    reg = ToolRegistry()
    ApplyPlugin().register(reg)
    names = {t for t in ("apply_job", "job_from_sheet", "submit_application", "submit_job") if reg.get(t)}
    assert names == {"apply_job", "job_from_sheet"}


# ── run log (live dashboard) ────────────────────────────────────────────────

def test_apply_run_logs_steps_decisions_and_labels_llm_calls(monkeypatch, env):
    from aria.llm import context as llm_context
    from aria.observability import run_log
    seen_labels = []

    def fill(job, progress_fn=None, tab=None):
        seen_labels.append(llm_context.current())
        return FillResult(ats="greenhouse", form_url=FORM, title="ML Engineer", company="Acme",
                          review=list(REVIEW), filled=dict(FILLED), tab=tab)
    monkeypatch.setattr(applicator.ats, "fill_application", fill)
    skipped = dict(JOB, row=6, company="Old Co", status="Applied", alignment="2")
    monkeypatch.setattr(sheets, "get_next_job", lambda: dict(JOB))
    monkeypatch.setattr(sheets, "default_sheet", lambda: type("S", (), {"jobs": lambda self: [skipped, dict(JOB)]})())
    _tool().execute({})
    [run] = run_log.read_runs()
    assert run["trigger"] == "apply_job" and run["status"] == "filled"
    assert run["job"] == {"company": "Acme", "role": "ML Engineer", "row": 7}
    steps = {s["step"]: s for s in run["steps"]}
    assert steps["read sheet"]["status"] == "ok" and "picked row 7" in steps["read sheet"]["detail"]
    assert steps["fill form"]["status"] == "ok" and "need you" in steps["fill form"]["detail"]
    assert steps["update sheet"]["status"] == "ok" and "In review" in steps["update sheet"]["detail"]
    assert steps["find email"]["status"] == "not in this run"
    picked = [d for d in run["decisions"] if d["picked"]]
    assert [d["row"] for d in picked] == [7] and any("picked:" in r for r in picked[0]["reasons"])
    assert any("Status: Applied" in r for d in run["decisions"] if not d["picked"] for r in d["reasons"])
    assert seen_labels[0]["job"] == "Acme | ML Engineer | row 7" and seen_labels[0]["run"] == run["id"]
    assert llm_context.current() == {}                       # labels cleared after the run


def test_skip_is_its_own_run(env):
    from aria.observability import run_log
    applicator.skip_entry(_entry(env))
    run = run_log.read_runs()[0]
    assert run["trigger"] == "skip" and run["status"] == "skipped"
    assert {s["step"]: s["status"] for s in run["steps"]}["update sheet"] == "ok"


def test_followups_never_relabel_a_running_job(monkeypatch, env):
    """A voice follow-up mid-run must not swap the global LLM job label the run's calls use."""
    from aria.llm import context as llm_context
    e = _entry(env)
    during = []
    monkeypatch.setattr(sheets, "set_status", lambda row, status, note="", company=None, role=None, job_id=None:
                        during.append(llm_context.current().get("job")) or row)
    with llm_context.job("Ookla | Data Engineer | row 14", "run-a"):
        applicator.skip_entry(e)
    assert during == ["Ookla | Data Engineer | row 14"]


def test_pipeline_run_is_per_thread(env):
    import threading

    from aria.observability import run_log
    run = run_log.start("x")
    seen = []
    applicator._set_run(run)
    t = threading.Thread(target=lambda: seen.append(applicator._current_run()))
    t.start()
    t.join()
    assert seen == [None] and applicator._current_run() is run
    applicator._set_run(None)
