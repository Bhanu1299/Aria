"""tests/test_outreach_dry_run.py — step 6: dry run does everything but write drafts, the sheet or feedback."""
from __future__ import annotations

from datetime import date

from aria.features.outreach import pipeline
from aria.integrations.sheets import JobSheet
from tests.test_outreach_feedback import _gmail
from tests.test_outreach_pipeline import BODY, _llm, _people, profile  # noqa: F401 (fixture)
from tests.test_sheets import HEADERS, FakeSheetsService


def _row(company, role, fit="4"):
    return [company, role, "NYC", fit, "", "", "https://boards.greenhouse.io/x/jobs/1", "Via Greenhouse", "", "", "", ""]


class NoGmail:
    def create_draft(self, *a, **k):
        raise AssertionError("dry run must not touch Gmail")


def test_dry_run_writes_nothing_and_reports_everything(monkeypatch, tmp_path):
    _people(monkeypatch, {"email": "pat@acme.com", "status": "valid", "confidence": 0.97})
    calls = _llm(monkeypatch, BODY)
    recorded = []
    import coldstart.api
    monkeypatch.setattr(coldstart.api, "record_feedback", lambda *a, **k: recorded.append(a))
    monkeypatch.setattr(pipeline.feedback, "_coldstart_state", lambda path=None: ({}, {}))
    svc = FakeSheetsService([HEADERS, _row("Acme", "ML Engineer", "5"), _row("Acme", "AI Engineer")])
    out = pipeline.run_outreach(sheet=JobSheet("s", "Sheet1", service=svc, today=lambda: date(2026, 10, 20)),
                                search_fn=lambda q: [], gmail_data=_gmail(), gmail=NoGmail(),
                                dry_run=True, report_dir=tmp_path)
    assert svc.writes == [] and recorded == []
    assert calls and calls[0]["task"] == "outreach_email"            # the LLM writing still runs
    assert out.startswith("Outreach dry run: 1 email draft written, not saved. Nothing was written.")
    assert "Would tell Coldstart about 1 reply and 1 bounce." in out
    [report] = list(tmp_path.glob("dry-run-*.md"))
    text = report.read_text()
    for want in ("## Acme", "row 2 ML Engineer (fit 5), row 3 AI Engineer (fit 4)", "Pat Lee (Recruiter) via email",
                 "**Subject:** ML Engineer and AI Engineer — Alex Doe", "Hi Pat,",
                 "row 2 Outreach_2: 'Email draft ready — Pat Lee (Recruiter), pat@acme.com'",
                 "row 3 Outreach_2: 'Outreach via row 2 (ML Engineer)'", "- pat@acme.com: replied",
                 "- ana@globex.com: bounced"):
        assert want in text, want


def test_dry_run_linkedin_note_lands_in_the_report(monkeypatch, tmp_path):
    _people(monkeypatch, {"email": None, "status": "unknown", "confidence": 0.0})
    _llm(monkeypatch, "I build voice agents and the ML Engineer role at Acme fits my MS work. Open to connecting?")
    svc = FakeSheetsService([HEADERS, _row("Acme", "ML Engineer")])
    out = pipeline.run_outreach(sheet=JobSheet("s", "Sheet1", service=svc), search_fn=lambda q: [], gmail_data={},
                                dry_run=True, report_dir=tmp_path)
    assert "1 LinkedIn note written, not saved" in out and svc.writes == []
    text = next(tmp_path.glob("dry-run-*.md")).read_text()
    assert "> Hi Pat, I build voice agents" in text


def test_dry_sheet_still_refuses_non_outreach_columns():
    dry = pipeline.DrySheet(JobSheet("s", "Sheet1", service=FakeSheetsService([HEADERS])))
    try:
        dry.set_fields(2, {"Status": "Applied"})
    except pipeline.sheets.SheetError:
        pass
    else:
        raise AssertionError("expected SheetError")
    assert dry.writes == []


def test_job_outreach_tool_passes_dry_run(monkeypatch):
    from aria.plugins.apply import ApplyPlugin
    seen = []
    monkeypatch.setattr(pipeline, "run_outreach", lambda progress_fn=None, dry_run=False: seen.append(dry_run) or "ok")
    tool = ApplyPlugin()._job_outreach_tool()
    assert tool.execute({"dry_run": True}) == "ok" and tool.execute({}) == "ok"
    assert seen == [True, False]
