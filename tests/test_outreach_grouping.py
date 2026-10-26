"""tests/test_outreach_grouping.py — step 5: one outreach per company, not one per role."""
from __future__ import annotations

from datetime import date, datetime

from aria.dashboard import model as m
from aria.dashboard.sources import Source
from aria.features.outreach import pipeline
from aria.integrations.sheets import JobSheet
from tests.test_outreach_pipeline import BODY, FakeGmail, _llm, _people, profile  # noqa: F401 (fixture)
from tests.test_sheets import HEADERS, FakeSheetsService


def _row(company, role, fit="4", who="Via Greenhouse", outreach2="", updates=""):
    return [company, role, "NYC", fit, "", "", "https://boards.greenhouse.io/x/jobs/1", who, "", "", outreach2, updates]


def _jobs(rows):
    return JobSheet("s", "Sheet1", service=FakeSheetsService([HEADERS] + rows)).jobs()


def test_roles_at_one_company_become_one_target_led_by_the_best_fit():
    [t] = pipeline.targets(_jobs([_row("Acme Inc", "SWE", fit="3"), _row("Acme", "ML Engineer", fit="5"),
                                  _row("acme", "Data Engineer", fit="2")]))
    assert [j["role"] for j in t.jobs] == ["ML Engineer", "SWE"]           # fit-2 row isn't outreach-worthy
    assert t.primary["row"] == 3


def test_company_already_contacted_on_any_row_is_skipped():
    todo, skipped = pipeline.plan(_jobs([_row("Acme", "SWE"), _row("Acme", "Intern", fit="1", who="Jo Park (CTO)"),
                                         _row("Globex", "SWE"), _row("Globex", "PM", fit="2", outreach2="LinkedIn: sent")]))
    assert todo == []
    assert [(j["company"], why.split(" on ")[0]) for j, why in skipped] == [
        ("Acme", "company already contacted"), ("Globex", "company already contacted")]


def test_aria_named_rows_do_not_count_as_contacted():
    rows = [_row("Acme", "SWE", who="Pat Lee (Recruiter) | Via Greenhouse", updates="2026-10-20: Aria found contacts"),
            _row("Acme", "ML Engineer")]
    [t] = pipeline.targets(_jobs(rows))
    assert [p.name for p in t.people] == ["Pat Lee"] and len(t.jobs) == 2


def test_one_draft_names_every_role_and_siblings_point_to_it(monkeypatch):
    _people(monkeypatch, {"email": "pat@acme.com", "status": "valid", "confidence": 0.97})
    calls = _llm(monkeypatch, BODY)
    svc = FakeSheetsService([HEADERS, _row("Acme", "ML Engineer", fit="5"), _row("Acme", "AI Engineer", fit="4")])
    gmail = FakeGmail()
    out = pipeline.run_outreach(sheet=JobSheet("s", "Sheet1", service=svc, today=lambda: date(2026, 10, 20)),
                                search_fn=lambda q: [], gmail_data={}, gmail=gmail)
    assert out.startswith("Outreach: 1 email draft")
    assert len(gmail.drafts) == 1 and gmail.drafts[0]["subject"] == "ML Engineer and AI Engineer — Alex Doe, agents + model evaluation"
    assert "Role(s): ML Engineer and AI Engineer" in calls[0]["content"]
    w = svc.written()
    assert w["'Sheet1'!K2"].startswith("Email draft ready — Pat Lee")
    assert w["'Sheet1'!K3"] == "Outreach via row 2 (ML Engineer)"
    assert w["'Sheet1'!H2"] == w["'Sheet1'!H3"] == "Pat Lee (Recruiter) | Via Greenhouse"


def test_dashboard_marks_the_sibling_as_covered():
    row = {"_row": 3, "Company": "Acme", "Role": "AI Engineer", "Status": "Applied", "Profile alignment /5": "4",
           "Apply link": "", "Post link": "", "Who to contact": "Pat Lee (Recruiter)",
           "Outreach_2": "Outreach via row 2 (ML Engineer)"}
    mdl = m.build({"sheet": Source(True, {"current": [row], "applied": []})}, now=datetime(2026, 10, 20))
    [job] = mdl["jobs"]
    assert m.next_action(job, mdl["now"]) == ("Covered by the outreach for ML Engineer", "waiting")
