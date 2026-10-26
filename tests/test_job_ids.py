"""tests/test_job_ids.py — stable job IDs and finding a job by what the user calls it (mocked API)."""
from __future__ import annotations

from datetime import date

import pytest

import aria.core.config as config
from aria.integrations import sheets
from aria.integrations.sheets import JobSheet
from tests.test_sheets import FakeSheetsService

JOBS = ["Company", "Role", "Location", "Status", "Apply link"]


def _tab(name, rows, headers=JOBS):
    svc = FakeSheetsService([headers] + rows)
    return JobSheet("sheet-id", name, service=svc, today=lambda: date(2026, 10, 23)), svc


@pytest.fixture
def two_tabs(monkeypatch):
    monkeypatch.setattr(config, "MUSE_SHEET_TAB", "Sheet1")
    monkeypatch.setattr(config, "MUSE_APPLIED_TAB", "Applied")
    jobs, jsvc = _tab("Sheet1", [
        ["Plaid", "Software Engineering, New Grad", "NYC", "Not gonna apply"],
        ["State of Michigan DTMB", "Information Technology Programmer Analyst 12", "Lansing", "In review"],
        ["John Deere", "IT Development Program - Software Engineering", "Moline", "Not applied"],
        ["RTX", "Software Engineer I (Onsite)", "Tucson", "Not applied"],
        ["eBay", "Software Engineer - Recent Grad", "Austin", ""],
    ])
    applied, asvc = _tab("Applied", [
        ["Magna", "Junior AI Engineer", "Troy", "Applied"],
        ["RTX", "Software Engineer I – Next Gen Systems", "Aurora", "Applied"],
        ["Sallie Mae", "Data Engineer", "Newark", "Applied"],
    ])
    return [jobs, applied], jsvc, asvc


# ── IDs ─────────────────────────────────────────────────────────────────────

def test_ensure_ids_adds_header_and_numbers_both_tabs_in_one_sequence(two_tabs):
    tabs, jsvc, asvc = two_tabs
    assert sheets.ensure_ids(tabs) == 8
    j, a = jsvc.written(), asvc.written()
    assert j["'Sheet1'!F1"] == "ID" and a["'Applied'!F1"] == "ID"     # after the last used column
    assert [j[f"'Sheet1'!F{r}"] for r in range(2, 7)] == ["J0001", "J0002", "J0003", "J0004", "J0005"]
    assert [a[f"'Applied'!F{r}"] for r in range(2, 5)] == ["J0006", "J0007", "J0008"]


def test_ensure_ids_never_rewrites_and_continues_after_the_highest(monkeypatch):
    monkeypatch.setattr(config, "MUSE_SHEET_TAB", "Sheet1")
    headers = JOBS + ["ID"]
    jobs, jsvc = _tab("Sheet1", [["A", "SWE", "", "", "", "J0007"], ["B", "SWE", "", "", "", ""]], headers)
    applied, asvc = _tab("Applied", [["C", "SWE", "", "Applied", "", "J0012"]], headers)
    assert sheets.ensure_ids([jobs, applied]) == 1
    assert jsvc.written() == {"'Sheet1'!F3": "J0013"}
    assert asvc.writes == []


def test_ensure_ids_renumbers_a_copied_rows_repeat(monkeypatch):
    headers = JOBS + ["ID"]
    jobs, jsvc = _tab("Sheet1", [["A", "SWE", "", "", "", "J0003"], ["A", "SWE", "", "", "", "J0003"]], headers)
    assert sheets.ensure_ids([jobs]) == 1
    assert jsvc.written() == {"'Sheet1'!F3": "J0004"}


def test_ensure_ids_puts_the_column_after_data_wider_than_the_header():
    jobs, jsvc = _tab("Sheet1", [["A", "SWE", "", "", "", "", "stray note"]])
    sheets.ensure_ids([jobs])
    assert jsvc.written()["'Sheet1'!H1"] == "ID"


def test_rows_carry_their_id_and_writes_find_a_moved_row_by_id():
    headers = JOBS + ["ID"]
    sheet, svc = _tab("Sheet1", [["Magna", "AI Eng", "", "", "", "J0001"], ["RTX", "SWE", "", "", "", "J0002"]], headers)
    assert sheet.jobs()[1]["job_id"] == "J0002"
    assert sheet.table()["rows"][0]["id"] == "J0001"
    # Magna moved away: RTX is now row 2, but the caller still says row 3.
    svc.values_data = [headers, ["RTX", "SWE", "", "", "", "J0002"]]
    assert sheet.set_status(3, "Skipped", job_id="J0002") == 2
    assert svc.written()["'Sheet1'!D2"] == "Skipped"


def test_id_beats_company_and_role_when_both_are_given():
    headers = JOBS + ["ID"]
    sheet, svc = _tab("Sheet1", [["RTX", "SWE", "", "", "", "J0001"], ["RTX", "SWE", "", "", "", "J0002"]], headers)
    assert sheet.set_status(2, "Skipped", company="RTX", role="SWE", job_id="J0002") == 3


# ── find / resolve ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("said,company,row", [
    ("let's apply to the software engineer one on-site from RTX", "RTX", 5),
    ("the Michigan job", "State of Michigan DTMB", 3),
    ("john deer", "John Deere", 4),
    ("the ebay one", "eBay", 6),
    ("row 5", "RTX", 5),
])
def test_resolve_one(two_tabs, said, company, row):
    tabs, _, _ = two_tabs
    r = sheets.resolve_job(said, tabs)
    assert (r["kind"], r["sure"]) == ("one", True)
    assert (r["jobs"][0]["company"], r["jobs"][0]["row"]) == (company, row)


def test_resolve_already_applied(two_tabs):
    tabs, _, _ = two_tabs
    r = sheets.resolve_job("apply to the Magna job", tabs)
    assert r["kind"] == "applied" and r["sure"]
    assert r["jobs"][0]["company"] == "Magna" and r["jobs"][0]["tab"] == "Applied"


def test_misheard_name_is_found_but_not_sure(two_tabs):
    tabs, _, _ = two_tabs
    r = sheets.resolve_job("the magnet job", tabs)
    assert r["jobs"][0]["company"] == "Magna" and r["sure"] is False


def test_open_job_wins_over_an_applied_one_from_the_same_company(two_tabs):
    tabs, _, _ = two_tabs
    r = sheets.resolve_job("RTX", tabs)
    assert r["kind"] == "one" and r["jobs"][0]["tab"] == "Sheet1"


def test_generic_words_ask_which(two_tabs):
    tabs, _, _ = two_tabs
    r = sheets.resolve_job("a software engineer job", tabs)
    assert r["kind"] == "many" and len(r["jobs"]) >= 2


def test_unknown_and_empty(two_tabs):
    tabs, _, _ = two_tabs
    assert sheets.resolve_job("the Stripe job", tabs)["kind"] == "none"
    assert sheets.resolve_job("the job", tabs)["kind"] == "none"


def test_job_id_in_words(monkeypatch):
    monkeypatch.setattr(config, "MUSE_SHEET_TAB", "Sheet1")
    headers = JOBS + ["ID"]
    sheet, _ = _tab("Sheet1", [["A", "SWE", "", "", "", "J0041"], ["B", "SWE", "", "", "", "J0042"]], headers)
    r = sheets.resolve_job("apply to j42", [sheet])
    assert r["kind"] == "one" and r["jobs"][0]["company"] == "B"


# ── agent tools: job_find / apply_job(job=...) ──────────────────────────────

def _tools(monkeypatch, tabs):
    from aria.core.tool import ToolRegistry
    from aria.features import applicator
    import aria.plugins.apply as apply_plugin
    monkeypatch.setattr(sheets, "ensure_ids", lambda tabs=None: 0)
    real_find = sheets.find_jobs
    monkeypatch.setattr(sheets, "find_jobs", lambda q, t=None: real_find(q, tabs))
    real_resolve = sheets.resolve_job
    monkeypatch.setattr(sheets, "resolve_job", lambda q, t=None: real_resolve(q, tabs))
    monkeypatch.setattr(apply_plugin.memory, "set_persistent", lambda k, v: None)
    filled = []
    monkeypatch.setattr(applicator, "run_application", lambda job, **kw: filled.append(job) or f"filled {job['company']}")
    reg = ToolRegistry()
    apply_plugin.ApplyPlugin().register(reg)
    return reg.get("job_find"), reg.get("apply_job"), filled


def test_apply_job_by_the_users_words_fills_that_job(two_tabs, monkeypatch):
    tabs, _, _ = two_tabs
    _, apply_job, filled = _tools(monkeypatch, tabs)
    assert apply_job.execute({"job": "the software engineer one on-site from RTX"}) == "filled RTX"
    assert filled[0]["role"] == "Software Engineer I (Onsite)"


@pytest.mark.parametrize("said,expect", [
    ("the Magna job", "Already applied"),
    ("the magnet job", "Not sure"),
    ("a software engineer job", "Several jobs fit"),
    ("the Stripe job", "No job in the sheet"),
])
def test_apply_job_never_guesses(two_tabs, monkeypatch, said, expect):
    tabs, _, _ = two_tabs
    _, apply_job, filled = _tools(monkeypatch, tabs)
    assert apply_job.execute({"job": said}).startswith(expect)
    assert filled == []


def test_job_find_lists_both_tabs(two_tabs, monkeypatch):
    tabs, _, _ = two_tabs
    job_find, _, _ = _tools(monkeypatch, tabs)
    out = job_find.execute({"query": "RTX"})
    assert "Onsite" in out and "Next Gen Systems" in out and "Applied tab" in out


# ── dashboard ───────────────────────────────────────────────────────────────

def test_dashboard_never_edits_the_id_column():
    from aria.dashboard import sheet_tab
    sheet, _ = _tab("Sheet1", [["RTX", "SWE", "", "", "", "J0001"]], JOBS + ["ID"])
    with pytest.raises(sheet_tab.SheetEditError):
        sheet_tab.edit({"tab": "jobs", "row": 2, "id": "J0001", "column": "ID", "value": "J9"}, sheet=sheet)


def test_dashboard_edit_finds_the_row_by_id():
    from aria.dashboard import sheet_tab
    headers = JOBS + ["ID"]
    sheet, svc = _tab("Sheet1", [["Plaid", "SWE", "", "", "", "J0001"], ["RTX", "SWE", "", "", "", "J0002"]], headers)
    out = sheet_tab.edit({"tab": "jobs", "row": 2, "id": "J0002", "column": "Location", "value": "Tucson",
                          "company": "RTX", "role": "SWE"}, sheet=sheet)
    assert out == {"ok": True, "row": 3} and svc.written() == {"'Sheet1'!C3": "Tucson"}


def test_dashboard_page_shows_the_id_not_the_row():
    from aria.dashboard import sheet_tab
    assert "th.textContent=cur.id_col||'ID'" in sheet_tab.JS and "n.textContent=r.id||''" in sheet_tab.JS
    assert "id:r.id||''" in sheet_tab.JS


@pytest.mark.parametrize("query", ["", "how many jobs are left", "what's on my list", "all"])
def test_job_find_without_a_job_named_gives_the_overview(two_tabs, monkeypatch, query):
    tabs, _, _ = two_tabs
    job_find, _, _ = _tools(monkeypatch, tabs)
    monkeypatch.setattr(sheets, "default_sheet", lambda: tabs[0])
    monkeypatch.setattr(sheets, "tab_sheet", lambda t: tabs[1])
    out = job_find.execute({"query": query})
    assert out.startswith("Jobs tab: 5 rows") and "Applied tab: 3 jobs" in out
    assert "3 still to apply to" in out and "John Deere" in out and "Plaid" not in out.split("still to apply to")[1]
