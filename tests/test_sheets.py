"""tests/test_sheets.py — job sheet header mapping, next-job selection, status writes (mocked API)."""
from __future__ import annotations

import json
from datetime import date

import pytest

from aria.integrations import google_auth
from aria.integrations.sheets import JobSheet, SheetError, col_letter, dedupe_headers

# The real Sheet1 header row (A–L), including the duplicated "Outreach".
HEADERS = [
    "Company", "Role", "Location", "Profile alignment /5", "Status", "Apply link",
    "Post link", "Who to contact", "Outreach", "Why it's a fit + draft", "Outreach", "Updates",
]


def _row(company, role, status="", apply_link="", post_link="", updates=None, **extra):
    r = [company, role, "NYC", "4", status, apply_link, post_link, "", "", "", ""]
    if updates is not None:
        r.append(updates)
    return r  # the API drops trailing empty cells, so rows are ragged


class FakeSheetsService:
    """Just enough of googleapiclient's spreadsheets().values() surface."""

    def __init__(self, values):
        self.values_data = values
        self.writes = []      # list of batchUpdate bodies
        self.read_ranges = []

    def spreadsheets(self):
        return self

    def values(self):
        return self

    def get(self, spreadsheetId, range):
        self.read_ranges.append(range)
        data = {"values": self.values_data}
        return _Exec(data)

    def batchUpdate(self, spreadsheetId, body):
        self.writes.append(body)
        return _Exec({})

    def written(self) -> dict:
        return {d["range"]: d["values"][0][0] for b in self.writes for d in b["data"]}


class _Exec:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


def _sheet(rows, headers=HEADERS):
    svc = FakeSheetsService([headers] + rows)
    return JobSheet("sheet-id", "Sheet1", service=svc, today=lambda: date(2026, 10, 19)), svc


# ── header helpers ──────────────────────────────────────────────────────────

def test_dedupe_headers_numbers_repeats():
    assert dedupe_headers(HEADERS)[8] == "Outreach"
    assert dedupe_headers(HEADERS)[10] == "Outreach_2"
    assert dedupe_headers(["A", "a", "A"]) == ["A", "a_2", "A_3"]


@pytest.mark.parametrize("index,letters", [(0, "A"), (4, "E"), (11, "L"), (25, "Z"), (26, "AA"), (51, "AZ"), (52, "BA")])
def test_col_letter(index, letters):
    assert col_letter(index) == letters


# ── get_next_job ────────────────────────────────────────────────────────────

def test_next_job_skips_rows_with_status_and_treats_blank_as_new():
    sheet, _ = _sheet([
        _row("Luminai", "FDE", status="Applied", apply_link="https://jobs.ashbyhq.com/x"),
        _row("Acme", "Engineer", status="Not gonna apply"),
        _row("Conviva", "AI/ML Engineer", apply_link="https://www.conviva.ai/careers/job/1?gh_jid=1"),
    ])
    job = sheet.get_next_job()
    assert job["company"] == "Conviva"
    assert job["row"] == 4
    assert job["url"] == "https://www.conviva.ai/careers/job/1?gh_jid=1"
    assert job["fields"]["Outreach_2"] == ""


def test_next_job_matches_explicit_new_status_case_insensitively():
    sheet, _ = _sheet([_row("Acme", "SWE", status="NEW", apply_link="https://a.com")])
    assert sheet.get_next_job()["company"] == "Acme"


def test_next_job_treats_not_applied_as_new():
    sheet, _ = _sheet([
        _row("Palantir", "SWE", status="Applied"),
        _row("Amazon", "SDE", status="Not applied", apply_link="https://amazon.jobs/1"),
    ])
    assert sheet.get_next_job()["company"] == "Amazon"


def test_next_job_falls_back_to_post_link_when_apply_link_is_not_a_url():
    sheet, _ = _sheet([
        _row("GovStar", "Dev", apply_link="Email resume to x@govstar.us", post_link="https://news.ycombinator.com/item?id=1"),
    ])
    job = sheet.get_next_job()
    assert job["url"] == "https://news.ycombinator.com/item?id=1"
    assert job["apply_link"].startswith("Email resume")


def test_next_job_without_any_url():
    sheet, _ = _sheet([_row("Converge", "NEXT program", apply_link="Email careers@converge.ai")])
    assert sheet.get_next_job()["url"] == ""


def test_next_job_none_when_every_row_has_status():
    sheet, _ = _sheet([_row("A", "B", status="Applied"), _row("C", "D", status="Skipped")])
    assert sheet.get_next_job() is None


def test_header_order_is_read_not_assumed():
    headers = ["Status", "Role", "Company", "Apply link"]
    sheet, _ = _sheet([["", "SWE", "Stripe", "https://stripe.com/jobs/1"]], headers=headers)
    job = sheet.get_next_job()
    assert (job["company"], job["role"], job["url"]) == ("Stripe", "SWE", "https://stripe.com/jobs/1")


def test_blank_rows_are_ignored():
    sheet, _ = _sheet([[], ["", ""], _row("Acme", "SWE", apply_link="https://a.com")])
    assert sheet.get_next_job()["row"] == 4


def test_missing_company_column_raises():
    sheet, _ = _sheet([["SWE"]], headers=["Role"])
    with pytest.raises(SheetError):
        sheet.get_next_job()


def test_reads_the_configured_tab_quoted():
    svc = FakeSheetsService([HEADERS])
    JobSheet("id", "Alex's Jobs", service=svc).jobs()
    assert svc.read_ranges == ["'Alex''s Jobs'"]


# ── set_status ──────────────────────────────────────────────────────────────

def test_set_status_writes_status_cell_only():
    sheet, svc = _sheet([_row("Acme", "SWE")])
    assert sheet.set_status(2, "In review") == 2
    assert svc.written() == {"'Sheet1'!E2": "In review"}
    assert svc.writes[0]["valueInputOption"] == "RAW"


def test_set_status_appends_note_to_updates_with_date():
    sheet, svc = _sheet([_row("Acme", "SWE", updates="2026-10-18: confirmation email")])
    sheet.set_status(2, "Needs human", note="resume upload failed")
    assert svc.written() == {
        "'Sheet1'!E2": "Needs human",
        "'Sheet1'!L2": "2026-10-18: confirmation email | 2026-10-19: resume upload failed",
    }


def test_set_status_first_note_has_no_separator():
    sheet, svc = _sheet([_row("Acme", "SWE")])
    sheet.set_status(2, "Skipped", note="not a fit")
    assert svc.written()["'Sheet1'!L2"] == "2026-10-19: not a fit"


def test_set_status_adds_status_column_when_missing():
    headers = ["Company", "Role", "Apply link"]
    sheet, svc = _sheet([["Acme", "SWE", "https://a.com"]], headers=headers)
    sheet.set_status(2, "In review")
    assert svc.written() == {"'Sheet1'!D1": "Status", "'Sheet1'!D2": "In review"}


def test_set_status_refinds_job_after_sheet_was_sorted():
    sheet, svc = _sheet([_row("Other", "PM"), _row("Acme", "SWE")])
    row = sheet.set_status(2, "Applied", company="Acme", role="SWE")
    assert row == 3
    assert svc.written() == {"'Sheet1'!E3": "Applied"}


def test_set_status_raises_when_job_is_gone():
    sheet, svc = _sheet([_row("Other", "PM")])
    with pytest.raises(SheetError):
        sheet.set_status(2, "Applied", company="Acme", role="SWE")
    assert svc.writes == []


def test_set_status_rejects_header_row():
    sheet, _ = _sheet([_row("Acme", "SWE")])
    with pytest.raises(SheetError):
        sheet.set_status(1, "Applied")


def test_no_sheet_id_is_a_spoken_error():
    with pytest.raises(SheetError, match="MUSE_SHEET_ID"):
        JobSheet("", "Sheet1").get_next_job()


# ── google_auth scope tracking ──────────────────────────────────────────────

def test_missing_scopes_reads_the_recorded_grant(tmp_path):
    token = tmp_path / "google.json"
    token.write_text(json.dumps({"scopes": [google_auth.GMAIL_MODIFY, google_auth.CALENDAR]}))
    assert google_auth.missing_scopes([google_auth.SPREADSHEETS, google_auth.CALENDAR], token) == [
        google_auth.SPREADSHEETS
    ]


def test_old_token_without_sheets_scope_asks_for_reauth(tmp_path):
    token = tmp_path / "google.json"
    token.write_text(json.dumps({"scopes": [google_auth.GMAIL_MODIFY], "token": "t"}))
    with pytest.raises(google_auth.GoogleAuthError, match="python -m aria.setup google"):
        google_auth.load_credentials([google_auth.SPREADSHEETS], google_auth.PERSONAL, token)


def test_not_connected_asks_for_setup(tmp_path):
    with pytest.raises(google_auth.GoogleAuthError, match="isn't connected"):
        google_auth.load_credentials([google_auth.SPREADSHEETS], google_auth.PERSONAL, tmp_path / "nope.json")


def test_auth_error_surfaces_as_sheet_error(monkeypatch):
    def boom(*a, **k):
        raise google_auth.GoogleAuthError("Google needs new permissions (spreadsheets). Run: python -m aria.setup google")
    monkeypatch.setattr(google_auth, "build_service", boom)
    with pytest.raises(SheetError, match="new permissions"):
        JobSheet("id", "Sheet1").get_next_job()


def test_setup_requests_every_scope_aria_uses():
    assert set(google_auth.SCOPES) == {
        google_auth.GMAIL_MODIFY, google_auth.GMAIL_LABELS, google_auth.GMAIL_COMPOSE,
        google_auth.CALENDAR, google_auth.SPREADSHEETS,
    }


# ── job_from_sheet tool ─────────────────────────────────────────────────────

def _tool():
    from aria.core.tool import ToolRegistry
    from aria.plugins.apply import ApplyPlugin
    reg = ToolRegistry()
    ApplyPlugin().register(reg)
    return reg.get("job_from_sheet")


def test_tool_speaks_next_job_and_remembers_it(monkeypatch):
    import aria.plugins.apply as apply_plugin
    job = {"row": 7, "company": "Conviva", "role": "AI/ML Engineer", "location": "Foster City, CA",
           "url": "https://www.conviva.ai/careers/job/1?gh_jid=1", "apply_link": "", "fields": {}}
    saved = {}
    monkeypatch.setattr(apply_plugin.sheets, "get_next_job", lambda: job)
    monkeypatch.setattr(apply_plugin.memory, "set_persistent", lambda k, v: saved.update({k: v}))
    out = _tool().execute({})
    assert "AI/ML Engineer at Conviva" in out and "row 7" in out and "conviva.ai" in out
    assert saved[apply_plugin.CURRENT_JOB_KEY] is job


def test_tool_reports_empty_sheet(monkeypatch):
    import aria.plugins.apply as apply_plugin
    monkeypatch.setattr(apply_plugin.sheets, "get_next_job", lambda: None)
    assert "no new jobs" in _tool().execute({})


def test_tool_never_raises(monkeypatch):
    import aria.plugins.apply as apply_plugin

    def boom():
        raise apply_plugin.sheets.SheetError("Google isn't connected yet. Run: python -m aria.setup google")
    monkeypatch.setattr(apply_plugin.sheets, "get_next_job", boom)
    assert "aria.setup google" in _tool().execute({})


# ── python -m aria.setup ────────────────────────────────────────────────────

def test_setup_without_client_secret_says_where_to_put_it(tmp_path, monkeypatch, capsys):
    import aria.setup as setup
    monkeypatch.setattr(google_auth, "CLIENT_SECRETS_PATH", tmp_path / "google_client_secret.json")
    assert setup.main(["gmail"]) == 1
    assert str(tmp_path / "google_client_secret.json") in capsys.readouterr().out


def test_setup_picks_account(monkeypatch):
    import aria.setup as setup
    seen = []
    monkeypatch.setattr(google_auth, "run_setup", lambda account, email=None: seen.append((account, email)) or 0)
    assert setup.main(["google"]) == 0
    assert setup.main(["google", "--account", "jobs"]) == 0
    assert setup.main(["google", "--account", "jobs", "--email", "j@example.com"]) == 0
    assert seen == [("personal", None), ("jobs", None), ("jobs", "j@example.com")]


def test_setup_rejects_unknown_account(monkeypatch, capsys):
    import aria.setup as setup
    monkeypatch.setattr(google_auth, "run_setup", lambda *a, **k: pytest.fail("must not run"))
    assert setup.main(["google", "--account", "me@example.com"]) == 1
    assert "--email" in capsys.readouterr().out
