"""tests/test_sheet_tab.py — the Muse sheet in the dashboard (read whole tab, edit one cell),
"mark … as applied" by voice, and the agent's job_status tool. Fake Sheets service, never Google."""
from __future__ import annotations

import json
import threading
from datetime import date

import pytest

from aria.dashboard import sheet_tab, sources
from aria.dashboard.__main__ import page_from, serve, settings_action
from aria.features.applying import commands, tabs
from aria.integrations import sheets
from tests.test_dashboard_tabs import _req
from tests.test_sheets import HEADERS, FakeSheetsService, _row


def _sheet(rows):
    svc = FakeSheetsService([HEADERS] + rows)
    return sheets.JobSheet("sheet-id", "Sheet1", service=svc, today=lambda: date(2026, 10, 22)), svc


ROWS = [_row("University at Buffalo (SUNY)", "Application Developer", "Applied"),
        _row("John Deere", "Data Analyst", "Not applied"),
        _row("Epic", "SWE", "Not applied", post_link="https://epic.com/jobs/1")]


# ── JobSheet.table / set_cell ───────────────────────────────────────────────

def test_table_is_every_row_padded_with_deduped_headers():
    sh, _ = _sheet(ROWS + [[""] * 3])
    t = sh.table()
    assert t["headers"][8] == "Outreach" and t["headers"][10] == "Outreach_2"
    assert [r["row"] for r in t["rows"]] == [2, 3, 4]                 # the blank row is left out
    assert all(len(r["cells"]) == len(t["headers"]) for r in t["rows"])
    assert t["rows"][2]["cells"][6] == "https://epic.com/jobs/1"


def test_set_cell_writes_one_cell_raw():
    sh, svc = _sheet(ROWS)
    assert sh.set_cell(3, "Status", "Applied", expect="Not applied", company="John Deere", role="Data Analyst") == 3
    assert svc.written() == {"'Sheet1'!E3": "Applied"} and svc.writes[0]["valueInputOption"] == "RAW"


def test_set_cell_follows_a_row_that_moved():
    sh, svc = _sheet(ROWS)
    sh.set_cell(2, "Location", "Moline, IL", company="John Deere", role="Data Analyst")
    assert svc.written() == {"'Sheet1'!C3": "Moline, IL"}


def test_set_cell_refuses_when_the_sheet_changed_meanwhile():
    sh, svc = _sheet(ROWS)
    with pytest.raises(sheets.SheetError, match="changed in the sheet"):
        sh.set_cell(3, "Status", "Skipped", expect="", company="John Deere", role="Data Analyst")
    assert svc.writes == []


def test_set_cell_never_makes_a_column():
    sh, svc = _sheet(ROWS)
    with pytest.raises(sheets.SheetError, match="no 'Salary' column"):
        sh.set_cell(3, "Salary", "100k")
    assert svc.writes == []


def test_tab_sheet_only_the_two_job_tabs():
    with pytest.raises(sheets.SheetError):
        sheets.tab_sheet("Secrets")


# ── dashboard: data / edit / endpoints ──────────────────────────────────────

def test_data_offers_every_status_in_the_column():
    sh, _ = _sheet(ROWS + [_row("Odd Co", "SWE", "Interview scheduled")])
    d = sheet_tab.data("jobs", sheet=sh)
    assert d["rows"][0]["cells"][4] == "Applied"
    assert "Applied" in d["statuses"] and "Interview scheduled" in d["statuses"]


def test_data_reports_a_read_failure_without_raising():
    class Broken:
        def table(self):
            raise sheets.SheetError("I couldn't read the job sheet: 403")
    assert sheet_tab.data("jobs", sheet=Broken()) == {"tab": "jobs", "error": "I couldn't read the job sheet: 403"}
    assert "error" in sheet_tab.data("nope")


def test_edit_writes_and_drops_the_page_cache(monkeypatch):
    sh, svc = _sheet(ROWS)
    sources.CACHE._items["sheet"] = (0.0, sources.Source(True, {}))
    out = sheet_tab.edit({"tab": "jobs", "row": 3, "column": "Status", "value": "Skipped", "expect": "Not applied",
                          "company": "John Deere", "role": "Data Analyst"}, sheet=sh)
    assert out == {"ok": True, "row": 3} and svc.written() == {"'Sheet1'!E3": "Skipped"}
    assert "sheet" not in sources.CACHE._items


@pytest.mark.parametrize("body,err", [
    ({"row": "3", "column": "Status", "value": "x"}, "row"),
    ({"row": 1, "column": "Status", "value": "x"}, "row"),
    ({"row": 3, "column": "", "value": "x"}, "column"),
    ({"row": 3, "column": "Status", "value": 5}, "value"),
    ({"row": 3, "column": "Status", "value": "x" * 5001}, "longer"),
])
def test_edit_rejects_bad_requests(body, err):
    sh, svc = _sheet(ROWS)
    with pytest.raises(sheet_tab.SheetEditError, match=err):
        sheet_tab.edit(dict(body, tab="jobs"), sheet=sh)
    assert svc.writes == []


def test_post_cell_maps_errors_to_400(monkeypatch):
    def refuse(body, sheet=None):
        raise sheets.SheetError("Your jobs Google account can read the job sheet but can't edit it.")
    monkeypatch.setattr(sheet_tab, "edit", refuse)
    status, out = settings_action("/api/sheet/cell", {"row": 3})
    assert status == 400 and "can't edit" in out["error"]


@pytest.fixture
def server():
    asked = []
    srv = serve(0, build=lambda: "<p>page</p>", token="tok123",
                sheet=lambda tab: asked.append(tab) or {"tab": tab, "headers": ["Company"], "rows": []},
                act=lambda path, body: (200, {"path": path}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, asked
    srv.shutdown()
    srv.server_close()


def test_sheet_endpoints(server):
    srv, asked = server
    status, body = _req(srv, "GET", "/api/sheet?tab=applied")
    assert status == 200 and json.loads(body)["tab"] == "applied" and asked == ["applied"]
    assert _req(srv, "POST", "/api/sheet/cell", {"row": 3})[0] == 403             # no token
    assert _req(srv, "GET", "/api/sheet", headers={"Host": "evil.test:80"})[0] == 403


def test_page_has_the_sheet_tab():
    html = page_from({}, token="t")
    assert 'id="sheet-tab"' in html and 'href="#sheet"' in html and "/api/sheet?tab=" in html


# ── voice: mark … as applied ────────────────────────────────────────────────

@pytest.fixture
def marks(monkeypatch, tmp_path):
    monkeypatch.setattr(tabs, "REGISTRY_PATH", tmp_path / "apply_tabs.json")
    jobs = [{"row": 2, "company": "University at Buffalo (SUNY)", "role": "Application Developer"},
            {"row": 19, "company": "1 Data Analytics Inc", "role": "Junior Data Analyst"},
            {"row": 22, "company": "Peregrine Advisors Benefit Inc.", "role": "Data Analyst"},
            {"row": 23, "company": "Peregrine Advisors Benefit Inc.", "role": "BI Analyst"}]
    written = []
    monkeypatch.setattr(sheets, "get_job", lambda row: next((j for j in jobs if j["row"] == row), None))
    monkeypatch.setattr(sheets, "default_sheet", lambda: type("S", (), {"jobs": lambda self: jobs})())
    monkeypatch.setattr(sheets, "set_status", lambda row, status, note="", company=None, role=None:
                        written.append((row, status, company)) or row)
    last = {"job": None}
    monkeypatch.setattr(commands, "_last_job", lambda: last["job"])
    monkeypatch.setattr(commands.agent_browser, "front_tabs", lambda names: [])
    commands._asking.clear()
    return {"written": written, "last": last, "jobs": jobs}


def test_the_users_own_sentence_marks_the_job_aria_just_filled(marks):
    marks["last"]["job"] = marks["jobs"][1]
    out = commands.handle("Yeah, I've submitted right now. You can mark the field as applied.")
    assert marks["written"] == [(19, "Applied", "1 Data Analytics Inc")]
    assert out == "Marked Junior Data Analyst at 1 Data Analytics Inc Applied in the sheet, row 19."


@pytest.mark.parametrize("text,row,status", [
    ("mark number 19 as applied", 19, "Applied"),
    ("mark row 2 as not applied", 2, "Not applied"),
    ("set 1 data analytics to skipped", 19, "Skipped"),
    ("mark university at buffalo as applied", 2, "Applied"),
])
def test_mark_by_row_or_company(marks, text, row, status):
    commands.handle(text)
    assert [(r, s) for r, s, _ in marks["written"]] == [(row, status)]


def test_mark_asks_when_unclear(marks):
    assert "rows 22, 23" in commands.handle("mark peregrine as applied")
    assert commands.handle("mark it as applied").startswith("Which job?")
    assert "isn't a job" in commands.handle("mark row 99 as applied")
    assert marks["written"] == []


def test_mark_uses_the_application_tab_and_flags_it_submitted(marks):
    e = tabs.add({"row": 19, "company": "1 Data Analytics Inc", "role": "Junior Data Analyst", "url": "https://x/1"})
    tabs.update(e["id"], state=tabs.FILLED)
    commands.handle("mark it as applied")
    assert marks["written"] == [(19, "Applied", "1 Data Analytics Inc")]
    got = tabs.get(e["id"])
    assert got["state"] == tabs.SUBMITTED and got["sheet_marked"]


@pytest.mark.parametrize("text", ["don't mark it as applied", "mark my calendar for tuesday",
                                  "what did you mark as applied"])
def test_mark_leaves_other_talk_alone(marks, text):
    marks["last"]["job"] = marks["jobs"][1]
    out = commands.handle(text)
    assert marks["written"] == [] and (out is None or "Which job" in out or "couldn't find" in out)


# ── agent tool: job_status ──────────────────────────────────────────────────

def _tool(marks):
    from aria.core.tool import ToolRegistry
    from aria.plugins.apply import ApplyPlugin
    reg = ToolRegistry()
    ApplyPlugin(None, None, None, "").register(reg)
    return reg.get("job_status")


def test_job_status_tool_writes_through_the_api(marks):
    out = _tool(marks).execute({"row": 22, "status": "Skipped"})
    assert marks["written"] == [(22, "Skipped", "Peregrine Advisors Benefit Inc.")] and "is now Skipped" in out


def test_job_status_tool_never_sets_applied(marks):
    out = _tool(marks).execute({"row": 19, "status": "Applied"})
    assert marks["written"] == [] and "mark row 19 as applied" in out


# ── Applied → moved to the Applied tab ──────────────────────────────────────

APPLIED_HEADERS = ["Company", "Role", "Profile alignment /5", "Status", "Channel", "Apply link", "Post link",
                   "Who to contact", "Source", "Why it's a fit + draft", "Outcome"]


class Book:
    """Two tabs of one spreadsheet: values get/append/batchUpdate + spreadsheet get/batchUpdate."""

    def __init__(self, tabs):
        self.tabs = tabs              # {title: [[...], ...]} header row first
        self.calls = []

    def for_tab(self, title):
        book = self

        class Svc:
            def spreadsheets(self):
                return self

            def values(self):
                return self

            def get(self, spreadsheetId, range=None, fields=None):
                if fields:
                    return _X({"sheets": [{"properties": {"title": t, "sheetId": i}} for i, t in enumerate(book.tabs)]})
                return _X({"values": [list(r) for r in book.tabs[title]]})

            def append(self, spreadsheetId, range, valueInputOption, insertDataOption, body):
                book.calls.append(("append", title, body["values"][0]))
                book.tabs[title].append(body["values"][0])
                return _X({"updates": {"updatedRange": f"'{title}'!A{len(book.tabs[title])}:K{len(book.tabs[title])}"}})

            def batchUpdate(self, spreadsheetId, body):
                for req in body.get("requests", []):
                    rng = req["deleteDimension"]["range"]
                    t = list(book.tabs)[rng["sheetId"]]
                    book.calls.append(("delete", t, rng["startIndex"] + 1))
                    del book.tabs[t][rng["startIndex"]]
                for d in body.get("data", []):
                    t, a1 = d["range"].split("!")
                    t = t.strip("'")
                    m = __import__("re").match(r"([A-Z]+)(\d+)", a1)
                    col, row = ord(m.group(1)) - 65, int(m.group(2))
                    r = book.tabs[t][row - 1]
                    r += [""] * (col + 1 - len(r))
                    r[col] = d["values"][0][0]
                    book.calls.append(("write", t, a1, d["values"][0][0]))
                return _X({})
        return sheets.JobSheet("sid", title, service=Svc())


class _X:
    def __init__(self, r):
        self.r = r

    def execute(self):
        return self.r


def _book():
    jobs = [HEADERS, ["Acme", "SWE", "NYC", "4", "Not applied", "https://a/apply", "https://a/post", "Jane", "", "fit", "",
                      "2026-10-20: found"],
            ["John Deere", "Data Analyst", "Moline", "3", "Not applied", "https://jd/apply"]]
    applied = [APPLIED_HEADERS, ["Old Co", "SWE", "3", "Rejected"]]
    b = Book({"Sheet1": jobs, "Applied": applied})
    return b, b.for_tab("Sheet1"), b.for_tab("Applied")


def test_applied_moves_the_row_by_column_name():
    book, jobs, applied = _book()
    out = sheet_tab.edit({"tab": "jobs", "row": 2, "column": "Status", "value": "Applied", "expect": "Not applied",
                          "company": "Acme", "role": "SWE"}, sheet=jobs, applied_sheet=applied)
    assert out == {"ok": True, "moved": "applied", "row": 3}
    assert book.tabs["Applied"][2] == ["Acme", "SWE", "4", "Applied", "", "https://a/apply", "https://a/post", "Jane",
                                       "", "fit", ""]
    assert [r[0] for r in book.tabs["Sheet1"][1:]] == ["John Deere"]
    assert [c[0] for c in book.calls] == ["append", "delete"]         # added before it's removed


def test_move_follows_a_sort_and_never_deletes_the_wrong_row():
    book, jobs, applied = _book()
    book.tabs["Sheet1"][1], book.tabs["Sheet1"][2] = book.tabs["Sheet1"][2], book.tabs["Sheet1"][1]
    sheet_tab.edit({"tab": "jobs", "row": 2, "column": "Status", "value": "Applied", "expect": "Not applied",
                    "company": "Acme", "role": "SWE"}, sheet=jobs, applied_sheet=applied)
    assert [r[0] for r in book.tabs["Sheet1"][1:]] == ["John Deere"] and ("delete", "Sheet1", 3) in book.calls


def test_move_refuses_when_status_changed_meanwhile():
    book, jobs, applied = _book()
    book.tabs["Sheet1"][1][4] = "Skipped"
    with pytest.raises(sheets.SheetError, match="changed"):
        sheet_tab.edit({"tab": "jobs", "row": 2, "column": "Status", "value": "Applied", "expect": "Not applied",
                        "company": "Acme", "role": "SWE"}, sheet=jobs, applied_sheet=applied)
    assert book.calls == []


def test_already_in_applied_only_sets_its_status_then_removes_the_job_row():
    book, jobs, applied = _book()
    book.tabs["Applied"].append(["Acme", "SWE", "4", "Not applied"])
    out = sheet_tab.edit({"tab": "jobs", "row": 2, "column": "Status", "value": "Applied", "expect": "Not applied",
                          "company": "Acme", "role": "SWE"}, sheet=jobs, applied_sheet=applied)
    assert out["row"] == 3 and book.tabs["Applied"][2][3] == "Applied" and len(book.tabs["Applied"]) == 3
    assert [c[0] for c in book.calls] == ["write", "delete"]


def test_other_statuses_and_the_applied_tab_stay_plain_cell_edits():
    book, jobs, applied = _book()
    sheet_tab.edit({"tab": "jobs", "row": 3, "column": "Status", "value": "Skipped", "expect": "Not applied",
                    "company": "John Deere", "role": "Data Analyst"}, sheet=jobs, applied_sheet=applied)
    sheet_tab.edit({"tab": "applied", "row": 2, "column": "Status", "value": "Applied", "expect": "Rejected",
                    "company": "Old Co", "role": "SWE"}, sheet=applied)
    assert [c[0] for c in book.calls] == ["write", "write"]


def test_page_has_a_refresh_button():
    assert 'id="shrefresh"' in sheet_tab.render()
