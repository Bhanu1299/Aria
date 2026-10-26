"""tests/test_dashboard_contacts.py — Muse's Companies + Contacts tabs in the dashboard: read with
the job tabs, joined to jobs by ID, shown on Outreach, editable on the Sheet tab."""
from __future__ import annotations

from datetime import datetime

import pytest

from aria.dashboard import model as m
from aria.dashboard import outreach, sheet_tab, sources
from aria.dashboard.__main__ import badges, page_from
from aria.dashboard.sources import Source
from aria.integrations.header_sheet import HeaderSheet
from aria.integrations.sheets import SheetError
from tests.test_contacts_tab import CO_HEADERS, PEOPLE_HEADERS, LiveSheet, company, person

NOW = datetime(2026, 10, 26, 12, 0)


def job(n, jid, name, role):
    return {"_row": n, "ID": jid, "Company": name, "Role": role, "Status": "Applied", "Apply link": "",
            "Post link": "", "Who to contact": "", "Profile alignment /5": "4"}


def rows(headers, *lines):
    return [{**dict(zip(headers, line)), "_row": i} for i, line in enumerate(lines, start=2)]


@pytest.fixture
def sheet():
    return {
        "current": [job(2, "J1", "Acme", "Founding Engineer"), job(3, "J2", "Statejobs", "Analyst"),
                    job(4, "J3", "Beta", "SWE")],
        "applied": [],
        "companies": rows(CO_HEADERS, company("J1", "Acme"),
                          company("J2", "Statejobs", outreach="skip", blocker="Government agency: portal only"),
                          company("J3", "Beta")),
        "contacts": rows(PEOPLE_HEADERS,
                         person("J1", "Acme", "Pat Lee", "1", Email="pat@acme.com", Confidence="verified",
                                **{"Find status": "done", "Draft status": "drafted",
                                   "Gmail draft link": "https://mail.google.com/mail/u/x/#drafts?compose=1"}),
                         person("J1", "Acme", "Ana Diaz", "2", Email="ana@acme.com", Confidence="pattern",
                                **{"Find status": "done"}),
                         person("J2", "Statejobs", "Ed Fu", "1"),
                         person("J3", "Beta", "—", "1"),
                         person("J3", "Beta", "Cy Do", "2", Email="not found", Confidence="not found",
                                **{"Find status": "done", "Notes": "2026-10-25: Coldstart found no address; no draft"}),
                         person("", "Nowhere", "Bo Li", "", **{"Added by": "Bhanu"})),
    }


def mdl_of(sheet):
    return m.build({"sheet": Source(True, sheet)}, NOW)


def test_contacts_join_jobs_by_id_and_move_their_stage(sheet):
    mdl = mdl_of(sheet)
    jobs = {j.company: j for j in mdl["jobs"]}
    acme = jobs["Acme"]
    assert acme.stage_label == "Draft ready" and [p["Person"] for p in acme.people] == ["Pat Lee", "Ana Diaz"]
    assert acme.company_row["Outreach status"] == "active" and acme.email == "pat@acme.com"
    assert jobs["Beta"].stage_label == "Contact found"
    assert [c["job"] is None for c in mdl["contacts"]] == [False] * 5 + [True]


def test_outreach_card_sorts_people_by_state(sheet):
    mdl = mdl_of(sheet)
    states = [outreach.contact_state(c) for c in mdl["contacts"]]
    assert states == ["review", "waiting", "held", "lookup", "needs", "needs"]
    html = outreach.contacts_card(mdl)
    assert 'data-p="review"' in html and "open draft" in html and "Pat Lee" in html
    assert "Government agency: portal only" in html                    # Held shows the Blocker
    assert "no Job ID" in html and "team inbox" in html
    assert outreach.review_count(mdl) == 1
    assert badges(mdl, 0)["outreach"] == 1


def test_outreach_tab_renders_in_the_page(sheet):
    html = page_from({"sheet": Source(True, sheet)}, NOW, calls={"calls": [], "via": ""}, flight=[], runs=[])
    assert "Drafts to review" in html and "Muse sheet, Contacts + Companies" in html


def test_fetch_sheet_reads_the_outreach_tabs_and_falls_back_without_them(monkeypatch):
    import aria.core.config as config
    monkeypatch.setattr(config, "MUSE_SHEET_ID", "sheet-id")
    seen = []

    class Svc:
        def __init__(self, fail_outreach):
            self.fail = fail_outreach

        def spreadsheets(self):
            return self

        def values(self):
            return self

        def batchGet(self, spreadsheetId, ranges):
            seen.append(len(ranges))
            if self.fail and len(ranges) == 4:
                raise RuntimeError("Unable to parse range: 'Companies'")
            self.n = len(ranges)
            return self

        def execute(self):
            return {"valueRanges": [{"values": [["Company"], [f"Tab{i}"]]} for i in range(self.n)]}

    out = sources.fetch_sheet(Svc(False))
    assert out["companies"][0]["Company"] == "Tab2" and out["contacts"][0]["Company"] == "Tab3"
    out = sources.fetch_sheet(Svc(True))
    assert "companies" not in out and out["current"][0]["Company"] == "Tab0" and seen == [4, 4, 2]


# ── Sheet tab: Companies / Contacts ─────────────────────────────────────────

def _tab(headers, *lines, tab="Contacts"):
    svc = LiveSheet([headers, *lines])
    return HeaderSheet("sheet-id", tab, service=svc), svc


def test_sheet_tab_shows_contacts_with_their_dropdowns():
    sh, _ = _tab(PEOPLE_HEADERS, person("J1", "Acme", "Pat Lee", "1", **{"Draft status": "drafted"}))
    d = sheet_tab.data("contacts", sheet=sh)
    ds = PEOPLE_HEADERS.index("Draft status")
    assert d["id_col"] == "Job ID" and d["rows"][0]["id"] == "J1"
    assert d["choices"][str(ds)][:6] == ["none", "drafted", "approved", "sent", "replied", "no reply"]
    co, _ = _tab(CO_HEADERS, company("J1", "Acme"), tab="Companies")
    assert sheet_tab.data("companies", sheet=co)["choices"][str(CO_HEADERS.index("Outreach status"))][0] == "active"


def test_sheet_tab_edits_a_contact_found_by_job_id_and_person():
    sh, svc = _tab(PEOPLE_HEADERS, person("J1", "Acme", "Ana Diaz", "2"),
                   person("J1", "Acme", "Pat Lee", "1", **{"Draft status": "drafted"}))
    out = sheet_tab.edit({"tab": "contacts", "row": 2, "id": "J1", "column": "Draft status", "value": "sent",
                          "expect": "drafted", "company": "Acme", "person": "Pat Lee"}, sheet=sh)
    assert out["row"] == 3 and svc.cell(3, "Draft status") == "sent" and svc.cell(2, "Draft status") == "none"
    with pytest.raises(SheetError, match="changed in the sheet"):
        sheet_tab.edit({"tab": "contacts", "row": 3, "id": "J1", "column": "Draft status", "value": "replied",
                        "expect": "drafted", "company": "Acme", "person": "Pat Lee"}, sheet=sh)


def test_archive_tabs_are_never_reachable():
    assert "error" in sheet_tab.data("archive")
    with pytest.raises(sheet_tab.SheetEditError):
        sheet_tab._tab_name("Contacts backup")
