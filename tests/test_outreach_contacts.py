"""tests/test_outreach_contacts.py — step 1: find people to contact (pure code, mocked HTTP + sheet)."""
from __future__ import annotations

from datetime import date

import httpx
import pytest

from aria.features.outreach import contacts as c
from aria.features.outreach import pipeline
from aria.integrations.sheets import JobSheet, SheetError
from tests.test_sheets import HEADERS, FakeSheetsService


def _job(row=2, company="Acme", role="ML Engineer", fit="4", who="Via Greenhouse", outreach2="", **links):
    fields = {"Company": company, "Role": role, "Profile alignment /5": fit, "Who to contact": who,
              "Outreach_2": outreach2, "Status": links.pop("status", "")}
    return {"row": row, "company": company, "role": role, "alignment": fit, "status": fields["Status"],
            "post_link": links.get("post_link", ""), "apply_link": links.get("apply_link", ""), "fields": fields}


# ── ranking ─────────────────────────────────────────────────────────────────

def test_recruiters_outrank_leaders_outrank_engineers():
    people = [c.Person("Sam Roe", "Senior Software Engineer", "team"),
              c.Person("Ana Diaz", "VP of Engineering", "team"),
              c.Person("Pat Lee", "Technical Recruiter", "search"),
              c.Person("Kim Wu", "Account Executive", "team"),
              c.Person("Bo Chan", "Founder & CEO", "team")]
    ranked = c.rank(people)
    assert [p.name for p in ranked] == ["Pat Lee", "Ana Diaz", "Bo Chan"]


def test_rank_dedupes_and_keeps_best_source_and_email():
    people = [c.Person("Pat Lee", "Recruiter", "search", url="https://linkedin.com/in/patlee"),
              c.Person("Pat Lee", "Recruiter", "posting", email="pat.lee@acme.com")]
    [p] = c.rank(people)
    assert p.source == "posting" and p.email == "pat.lee@acme.com" and p.url.endswith("patlee")


def test_title_score_drops_irrelevant_titles():
    assert c.title_score("Sales Engineer") == 0
    assert c.title_score("") == 0
    assert c.title_score("University Recruiter") > c.title_score("Recruiter") > c.title_score("CTO")


@pytest.mark.parametrize("text,ok", [("Pat Lee", True), ("Mary-Jane O'Neil", True), ("About Us", False),
                                     ("Senior Engineer", False), ("pat lee", False), ("Pat", False)])
def test_looks_like_name(text, ok):
    assert c.looks_like_name(text) is ok


# ── sources ─────────────────────────────────────────────────────────────────

def test_posting_names_and_published_addresses():
    text = ("Questions? Reach out to Pat Lee (Technical Recruiter). You'll report to Ana Diaz, our "
            "Head of AI, at Acme. Or email sam.roe@acme.com. Apply at jobs@acme.com.")
    people = {p.name: p for p in c.from_posting(text, "acme.com")}
    assert people["Pat Lee"].title == "Technical Recruiter"
    assert people["Ana Diaz"].title == "Head of AI"
    assert people["Sam Roe"].email == "sam.roe@acme.com"
    assert "Jobs" not in " ".join(people)


def test_team_page_two_line_and_one_line_layouts():
    html = """<html><script>var x = "Fake Person";</script>
      <div><h3>Ana Diaz</h3><p>Co-founder &amp; CTO</p></div>
      <div><h3>Our Team</h3><p>Engineering</p></div>
      <li>Pat Lee — Talent Partner</li><li>Kim Wu, Head of Marketing</li></html>"""
    people = {p.name: p.title for p in c.from_team_page(html)}
    assert people == {"Ana Diaz": "Co-founder & CTO", "Pat Lee": "Talent Partner"}


def test_search_results_parse_linkedin_titles_for_this_company_only():
    results = [
        {"title": "Pat Lee - Technical Recruiter - Acme | LinkedIn", "url": "https://www.linkedin.com/in/patlee", "snippet": ""},
        {"title": "Ana Diaz – Engineering Manager at Acme | LinkedIn", "url": "https://linkedin.com/in/anadiaz", "snippet": ""},
        {"title": "Bo Chan - Recruiter - Globex | LinkedIn", "url": "https://linkedin.com/in/bochan", "snippet": "Globex"},
        {"title": "Acme careers", "url": "https://acme.com/careers", "snippet": "Acme"},
    ]
    people = c.from_search_results(results, "Acme")
    assert [(p.name, p.title) for p in people] == [("Pat Lee", "Technical Recruiter"), ("Ana Diaz", "Engineering Manager")]


def test_company_domain_order():
    assert c.company_domain(_job(post_link="https://boards.greenhouse.io/acme/jobs/1",
                                 apply_link="https://careers.acme.ai/jobs/1")) == "acme.ai"
    j = _job(post_link="https://jobs.lever.co/acme/1")
    j["fields"]["Outreach"] = "write to pat@acme.io"
    assert c.company_domain(j) == "acme.io"
    search = lambda q: [{"url": "https://www.linkedin.com/company/acme"}, {"url": "https://www.acmerobotics.com/"}]
    assert c.company_domain(_job(company="Acme Robotics", post_link="https://jobs.lever.co/x"), search) == "acmerobotics.com"
    assert c.company_domain(_job(post_link="https://jobs.lever.co/x"), lambda q: []) == ""
    assert c.registrable("jobs.eu.acme.co.uk") == "acme.co.uk"


def _transport(routes: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for prefix, (status, body, ctype) in routes.items():
            if url.startswith(prefix):
                return httpx.Response(status, text=body if isinstance(body, str) else None,
                                      json=None if isinstance(body, str) else body,
                                      headers={"content-type": ctype})
        return httpx.Response(404, text="")
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_github_members_need_an_org_that_points_at_the_company():
    client = _transport({
        "https://api.github.com/orgs/acme/public_members": (200, [{"login": "ana"}, {"login": "anon"}], "application/json"),
        "https://api.github.com/orgs/acme": (200, {"login": "acme", "blog": "https://acme.com", "name": "Acme"}, "application/json"),
        "https://api.github.com/users/ana": (200, {"name": "Ana Diaz", "bio": "CTO @ Acme", "html_url": "https://github.com/ana"}, "application/json"),
        "https://api.github.com/users/anon": (200, {"name": None}, "application/json"),
    })
    people = c.github_people("Acme", "acme.com", client)
    assert [(p.name, p.title, p.source) for p in people] == [("Ana Diaz", "CTO", "github")]
    wrong = _transport({"https://api.github.com/orgs/acme": (200, {"login": "acme", "blog": "https://other.org", "name": "ACME Corp Intl"}, "application/json")})
    assert c.github_people("Acme", "acme.com", wrong) == []


def test_find_people_combines_sources_and_respects_robots():
    client = _transport({
        "https://acme.com/robots.txt": (200, "User-agent: *\nDisallow: /people\n", "text/plain"),
        "https://acme.com/team": (200, "<h3>Ana Diaz</h3><p>Head of Engineering</p>", "text/html"),
        "https://acme.com/people": (200, "<h3>Hidden Person</h3><p>Recruiter</p>", "text/html"),
        "https://boards.greenhouse.io/acme/jobs/1": (200, "<p>Contact Pat Lee (Recruiter) with questions.</p>", "text/html"),
    })
    search = lambda q: [{"title": "Sam Roe - Engineering Manager - Acme | LinkedIn", "url": "https://linkedin.com/in/samroe"}]
    people = c.find_people("Acme", "acme.com", ["https://boards.greenhouse.io/acme/jobs/1"], search_fn=search, client=client)
    assert [p.name for p in people] == ["Pat Lee", "Ana Diaz", "Sam Roe"]
    assert all(p.name != "Hidden Person" for p in people)


def test_find_people_never_raises():
    def boom(_q):
        raise RuntimeError("search down")
    client = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))))
    assert c.find_people("Acme", "acme.com", ["https://acme.com/jobs/1"], search_fn=boom, client=client) == []


# ── row selection + sheet write ─────────────────────────────────────────────

def test_needs_contact_rules():
    assert pipeline.needs_contact(_job())
    assert not pipeline.needs_contact(_job(fit="2"))
    assert not pipeline.needs_contact(_job(who="Pat Lee (Recruiter)"))
    assert not pipeline.needs_contact(_job(outreach2="LinkedIn: Pat Lee — hi"))
    assert not pipeline.needs_contact(_job(status="Not gonna apply"))
    assert pipeline.needs_contact(_job(fit="3.5", status="Applied"))


def test_who_cell_keeps_the_placeholder():
    people = [c.Person("Pat Lee", "Recruiter"), c.Person("Ana Diaz", "CTO")]
    assert pipeline.who_cell(people, "Via Breezy ATS") == "Pat Lee (Recruiter); Ana Diaz (CTO) | Via Breezy ATS"
    assert pipeline.who_cell(people[:1], "") == "Pat Lee (Recruiter)"


def _row(company, role, fit="4", who="Via Greenhouse", post="https://boards.greenhouse.io/x/jobs/1"):
    return [company, role, "NYC", fit, "", "", post, who]


def _sheet(rows):
    svc = FakeSheetsService([HEADERS] + rows)
    return JobSheet("sheet-id", "Sheet1", service=svc, today=lambda: date(2026, 10, 20)), svc


def test_run_outreach_writes_who_to_contact_for_qualifying_rows(monkeypatch):
    sheet, svc = _sheet([_row("Acme", "ML Engineer"), _row("Globex", "SWE", fit="2"),
                         _row("Initech", "AI Eng", who="Bo Chan (Recruiter)")])
    monkeypatch.setattr(pipeline.finder, "company_domain", lambda job, search_fn=None: "acme.com")
    monkeypatch.setattr(pipeline.finder, "find_people",
                        lambda *a, **k: [c.Person("Pat Lee", "Recruiter", "search")])
    monkeypatch.setattr(pipeline.channel, "find_emails", lambda domain, people, **k: people)
    monkeypatch.setattr(pipeline.writer, "linkedin_note", lambda t: None)
    pipeline.run_outreach(sheet=sheet, search_fn=lambda q: [], gmail_data={}, gmail=object())
    assert svc.written() == {"'Sheet1'!H2": "Pat Lee (Recruiter) | Via Greenhouse",
                             "'Sheet1'!L2": "2026-10-20: Aria found contacts"}


def test_run_outreach_with_nothing_to_do():
    sheet, svc = _sheet([_row("Acme", "ML Engineer", who="Pat Lee (Recruiter)")])
    assert "already has outreach" in pipeline.run_outreach(sheet=sheet, search_fn=lambda q: [], gmail_data={}, gmail=object())
    assert svc.writes == []


def test_set_fields_refuses_columns_outside_outreach():
    sheet, svc = _sheet([_row("Acme", "ML Engineer")])
    with pytest.raises(SheetError):
        sheet.set_fields(2, {"Status": "Applied"})
    assert sheet.set_fields(2, {"Outreach_2": "LinkedIn: hi"}, note="outreach planned") == 2
    assert svc.written() == {"'Sheet1'!K2": "LinkedIn: hi", "'Sheet1'!L2": "2026-10-20: outreach planned"}
