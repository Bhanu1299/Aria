"""tests/test_outreach_pipeline.py — step 3: outreach writing, drafts, sheet cells, dashboard (all mocked)."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from aria.dashboard import model as m
from aria.dashboard import outreach as outreach_tab
from aria.dashboard import today
from aria.dashboard.sources import Source
from aria.features.outreach import pipeline, writer
from aria.features.outreach.contacts import Person
from aria.integrations.sheets import JobSheet
from types import SimpleNamespace
from tests.test_sheets import HEADERS, FakeSheetsService

BODY = ("I'm a recent MS CS graduate who has shipped a voice agent with a tool-calling loop and "
        "an evaluation harness, and the ML Engineer role at Acme lines up closely with that work. "
        "I'd love to hear how the team thinks about evaluation for production models.\n\n"
        "Would you have fifteen minutes in the next couple of weeks, or could you point me to the "
        "hiring manager for this role? Thank you for considering it.")


class FakeGmail:
    def __init__(self, result="Draft saved for x: y. It's in the drafts of jobs; nothing was sent."):
        self.drafts, self.result = [], result

    def create_draft(self, to, subject, body, cc="", reply_to_id=""):
        self.drafts.append({"to": to, "subject": subject, "body": body})
        return self.result


@pytest.fixture(autouse=True)
def profile(monkeypatch):
    bank = {"answers": {"first_name": "Alex", "last_name": "Doe", "school": "State University",
                        "linkedin": "https://linkedin.com/in/me", "sponsorship": "Yes", "phone": "555"}}
    monkeypatch.setattr(writer.answer_bank, "load", lambda: bank)
    monkeypatch.setattr(writer, "_identity", lambda: {"summary": "Builds voice agents."})
    monkeypatch.setattr(writer, "_resume", lambda: '')


def _llm(monkeypatch, text):
    calls = []

    def complete_checked(task, messages, validate, system=None, max_tokens=1024):
        calls.append({"task": task, "content": messages[0]["content"]})
        return SimpleNamespace(text=text, stop_reason="end_turn"), validate(text)
    monkeypatch.setattr(writer.llm_client, "complete_checked", complete_checked)
    return calls


def _target(**contact):
    job = {"row": 2, "company": "Acme", "role": "ML Engineer",
           "fields": {"Why it's a fit + draft": "Strong match: agents + eval."}}
    t = pipeline.Target("Acme", [job])
    t.contact = Person(contact.get("name", "Pat Lee"), "Recruiter", "search", url=contact.get("url", ""))
    return t


# ── writer ──────────────────────────────────────────────────────────────────

def test_email_wraps_llm_body_with_code_greeting_subject_and_signoff(monkeypatch):
    calls = _llm(monkeypatch, BODY)
    msg = writer.email(_target(), "pat@acme.com")
    assert msg["subject"] == "ML Engineer — Alex Doe"
    assert msg["body"].startswith("Hi Pat,\n\nI'm a recent")
    assert msg["body"].endswith("Best,\nAlex Doe\nhttps://linkedin.com/in/me")
    assert calls[0]["task"] == "outreach_email"
    assert "Strong match" in calls[0]["content"]
    assert "sponsorship" not in calls[0]["content"] and "555" not in calls[0]["content"]


@pytest.mark.parametrize("bad", ["Hi Pat, " + BODY, BODY + "\n\nBest,", "Too short.", BODY.replace("Acme", "[Company]")])
def test_email_rejects_invalid_bodies(monkeypatch, bad):
    _llm(monkeypatch, bad)
    assert writer.email(_target(), "pat@acme.com") is None


def test_linkedin_note_respects_the_300_char_cap(monkeypatch):
    calls = _llm(monkeypatch, "I just finished my MS CS and build voice agents; the ML Engineer role at Acme fits. Open to a quick chat?")
    note = writer.linkedin_note(_target())
    assert note.startswith("Hi Pat, I just finished") and len(note) <= writer.NOTE_LIMIT
    assert calls[0]["task"] == "outreach_linkedin_note"
    _llm(monkeypatch, "x " * 200)
    assert writer.linkedin_note(_target()) is None


def test_writer_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("gateway down")
    monkeypatch.setattr(writer.llm_client, "complete_checked", boom)
    assert writer.email(_target(), "pat@acme.com") is None
    assert writer.linkedin_note(_target()) is None


# ── pipeline ────────────────────────────────────────────────────────────────

def _sheet(rows):
    svc = FakeSheetsService([HEADERS] + rows)
    return JobSheet("sheet-id", "Sheet1", service=svc, today=lambda: date(2026, 10, 20)), svc


def _row(company="Acme", role="ML Engineer", who="Via Greenhouse", updates=None):
    r = [company, role, "NYC", "4", "", "", "https://boards.greenhouse.io/acme/jobs/1", who, "", "", ""]
    return r + ([updates] if updates else [])


def _people(monkeypatch, lookup):
    def find(company, domain, urls, **k):
        p = Person("Pat Lee", "Recruiter", "search", url="https://linkedin.com/in/patlee")
        return [p]

    def emails(domain, people, **k):
        for p in people:
            p.lookup = dict(lookup)
        return people
    monkeypatch.setattr(pipeline.finder, "company_domain", lambda job, search_fn=None: "acme.com")
    monkeypatch.setattr(pipeline.finder, "find_people", find)
    monkeypatch.setattr(pipeline.channel, "find_emails", emails)


def test_email_channel_creates_a_jobs_gmail_draft_and_marks_the_sheet(monkeypatch):
    _people(monkeypatch, {"email": "pat@acme.com", "status": "valid", "confidence": 0.97})
    _llm(monkeypatch, BODY)
    sheet, svc = _sheet([_row()])
    gmail = FakeGmail()
    out = pipeline.run_outreach(sheet=sheet, search_fn=lambda q: [], gmail_data={}, gmail=gmail)
    assert out == "Outreach: 1 email draft in your jobs Gmail. Nothing was sent."
    assert [d["to"] for d in gmail.drafts] == ["pat@acme.com"]
    w = svc.written()
    assert w["'Sheet1'!K2"] == "Email draft ready — Pat Lee (Recruiter), pat@acme.com"
    assert "Aria outreach: email draft in the jobs Gmail (Pat Lee)" in w["'Sheet1'!L2"]


def test_linkedin_channel_writes_the_note_to_the_sheet_not_gmail(monkeypatch):
    _people(monkeypatch, {"email": "pat@acme.com", "status": "invalid", "confidence": 0.0})
    _llm(monkeypatch, "I build voice agents and the ML Engineer role at Acme fits my MS work. Open to connecting?")
    sheet, svc = _sheet([_row()])
    gmail = FakeGmail()
    out = pipeline.run_outreach(sheet=sheet, search_fn=lambda q: [], gmail_data={}, gmail=gmail)
    assert "1 LinkedIn note in the sheet" in out
    assert gmail.drafts == []
    cell = svc.written()["'Sheet1'!K2"]
    assert cell.startswith("LinkedIn (Aria): Pat Lee (Recruiter) — https://linkedin.com/in/patlee\nNote: Hi Pat, I build")


def test_failed_draft_leaves_outreach_empty_and_retries_next_run(monkeypatch):
    _people(monkeypatch, {"email": "pat@acme.com", "status": "valid", "confidence": 0.97})
    _llm(monkeypatch, BODY)
    sheet, svc = _sheet([_row()])
    out = pipeline.run_outreach(sheet=sheet, search_fn=lambda q: [], gmail_data={}, gmail=FakeGmail("Jobs Gmail isn't connected yet."))
    assert "1 failed (Acme)" in out
    assert "'Sheet1'!K2" not in svc.written()
    # next run: the sheet now names Pat (written by Aria) — reuse them, don't search again
    rerun = _row(who="Pat Lee (Recruiter) | Via Greenhouse", updates="2026-10-20: Aria found contacts")
    [t] = pipeline.targets(JobSheet("s", "Sheet1", service=FakeSheetsService([HEADERS, rerun])).jobs())
    assert [p.name for p in t.people] == ["Pat Lee"] and not t.new_people


def test_user_named_contacts_are_left_alone():
    job = {"row": 2, "company": "Acme", "role": "R", "alignment": "4", "status": "",
           "fields": {"Who to contact": "Jo Park (CTO)", "Outreach_2": "", "Updates": ""}}
    assert pipeline.targets([job]) == []


def test_cost_goes_to_the_flight_log_per_company(monkeypatch):
    _people(monkeypatch, {"email": "pat@acme.com", "status": "valid", "confidence": 0.97})
    _llm(monkeypatch, BODY)
    logged = []
    monkeypatch.setattr(pipeline.flight_recorder, "record_job", lambda *a, **k: logged.append(a))
    sheet, _svc = _sheet([_row()])
    pipeline.run_outreach(sheet=sheet, search_fn=lambda q: [], gmail_data={}, gmail=FakeGmail())
    assert [(a[0], a[1], a[3], a[6]) for a in logged] == [(2, "Acme", "outreach", True)]


# ── dashboard ───────────────────────────────────────────────────────────────

def test_outreach_tab_lists_aria_linkedin_notes_with_the_text():
    row = {"_row": 2, "Company": "Acme", "Role": "ML Engineer", "Status": "", "Profile alignment /5": "4",
           "Apply link": "", "Post link": "", "Who to contact": "Pat Lee (Recruiter)",
           "Outreach_2": "LinkedIn (Aria): Pat Lee (Recruiter) — https://linkedin.com/in/patlee\nNote: Hi Pat, hello."}
    mdl = m.build({"sheet": Source(True, {"current": [row], "applied": []})}, now=datetime(2026, 10, 20))
    [job] = mdl["jobs"]
    assert job.linkedin_note == "Hi Pat, hello."
    assert m.next_action(job, mdl["now"])[0] == "Send the LinkedIn note"
    d = today.build(mdl)
    assert d["linkedin"] == [job]          # listed even before applying
    html = outreach_tab.linkedin_row(job)               # the notes live on the Outreach tab now
    assert "Hi Pat, hello." in html and 'href="https://linkedin.com/in/patlee"' in html
