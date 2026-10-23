"""tests/test_ats_adapters.py — the ATS adapters: field-key maps, their pre-steps and Handshake's
resume check, on mocked pages (routed URLs, headless Chromium in tests only). No live sites."""
from __future__ import annotations

import json

import pytest

import aria.features.answer_bank as ab
import aria.state.db as db
from aria.core import paths
from aria.features.applying import walls
from aria.features.ats import adapters, apply_target, corrections, form_engine, planner
from aria.features.ats.adapters import ApplyContext
from aria.features.ats.adapters.ashby import AshbyAdapter
from aria.features.ats.adapters.ashby import form_url as ashby_form_url
from aria.features.ats.adapters.handshake import HandshakeAdapter, required_documents, resume_choice
from aria.features.ats.adapters.icims import IcimsAdapter
from aria.features.ats.adapters.lever import LeverAdapter
from aria.features.ats.adapters.lever import form_url as lever_form_url
from aria.features.ats.apply_target import ApplyTarget, Tab
from aria.features.ats.planner import FormField, ParsedForm

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(autouse=True)
def bank(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "identity.json")
    monkeypatch.setattr(corrections, "CORRECTIONS_DIR", tmp_path / "corrections")
    monkeypatch.setattr(apply_target, "SETTLE_STABLE_SECS", 0.4)
    monkeypatch.setattr(apply_target, "SETTLE_MAX_SECS", 3.0)
    monkeypatch.setattr(apply_target, "CLICK_WAIT_SECS", 1.5)
    cv = tmp_path / "Alex_Rivera_Resume.pdf"
    cv.write_bytes(b"%PDF-1.4 fake")
    path = tmp_path / "answer_bank.json"
    path.write_text(json.dumps({"answers": {
        "first_name": "Alex", "last_name": "Rivera", "full_name": "Alex Rivera", "email": "me@example.com",
        "phone": "+1-555-555-0100", "linkedin": "https://linkedin.com/in/me", "github": "https://github.com/me",
        "location": "Buffalo, NY", "current_company": "Acme Robotics", "resume_path": str(cv),
        "sponsorship": None, "work_authorization": None}}))
    monkeypatch.setattr(ab, "BANK_PATH", path)
    return cv


# ── field-key maps (pure) ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw,key", [
    ({"name": "name"}, "full_name"), ({"name": "email"}, "email"), ({"name": "org"}, "current_company"),
    ({"name": "urls[LinkedIn]"}, "linkedin"), ({"name": "urls[GitHub]"}, "github"),
    ({"name": "cards[abc][field0]"}, ""), ({"name": "names"}, ""),
])
def test_lever_field_keys(raw, key):
    assert LeverAdapter().field_key(raw) == key


@pytest.mark.parametrize("raw,key", [
    ({"name": "_systemfield_name"}, "full_name"), ({"name": "_systemfield_email"}, "email"),
    ({"name": "_systemfield_location"}, "location"), ({"name": "1f0c-uuid"}, ""),
])
def test_ashby_field_keys(raw, key):
    assert AshbyAdapter().field_key(raw) == key


def test_automation_id_is_checked_before_the_name():
    class A(adapters.Adapter):
        FIELD_KEYS = ((r"email", "email"), (r"legalNameSection_firstName", "first_name"))
    assert A().field_key({"aid": "legalNameSection_firstName", "name": "email"}) == "first_name"


def test_a_keyed_field_is_filled_by_key_whatever_its_label_says():
    f = FormField("f1", "Your details", "text", required=True, key="full_name")
    (p,) = planner.plan(ParsedForm([f]), {}, "", llm_available=False)
    assert (p.value, p.source) == ("Alex Rivera", "bank")


def test_a_keyed_field_missing_from_the_bank_is_asked_never_guessed():
    f = FormField("f1", "Visa status", "text", required=True, key="sponsorship")
    (p,) = planner.plan(ParsedForm([f]), {}, "", guess_fn=lambda *a: pytest.fail("no LLM"),
                        draft_fn=lambda *a, **k: pytest.fail("no LLM"))
    assert p.source == "needs_me" and "in your bank" in p.note


@pytest.mark.parametrize("url,want", [
    ("https://jobs.lever.co/acme/1234-abcd", "https://jobs.lever.co/acme/1234-abcd/apply"),
    ("https://jobs.lever.co/acme/1234-abcd?lever-source=x", "https://jobs.lever.co/acme/1234-abcd/apply"),
    ("https://jobs.lever.co/acme/1234-abcd/apply", "https://jobs.lever.co/acme/1234-abcd/apply"),
    ("https://jobs.lever.co/acme", "https://jobs.lever.co/acme"),
])
def test_lever_form_url(url, want):
    assert lever_form_url(url) == want


def test_ashby_form_url():
    assert ashby_form_url("https://jobs.ashbyhq.com/acme/9f2e") == "https://jobs.ashbyhq.com/acme/9f2e/application"
    assert ashby_form_url("https://jobs.ashbyhq.com/acme/9f2e/application") == "https://jobs.ashbyhq.com/acme/9f2e/application"


# ── Handshake resume check (pure) ───────────────────────────────────────────

@pytest.mark.parametrize("picked,status", [
    (["Alex_Rivera_Resume.pdf"], "ok"),
    (["Alex Rivera Resume.pdf · uploaded Oct 2"], "ok"),
    (["Old_CV_2023.pdf"], "wrong"),
    (["Spring 2026 Resume"], "wrong"),
    ([], "none"),
    (["English"], "none"),
])
def test_resume_choice(bank, picked, status):
    assert resume_choice(picked, str(bank))[0] == status


def test_required_documents():
    text = "Resume (required) Alex_Rivera_Resume.pdf Transcript is required for this job. Cover letter (optional)"
    assert required_documents(text) == ["transcript"]
    assert required_documents("Required: cover letter") == ["cover letter"]


def test_handshake_and_sso_sign_in_pages_are_login_walls():
    assert walls.detect("Sign in with your school email", "https://app.joinhandshake.com/login", []) == "login"
    assert walls.detect("Log in", "https://login.usajobs.gov/Account/Login?ReturnUrl=x", []) == "login"
    assert walls.detect("Log in Username Password", "https://www.ubjobs.buffalo.edu/login", []) == "login"
    assert walls.detect("Sign in to apply", "https://app.joinhandshake.com/stu/jobs/1", []) is None
    assert walls.detect("How to log in faster", "https://blog.example/login-tips", []) is None


# ── on mocked pages ─────────────────────────────────────────────────────────

LEVER_FORM = """<form><label>Full name✱ <input name="name"></label><label>Email✱ <input name="email"></label>
<label>Current company <input name="org"></label><label>LinkedIn URL <input name="urls[LinkedIn]"></label>
<button type="submit" onclick="window.submitted = true; return false">Submit application</button></form>"""

HS_DIALOG = """<h1>Engineer</h1><input type="text" placeholder="Search jobs">
<button onclick="document.getElementById('d').style.display='block'">Apply</button>
<div id="d" role="dialog" style="display:none"><h2>Apply to Engineer</h2>
  <p>Resume (required)</p>%s
  <button onclick="window.submitted = true">Submit Application</button></div>"""

PAGES = {
    "https://jobs.lever.co/acme/1": "<h1>Engineer</h1><p>Lever posting</p>",
    "https://jobs.lever.co/acme/1/apply": LEVER_FORM,
    "https://jobs.ashbyhq.com/acme/9f": "<h1>Engineer</h1><a href='#'>Overview</a>",
    "https://jobs.ashbyhq.com/acme/9f/application": '<label>Name <input name="_systemfield_name"></label>'
                                                    '<label>Email <input name="_systemfield_email"></label>',
    "https://careers.acme.com/icims": '<iframe id="icims_content_iframe" src="https://careers-acme.icims.com/jobs/1/eng/job?in_iframe=1"></iframe>',
    "https://careers-acme.icims.com/jobs/1/eng/job": "<label>First <input></label><label>Last <input></label>",
    "https://app.joinhandshake.com/jobs/ok": HS_DIALOG % '<div class="select__single-value">Alex_Rivera_Resume.pdf</div>',
    "https://app.joinhandshake.com/jobs/wrong": HS_DIALOG % '<div class="select__single-value">Old_CV_2023.pdf</div>',
    "https://app.joinhandshake.com/jobs/upload": HS_DIALOG % '<input type="file" name="doc">'
                                                 '<p>Transcript is required for this job.</p>',
    "https://app.joinhandshake.com/jobs/external": """<h1>Engineer</h1>
        <button onclick="window.open('https://jobs.lever.co/acme/1/apply', '_blank')">Apply Externally</button>""",
}


@pytest.fixture(scope="module")
def browser():
    try:
        pw = sync_api.sync_playwright().start()
        b = pw.chromium.launch(headless=True)
    except Exception as exc:
        pytest.skip(f"chromium unavailable: {exc}")
    yield b
    b.close()
    pw.stop()


@pytest.fixture
def context(browser):
    ctx = browser.new_context()

    def handle(route):
        body = PAGES.get(route.request.url.split("?")[0])
        route.fulfill(status=200 if body else 404, content_type="text/html; charset=utf-8", body=body or "not found")
    ctx.route("**/*", handle)
    yield ctx
    ctx.close()


def open_tab(context, url):
    holder = {"page": context.new_page()}
    holder["page"].goto(url)
    tab = Tab("apply-test", run=lambda fn: fn(holder["page"]),
              navigate=lambda u, settle_secs=0: holder["page"].goto(u),
              adopt=lambda p: holder.__setitem__("page", p))
    return tab, holder


def ctx_for(url, ats="generic"):
    return ApplyContext(job={"url": url, "company": "Acme", "role": "Engineer"},
                        target=ApplyTarget(tab=None, ats=ats, kind="inline", url=url, posting_text="posting"))


def test_lever_opens_apply_and_fills_its_fields_by_name(context, monkeypatch):
    monkeypatch.setattr("aria.features.ats.greenhouse._gateway_up", lambda: False)
    tab, holder = open_tab(context, "https://jobs.lever.co/acme/1")
    ctx = ctx_for("https://jobs.lever.co/acme/1", "lever")
    report = LeverAdapter().fill(tab, ctx)
    assert ctx.target.url == "https://jobs.lever.co/acme/1/apply"
    got = {r["label"]: r["value"] for r in report.rows}
    assert got == {"Full name": "Alex Rivera", "Email": "me@example.com", "Current company": "Acme Robotics",
                   "LinkedIn URL": "https://linkedin.com/in/me"}
    assert holder["page"].evaluate("window.submitted === true") is False
    assert LeverAdapter().verify(tab, ctx).ok


def test_ashby_opens_the_application_tab(context, monkeypatch):
    monkeypatch.setattr("aria.features.ats.greenhouse._gateway_up", lambda: False)
    tab, holder = open_tab(context, "https://jobs.ashbyhq.com/acme/9f")
    report = AshbyAdapter().fill(tab, ctx_for("https://jobs.ashbyhq.com/acme/9f", "ashby"))
    assert {r["label"]: r["value"] for r in report.rows} == {"Name": "Alex Rivera", "Email": "me@example.com"}


def test_icims_form_is_opened_outside_its_frame(context):
    tab, holder = open_tab(context, "https://careers.acme.com/icims")
    ctx = ctx_for("https://careers.acme.com/icims", "icims")
    IcimsAdapter().prepare(tab, ctx)
    assert ctx.target.url.startswith("https://careers-acme.icims.com/jobs/1/eng/job")


def test_handshake_keeps_the_selected_resume_and_never_submits(context):
    tab, holder = open_tab(context, "https://app.joinhandshake.com/jobs/ok")
    report = HandshakeAdapter().fill(tab, ctx_for("https://app.joinhandshake.com/jobs/ok", "handshake"))
    assert [(r["label"], r["value"], r["source"]) for r in report.rows] == [("Resume", "Alex_Rivera_Resume.pdf", "bank")]
    assert "Submit is yours" in report.stopped_at
    assert holder["page"].evaluate("window.submitted === true") is False


def test_handshake_flags_a_different_resume(context):
    tab, holder = open_tab(context, "https://app.joinhandshake.com/jobs/wrong")
    report = HandshakeAdapter().fill(tab, ctx_for("https://app.joinhandshake.com/jobs/wrong", "handshake"))
    (row,) = report.rows
    assert row["source"] == "needs_me" and "Old_CV_2023.pdf" in row["note"]


def test_handshake_uploads_when_nothing_is_selected_and_asks_for_other_documents(context, bank):
    tab, holder = open_tab(context, "https://app.joinhandshake.com/jobs/upload")
    report = HandshakeAdapter().fill(tab, ctx_for("https://app.joinhandshake.com/jobs/upload", "handshake"))
    assert [(r["label"], r["source"]) for r in report.rows] == [("Resume", "bank"), ("Transcript", "needs_me")]
    assert holder["page"].evaluate("document.querySelector('input[type=file]').files[0].name") == bank.name


def test_handshake_apply_externally_goes_to_that_sites_adapter(context, monkeypatch):
    seen = []
    monkeypatch.setattr(LeverAdapter, "fill", lambda self, tab, ctx: seen.append(ctx.target.url) or
                        adapters.FillReport(form_url=ctx.target.url))
    tab, holder = open_tab(context, "https://app.joinhandshake.com/jobs/external")
    HandshakeAdapter().fill(tab, ctx_for("https://app.joinhandshake.com/jobs/external", "handshake"))
    assert seen == ["https://jobs.lever.co/acme/1/apply"]


def test_registry_has_every_adapter():
    assert {n: adapters.pick(n).name for n in ("greenhouse", "lever", "ashby", "icims", "handshake", "taleo")} == {
        "greenhouse": "greenhouse", "lever": "lever", "ashby": "ashby", "icims": "icims",
        "handshake": "handshake", "taleo": "generic"}


def test_scan_reports_automation_ids(context):
    page = context.new_page()
    page.set_content('<div data-automation-id="formField-legalNameSection_firstName"><label for="a">First</label>'
                     '<input id="a" data-automation-id="legalNameSection_firstName"></div>')
    (raw,) = form_engine.scan(page)
    assert raw["aid"] == "legalNameSection_firstName"
