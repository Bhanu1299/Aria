"""tests/test_situations.py — the pages between an Apply link and the form: rules, the
small-model fallback (mocked), and the walk through them on mocked pages (headless Chromium)."""
from __future__ import annotations

import json

import pytest

import aria.state.db as db
from aria.features import ats
from aria.features.ats import apply_target, corrections, situations
from aria.features.ats.apply_target import Tab, resolve_apply_target
from aria.features.ats.situations import Situation, look

JOB = {"company": "Acme", "role": "Software Engineer I", "url": "https://careers.acme.example/jobs"}


def snap(text="", clickables=(), fields=0, url="https://careers.acme.example/x"):
    return {"url": url, "title": "", "fields": fields, "text": text,
            "clickables": [dict({"visible": True, "tag": "button", "href": "", "banner": False}, i=n, **c)
                           for n, c in enumerate(clickables)]}


# ── rules (pure) ─────────────────────────────────────────────────────────────

def test_cookie_banner_prefers_reject_and_only_clicks_inside_the_banner():
    s = look(snap("We use cookies to improve your experience.",
                  [{"text": "OK"}, {"text": "Accept all cookies", "banner": True},
                   {"text": "Reject all", "banner": True}]), JOB)
    assert (s.kind, s.click) == ("cookie_banner", 2)
    # a footer "Cookie policy" + an unrelated OK: nothing to click
    assert look(snap("Footer · Cookie policy", [{"text": "OK"}]), JOB).kind == "unknown"


@pytest.mark.parametrize("text,kind", [
    ("Sorry, this job is no longer accepting applications.", "closed"),
    ("This position has been filled.", "closed"),
    ("Error 404 not found", "closed"),
    ("You have already applied to this job.", "already_applied"),
    ("To apply, send your resume to jobs@acme.example", "email_only"),
    ("Step 1: Start the assessment to continue.", "assessment"),
])
def test_stops(text, kind):
    clickables = [{"text": "jobs@acme.example", "tag": "a", "href": "mailto:jobs@acme.example"}]
    assert look(snap(text, clickables), JOB).kind == kind


def test_stops_never_fire_on_a_page_with_a_form():
    assert look(snap("This position has been filled.", fields=5), JOB).kind == "unknown"


def test_job_id_404_is_not_a_missing_page():
    assert look(snap("Requisition 404 · Software Engineer I"), JOB).kind != "closed"


def test_leaving_site_notice_continues():
    s = look(snap("You are now leaving Acme Careers for our partner's site.", [{"text": "Cancel"}, {"text": "Continue"}]), JOB)
    assert (s.kind, s.click) == ("leaving_site", 1)


def test_careers_list_picks_this_jobs_link():
    s = look(snap("Open positions", [{"text": "Data Analyst", "tag": "a", "href": "/j/1"},
                                     {"text": "Software Engineer I - Platform", "tag": "a", "href": "/j/2"},
                                     {"text": "Senior Software Engineer", "tag": "a", "href": "/j/3"}]), JOB)
    assert (s.kind, s.click) == ("pick_job", 1)


def test_never_clicks_submit_like_controls():
    s = look(snap("You are leaving this site", [{"text": "Submit application"}]), JOB)
    assert s.kind == "unknown"


# ── the small-model fallback (mocked) ────────────────────────────────────────

class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete_checked(self, task, messages, validate, **kw):
        self.calls.append(task)
        return None, validate(self.reply)


@pytest.fixture
def llm(monkeypatch):
    import aria.llm as llm_pkg

    def use(reply):
        fake = FakeLLM(reply)
        monkeypatch.setattr(llm_pkg, "llm_client", fake)
        return fake
    return use


def test_ask_picks_a_listed_control(llm):
    fake = llm('{"kind": "click", "click": 1, "why": "the job link"}')
    out = situations.ask(snap("Careers", [{"text": "Home"}, {"text": "View role"}]), JOB)
    assert (out.kind, out.click, out.source) == ("click", 1, "llm") and fake.calls == ["apply_page_situation"]


@pytest.mark.parametrize("reply", [
    '{"kind": "click", "click": 0}',          # 0 is "Submit application": never offered, never accepted
    '{"kind": "click", "click": 99}',         # not on the page
    '{"kind": "delete everything"}',
    "not json",
])
def test_ask_rejects_what_code_cant_back(llm, reply):
    llm(reply)
    assert situations.ask(snap("x", [{"text": "Submit application"}, {"text": "Next"}]), JOB) is None


def test_ask_never_runs_on_a_page_with_fields(llm):
    fake = llm('{"kind": "click", "click": 0}')
    assert situations.ask(snap("x", [{"text": "Next"}], fields=4), JOB) is None and fake.calls == []


def test_ask_names_a_stop(llm):
    llm('{"kind": "closed", "why": "posting expired"}')
    assert situations.ask(snap("x", [{"text": "Back"}]), JOB) == Situation("closed", None, "posting expired", "llm")


# ── the walk, on mocked pages ────────────────────────────────────────────────

BASE = "https://careers.acme.example"
FORM = """<h1>Apply</h1><form><label>First <input name="f"></label><label>Last <input name="l"></label>
<label>Email <input name="e"></label></form>"""
PAGES = {
    # cookie banner over a careers list → this job's link → leaving-site notice → the form
    f"{BASE}/jobs": """<div id="onetrust-banner-sdk"><p>We use cookies.</p>
        <button onclick="this.parentNode.remove()">Reject All</button><button>Accept</button></div>
        <h1>Open roles</h1><a href="/jobs/1">Data Analyst</a> <a href="/jobs/2">Software Engineer I</a>""",
    f"{BASE}/jobs/2": """<p>You are now leaving Acme Careers.</p><a href="/form">Continue</a>""",
    f"{BASE}/form": FORM,
    f"{BASE}/closed": "<h1>Software Engineer I</h1><p>This job is no longer accepting applications.</p>",
    f"{BASE}/mystery": """<h1>Welcome</h1><button onclick="location.href='/form'">Begin</button>""",
}


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    try:
        pw = sync_api.sync_playwright().start()
        b = pw.chromium.launch(headless=True)
    except Exception as exc:
        pytest.skip(f"chromium unavailable: {exc}")
    yield b
    b.close()
    pw.stop()


@pytest.fixture
def context(browser, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(corrections, "CORRECTIONS_DIR", tmp_path / "corrections")
    monkeypatch.setattr(apply_target, "SETTLE_STABLE_SECS", 0.4)
    monkeypatch.setattr(apply_target, "SETTLE_MAX_SECS", 3.0)
    monkeypatch.setattr(apply_target, "CLICK_WAIT_SECS", 1.5)
    ctx = browser.new_context()
    ctx.route("**/*", lambda route: route.fulfill(
        status=200 if route.request.url.split("?")[0] in PAGES else 404, content_type="text/html",
        body=PAGES.get(route.request.url.split("?")[0], "not found")))
    yield ctx
    ctx.close()


def open_tab(context, url):
    holder = {"page": context.new_page()}
    holder["page"].goto(url)
    return Tab("apply-test", run=lambda fn: fn(holder["page"]), navigate=lambda u, settle_secs=0: holder["page"].goto(u),
               adopt=lambda p: holder.__setitem__("page", p)), holder


def test_walks_cookie_banner_careers_list_and_leaving_notice_to_the_form(context, llm):
    fake = llm("never asked")
    said = []
    tab, holder = open_tab(context, f"{BASE}/jobs")
    t = resolve_apply_target(tab, JOB, said.append)
    assert holder["page"].url == f"{BASE}/form" and not t.stopped
    assert said[:3] == ["Closing the cookie notice.", "Picking this job from the careers list.",
                        "Continuing past the leaving-this-site notice."]
    assert fake.calls == []                                # rules handled it all: zero LLM calls


def test_closed_posting_stops_with_a_reason(context, llm):
    llm("never asked")
    tab, _ = open_tab(context, f"{BASE}/closed")
    t = resolve_apply_target(tab, JOB)
    assert t.stopped == "closed" and "no longer accepting" in t.stopped_why


def test_unreadable_page_gets_one_small_model_look(context, llm):
    fake = llm(json.dumps({"kind": "click", "click": 0, "why": "starts the application"}))
    tab, holder = open_tab(context, f"{BASE}/mystery")
    # the model's index is validated against the page's own numbering, so ask for the real one
    real = holder["page"].evaluate(situations.SNAPSHOT_JS)["clickables"][0]["i"]
    fake.reply = json.dumps({"kind": "click", "click": real, "why": "starts the application"})
    resolve_apply_target(tab, JOB)
    assert holder["page"].url == f"{BASE}/form" and fake.calls == ["apply_page_situation"]


def test_generic_fill_reports_a_stop_instead_of_an_empty_form(monkeypatch):
    target = apply_target.ApplyTarget(tab=None, ats="generic", kind="inline", url=f"{BASE}/closed",
                                      posting_text="", stopped="closed", stopped_why="no longer accepting applications")
    import aria.web.agent_browser as agent_browser
    from contextlib import contextmanager

    @contextmanager
    def use_tab(tab, gate=None):
        yield
    monkeypatch.setattr(agent_browser, "use_tab", use_tab)
    monkeypatch.setattr(agent_browser, "current_gate", lambda: None)
    monkeypatch.setattr(apply_target.Tab, "of", classmethod(lambda cls, n: Tab(n, lambda fn: None, lambda u, settle_secs=0: None,
                                                                             lambda p: None)))
    monkeypatch.setattr("aria.features.ats.apply_target.resolve_apply_target", lambda *a, **k: target)
    r = ats._generic(dict(JOB, url=f"{BASE}/closed"), None, tab="apply-x")
    assert r.stopped_at == "closed" and r.error.startswith("The posting looks closed")
