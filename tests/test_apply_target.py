"""tests/test_apply_target.py — posting → application form: affordance choice (pure) and the
three click outcomes + redirects on mocked pages (routed URLs, headless Chromium in tests
only). Nothing here reaches a real site."""
from __future__ import annotations

import pytest

import aria.state.db as db
from aria.features import ats
from aria.features.ats import adapters, apply_target, corrections
from aria.features.ats.apply_target import Tab, pick_affordance, resolve_apply_target

sync_api = pytest.importorskip("playwright.sync_api")


# ── pick_affordance (pure) ──────────────────────────────────────────────────

def c(i, text, href="", tag="a", visible=True):
    return {"i": i, "text": text, "href": href, "tag": tag, "visible": visible}


POSTING = "https://jobs.jobvite.com/ookla/job/oAaCAfwG"


def test_follows_the_postings_apply_link():
    got = pick_affordance([c(0, "Apply", "/ookla/job/oAaCAfwG/apply"), c(1, "Apply Later", "mailto:?subject=x")], POSTING)
    assert got == {"i": 0, "text": "Apply", "action": "follow", "url": "https://jobs.jobvite.com/ookla/job/oAaCAfwG/apply"}


@pytest.mark.parametrize("cand", [
    c(0, "Apply later", "/later"),
    c(0, "Apply with LinkedIn", "/li"),
    c(0, "Apply with Indeed", tag="button"),
    c(0, "Applied", "/x"),
    c(0, "Share this job", "/share"),
    c(0, "See all jobs", "/jobs"),
    c(0, "Save job alert", tag="button"),
    c(0, "Apply now", "/apply", visible=False),
])
def test_never_picks_non_apply_or_hidden_controls(cand):
    assert pick_affordance([cand], POSTING) is None


@pytest.mark.parametrize("text", ["Apply", "Apply now", "APPLY NOW", "Apply for this job", "Easy Apply", "Quick apply",
                                  "Apply manually", "Start application", "Start your application", "I'm interested",
                                  "Apply to this position"])
def test_common_apply_wordings(text):
    assert pick_affordance([c(3, text, tag="button")], POSTING)["action"] == "click"


@pytest.mark.parametrize("href", ["mailto:jobs@x.com", "javascript:void(0)", "#apply", POSTING])
def test_a_link_that_goes_nowhere_is_clicked_not_followed(href):
    got = pick_affordance([c(0, "Apply now", href)], POSTING)
    assert got["action"] == "click" and got["url"] == ""


def test_prefers_apply_manually_over_plain_apply():
    got = pick_affordance([c(0, "Apply", tag="button"), c(1, "Apply Manually", tag="button")], POSTING)
    assert got["i"] == 1


def test_first_of_equal_links_wins():
    got = pick_affordance([c(0, "Apply for this job", "https://x.example/job/1/form"), c(1, "Apply now", "/other")],
                          "https://x.example/job/1")
    assert got["url"] == "https://x.example/job/1/form"


def test_ats_apply_href_counts_even_with_other_text():
    got = pick_affordance([c(0, "Click here", "https://boards.greenhouse.io/acme/jobs/1/application")], POSTING)
    assert got["action"] == "follow"
    # ...and an ATS apply href beats a plain "Apply" link elsewhere
    got = pick_affordance([c(0, "Apply", "/somewhere"), c(1, "Apply", "https://jobs.lever.co/acme/1/apply")], POSTING)
    assert got["i"] == 1


# ── resolve_apply_target on mocked pages ────────────────────────────────────

FORM = """<html><body><h1>Application</h1><form>
<label>First name <input name="first"></label><label>Last name <input name="last"></label>
<label>Email <input type="email" name="email"></label></form></body></html>"""

PAGES = {
    "https://careers.acme.test/inline": FORM,
    "https://careers.acme.test/form": FORM,
    "https://careers.acme.test/link": '<h1>Engineer</h1><p>Great job.</p><a href="/form">Apply now</a>',
    "https://careers.acme.test/popup": """<h1>Engineer</h1>
        <button onclick="window.open('/form', '_blank')">Apply</button>""",
    "https://careers.acme.test/modal": """<h1>Engineer</h1><button onclick="document.getElementById('d').style.display='block'">Apply</button>
        <div id="d" role="dialog" style="display:none"><input name="a"><input name="b"><input name="c"></div>""",
    "https://careers.acme.test/redirect": '<h1>Engineer</h1><a href="/r1">Apply</a>',
    "https://careers.acme.test/r1": "<script>setTimeout(() => location.replace('/r2'), 150)</script>",
    "https://careers.acme.test/r2": "<script>setTimeout(() => location.replace('/form'), 150)</script>",
    "https://careers.acme.test/fields_and_button": """<h1>Engineer</h1><label>Email for job alerts <input type="email"></label>
        <button onclick="window.clicked = true">Apply</button>""",
    "https://careers.acme.test/dead": '<h1>Engineer</h1><button onclick="void 0">Apply</button>',
    "https://careers.acme.test/nothing": "<h1>Engineer</h1><p>Email us your resume.</p>",
    "https://careers.acme.test/to_lever": '<h1>Engineer</h1><a href="https://jobs.lever.co/acme/1/apply">Apply</a>',
    "https://jobs.lever.co/acme/1/apply": FORM,
    "https://careers.acme.test/embed": '<h1>Engineer</h1><iframe src="https://jobs.lever.co/acme/1/apply"></iframe>',
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
        url = route.request.url.split("?")[0]
        body = PAGES.get(url)
        if body is None:
            route.fulfill(status=404, body="not found")
        else:
            route.fulfill(status=200, content_type="text/html", body=body)
    ctx.route("**/*", handle)
    yield ctx
    ctx.close()


@pytest.fixture(autouse=True)
def quick(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(corrections, "CORRECTIONS_DIR", tmp_path / "corrections")
    monkeypatch.setattr(apply_target, "SETTLE_STABLE_SECS", 0.5)
    monkeypatch.setattr(apply_target, "SETTLE_MAX_SECS", 4.0)
    monkeypatch.setattr(apply_target, "CLICK_WAIT_SECS", 1.5)


def open_tab(context, url):
    holder = {"page": context.new_page()}
    holder["page"].goto(url)
    tab = Tab("apply-test", run=lambda fn: fn(holder["page"]),
              navigate=lambda u, settle_secs=0: holder["page"].goto(u),
              adopt=lambda p: holder.__setitem__("page", p))
    return tab, holder


def kinds():
    return [r["kind"] for r in corrections.recent(1)]


def test_form_already_on_the_page_is_inline(context):
    tab, holder = open_tab(context, "https://careers.acme.test/inline")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/inline"})
    assert (t.kind, t.url) == ("inline", "https://careers.acme.test/inline")


def test_apply_link_navigates_the_same_tab(context):
    tab, holder = open_tab(context, "https://careers.acme.test/link")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/link"})
    assert (t.kind, t.url) == ("same_tab", "https://careers.acme.test/form")
    assert "Great job." in t.posting_text          # the posting's text, for open-text answers


def test_popup_becomes_the_applications_tab_and_the_posting_stays_open(context):
    tab, holder = open_tab(context, "https://careers.acme.test/popup")
    posting = holder["page"]
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/popup"})
    assert t.kind == "new_tab" and t.url == "https://careers.acme.test/form"
    assert holder["page"] is not posting and not posting.is_closed()


def test_in_page_modal_is_found_without_navigation(context):
    tab, holder = open_tab(context, "https://careers.acme.test/modal")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/modal"})
    assert (t.kind, t.url) == ("modal", "https://careers.acme.test/modal")


def test_redirect_chain_settles_on_the_final_url(context):
    tab, holder = open_tab(context, "https://careers.acme.test/redirect")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/redirect"})
    assert t.url == "https://careers.acme.test/form"


def test_a_button_is_never_clicked_on_a_page_that_has_fields(context):
    tab, holder = open_tab(context, "https://careers.acme.test/fields_and_button")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/fields_and_button"})
    assert t.kind == "inline" and holder["page"].evaluate("window.clicked === true") is False


def test_destination_ats_is_named_from_the_final_url(context):
    tab, holder = open_tab(context, "https://careers.acme.test/to_lever")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/to_lever"})
    assert (t.ats, t.identified, t.kind) == ("lever", "lever", "same_tab")
    assert "unknown_ats" not in kinds()


def test_embedded_ats_iframe_is_opened_directly(context):
    tab, holder = open_tab(context, "https://careers.acme.test/embed")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/embed"})
    assert (t.kind, t.url, t.ats) == ("iframe", "https://jobs.lever.co/acme/1/apply", "lever")


def test_unknown_ats_falls_back_to_generic_and_logs_the_domain(context):
    tab, holder = open_tab(context, "https://careers.acme.test/link")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/link", "company": "Acme"})
    assert (t.ats, t.identified) == ("generic", "")
    row = next(r for r in corrections.recent(1) if r["kind"] == "unknown_ats")
    assert row["detail"] == "acme.test" and row["company"] == "Acme" and row["step"] == "resolve"
    assert row["snapshot"] and (corrections.CORRECTIONS_DIR / (row["snapshot"].rsplit("/", 1)[1] + ".html")).exists()


def test_click_that_does_nothing_is_logged(context):
    tab, holder = open_tab(context, "https://careers.acme.test/dead")
    resolve_apply_target(tab, {"url": "https://careers.acme.test/dead"})
    assert "apply_click_nothing" in kinds()


def test_no_form_and_no_apply_control_is_logged(context):
    tab, holder = open_tab(context, "https://careers.acme.test/nothing")
    resolve_apply_target(tab, {"url": "https://careers.acme.test/nothing"})
    assert "apply_not_found" in kinds()


def test_a_wall_on_the_way_waits_for_the_user(context, monkeypatch):
    seen = []
    walls_seen = iter(["captcha", None, None, None, None, None])
    monkeypatch.setattr(apply_target, "_wall", lambda tab: next(walls_seen, None))
    tab, holder = open_tab(context, "https://careers.acme.test/link")
    t = resolve_apply_target(tab, {"url": "https://careers.acme.test/link"}, wait_fn=lambda k: seen.append(k) or True)
    assert seen == ["captcha"] and t.url == "https://careers.acme.test/form"


# ── fill_application → resolve → adapter ────────────────────────────────────

def test_generic_path_hands_the_resolved_target_to_its_adapter(monkeypatch):
    calls = []

    class Spy(adapters.Adapter):
        name = "spy"

        def login(self, tab, ctx):
            calls.append(("login", ctx.target.url))
            return True

        def fill(self, tab, ctx):
            calls.append(("fill", ctx.target.posting_text))
            ctx.report = adapters.FillReport(rows=[{"name": "q", "label": "Q", "source": "bank", "value": "x",
                                                    "status": "filled"}], form_url=ctx.target.url)
            return ctx.report

    target = apply_target.ApplyTarget(tab=None, ats="lever", kind="same_tab", url="https://jobs.lever.co/a/1/apply",
                                      posting_text="posting", identified="lever")
    monkeypatch.setattr(apply_target, "resolve_apply_target", lambda tab, job, progress_fn=None, wait_fn=None: target)
    monkeypatch.setattr(adapters, "pick", lambda name: Spy())
    monkeypatch.setattr(apply_target.Tab, "of", classmethod(lambda cls, name: cls(name, lambda fn: None,
                                                                                    lambda u, settle_secs=0: None,
                                                                                    lambda p: None)))
    result = ats.fill_application({"url": "https://jobs.jobvite.com/acme/job/1", "company": "Acme"}, tab="apply-1")
    assert calls == [("login", "https://jobs.lever.co/a/1/apply"), ("fill", "posting")]
    assert (result.ats, result.identified, result.form_url) == ("spy", "lever", "https://jobs.lever.co/a/1/apply")


def test_failed_login_leaves_the_form_empty_and_says_so(monkeypatch):
    class NoLogin(adapters.Adapter):
        def login(self, tab, ctx):
            return False

    target = apply_target.ApplyTarget(tab=None, ats="workday", kind="same_tab", url="https://x.wd1.myworkdayjobs.com/a",
                                      posting_text="", identified="workday")
    monkeypatch.setattr(apply_target, "resolve_apply_target", lambda tab, job, progress_fn=None, wait_fn=None: target)
    monkeypatch.setattr(adapters, "pick", lambda name: NoLogin())
    monkeypatch.setattr(apply_target.Tab, "of", classmethod(lambda cls, name: cls(name, lambda fn: None,
                                                                                    lambda u, settle_secs=0: None,
                                                                                    lambda p: None)))
    result = ats.fill_application({"url": "https://x.wd1.myworkdayjobs.com/a"}, tab="apply-1")
    assert "sign-in" in result.error and not result.review


def test_pick_falls_back_to_the_generic_adapter():
    assert adapters.pick("greenhouse").name == "greenhouse"
    assert adapters.pick("nonesuch").name == "generic"
    assert adapters.pick("jobvite").name == "generic"


def test_verify_reports_what_did_not_stick():
    ctx = adapters.ApplyContext(job={}, target=None)
    ctx.report = adapters.FillReport(rows=[{"label": "A", "value": "1", "source": "bank", "status": "filled"},
                                           {"label": "B", "value": "", "source": "failed", "status": "failed"}])
    rb = adapters.Adapter().verify(None, ctx)
    assert (rb.ok, rb.mismatches) == (False, ["B"])
