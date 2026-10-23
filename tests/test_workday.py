"""tests/test_workday.py — the Workday adapter on a mocked tenant (routed URLs, headless Chromium
in tests only; fake Keychain, fake inbox). Covers the tenant state machine, account creation
with the user's terms tick, the email code, a rejected login, the wizard, and that no
password ever reaches a log, a progress line or a snapshot."""
from __future__ import annotations

import json
import logging

import pytest

import aria.features.answer_bank as ab
import aria.state.db as db
from aria.core import paths
from aria.features.ats import apply_target, corrections, email_code
from aria.features.ats.adapters import ApplyContext
from aria.features.ats.adapters import workday as wd
from aria.features.ats.adapters.workday import WorkdayAdapter, classify, senders
from aria.features.ats.apply_target import ApplyTarget, Tab, resolve_apply_target
from aria.system import keychain

sync_api = pytest.importorskip("playwright.sync_api")

HOST = "acme.wd5.myworkdayjobs.com"
BASE = f"https://{HOST}/en-US/External"
SAVED = "Saved#Pass2026x"

# Inputs mirror what's typed into their value attribute, like React does.
MIRROR = """<script>document.addEventListener('input', e => e.target.setAttribute('value', e.target.value));</script>"""

POSTING = """<h1>Software Engineer</h1>
<a data-automation-id="adventureButton" href="#" onclick="document.getElementById('m').style.display='block'; return false">Apply</a>
<div id="m" role="dialog" style="display:none">
  <a data-automation-id="autofillWithResume" href="#" onclick="window.autofill = true">Autofill with Resume</a>
  <a data-automation-id="applyManually" href="/en-US/External/job/R1/apply/applyManually">Apply Manually</a>%s
</div>"""
LAST_APP = '<a data-automation-id="useMyLastApplication" href="/en-US/External/job/R1/apply/useMyLastApplication">Use My Last Application</a>'

SIGN_IN = MIRROR + """<div data-automation-id="signInContent"><h2>Sign In</h2>
<label>Email Address <input data-automation-id="email"></label>
<label>Password <input type="password" data-automation-id="password"></label>
<button data-automation-id="signInSubmitButton" onclick="go()">Sign In</button>
<button data-automation-id="createAccountLink" onclick="location.href='/en-US/External/create'">Create Account</button>
<div id="err"></div></div>
<script>function go() {
  const st = window.name ? JSON.parse(window.name) : {};
  const want = st.pw || window.atob('%s');
  if (document.querySelector('[data-automation-id=password]').value === want) location.href = '/en-US/External/wizard1';
  else document.getElementById('err').innerHTML = '<div data-automation-id="errorMessage">Wrong email address or password</div>';
}</script>"""

CREATE = MIRROR + """<h2>Create Account</h2>
<label>Email Address <input data-automation-id="email"></label>
<label>Password <input type="password" data-automation-id="password"></label>
<label>Verify New Password <input type="password" data-automation-id="verifyPassword"></label>
<label><input type="checkbox" data-automation-id="createAccountCheckbox"> I agree to the terms</label>
<button data-automation-id="createAccountSubmitButton" onclick="create()">Create Account</button>
<script>function create() {
  const v = a => document.querySelector('[data-automation-id=' + a + ']');
  if (!v('createAccountCheckbox').checked) return;
  window.name = JSON.stringify({email: v('email').value, pw: v('password').value, again: v('verifyPassword').value});
  location.href = '/en-US/External/verify';
}</script>"""

VERIFY = """<h2>Verify your account</h2><p>We've sent a verification code to your email.</p>
<input data-automation-id="verificationCode">
<button data-automation-id="verifyButton" onclick="if (document.querySelector('[data-automation-id=verificationCode]').value === '483920') location.href='/en-US/External/signin'">Verify</button>"""

WIZARD1 = """<div data-automation-id="progressBar">My Information</div>
<div data-automation-id="formField-legalNameSection_firstName"><label for="fn">Given Name(s)*</label>
  <input id="fn" data-automation-id="legalNameSection_firstName" required></div>
<div data-automation-id="formField-legalNameSection_lastName"><label for="ln">Family Name*</label>
  <input id="ln" data-automation-id="legalNameSection_lastName" required></div>
<div data-automation-id="formField-addressSection_city"><label for="c">Town*</label>
  <input id="c" data-automation-id="addressSection_city" required></div>
<button data-automation-id="bottom-navigation-next-button" onclick="location.href='/en-US/External/wizard2'">Save and Continue</button>"""

WIZARD2 = """<div data-automation-id="progressBar">My Experience</div>
<div data-automation-id="formField-phone-number"><label for="p">Phone*</label><input id="p" data-automation-id="phone-number" required></div>
<div id="err"></div>
<button data-automation-id="bottom-navigation-next-button" onclick="document.getElementById('err').innerHTML =
  '<div data-automation-id=&quot;errorMessage&quot;>Error: How Did You Hear About Us? is required</div>'">Save and Continue</button>"""

REVIEW = """<div data-automation-id="progressBar">Review</div><h2>Review</h2><p>Alex Rivera</p>
<button data-automation-id="bottom-navigation-next-button" onclick="window.submitted = true">Submit</button>"""

PAGES = {
    f"{BASE}/job/R1": POSTING % "",
    f"{BASE}/job/R2": POSTING % LAST_APP,
    f"{BASE}/job/R1/apply/applyManually": None,          # SIGN_IN, filled in per test (needs the saved password)
    f"{BASE}/job/R1/apply/useMyLastApplication": None,
    f"{BASE}/signin": None,
    f"{BASE}/create": CREATE,
    f"{BASE}/verify": VERIFY,
    f"{BASE}/wizard1": WIZARD1,
    f"{BASE}/wizard2": WIZARD2,
    f"{BASE}/review": REVIEW,
}


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, pw):
        self.store[(service, user)] = pw

    def delete_password(self, service, user):
        self.store.pop((service, user), None)


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "identity.json")
    monkeypatch.setattr(corrections, "CORRECTIONS_DIR", tmp_path / "corrections")
    monkeypatch.setattr(apply_target, "SETTLE_STABLE_SECS", 0.3)
    monkeypatch.setattr(apply_target, "SETTLE_MAX_SECS", 3.0)
    monkeypatch.setattr(wd, "AFTER_CLICK_SECS", 0.2)
    monkeypatch.setattr("aria.features.ats.greenhouse._gateway_up", lambda: False)
    ring = FakeKeyring()
    monkeypatch.setattr(keychain, "_backend", lambda: ring)
    bank = tmp_path / "answer_bank.json"
    bank.write_text(json.dumps({"answers": {
        "first_name": "Alex", "last_name": "Rivera", "full_name": "Alex Rivera", "email": "me@example.com",
        "phone": "+1-555-555-0100", "city": "Buffalo", "sponsorship": None, "work_authorization": None}}))
    monkeypatch.setattr(ab, "BANK_PATH", bank)
    return ring


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
    import base64
    ctx = browser.new_context()
    sign_in = SIGN_IN % base64.b64encode(SAVED.encode()).decode()

    def handle(route):
        url = route.request.url.split("?")[0]
        if url not in PAGES:
            return route.fulfill(status=404, body="not found")
        route.fulfill(status=200, content_type="text/html; charset=utf-8", body=PAGES[url] or sign_in)
    ctx.route("**/*", handle)
    yield ctx
    ctx.close()


def open_tab(context, url):
    holder = {"page": context.new_page()}
    holder["page"].goto(url)
    tab = Tab("apply-wd", run=lambda fn: fn(holder["page"]),
              navigate=lambda u, settle_secs=0: holder["page"].goto(u),
              adopt=lambda p: holder.__setitem__("page", p))
    return tab, holder


def ctx_for(url, wait_fn=None, said=None):
    return ApplyContext(job={"url": url, "company": "Acme", "role": "Engineer", "row": 3},
                        target=ApplyTarget(tab=None, ats="workday", kind="inline", url=url, posting_text="posting"),
                        progress_fn=(said.append if said is not None else (lambda m: None)), wait_fn=wait_fn)


def no_secret_anywhere(secret, caplog, said, tmp_dir):
    assert secret and secret not in caplog.text
    assert not any(secret in m for m in said)
    for f in (tmp_dir / "corrections").glob("*") if (tmp_dir / "corrections").exists() else []:
        assert secret.encode() not in f.read_bytes(), f.name
    with db.get_connection() as conn:
        for r in conn.execute("SELECT * FROM apply_corrections"):
            assert secret not in " ".join(str(v) for v in tuple(r))


# ── pure ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("state,want", [
    ({"posting": True}, "posting"),
    ({"posting": True, "start": True}, "start"),
    ({"sign_in": True, "create_link": True}, "sign_in"),
    ({"create": True, "sign_in": True}, "create_account"),
    ({"code_box": True}, "verify_email"),
    ({"sign_in": True, "text": "Please verify your account. We've sent an email."}, "verify_email"),
    ({"wizard": True, "sign_in": True}, "signed_in"),
    ({}, "unknown"),
])
def test_classify(state, want):
    assert classify(state) == want


def test_senders_are_workday_and_the_tenant():
    assert senders(HOST) == ["myworkday.com", "myworkdayjobs.com", "acme"]
    assert senders("wd3.myworkdaysite.com") == ["myworkday.com", "myworkdayjobs.com"]


def test_workday_field_ids_map_to_bank_keys():
    a = WorkdayAdapter()
    assert a.field_key({"aid": "legalNameSection_firstName"}) == "first_name"
    assert a.field_key({"aid": "addressSection_city"}) == "city"
    assert a.field_key({"aid": "phone-number"}) == "phone"
    assert a.field_key({"aid": "formField-someQuestion", "name": "q1"}) == ""


def test_resolve_leaves_workday_to_its_adapter(context):
    tab, holder = open_tab(context, f"{BASE}/job/R1")
    t = resolve_apply_target(tab, {"url": f"{BASE}/job/R1"})
    assert (t.ats, t.url) == ("workday", f"{BASE}/job/R1")
    assert holder["page"].evaluate("getComputedStyle(document.getElementById('m')).display") == "none"


# ── login state machine ─────────────────────────────────────────────────────

def test_new_tenant_creates_an_account_after_you_tick_the_terms(context, env, caplog, tmp_path, monkeypatch):
    caplog.set_level(logging.DEBUG)
    asked, said, looked = [], [], []
    monkeypatch.setattr(email_code, "wait_for", lambda email, who, since, **kw: looked.append((email, who)) or ("code", "483920"))
    tab, holder = open_tab(context, f"{BASE}/job/R1")

    def wait_fn(kind, cleared=None):
        asked.append(kind)
        if kind == "terms":                       # you tick it in the tab
            holder["page"].check('[data-automation-id="createAccountCheckbox"]')
        return cleared() if cleared else True

    ctx = ctx_for(f"{BASE}/job/R1", wait_fn, said)
    assert WorkdayAdapter().login(tab, ctx) is True
    assert asked == ["terms"] and ctx.target.url == f"{BASE}/wizard1"
    assert holder["page"].evaluate("window.autofill === true") is False      # never Autofill with Resume
    pw = env.store[(keychain.service(HOST), "me@example.com")]
    sent = json.loads(holder["page"].evaluate("window.name"))
    assert sent == {"email": "me@example.com", "pw": pw, "again": pw}
    assert looked == [("me@example.com", ["myworkday.com", "myworkdayjobs.com", "acme"])]
    no_secret_anywhere(pw, caplog, said, tmp_path)


def test_saved_login_and_last_application_are_preferred(context, env, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    keychain.save(HOST, "me@example.com", SAVED)
    said = []
    tab, holder = open_tab(context, f"{BASE}/job/R2")
    ctx = ctx_for(f"{BASE}/job/R2", lambda kind, cleared=None: pytest.fail(f"asked the user: {kind}"), said)
    assert WorkdayAdapter().login(tab, ctx) is True
    assert any("last application" in m for m in said)
    no_secret_anywhere(SAVED, caplog, said, tmp_path)


def test_rejected_login_is_logged_and_handed_to_you(context, env, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    keychain.save(HOST, "me@example.com", "Wrong#Pass2026y")
    said, asked = [], []
    tab, holder = open_tab(context, f"{BASE}/job/R1")
    ctx = ctx_for(f"{BASE}/job/R1", lambda kind, cleared=None: asked.append(kind) or False, said)
    assert WorkdayAdapter().login(tab, ctx) is False
    assert asked == ["login"]
    (row,) = [r for r in corrections.recent(1) if r["kind"] == "login_failed"]
    assert "Wrong email address or password" in row["detail"] and HOST in row["detail"]
    snap = (tmp_path / "corrections" / (row["snapshot"].rsplit("/", 1)[1] + ".html")).read_text()
    assert 'data-automation-id="password"' in snap                # the page was saved…
    no_secret_anywhere("Wrong#Pass2026y", caplog, said, tmp_path)  # …without the typed password


def test_terms_not_ticked_means_no_account(context, env, monkeypatch):
    monkeypatch.setattr(email_code, "wait_for", lambda *a, **k: pytest.fail("no account was made"))
    tab, holder = open_tab(context, f"{BASE}/job/R1")
    ctx = ctx_for(f"{BASE}/job/R1", lambda kind, cleared=None: False)
    assert WorkdayAdapter().login(tab, ctx) is False
    assert holder["page"].evaluate("window.name") == ""
    assert "login_failed" in [r["kind"] for r in corrections.recent(1)]


def test_no_verification_email_hands_it_to_you(context, env, monkeypatch):
    monkeypatch.setattr(email_code, "wait_for", lambda *a, **k: None)
    tab, holder = open_tab(context, f"{BASE}/job/R1")
    asked = []

    def wait_fn(kind, cleared=None):
        asked.append(kind)
        if kind == "terms":
            holder["page"].check('[data-automation-id="createAccountCheckbox"]')
            return cleared()
        return False
    assert WorkdayAdapter().login(tab, ctx_for(f"{BASE}/job/R1", wait_fn)) is False
    assert asked == ["terms", "email_code"]


# ── wizard ──────────────────────────────────────────────────────────────────

def test_wizard_fills_by_workday_ids_and_surfaces_what_it_cant_resolve(context):
    tab, holder = open_tab(context, f"{BASE}/wizard1")
    report = WorkdayAdapter().fill(tab, ctx_for(f"{BASE}/wizard1"))
    got = {r["label"]: (r["value"], r["source"]) for r in report.rows}
    assert got["Given Name(s)"] == ("Alex", "bank") and got["Family Name"] == ("Rivera", "bank")
    assert got["Town"] == ("Buffalo", "bank") and got["Phone"] == ("+1-555-555-0100", "bank")
    assert got["Error: How Did You Hear About Us? is required"] == ("", "needs_me")
    assert report.stopped_at.startswith("Workday page 2: 1 error(s)")


def test_review_page_is_the_end_and_submit_is_never_clicked(context):
    tab, holder = open_tab(context, f"{BASE}/review")
    report = WorkdayAdapter().fill(tab, ctx_for(f"{BASE}/review"))
    assert holder["page"].evaluate("window.submitted === true") is False
    assert report.stopped_at == "no fields" and not report.rows       # no vision loop on Workday


def test_password_boxes_are_never_form_fields(context):
    tab, holder = open_tab(context, f"{BASE}/signin")
    from aria.features.ats import form_engine
    assert [r["label"] for r in form_engine.scan(holder["page"])] == ["Email Address"]


def test_only_the_workday_adapter_reads_the_keychain():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "aria"
    readers = sorted(str(p.relative_to(root)) for p in root.rglob("*.py")
                     if "keychain.get(" in p.read_text() and p.name != "keychain.py")
    assert readers == ["features/ats/adapters/workday.py"], readers
