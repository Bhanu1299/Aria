"""tests/test_click_targets.py — live log 2026-10-22: the browser agent clicked a hidden
"Skip to main content" link instead of Apply, and refused Workday's "Apply Manually"."""
from __future__ import annotations

import pytest

from aria.web import dom_browser
from aria.web.computer_use import _TARGET_JS, is_submit_like

sync_api = pytest.importorskip("playwright.sync_api")

PAGE = """<a id="skip" href="#main" style="position:absolute;top:-50px">Skip to main content</a>
<div><a href="/jobs">Jobs</a></div>
<div><a href="/apply" class="x">Apply</a></div>
<form><input name="email" value=""><button>Apply Manually</button></form>
<form><input name="email" value="me@example.com"><button>Apply</button></form>
<div id="react"><input name="phone" value="555"><div role="button">Apply</div></div>"""


@pytest.fixture(scope="module")
def page():
    try:
        pw = sync_api.sync_playwright().start()
        b = pw.chromium.launch(headless=True)
    except Exception as exc:          # no browser installed on this machine
        pytest.skip(f"chromium unavailable: {exc}")
    p = b.new_page()
    p.set_content(PAGE)
    yield p
    b.close()
    pw.stop()


def test_every_scanned_selector_names_exactly_one_element(page):
    items = page.evaluate(dom_browser._DOM_EXTRACT_JS)
    for it in items:
        assert page.locator(it["selector"]).count() == 1, it
    apply = next(it for it in items if it["text"] == "Apply" and it["tag"] == "A")
    assert page.locator(apply["selector"]).get_attribute("href") == "/apply"      # not the skip link


def test_opening_a_form_is_allowed_but_not_once_it_is_filled(page):
    empty = page.locator("form").nth(0).locator("button").evaluate(_TARGET_JS)
    filled = page.locator("form").nth(1).locator("button").evaluate(_TARGET_JS)
    assert not is_submit_like(empty)                              # Workday "Apply Manually"
    assert filled["filled"] == 1 and is_submit_like(filled)       # a filled form's Apply sends it
    react = page.locator("#react [role=button]").evaluate(_TARGET_JS)
    assert react["filled"] >= 1 and is_submit_like(react)         # no <form>: the whole page counts


@pytest.mark.parametrize("info, refused", [
    ({"text": "Submit application"}, True), ({"text": "Send"}, True), ({"text": "Finish"}, True),
    ({"text": "Apply now", "explicit_submit": True}, True), ({"text": "Apply and submit"}, True),
    ({"text": "Apply for this job", "filled": 0}, False), ({"text": "Start application", "filled": 0}, False),
    ({"text": "Apply for this job"}, True), ({"text": "Next"}, False),
])
def test_submit_words(info, refused):
    assert is_submit_like(info) is refused
