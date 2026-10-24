"""tests/test_form_engine.py — the generic form engine on real HTML in headless Chromium.

Scan → plan → fill → re-scan → verify → next page, never submit. LLM calls are spies.
"""
from __future__ import annotations

import json

import pytest

import aria.features.answer_bank as ab
import aria.state.db as db
from aria.core import paths
from aria.features.ats import form_engine, planner

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def browser():
    try:
        pw = sync_api.sync_playwright().start()
        b = pw.chromium.launch(headless=True)
    except Exception as exc:          # no browser installed on this machine
        pytest.skip(f"chromium unavailable: {exc}")
    yield b
    b.close()
    pw.stop()


@pytest.fixture
def page(browser):
    pg = browser.new_page()
    yield pg
    pg.close()


@pytest.fixture(autouse=True)
def bank(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "identity.json")
    monkeypatch.setattr(planner, "COVER_LETTER_DIR", tmp_path / "cover_letters")
    cv = tmp_path / "Alex_Resume.txt"
    cv.write_text("Alex Rivera. Built a voice agent in Python. Intern at Acme Robotics 2024.")
    path = tmp_path / "answer_bank.json"
    path.write_text(json.dumps({"answers": {
        "first_name": "Alex", "last_name": "Rivera", "full_name": "Alex Rivera", "email": "me@example.com",
        "phone": "+1-555-555-0100", "linkedin": "https://linkedin.com/in/me", "location": "Buffalo, NY",
        "city": "Buffalo", "country": "United States", "relocation": "Yes", "resume_path": str(cv),
        "sponsorship": None, "work_authorization": None,
        "eeo": {"gender": "Male"}}}))
    monkeypatch.setattr(ab, "BANK_PATH", path)
    return path


class Spy:
    def __init__(self):
        self.drafts, self.guesses = [], []

    def draft(self, label, job_text, profile, max_chars=0, short=False, allow_unknown=False):
        self.drafts.append((label, max_chars, short))
        assert "voice agent" in profile                  # the resume reaches the writer
        return "I built a voice agent in Python." if short else "At Acme Robotics I built a voice agent in Python."

    def guess(self, questions, profile):
        self.guesses.append([q["label"] for q in questions])
        return {q["name"]: q["options"][0] for q in questions if q.get("options")}


LEVER_LIKE = """
<form>
 <ul>
  <li class="application-question"><div class="application-label">Full name<span>✱</span></div>
      <div class="application-field"><input type="text" name="name" required></div></li>
  <li class="application-question"><div class="application-label">Email ✱</div>
      <div class="application-field"><input type="email" name="email" required></div></li>
  <li class="application-question"><div class="application-label">LinkedIn URL</div>
      <div class="application-field"><input type="text" name="urls[LinkedIn]"></div></li>
  <li class="application-question"><div class="application-label">Resume/CV ✱</div>
      <div class="application-field"><button type="button">Attach</button>
      <input type="file" name="resume" style="display:none" required></div></li>
  <li><label for="why">Why do you want to work at Acme? *</label>
      <textarea id="why" maxlength="300" required></textarea></li>
  <li><div class="question-title">What's the best project you've built? *</div>
      <input type="text" placeholder="Type here..." required></li>
  <li><fieldset><legend>Will you require visa sponsorship? *</legend>
      <label><input type="radio" name="spons" value="y" required> Yes</label>
      <label><input type="radio" name="spons" value="n"> No</label></fieldset></li>
  <li><fieldset><legend>Are you open to relocation? *</legend>
      <label><input type="radio" name="reloc" value="y" required onchange="document.getElementById('more').style.display='block'"> Yes</label>
      <label><input type="radio" name="reloc" value="n"> No</label></fieldset></li>
  <li id="more" style="display:none"><label for="offices">Tell us what draws you to relocating *</label>
      <textarea id="offices" required></textarea></li>
  <li><label for="gender">Gender</label>
      <select id="gender"><option value="">Select...</option><option>Male</option><option>Female</option>
      <option>Decline to self-identify</option></select></li>
  <li><div class="question-label">Where are you based? *</div>
      <div class="dropdown"><input role="combobox" aria-autocomplete="list" id="loc" required
        oninput="document.getElementById('lb').style.display='block'"
        onclick="document.getElementById('lb').style.display='block'">
      <ul role="listbox" id="lb" style="display:none">
        <li role="option" onclick="document.getElementById('loc').value='Buffalo, New York, United States';this.parentElement.style.display='none'">Buffalo, New York, United States</li>
        <li role="option" onclick="document.getElementById('loc').value='Boston, Massachusetts';this.parentElement.style.display='none'">Boston, Massachusetts</li>
      </ul></div></li>
  <li><label><input type="checkbox" id="agree" required> I agree to the privacy policy *</label></li>
 </ul>
 <button type="submit" onclick="window.submitted=true;return false">Submit application</button>
</form>
"""


def _run(page):
    return lambda fn: fn(page)


def test_scan_reads_labels_kinds_options_and_limits(page):
    page.set_content(LEVER_LIKE)
    fields = {f.label: f for f in form_engine.to_fields(form_engine.scan(page))}
    assert fields["Full name"].kind == "text" and fields["Full name"].required
    assert fields["Resume/CV"].kind == "file"                         # hidden input still found
    assert fields["Why do you want to work at Acme?"].max_chars == 300
    assert "What's the best project you've built?" in fields          # label over the "Type here..." placeholder
    assert fields["Will you require visa sponsorship?"].options == ["Yes", "No"]
    assert fields["Gender"].options == ["Male", "Female", "Decline to self-identify"]
    assert fields["Where are you based?"].kind == "combobox"
    assert fields["I agree to the privacy policy"].kind == "checkbox"
    assert "Tell us what draws you to relocating" not in fields      # hidden until relocation = Yes


def test_fill_end_to_end_open_text_by_llm_unknowns_only_for_missing_facts(page):
    page.set_content(LEVER_LIKE)
    spy = Spy()
    res = form_engine.fill({"company": "Acme", "role": "SWE"}, "Acme builds robots.", run=_run(page),
                           guess_fn=spy.guess, draft_fn=spy.draft)
    rows = {r["label"]: r for r in res.review}
    assert rows["Full name"]["value"] == "Alex Rivera" and rows["Full name"]["source"] == "bank"
    assert rows["Resume/CV"]["value"] == "Alex_Resume.txt"
    assert rows["Why do you want to work at Acme?"]["source"] == "llm_draft"
    assert rows["What's the best project you've built?"]["source"] == "llm_draft"
    assert rows["Tell us what draws you to relocating"]["source"] == "llm_draft"   # revealed, re-scanned
    assert rows["Gender"]["value"] == "Male"
    assert rows["Where are you based?"]["value"].startswith("Buffalo")
    assert rows["Are you open to relocation?"]["value"] == "Yes"
    # Left for the user: the missing sponsorship fact, and the agreement (UNKNOWN, never ticked).
    assert sorted(r["label"] for r in res.review if r["source"] == "needs_me") == \
        ["I agree to the privacy policy", "Will you require visa sponsorship?"]
    assert rows["I agree to the privacy policy"]["category"] == "consent"
    assert not page.locator("#agree").is_checked()
    assert not any(r["source"] == "failed" for r in res.review)
    assert not page.evaluate("window.submitted === true")
    assert page.input_value("#why") == "At Acme Robotics I built a voice agent in Python."
    assert ("Why do you want to work at Acme?", 300, False) in spy.drafts
    assert ("What's the best project you've built?", 0, True) in spy.drafts
    assert spy.guesses == []                                           # no choice question needed the LLM


MULTI = """
<div id="p1"><label for="fn">First name *</label><input id="fn" required>
  <label for="em">Email *</label><input id="em" type="email" required>
  <button type="button" onclick="document.getElementById('p1').remove();document.getElementById('p2').style.display='block'">Next</button></div>
<div id="p2" style="display:none"><label for="q">Describe a hard bug you fixed *</label><textarea id="q" required></textarea>
  <button type="button" onclick="window.submitted=true">Submit</button></div>
"""


def test_multipage_clicks_next_only_after_verify_and_stops_before_submit(page):
    page.set_content(MULTI)
    spy = Spy()
    res = form_engine.fill({"company": "Acme", "role": "SWE"}, run=_run(page), guess_fn=spy.guess, draft_fn=spy.draft)
    assert res.pages == 2 and res.stopped_at == "final page"
    assert {r["label"]: r["page"] for r in res.review} == {"First name": 1, "Email": 1, "Describe a hard bug you fixed": 2}
    assert page.input_value("#q").startswith("At Acme Robotics")
    assert not page.evaluate("window.submitted === true")


def test_unknown_required_fact_stops_on_that_page(page):
    page.set_content("""<label for="ws">Work authorization number *</label><input id="ws" required>
      <button type="button" onclick="window.next=true">Continue</button>""")
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    assert res.stopped_at == "needs you" and not page.evaluate("window.next === true")
    assert res.review[0]["source"] == "needs_me"


def test_generated_answer_is_reused_for_the_same_generic_question(page):
    page.set_content(LEVER_LIKE)
    form_engine.fill({"company": "Acme", "role": "SWE"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    page.set_content(LEVER_LIKE.replace("Acme", "Globex"))
    spy = Spy()
    res = form_engine.fill({"company": "Globex", "role": "SWE"}, run=_run(page), guess_fn=spy.guess, draft_fn=spy.draft)
    rows = {r["label"]: r for r in res.review}
    assert rows["What's the best project you've built?"]["source"] == "reused"
    assert rows["Why do you want to work at Globex?"]["source"] == "llm_draft"      # company-specific: rewritten
    assert all(d[0] != "What's the best project you've built?" for d in spy.drafts)


def test_find_next_never_returns_a_submit(page):
    page.set_content("<button>Submit</button><button>Apply now</button><a role=button>Continue</a>")
    assert form_engine.find_next(page).inner_text() == "Continue"
    page.set_content("<button>Submit application</button>")
    assert form_engine.find_next(page) is None


# ── regressions from the live runs (2026-10-21) ─────────────────────────────

def test_jazzhr_labels_placeholders_and_checkbox_groups(page):
    """City/State/Postal have only placeholders; checkboxes q-1, q-2 … are one question whose
    label[for] points at no element; "-- No answer --" isn't an option."""
    page.set_content("""<h2>Apply for this position</h2><div class="job-form-fields">
      <div class="form-group"><label for="a">Address</label><input id="a">
        <div class="row"><div class="col"><input id="c" placeholder="City"></div>
        <div class="col"><input id="s" placeholder="State/Province"></div></div></div>
      <div class="form-group"><label for="y">Years?</label><select id="y"><option value="x">-- No answer --</option>
        <option>1-2 years</option></select></div>
      <div class="form-group"><label for="q-3011547" class="control-label">Which practices have you used?<i>*</i></label>
        <label class="checkbox"><input type="checkbox" name="cb-3011547-1" value="Git"> Git</label>
        <label class="checkbox"><input type="checkbox" name="cb-3011547-2" value="Docker"> Docker</label></div></div>""")
    raw = {r["label"]: r for r in form_engine.scan(page)}
    assert {"Address", "City", "State/Province"} <= set(raw)
    assert raw["Years?"]["options"] == ["1-2 years"]
    group = raw["Which practices have you used?"]
    assert group["kind"] == "checkgroup" and group["options"] == ["Git", "Docker"] and group["required"]


def test_privacy_notice_mid_form_waits_for_the_user_then_fills(page):
    page.set_content("""<div id="gate"><p>Privacy notice</p><button onclick="document.getElementById('gate').remove();
      document.getElementById('f').style.display='block'">I Accept</button><button>I Decline</button></div>
      <div id="f" style="display:none"><label for="fn">First name *</label><input id="fn" required></div>""")
    asked = []

    def wait(kind):                    # the user clears it in the tab
        asked.append(kind)
        page.click("text=I Accept")
        return True
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft, wait_fn=wait)
    assert asked == ["agreement"] and page.input_value("#fn") == "Alex"
    page.set_content("<button>I Accept</button><button>I Decline</button>")
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    assert res.stopped_at.startswith("blocked by a privacy notice")


def test_a_fact_in_a_text_box_comes_back_unknown_never_as_prose(page):
    page.set_content('<label for="hs">High School Name *</label><textarea id="hs" required></textarea>'
                     '<label for="pref">Which office would you prefer and why? *</label><textarea id="pref" required></textarea>')

    def draft(label, job_text, profile, max_chars=0, short=False, allow_unknown=False):
        return "UNKNOWN"
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=draft)
    rows = {r["label"]: r for r in res.review}
    assert rows["High School Name"]["source"] == "needs_me"                      # never the university
    assert rows["Which office would you prefer and why?"]["category"] == "fact"
    assert page.input_value("#hs") == "" and page.input_value("#pref") == ""


def test_open_question_is_always_written_even_if_the_llm_balks(page):
    page.set_content('<label for="r">Tell us about someone you look up to *</label><textarea id="r" required></textarea>')
    calls = []

    def draft(label, job_text, profile, max_chars=0, short=False, allow_unknown=False):
        calls.append(allow_unknown)
        return "UNKNOWN" if len(calls) == 1 else "My first manager at Acme Robotics, who taught me to test first."
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=draft)
    assert calls == [False, False] and res.review[0]["source"] == "llm_draft"     # never needs_me


def test_consent_checkbox_and_decline_option_are_not_a_privacy_wall(page):
    """#4: checkboxes and radios never count; only Accept + Decline buttons side by side."""
    from aria.features.applying import walls
    page.set_content("""<label><input type="checkbox"> I accept the privacy policy</label>
      <fieldset><legend>Veteran status</legend><label><input type="radio" name="v"> I decline to answer</label></fieldset>
      <p>Note: I accept that... I decline to ...</p>""")
    assert form_engine.wall(page) is None
    page.set_content("<button>I Accept</button><div style='height:2000px'></div><button>I Decline</button>")
    assert form_engine.wall(page) is None                        # far apart: not one notice
    page.set_content("<div><p>Privacy notice</p><button>I Accept</button><button>I Decline</button></div>")
    assert form_engine.wall(page) == "agreement"
    assert walls.detect(**{k: v for k, v in page.evaluate(walls.WALL_JS).items() if k != "agreement"}) is None


def test_dropdown_clicks_its_own_option_not_another_questions(page):
    """#6: option lookup was page-wide; an earlier Yes/No group (class *option*) got clicked."""
    page.set_content("""
      <div class="q"><div class="question-title">Have you worked here before? *</div>
        <div class="option-list"><span class="option" onclick="window.wrong=true">Yes</span>
        <span class="option" onclick="window.wrong=true">No</span></div></div>
      <div class="q"><label for="cb">Will you relocate? *</label>
        <input id="cb" role="combobox" aria-controls="lb" required
          onclick="document.getElementById('lb').style.display='block'">
        <ul role="listbox" id="lb" style="display:none">
          <li role="option" onclick="document.getElementById('cb').value='Yes';this.parentElement.style.display='none'">Yes</li>
          <li role="option" onclick="document.getElementById('cb').value='No';this.parentElement.style.display='none'">No</li>
        </ul></div>""")
    f = planner.FormField(form_engine.scan(page)[-1]["fid"], "Will you relocate?", "combobox", True)
    assert form_engine.fill_one(page, planner.Planned(f, "Yes", "bank"))
    assert page.input_value("#cb") == "Yes" and not page.evaluate("window.wrong === true")


@pytest.mark.parametrize("kind,want,actual,ok", [
    ("select", "Not applicable", "No", False), ("select", "No", "Not applicable", False),
    ("select", "Yes", "Yes", True), ("radio", "No", "No, I do not have a disability", False),
    ("combobox", "Male", "Gender Male", True), ("combobox", "No", "Not applicable", False),
    ("combobox", "No", "No, I do not have a disability", False),
])
def test_verify_needs_the_whole_option_not_a_fragment(kind, want, actual, ok):
    """#7: 'no' inside 'not applicable' passed verify."""
    f = planner.FormField("x", "Q", kind, True, ["No", "Not applicable", "Yes", "No, I do not have a disability"]
                          if kind != "combobox" else [])
    assert form_engine.matches(planner.Planned(f, want, "bank"), {"value": actual, "error": ""}) is ok


# ── Ashby-style forms (2026-10-23: work authorization, essays, links were silently skipped) ──

ASHBY_LIKE = """<style>._required_x::after{content:"*"}</style><form>
<div><label class="_heading _required_x" for="loc">Location</label><input id="loc"></div>
<div><label class="_heading" for="li">Link to your LinkedIn profile</label><input id="li"></div>
<div><label class="_heading _required_x" for="wa">Are you legally authorized to work in the US for any employer?</label>
  <div><button type="button" aria-pressed="false" data-option="yes"
        onclick="this.setAttribute('aria-pressed','true');this.nextElementSibling.setAttribute('aria-pressed','false')">Yes</button
  ><button type="button" aria-pressed="false" data-option="no"
        onclick="this.setAttribute('aria-pressed','true');this.previousElementSibling.setAttribute('aria-pressed','false')">No</button
  ><input type="checkbox" tabindex="-1" name="wa" style="position:absolute;opacity:0;width:1px;height:1px"></div></div>
<div><label class="_heading" for="sp">Will you now or in the future require visa sponsorship?</label>
  <div><button type="button" aria-pressed="false" data-option="yes">Yes</button><button type="button" aria-pressed="false"
  data-option="no">No</button><input type="checkbox" tabindex="-1" name="sp" style="opacity:0"></div></div>
<div><label class="_heading" for="ex">What makes you excited about the Acme mission?</label><textarea id="ex"></textarea></div>
<div><label class="_heading" for="ot">If other, please specify below</label><textarea id="ot"></textarea></div>
</form>"""


def test_ashby_required_marks_and_yes_no_buttons_are_read(page):
    page.set_content(ASHBY_LIKE)
    got = {r["label"]: r for r in form_engine.scan(page)}
    assert got["Location"]["required"] is True                        # a CSS-drawn asterisk
    assert got["Link to your LinkedIn profile"]["required"] is False
    wa = got["Are you legally authorized to work in the US for any employer?"]
    assert (wa["kind"], wa["options"], wa["required"]) == ("buttons", ["Yes", "No"], True)
    assert not [r for r in got.values() if r["kind"] == "checkbox"]  # the hidden checkboxes aren't fields


def test_ashby_like_form_fills_buttons_links_and_optional_essays(page, bank):
    data = json.loads(bank.read_text())
    data["answers"]["work_authorization"] = "Yes"
    bank.write_text(json.dumps(data))
    page.set_content(ASHBY_LIKE)
    spy = Spy()
    res = form_engine.fill({"company": "Acme", "role": "Engineer"}, "posting", run=_run(page),
                           draft_fn=spy.draft, guess_fn=spy.guess, llm_available=True)
    rows = {r["label"]: r for r in res.review}
    assert rows["Are you legally authorized to work in the US for any employer?"]["value"] == "Yes"
    assert page.evaluate("document.querySelector('[data-option=yes]').getAttribute('aria-pressed')") == "true"
    assert rows["Link to your LinkedIn profile"]["source"] == "bank"
    # eligibility is never skipped silently, even optional: no answer in the bank → yours
    assert rows["Will you now or in the future require visa sponsorship?"]["source"] == "needs_me"
    assert rows["What makes you excited about the Acme mission?"]["source"] == "llm_draft"
    assert rows["If other, please specify below"]["source"] == "skipped"
