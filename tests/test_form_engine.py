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


def test_unknown_required_fact_tries_next_and_stops_only_when_the_page_refuses(page):
    page.set_content("""<label for="ws">Work authorization number *</label><input id="ws" required>
      <button type="button" onclick="window.next=true">Continue</button>""")
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    assert page.evaluate("window.next === true")                      # tried, didn't move
    assert res.stopped_at == "needs you" and res.pages == 1
    assert res.review[0]["source"] == "needs_me"


def test_a_field_left_open_never_stops_the_run_short_of_the_last_page(page):
    # 2026-10-25: Meijer's State failed on page 1 and the run stopped there; the user wants every
    # page filled up to the Submit
    page.set_content("""<div id="p1"><label for="ws">Work authorization number *</label><input id="ws">
      <label for="fn">First name *</label><input id="fn" required>
      <button type="button" onclick="document.getElementById('p1').remove();document.getElementById('p2').style.display='block'">Next</button></div>
      <div id="p2" style="display:none"><label for="em">Email *</label><input id="em" type="email" required>
      <button type="button" onclick="window.submitted=true">Submit</button></div>""")
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    assert res.pages == 2 and res.stopped_at == "final page"
    rows = {r["label"]: r for r in res.review}
    assert rows["Work authorization number"]["source"] == "needs_me"      # still listed for you
    assert page.input_value("#em")                                         # page 2 got filled
    assert not page.evaluate("window.submitted === true")


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


# ── live run 2026-10-24: Workday re-draws a page it won't leave; dropdowns that hide their input ──

REDRAWN = """<div id="wrap"></div>
<script>
function draw() { document.getElementById('wrap').innerHTML = `
  <label for="fn">First name *</label><input id="fn" required>
  <button type="button" onclick="draw()">Save and Continue</button>`; }
draw();
</script>"""


def test_a_next_that_redraws_the_same_page_did_nothing(page, monkeypatch):
    """Meijer: Workday re-drew My Information 8 times (new elements, same questions) and every
    one counted as a new page."""
    monkeypatch.setattr(form_engine, "_settled_signature", lambda run, max_secs=10.0: run(form_engine.signature))
    page.set_content(REDRAWN)
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    assert res.pages == 1 and res.stopped_at.startswith("next did nothing")


def test_an_optional_question_left_for_you_does_not_stop_the_next_page(page, monkeypatch):
    monkeypatch.setattr(form_engine, "_settled_signature", lambda run, max_secs=10.0: run(form_engine.signature))
    page.set_content("""<div id="p1"><label for="fn">First name *</label><input id="fn" required>
      <label for="sp">Will you require visa sponsorship?</label><input id="sp">
      <button type="button" onclick="document.getElementById('p1').remove();document.getElementById('p2').style.display='block'">Next</button></div>
      <div id="p2" style="display:none"><label for="em">Email *</label><input id="em" type="email" required></div>""")
    res = form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    rows = {r["label"]: r for r in res.review}
    assert rows["Will you require visa sponsorship?"]["source"] == "needs_me"     # still asked at review
    assert res.pages == 2 and page.input_value("#em") == "me@example.com"


def test_react_select_with_its_placeholder_over_the_input_still_opens(page):
    """Nodal Exchange: the placeholder div took every click on the input (4 s timeouts × 3)."""
    page.set_content("""<form><label for="ed">What is your highest level of education? *</label>
      <div class="select__control" style="position:relative;width:300px;height:30px"
           onclick="document.getElementById('lb').style.display='block'">
        <div style="position:absolute;inset:0;z-index:2">Select...</div>
        <input id="ed" role="combobox" aria-autocomplete="list" style="width:280px">
      </div>
      <ul role="listbox" id="lb" style="display:none">
        <li role="option" onclick="document.getElementById('ed').value='Master\\'s Degree'">Master's Degree</li>
        <li role="option" onclick="document.getElementById('ed').value='Bachelor\\'s Degree'">Bachelor's Degree</li>
      </ul>
      <button type="submit" onclick="window.submitted=true;return false">Submit</button></form>""")
    f = planner.FormField(form_engine.scan(page)[0]["fid"], "What is your highest level of education?", "combobox", True)
    assert form_engine.fill_one(page, planner.Planned(f, "Master's Degree", "bank"))
    assert page.input_value("#ed") == "Master's Degree"
    assert not page.evaluate("window.submitted === true")          # no Enter inside a form


WORKDAY_PROMPT = """<div data-automation-id="formField-source"><label for="s">How Did You Hear About Us?*</label>
  <div data-automation-id="multiselectInputContainer">
    <input id="s" data-automation-id="searchBox" placeholder="Search" onkeydown="if (event.key === 'Enter') top_level()">
  </div>
  <div id="sel"></div>
  <div role="listbox" id="lb"></div></div>
<script>
const lb = () => document.getElementById('lb');
function top_level() { lb().innerHTML = ['Job Board', 'Recruiter', 'Social Media'].map(t =>
  `<div role="option" onclick="pick('${t}')">${t}</div>`).join(''); }
function pick(t) {
  if (t === 'Recruiter') { lb().innerHTML = ['Recruiter Contacted Me', 'Agency Recruiter'].map(x =>
      `<div role="option" onclick="done('${x}')">${x}</div>`).join(''); return; }
  done(t); }
function done(t) { document.getElementById('sel').innerText = t; lb().innerHTML = ''; window.picked = t; }
</script>"""


def test_workday_prompt_lists_on_enter_and_opens_the_recruiter_category(page):
    """GlobalFoundries: "How Did You Hear About Us?" was typed into the search box as text."""
    page.set_content(WORKDAY_PROMPT)
    raw = form_engine.scan(page)
    assert raw[0]["kind"] == "combobox"
    f = planner.FormField(raw[0]["fid"], "How Did You Hear About Us?", "combobox", True)
    assert form_engine.fill_one(page, planner.Planned(f, "Recruiter", "bank"))
    assert page.evaluate("window.picked") == "Recruiter Contacted Me"


def test_custom_element_buttons_count_as_next(page):
    page.set_content('<ukg-button onclick="window.next=true">Continue</ukg-button>')
    form_engine.find_next(page).click()
    assert page.evaluate("window.next === true")


# ── live run 2026-10-24 (2): Workday's autofilled My Experience, RippleMatch (eBay) ──

WORKDAY_MY_EXPERIENCE = """
<div data-fkit-id="workExperience-22--null">
  <div data-automation-id="formField-jobTitle" data-fkit-id="workExperience-22--jobTitle">
    <label for="w-title"><span>Job Title<abbr aria-hidden="true">*</abbr></span></label>
    <input type="text" id="w-title" name="jobTitle" aria-required="true" value="Software Engineer Intern"></div>
  <div data-automation-id="formField-companyName" data-fkit-id="workExperience-22--companyName">
    <label for="w-co"><span>Company<abbr aria-hidden="true">*</abbr></span></label>
    <input type="text" id="w-co" name="companyName" aria-required="true" value="N/A"></div>
  <div data-automation-id="formField-endDate" data-fkit-id="workExperience-22--endDate">
    <input role="spinbutton" aria-label="Year" aria-required="true" id="w-year" data-automation-id="dateSectionYear-input" value=""></div>
</div>
<div data-fkit-id="education-40--null">
  <div data-automation-id="formField-degree" data-fkit-id="education-40--degree">
    <label for="e-deg"><span>Degree<abbr aria-hidden="true">*</abbr></span></label>
    <button aria-haspopup="listbox" type="button" value="" aria-label="Degree Select One Required" name="degree" id="e-deg"
      onclick="document.getElementById('deg-lb').style.display='block'">Select One</button>
    <ul role="listbox" id="deg-lb" style="display:none">
      <li role="option" onclick="pick(this)">Bachelor's Degree</li><li role="option" onclick="pick(this)">Master's Degree</li></ul></div>
  <div data-automation-id="formField-fieldOfStudy" data-fkit-id="education-40--fieldOfStudy">
    <label for="e-fos">Field of Study</label>
    <div data-automation-id="multiSelectContainer">
      <div data-automation-id="multiselectInputContainer">
        <ul role="listbox" data-automation-id="selectedItemList"><li><div role="option" data-automation-id="selectedItem"
          aria-label="Computer and Information Science, press delete to clear value.">Computer and Information Science</div></li></ul>
        <input id="e-fos" data-automation-id="searchBox" placeholder="Search"></div></div>
    <div data-automation-id="activeListContainer"><div data-automation-id="promptOption">
      <input type="radio" data-automation-id="radioBtn" id="o1"><label for="o1">Accounting</label></div></div></div>
</div>
<div data-fkit-id="webAddress-19--null"><div data-automation-id="formField-url" data-fkit-id="webAddress-19--url">
  <label for="u1"><span>URL<abbr aria-hidden="true">*</abbr></span></label>
  <input type="text" id="u1" name="url" aria-required="true" value="https://www.linkedin.com/in/me"></div></div>
<script>function pick(li) { const b = document.getElementById('e-deg'); b.innerText = li.innerText; li.parentElement.style.display = 'none'; }</script>
"""


def test_workdays_autofilled_experience_is_kept_and_never_overwritten(page, bank):
    data = json.loads(bank.read_text())
    data["answers"].update({"degree": "Master of Science, Computer Science", "grad_date": "05/18/2026"})
    bank.write_text(json.dumps(data))
    page.set_content(WORKDAY_MY_EXPERIENCE)
    labels = [r["label"] for r in form_engine.scan(page)]
    assert "Accounting" not in labels                       # a picker's options aren't questions
    assert "Degree" in labels
    res = form_engine.fill({"company": "Meijer"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    rows = {r["label"]: r for r in res.review}
    assert rows["Job Title"]["source"] == "page" and rows["Job Title"]["value"] == "Software Engineer Intern"
    assert rows["Company"]["source"] == "needs_me"           # "N/A" is no answer: yours to fix
    assert rows["Year"]["source"] == "needs_me" and page.input_value("#w-year") == ""   # never the grad year
    assert rows["URL"]["source"] == "page"
    assert rows["Field of Study"]["source"] == "page" and "Computer and Information Science" in rows["Field of Study"]["value"]
    assert rows["Degree"]["source"] == "bank" and page.inner_text("#e-deg") == "Master's Degree"


RIPPLEMATCH = """<form>
<style>._visuallyHidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); left: -9999px; }</style>
<label for="school">University<span class="_required"> * </span></label>
<div aria-expanded="false"><input id="school" name="school" placeholder="Type in university" role="combobox"
  aria-required="true" value="University at Buffalo - SUNY"></div>
<label for="gm">Graduation month<span class="_required"> * </span></label>
<div><input id="gm" name="graduationMonth" readonly placeholder="Select month" role="combobox" aria-required="true"
  onclick="document.getElementById('months').style.display='block'"></div>
<ul id="months" style="display:none"><li onclick="document.getElementById('gm').value='April';this.parentElement.style.display='none'">April</li>
  <li onclick="document.getElementById('gm').value='May';this.parentElement.style.display='none'">May</li></ul>
<label for="q1">Are you authorized to work lawfully in the United States?<span class="_required"> * </span></label>
<fieldset id="q1" role="listbox" aria-label="Are you authorized to work lawfully in the United States?">
  <legend class="_visuallyHidden">Are you authorized to work lawfully in the United States?</legend>
  <label for="r-yes"><div></div>Yes, I am<input id="r-yes" class="_visuallyHidden" type="radio" name="q-569352" value="Yes, I am" aria-required="true"></label>
  <label for="r-no"><div></div>No, I am not<input id="r-no" class="_visuallyHidden" type="radio" name="q-569352" value="No, I am not" aria-required="true"></label>
</fieldset>
<button type="submit" onclick="window.submitted=true;return false">Submit application</button></form>"""


def test_ripplematch_hidden_radios_plain_list_dropdowns_and_a_prefilled_school(page, bank):
    data = json.loads(bank.read_text())
    data["answers"].update({"school": "University at Buffalo, SUNY", "work_authorization": "Yes", "grad_date": "05/18/2026"})
    bank.write_text(json.dumps(data))
    page.set_content(RIPPLEMATCH)
    res = form_engine.fill({"company": "eBay"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    rows = {r["label"]: r for r in res.review}
    assert page.is_checked("#r-yes") and not page.is_checked("#r-no")
    assert rows["Are you authorized to work lawfully in the United States?"]["status"] == "filled"
    assert rows["University"]["status"] == "filled"             # " - " vs ", " is the same school
    assert page.input_value("#gm") == "May"
    assert not page.evaluate("window.submitted === true")


def test_the_banks_answer_replaces_a_wrong_prefilled_one(page, bank):
    """User, 2026-10-24: Aria refills every field it knows. RippleMatch's profile had 'No, I am
    not' preselected; the bank says Yes — the page ends up saying Yes."""
    data = json.loads(bank.read_text())
    data["answers"].update({"work_authorization": "Yes"})
    bank.write_text(json.dumps(data))
    page.set_content(RIPPLEMATCH.replace('value="No, I am not" aria-required="true">',
                                         'value="No, I am not" aria-required="true" checked>'))
    assert page.is_checked("#r-no")
    form_engine.fill({"company": "eBay"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    assert page.is_checked("#r-yes") and not page.is_checked("#r-no")


# ── 2026-10-25: Workday's resume parse made jobs of the PROJECTS section; the bank had one school ──

WORKDAY_BLOCKS = r"""
<div id="root"></div>
<script>
let seq = 10;
function panel(kind, n, id, vals) {
  const prefix = kind === 'w' ? 'Work-Experience' : 'Education';
  const sec = kind === 'w' ? 'workExperience' : 'education';
  const fk = sec + '-' + id;
  const f = (sub, inner) => `<div data-fkit-id="${fk}--${sub}">${inner}</div>`;
  const body = kind === 'w'
    ? f('jobTitle', `<label for="${fk}-t">Job Title *</label><input id="${fk}-t" required value="${vals[0] || ''}">`) +
      f('companyName', `<label for="${fk}-c">Company *</label><input id="${fk}-c" required value="${vals[1] || ''}">`) +
      f('currentlyWorkHere', `<label for="${fk}-cur">I currently work here</label><input type="checkbox" name="currentlyWorkHere" id="${fk}-cur" ${vals[2] ? 'checked' : ''}>`) +
      f('startDate', `<fieldset><legend><label>From *</label></legend><input role="spinbutton" aria-label="Month" id="${fk}-sm" value="${vals[3] || ''}"><input role="spinbutton" aria-label="Year" id="${fk}-sy" value="${vals[4] || ''}"></fieldset>`) +
      f('roleDescription', `<label for="${fk}-d">Role Description</label><textarea id="${fk}-d"></textarea>`)
    : f('schoolName', `<label for="${fk}-s">School or University *</label><input id="${fk}-s" required value="${vals[0] || ''}">`) +
      f('degree', `<label for="${fk}-g">Degree *</label><select id="${fk}-g" required><option value="">Select One</option><option>MA</option><option>MS</option><option>BA</option><option>BS</option></select>`);
  return `<div role="group" aria-labelledby="${prefix}-${n}-panel"><h5 id="${prefix}-${n}-panel">${prefix} ${n}</h5>
    <button type="button" onclick="this.closest('[role=group]').remove(); renumber()"><svg class="wd-icon-trash"></svg></button>${body}</div>`;
}
function renumber() {}
function section(kind, title, panels) {
  return `<div role="group" aria-labelledby="${title}-section"><h4 id="${title}-section">${title}</h4><div class="panels">${panels}</div>
    <button type="button" data-automation-id="add-button" onclick="add('${kind}', this)">Add Another</button></div>`;
}
function add(kind, btn) {
  const box = btn.parentElement.querySelector('.panels');
  const n = box.children.length + 1;
  box.insertAdjacentHTML('beforeend', panel(kind, n, ++seq, []));
}
document.getElementById('root').innerHTML =
  section('w', 'Work-Experience', panel('w', 1, 12, ['Software Engineer Intern', 'Northwind', true, '1', '2025']) +
    panel('w', 2, 13, ['Software Engineer Intern', 'AppsTek Corp', false, '9', '2023']) +
    panel('w', 3, 14, ['PROJECTS', 'LLM Gateway - Cost', false, '9', '2026']) +
    panel('w', 4, 15, ['', 'Draw.AI', false, '8', '2026'])) +
  section('e', 'Education', panel('e', 1, 30, ['University at Buffalo, SUNY']));
</script>
<button type="button" onclick="window.submitted=true">Submit</button>
"""


def test_workday_blocks_are_replaced_with_your_lists(page, monkeypatch):
    lists = {
        "experience": [
            {"title": "Software Engineer Intern", "company": "Globex", "location": "", "start": "Jan 2025",
             "end": "May 2025", "description": "• Built billing"},
            {"title": "Data Engineer", "company": "Initech", "location": "", "start": "Sep 2023",
             "end": "Present", "description": "• Spark pipelines"}],
        "education": [
            {"school": "State University of Somewhere", "degree": "M.S.", "field": "Computer Science",
             "start": "Aug 2024", "end": "May 2026", "gpa": ""},
            {"school": "Some Institute of Technology", "degree": "B.Tech.", "field": "EE",
             "start": "Sep 2020", "end": "May 2024", "gpa": ""}],
    }
    monkeypatch.setattr(form_engine.answer_bank, "get_list", lambda name: lists.get(name, []))
    page.set_content(WORKDAY_BLOCKS)
    res = form_engine.fill({"company": "Acme", "role": "SWE"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    work = page.locator('[aria-labelledby="Work-Experience-section"] [role=group]')
    edu = page.locator('[aria-labelledby="Education-section"] [role=group]')
    assert work.count() == 2 and edu.count() == 2                 # the projects deleted, the B.Tech added
    assert page.input_value("#workExperience-12-c") == "Globex"   # Workday's parse overwritten
    assert page.input_value("#workExperience-13-t") == "Data Engineer"
    assert not page.is_checked("#workExperience-12-cur")           # Jan–May 2025 is not current
    assert page.is_checked("#workExperience-13-cur")               # "Present"
    assert page.input_value("#workExperience-12-sm") in ("01", "1") and page.input_value("#workExperience-12-sy") == "2025"
    assert page.input_value("#workExperience-13-d") == "• Spark pipelines"
    new_school = page.locator('[aria-labelledby="Education-section"] [role=group]').nth(1).locator("input").first
    assert new_school.input_value() == "Some Institute of Technology"
    degrees = page.locator('[aria-labelledby="Education-section"] select')
    assert [degrees.nth(i).input_value() for i in range(2)] == ["MS", "BS"]   # never "MA" for an M.S.
    assert "PROJECTS" not in [page.locator("input").nth(i).input_value() for i in range(page.locator("input").count())]
    assert not any(r["source"] == "failed" for r in res.review), [r for r in res.review if r["source"] == "failed"]
    assert not page.evaluate("window.submitted === true")


def test_no_list_leaves_the_blocks_as_the_page_has_them(page, monkeypatch):
    monkeypatch.setattr(form_engine.answer_bank, "get_list", lambda name: [])
    page.set_content(WORKDAY_BLOCKS)
    form_engine.fill({"company": "Acme"}, run=_run(page), guess_fn=Spy().guess, draft_fn=Spy().draft)
    assert page.locator('[aria-labelledby="Work-Experience-section"] [role=group]').count() == 4
    assert page.input_value("#workExperience-12-c") == "Northwind"


def test_each_jobs_current_box_is_its_own_question(page):
    page.set_content(WORKDAY_BLOCKS)
    rows = [r for r in form_engine.scan(page) if r["kind"] in ("checkbox", "checkgroup")]
    assert len(rows) == 4 and all(r["kind"] == "checkbox" for r in rows)
