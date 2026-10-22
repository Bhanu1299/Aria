"""tests/test_ats.py — ATS detection, Greenhouse parsing (saved real pages), planning, review."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import aria.features.answer_bank as ab
import aria.state.db as db
from aria.core import paths
from aria.features import ats
from aria.features.ats import answers as llm_answers
from aria.features.ats import greenhouse as gh
from aria.features.ats.detect import detect, greenhouse_form_url, identify

FIX = Path(__file__).parent / "fixtures" / "greenhouse"


def page(name: str) -> str:
    return (FIX / f"{name}.html").read_text()


@pytest.fixture(autouse=True)
def bank(tmp_path, monkeypatch):
    """A realistic bank (the user's shape: legal answers left null) + empty identity/db."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "identity.json")
    path = tmp_path / "answer_bank.json"
    path.write_text(json.dumps({
        "answers": {
            "first_name": "Alex Jordan", "last_name": "Rivera",
            "legal_name": "Alex Jordan Rivera", "preferred_name": "Alex Jordan",
            "email": "me@example.com", "phone": "+1-555-555-0100",
            "linkedin": "https://linkedin.com/in/me", "github": "https://github.com/me",
            "city": "Springfield", "country": "United States", "location": "Springfield, IL",
            "degree": "Bachelor of Arts, History", "relocation": "Yes",
            "sponsorship": None, "work_authorization": None, "years_experience": None, "start_date": None,
            "eeo": {"gender": "Female", "race": "White", "hispanic_latino": "No",
                    "veteran": "I am not a protected veteran", "disability": "No, I do not have a disability",
                    "lgbtq": "No"},
        },
        "aliases": {"location": ["city", "where are you based"]},
    }))
    monkeypatch.setattr(ab, "BANK_PATH", path)
    return path


# ── detection ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,name", [
    ("https://job-boards.greenhouse.io/faradayfuture/jobs/8036218003", "greenhouse"),
    ("https://boards.greenhouse.io/discord/jobs/8806482002", "greenhouse"),
    ("https://job-boards.eu.greenhouse.io/acme/jobs/1", "greenhouse"),
    ("https://www.conviva.ai/careers/job/8036615003?gh_jid=8036615003", "greenhouse"),
    ("https://abnormal.ai/careers/jobs/8015179003?gh_jid=8015179003", "greenhouse"),
    ("https://jobs.lever.co/acme/123", "lever"),
    ("https://jobs.ashbyhq.com/luminai/ba606839", "ashby"),
    ("https://nike.wd1.myworkdayjobs.com/en-US/nke/job/x", "workday"),
    ("https://amazon.jobs/en/jobs/10581943", None),
])
def test_identify(url, name):
    assert identify(url) == name


def test_detect_only_names_atses_with_a_filler():
    assert detect("https://jobs.lever.co/acme/123") is None
    assert detect("https://jobs.ashbyhq.com/x/y") is None
    assert detect("https://job-boards.greenhouse.io/x/jobs/1") == "greenhouse"


def test_detect_embedded_greenhouse_from_page_html():
    html = '<div id="grnhse_app"></div><iframe id="grnhse_iframe" src="https://job-boards.greenhouse.io/embed/job_app?for=acme&amp;token=42"></iframe>'
    assert detect("https://acme.com/careers/swe", html) == "greenhouse"
    assert detect("https://acme.com/careers/swe", "<html>no ats</html>") is None


def test_form_url_for_boards_is_the_url_itself():
    url = "https://job-boards.greenhouse.io/faradayfuture/jobs/8036218003"
    assert greenhouse_form_url(url) == url


def test_form_url_from_embed_iframe():
    html = '<iframe id="grnhse_iframe" src="https://job-boards.greenhouse.io/embed/job_app?for=acme&amp;token=42"></iframe>'
    assert greenhouse_form_url("https://acme.com/jobs?gh_jid=42", html) == \
        "https://job-boards.greenhouse.io/embed/job_app?for=acme&token=42"


def test_form_url_from_gh_jid_with_and_without_board():
    assert greenhouse_form_url("https://www.conviva.ai/careers/job/1?gh_jid=8036615003") == \
        "https://job-boards.greenhouse.io/embed/job_app?token=8036615003"
    html = '<script src="https://boards.greenhouse.io/embed/job_board/js?for=conviva"></script>'
    assert greenhouse_form_url("https://www.conviva.ai/careers/job/1?gh_jid=8036615003", html) == \
        "https://job-boards.greenhouse.io/embed/job_app?token=8036615003&for=conviva"


# ── parsing real pages ──────────────────────────────────────────────────────

def _by_label(form, label):
    return next(f for f in form.fields if f.label.startswith(label))


def test_parse_conviva_embed():
    form = gh.parse_form(page("conviva_embed"))
    assert (form.title, form.company) == ("AI/ML Engineer", "Conviva")
    assert "machine learning" in form.description.lower()
    names = [f.name for f in form.fields]
    assert names[0] == "hispanic_ethnicity"                       # before race: race renders after it
    assert "resume_text" not in names and "cover_letter_text" not in names
    first = _by_label(form, "First Name")
    assert (first.kind, first.required, first.section) == ("text", True, "questions")
    sponsor = _by_label(form, "Will you now or in the future require sponsorship")
    assert (sponsor.kind, sponsor.required, sponsor.options) == ("select", True, ["Yes", "No"])
    degree = _by_label(form, "What is your highest completed degree?")
    assert degree.options == ["PhD/Doctorate", "Master's", "Bachelor's", "Other"]
    essay = _by_label(form, "Briefly describe an ML model")
    assert (essay.kind, essay.required) == ("textarea", True)
    resume = next(f for f in form.fields if f.name == "resume")
    assert (resume.kind, resume.required) == ("file", True)
    gender = next(f for f in form.fields if f.name == "gender")
    assert gender.section == "eeoc" and "Decline To Self Identify" in gender.options
    country = next(f for f in form.fields if f.name == "country")
    assert (country.kind, country.section, country.options) == ("select", "dom", [])


def test_parse_discord_demographics_and_location():
    form = gh.parse_form(page("discord_jobboards"))
    demo = [f for f in form.fields if f.section == "demographic"]
    assert len(demo) == 7
    disability = next(f for f in demo if f.label == "Disability Status")
    assert disability.required and disability.options[1].startswith("No, I do not have a disability")
    assert next(f for f in demo if f.label.startswith("Race or Ethnicity")).kind == "multiselect"
    locs = [f for f in form.fields if "location" in f.name]
    assert [(f.name, f.kind) for f in locs] == [("candidate-location", "location")]   # schema dup dropped
    assert _by_label(form, "Why do you want to work at Discord?").kind == "textarea"


def test_parse_minimal_form():
    form = gh.parse_form(page("faraday_jobboards"))
    assert [f.name for f in form.fields][:5] == ["first_name", "last_name", "preferred_name", "email", "phone"]
    assert not any(f.section in ("eeoc", "demographic") for f in form.fields)


def test_parse_pay_range():
    assert gh.parse_form(page("anthropic_jobboards")).salary_range == (190000, 230000)
    assert gh.parse_form(page("conviva_embed")).salary_range is None


def test_parse_legacy_dom_form_without_remix_context():
    html = """
    <form id="application_form">
      <label for="first_name">First Name *</label><input id="first_name" type="text" aria-required="true">
      <label for="email">Email</label><input id="email" type="text">
      <label for="job_application_answers_attributes_0_text_value">Why us?</label>
      <textarea id="job_application_answers_attributes_0_text_value"></textarea>
      <label for="sponsor">Do you require sponsorship?</label>
      <select id="sponsor" required><option value="">Please select</option><option>Yes</option><option>No</option></select>
      <input id="csrf" type="hidden" value="x">
    </form>"""
    form = gh.parse_form(html)
    got = {f.name: (f.kind, f.label, f.required, f.options) for f in form.fields}
    assert got["first_name"] == ("text", "First Name", True, [])
    assert got["sponsor"] == ("select", "Do you require sponsorship?", True, ["Yes", "No"])
    assert got["job_application_answers_attributes_0_text_value"][0] == "textarea"
    assert "csrf" not in got


# ── planning ────────────────────────────────────────────────────────────────

class Spy:
    def __init__(self, guesses=None, draft="Drafted answer."):
        self.guess_calls, self.draft_calls = [], []
        self.guesses, self.draft = guesses or {}, draft

    def guess(self, questions, profile):
        self.guess_calls.append(questions)
        return {q["name"]: self.guesses[q["label"]] for q in questions if q["label"] in self.guesses}

    def draft_fn(self, label, job_text, profile, max_chars=0, short=False, allow_unknown=False):
        self.draft_calls.append(label)
        return self.draft


def _plan(name, spy, resume=None, job=None):
    form = gh.parse_form(page(name))
    plans = gh.plan(form, job or {"company": form.company, "role": form.title}, "profile",
                    spy.guess, spy.draft_fn, resume=resume)
    return {p.field.label: p for p in plans}


def test_plan_bank_answers_and_null_legal_answers_are_needs_me():
    spy = Spy()
    plans = _plan("conviva_embed", spy)
    assert plans["First Name"].value == "Alex Jordan" and plans["First Name"].source == "bank"
    assert plans["Gender"].value == "Female"
    sponsor = plans["Will you now or in the future require sponsorship for employment visa status?"]
    assert (sponsor.source, sponsor.value) == ("needs_me", "")
    asked = [q["label"] for call in spy.guess_calls for q in call]
    assert not any("sponsorship" in a.lower() for a in asked)        # never sent to the LLM


def test_plan_agreements_are_never_filled():
    spy = Spy()
    plans = _plan("anthropic_jobboards", spy)
    for label in ("AI Policy for Application", "Please read the arbitration agreement below", "Agreement to Arbitrate"):
        assert (plans[label].source, plans[label].value) == ("consent", ""), label
    asked = [q["label"] for call in spy.guess_calls for q in call]
    assert not any("arbitrat" in a.lower() or "policy" in a.lower() for a in asked)


def test_plan_unknown_required_short_fields_go_to_one_small_batch():
    spy = Spy(guesses={"What is your highest completed degree?": "Master's"})
    plans = _plan("conviva_embed", spy)
    assert len(spy.guess_calls) == 1
    assert plans["What is your highest completed degree?"].source == "bank"   # the bank's degree, not a guess
    hands_on = next(p for label, p in plans.items() if label.startswith("Do you have hands-on experience"))
    assert hands_on.source == "needs_me"                              # model said nothing → yours


def test_plan_required_essay_is_drafted_and_optional_extra_info_skipped():
    spy = Spy()
    plans = _plan("anthropic_jobboards", spy)
    assert plans["Why Anthropic?"].source == "llm_draft"
    assert plans["Additional Information"].source == "skipped"
    assert spy.draft_calls == ["Why Anthropic?"]


def test_plan_optional_unknowns_are_skipped_without_llm():
    spy = Spy()
    plans = _plan("discord_jobboards", spy)
    assert plans["How did you hear about this job?"].source == "skipped"
    asked = [q["label"] for call in spy.guess_calls for q in call]
    assert "How did you hear about this job?" not in asked


def test_plan_open_question_does_not_take_a_yes_no_answer():
    spy = Spy()
    plans = _plan("anthropic_jobboards", spy)
    addr = next(p for label, p in plans.items() if label.startswith("What is the address"))
    assert addr.value == ""


def test_plan_resume_from_path_or_needs_me(tmp_path):
    cv = tmp_path / "Alex_Resume.pdf"
    cv.write_bytes(b"%PDF-1.4")
    assert _plan("conviva_embed", Spy(), resume=str(cv))["Resume/CV"].value == str(cv)
    no_cv = _plan("conviva_embed", Spy())["Resume/CV"]
    assert (no_cv.source, no_cv.note) == ("needs_me", "no resume_path set")


def test_plan_demographic_options_map_to_bank_values():
    plans = _plan("discord_jobboards", Spy())
    assert plans["Disability Status"].value.startswith("No, I do not have a disability")
    assert plans["Race and Ethnicity"].value == "White"
    assert plans["Gender Identity (optional)"].source == "skipped"     # "Female" isn't an option there


def test_resume_path_requires_an_existing_file(tmp_path, bank):
    data = json.loads(bank.read_text())
    data["answers"]["resume_path"] = str(tmp_path / "missing.pdf")
    bank.write_text(json.dumps(data))
    assert gh.resume_path() is None
    (tmp_path / "missing.pdf").write_bytes(b"%PDF")
    assert gh.resume_path() == str(tmp_path / "missing.pdf")


def test_profile_includes_sheet_notes_but_not_eeo():
    profile = gh.build_profile({"fields": {"Why it's a fit + draft": "Built a LangGraph voice agent."}})
    assert "LangGraph voice agent" in profile and "first_name: Alex Jordan" in profile
    assert "White" not in profile and "sponsorship" not in profile


# ── review ──────────────────────────────────────────────────────────────────

def test_review_reads_actual_dom_values():
    f1 = gh.FormField("first_name", "First Name", "text", True)
    f2 = gh.FormField("q1", "Degree", "select", True, ["Master's"])
    f3 = gh.FormField("race", "Race", "select", False, ["White"], "eeoc")
    f4 = gh.FormField("resume", "Resume/CV", "file", True)
    plans = [gh.Planned(f1, "Alex Jordan", "bank"), gh.Planned(f2, "Master's", "llm_guess"),
             gh.Planned(f3, "White", "bank"), gh.Planned(f4, "/x/Alex_Resume.pdf", "bank")]
    rows = {r["name"]: r for r in gh.build_review(plans, {"first_name": "Alex T", "q1": "", "race": None,
                                                           "resume": "Alex_Resume.pdf"})}
    assert rows["first_name"]["value"] == "Alex T"                   # what the DOM holds, not the plan
    assert (rows["q1"]["source"], rows["q1"]["note"]) == ("failed", "didn't stick after 2 retries")
    assert (rows["race"]["source"], rows["race"]["note"]) == ("skipped", "not on the form")
    assert rows["resume"]["value"] == "Alex_Resume.pdf" and rows["resume"]["source"] == "bank"


# ── LLM guess validation ────────────────────────────────────────────────────

QS = [{"name": "a", "label": "Degree?", "options": ["PhD", "Master's"]},
      {"name": "b", "label": "Current title?", "options": []}]


def test_validator_accepts_options_and_drops_unknown():
    v = llm_answers._validator(QS)
    assert v('{"a": "Master\'s", "b": "UNKNOWN"}') == {"a": "Master's"}
    assert v('```json\n{"a": "PhD", "b": "Engineer"}\n```') == {"a": "PhD", "b": "Engineer"}


@pytest.mark.parametrize("text", ['{"a": "Bachelor"}', "not json", "[]", '{"b": "' + "x" * 300 + '"}'])
def test_validator_rejects_bad_output(text):
    assert llm_answers._validator(QS)(text) is None


def test_guess_short_uses_checked_small_task(monkeypatch):
    seen = {}

    def fake_checked(task, messages, validate, system=None, max_tokens=0):
        seen["task"] = task
        return None, validate('{"a": "PhD"}')
    monkeypatch.setattr(llm_answers.llm_client, "complete_checked", fake_checked)
    assert llm_answers.guess_short(QS, "profile") == {"a": "PhD"}
    assert seen["task"] == "form_short_answer"


# ── dispatch + never submit ─────────────────────────────────────────────────

def test_fill_application_requires_url():
    assert ats.fill_application({"company": "X"}).error == "That job has no application URL."


def test_fill_application_routes_greenhouse_to_adapter(monkeypatch):
    called = {}
    monkeypatch.setattr(gh, "fill", lambda job, progress_fn=None, tab=None: called.setdefault("job", job) and gh.GreenhouseResult(
        form_url="https://job-boards.greenhouse.io/x", review=[{"source": "needs_me"}]))
    result = ats.fill_application({"url": "https://job-boards.greenhouse.io/x/jobs/1"})
    assert result.ats == "greenhouse" and result.count("needs_me") == 1 and called


def test_fill_application_falls_back_to_generic_for_other_atses(monkeypatch):
    monkeypatch.setattr(ats, "_generic", lambda job, progress_fn, tab=None: ats.FillResult(ats="generic"))
    said = []
    result = ats.fill_application({"url": "https://nike.wd1.myworkdayjobs.com/job/1"}, said.append)
    assert (result.ats, result.identified) == ("generic", "workday")
    assert "Workday" in said[0]


def test_greenhouse_filler_only_clicks_field_inputs_and_their_options():
    code = re.sub(r'"""[\s\S]*?"""', "", Path(gh.__file__).read_text())
    clicked = set(re.findall(r"([\w.\[\]()]+?)\.click\(", code))
    assert clicked == {"loc.first", "opts.nth(texts.index(target))"}, clicked
    assert "get_by_role" not in code and "press(\"Enter\")" not in code
    assert not re.search(r"click\([^)]*submit|submit[^\n]*\.click", code, re.I)


def test_generic_fallback_goal_forbids_submit():
    assert "Do NOT click Submit" in ats.NEVER_SUBMIT


# ── live-run fixes ──────────────────────────────────────────────────────────

# Real Greenhouse autocomplete options for "Buffalo" (2026-10-19).
BUFFALO = [
    "Buffalo City, Eastern Cape, South Africa", "Buffalo, New York, United States",
    "City of Buffalo, New York, United States", "Buffalo Grove, Illinois, United States",
    "Buffalo, Minnesota, United States", "University at Buffalo, New York, United States",
]


@pytest.mark.parametrize("value,expected", [
    ("Buffalo, NY", "Buffalo, New York, United States"),
    ("Buffalo, New York", "Buffalo, New York, United States"),
    ("Buffalo, MN", "Buffalo, Minnesota, United States"),
    ("Buffalo, CA", None),                       # no such option → leave it, don't pick South Africa
    ("Springfield, IL", None),
])
def test_match_location(value, expected):
    assert gh.match_location(value, BUFFALO) == expected


def test_schema_comes_from_the_http_copy_when_the_live_dom_lost_it():
    live_dom = re.sub(r"window\.__remixContext\s*=.*?</script>", "</script>", page("discord_jobboards"), flags=re.S)
    dom_only = gh.parse_form(live_dom)
    assert dom_only.title == ""                                       # what the live run saw
    both = gh.parse_form(live_dom, page("discord_jobboards"))
    assert (both.title, both.company) == ("Commercial Policy Lead", "Discord")
    assert next(f for f in both.fields if f.name == "resume").label == "Resume/CV"
    assert any(f.name == "race" for f in both.fields)


def test_review_shows_country_not_just_dial_code():
    f = gh.FormField("country", "Country", "select", True)
    row = gh.build_review([gh.Planned(f, "United States", "bank")], {"country": "+1"})[0]
    assert row["value"] == "United States (+1)"


def test_review_country_option_already_has_dial_code():
    f = gh.FormField("country", "Country", "select", True)
    assert gh.build_review([gh.Planned(f, "United States +1", "bank")], {"country": "+1"})[0]["value"] == "United States +1"


def test_plan_without_gateway_marks_open_text_failed_never_needs_me():
    spy = Spy(guesses={"What is your highest completed degree?": "Master's"})
    form = gh.parse_form(page("anthropic_jobboards"))
    plans = {p.field.label: p for p in gh.plan(form, {}, "profile", spy.guess, spy.draft_fn, llm_available=False)}
    assert (plans["Why Anthropic?"].source, plans["Why Anthropic?"].note) == ("failed", "LLM gateway was down")
    assert spy.guess_calls == [] and spy.draft_calls == []
    assert plans["First Name"].source == "bank"


def test_upload_confirmed_by_filename_chip_when_input_is_gone():
    f = gh.FormField("resume", "Resume/CV", "file", True)
    plans = [gh.Planned(f, "/Users/me/Downloads/Alex_Resume.pdf", "bank")]
    values = gh.confirm_uploads({"resume": None}, plans, "Resume/CV\nAlex_Resume.pdf ×\nCover Letter")
    row = gh.build_review(plans, values)[0]
    assert (row["value"], row["source"]) == ("Alex_Resume.pdf", "bank")
    missing = gh.confirm_uploads({"resume": ""}, plans, "Resume/CV Attach Dropbox")
    assert gh.build_review(plans, missing)[0]["source"] == "failed"


def test_fixtures_carry_no_keys_or_tokens():
    pattern = re.compile(r'AIza[0-9A-Za-z_-]{20,}|GOCSPX|sk-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}|ya29\.|'
                         r'"[A-Z0-9_]*(?:KEY|TOKEN|SECRET|DSN|PASSWORD)[A-Z0-9_]*":"(?!REDACTED")[^"]+"')
    for path in FIX.glob("*.html"):
        hit = pattern.search(path.read_text())
        assert hit is None, f"{path.name}: {hit.group(0)[:40]}"


# ── generic filler: posting page → the form behind its Apply link ───────────

JOBVITE_POSTING = """<div class="jv-job-detail-top-actions">
<a class="jv-button jv-button-primary jv-button-apply" href="/ookla/job/oAaCAfwG/apply">
  Apply
</a>
<a class="jv-button jv-button-apply-later" ng-href="mailto:?subject=Apply for Data Engineer at Ookla">Apply Later</a>
</div>"""


def test_apply_form_link_follows_the_postings_apply_link():
    url = "https://jobs.jobvite.com/ookla/job/oAaCAfwG"
    assert ats.apply_form_link(JOBVITE_POSTING, url) == "https://jobs.jobvite.com/ookla/job/oAaCAfwG/apply"


@pytest.mark.parametrize("html", [
    '<a href="mailto:jobs@x.com">Apply</a>',
    '<a href="javascript:void(0)">Apply now</a>',
    '<a href="/careers">Apply later</a>',
    '<a href="/jobs">See all jobs</a>',
    '<button type="submit">Apply</button>',
])
def test_apply_form_link_ignores_non_links_and_other_buttons(html):
    assert ats.apply_form_link(html, "https://x.example/job/1") is None


def test_apply_form_link_accepts_common_wordings():
    html = '<a href="https://x.example/job/1/form">Apply for this job</a><a href="/other">Apply now</a>'
    assert ats.apply_form_link(html, "https://x.example/job/1") == "https://x.example/job/1/form"


JOBVITE_COUNTRY = ["Select your location of residence and language", "US", "Canadian-French",
                   "European Economic Area", "Other Areas"]


@pytest.mark.parametrize("options,country,want", [
    (JOBVITE_COUNTRY, "United States", "US"),
    (["Choose…", "Canada", "United States of America", "Mexico"], "United States", "United States of America"),
    (["Select", "USA", "UK"], "united states", "USA"),
    (["Select", "Canada", "Mexico"], "United States", None),
    (JOBVITE_COUNTRY, "", None),
])
def test_match_country_option(options, country, want):
    assert ats.match_country(options, country) == want


def test_country_label_pattern():
    assert ats.is_country_label("Location of Residence and Language:")
    assert ats.is_country_label("Country")
    assert not ats.is_country_label("Are you legally authorized to work in the United States?")


# ── education block: matched by field name; school/discipline typed to search ──

GH_DEGREES = ["Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctor of Philosophy (Ph.D.)", "Other"]


def _edu_form():
    F = gh.FormField
    return gh.ParsedForm(fields=[
        F("school--0", "School", "select", True, [f"School {i}" for i in range(100)]),   # truncated list
        F("degree--0", "Degree", "select", True, GH_DEGREES),
        F("discipline--0", "Discipline", "select", False, []),
        F("start-month--0", "Start date month", "select", False, ["January", "May", "August"]),
        F("start-year--0", "Start date year", "text", True),
        F("end-month--0", "End date month", "select", False, ["January", "May", "August"]),
        F("end-year--0", "End date year", "text", False),
        F("question_1", "How did you hear about this opportunity?", "select", True, ["Instagram", "LinkedIn", "Other"]),
    ], title="SWE", company="Acme")


def _with_education(bank):
    data = json.loads(bank.read_text())
    data["answers"].update({"school": "University at Buffalo, SUNY", "degree": "Master of Science, Computer Science",
                            "grad_date": "05/18/2026", "edu_start_month": None, "edu_start_year": None})
    bank.write_text(json.dumps(data))


def test_plan_fills_the_education_block_by_field_name(bank):
    _with_education(bank)
    spy = Spy()
    plans = {p.field.name: p for p in gh.plan(_edu_form(), {"company": "Acme", "role": "SWE", "source": "Instagram"},
                                             "profile", spy.guess, spy.draft_fn)}
    assert (plans["school--0"].source, plans["school--0"].value) == ("bank", "University at Buffalo, SUNY")
    assert (plans["degree--0"].source, plans["degree--0"].value) == ("bank", "Master's Degree")
    assert plans["discipline--0"].value == "Computer Science"
    assert (plans["end-month--0"].value, plans["end-year--0"].value) == ("May", "2026")
    assert plans["start-year--0"].source == "needs_me" and "education start" in plans["start-year--0"].note
    assert plans["start-month--0"].source == "skipped"                  # optional, no answer
    assert plans["question_1"].value == "Instagram"                      # the sheet's source
    assert spy.guess_calls == []                                         # nothing went to the LLM


def test_plan_note_says_when_the_bank_value_fits_no_option(bank):
    data = json.loads(bank.read_text())
    data["answers"]["degree"] = "Certificate in Welding"
    bank.write_text(json.dumps(data))
    spy = Spy()
    plans = {p.field.name: p for p in gh.plan(_edu_form(), {"company": "Acme", "role": "SWE"}, "profile",
                                             spy.guess, spy.draft_fn)}
    assert plans["degree--0"].source == "needs_me"
    assert "Certificate in Welding" in plans["degree--0"].note and "options" in plans["degree--0"].note


def test_fill_types_to_search_for_school(monkeypatch):
    seen = {}

    def pick(page, name, value, type_text="", search=False):
        seen.update(name=name, value=value, type_text=type_text, search=search)
        return True

    class Loc:
        def count(self):
            return 1

        @property
        def first(self):
            return self

        def evaluate(self, js):
            return "INPUT"

    class Page:
        def locator(self, sel):
            return Loc()
    monkeypatch.setattr(gh, "_pick", pick)
    p = gh.Planned(gh.FormField("school--0", "School", "select", True), "University at Buffalo, SUNY", "bank")
    assert gh._fill_one(Page(), p)
    assert seen == {"name": "school--0", "value": "University at Buffalo, SUNY",
                    "type_text": "University at Buffalo", "search": True}
