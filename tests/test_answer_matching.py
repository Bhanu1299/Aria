"""tests/test_answer_matching.py — field matching: synonyms, education, option choosers,
match steps (exact → alias → synonym → small-LLM key pick), aliases learned from answers."""
from __future__ import annotations

import json
import logging
from datetime import date

import pytest

import aria.features.answer_bank as ab
import aria.state.db as db
from aria.core import paths

DEGREES = ["Associate's Degree", "Bachelor's Degree", "Doctor of Medicine (M.D.)", "Doctor of Philosophy (Ph.D.)",
           "Engineer's Degree", "High School", "Juris Doctor (J.D.)", "Master of Business Administration (M.B.A.)",
           "Master's Degree", "Other"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
ZIP_GRAD = ["I have already graduated", "December 2026 - November 2027", "I graduate after December 2027"]


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "identity.json")
    monkeypatch.setattr(ab, "_llm_cache", {})


@pytest.fixture
def bank_file(tmp_path):
    path = tmp_path / "answer_bank.json"
    path.write_text(json.dumps({
        "answers": {
            "first_name": "Alex", "school": "University at Buffalo, SUNY",
            "degree": "Master of Science, Computer Science", "grad_date": "05/18/2026",
            "edu_start_month": None, "edu_start_year": None, "gpa": None,
            "work_authorization": None, "sponsorship": None,
        },
        "aliases": {"first_name": ["what should we call you"]},
    }))
    return path


def look(label, bank_file, **kw):
    return ab.lookup(label, path=bank_file, **kw)


# ── synonyms: every wording from the four real forms ────────────────────────

@pytest.mark.parametrize("label,key", [
    ("What school are you currently attending / did you graduate from?", "school"),
    ("Institution", "school"),
    ("Name of college or university", "school"),
    ("Level of education", "degree"),
    ("Highest degree earned", "degree"),
    ("When is your anticipated graduation date?", "grad_date"),
    ("When do you graduate?", "grad_date"),
    ("Expected graduation", "grad_date"),
    ("Field of study", "major"),
    ("Discipline", "major"),
    ("GPA", "gpa"),
    ("How many years of professional experience do you have?", "years_experience"),
    ("How did you hear about this opportunity?", "referral_source"),
    ("How did you find out about this position?", "referral_source"),
    ("List the specific source where you learned about this position (ex. Job posting on Indeed, Google search)",
     "referral_source"),
])
def test_synonyms_map_to_bank_fields(bank_file, label, key):
    assert ab.matched_key(label, path=bank_file) == key


def test_yes_no_question_never_takes_a_value_field(bank_file):
    assert ab.matched_key("Do you have a bachelor's degree in computer science?", path=bank_file) != "degree"


# ── education values, derived from what the bank already has ────────────────

def test_school_and_degree_fill_from_live_options(bank_file):
    assert look("School", bank_file, options=["Aalborg University", "University at Buffalo - SUNY"]).value \
        == "University at Buffalo - SUNY"
    assert look("Degree", bank_file, options=DEGREES).value == "Master's Degree"


def test_degree_level_never_picks_mba_for_ms():
    assert ab.choose_degree("Master of Science, Computer Science", DEGREES) == "Master's Degree"
    assert ab.choose_degree("MBA", DEGREES) == "Master of Business Administration (M.B.A.)"
    assert ab.choose_degree("B.Tech in CSE", DEGREES) == "Bachelor's Degree"
    assert ab.choose_degree("Some certificate", DEGREES) is None


def test_major_comes_from_degree_when_unset(bank_file):
    assert look("Field of study", bank_file).value == "Computer Science"


def test_grad_month_and_year_come_from_grad_date(bank_file):
    assert ab.lookup_key("grad_month", options=MONTHS, path=bank_file).value == "May"
    assert ab.lookup_key("grad_year", path=bank_file).value == "2026"


def test_unset_education_start_is_none(bank_file):
    assert ab.lookup_key("edu_start_year", path=bank_file) is None


@pytest.mark.parametrize("options,today,want", [
    (ZIP_GRAD, date(2026, 10, 21), "I have already graduated"),
    (ZIP_GRAD, date(2026, 1, 10), None),                       # not graduated, no range covers May 2026
    (["May 2026 - August 2026", "September 2026 or later"], date(2026, 1, 1), "May 2026 - August 2026"),
])
def test_grad_date_picks_the_right_range_option(options, today, want):
    assert ab.choose_date_option("05/18/2026", options, today=today) == want


@pytest.mark.parametrize("label,want", [
    ("I confirm that my graduation date will be either Fall 2026 or Spring 2027", "No"),
    ("I confirm that my graduation date will be either Spring 2026 or Fall 2026", "Yes"),
])
def test_graduation_window_yes_no(bank_file, label, want):
    assert look(label, bank_file, options=["Yes", "No"]).value == want


# ── how did you hear: the sheet's source ────────────────────────────────────

def test_referral_source_comes_from_the_job(bank_file):
    job = {"company": "Ookla", "role": "DE", "source": "Instagram"}
    assert look("How did you find out about this position?", bank_file, job=job).value == "Instagram"
    opts = ["Facebook", "Friend/Referral", "Instagram", "LinkedIn", "Other"]
    assert look("How did you hear about this opportunity?", bank_file, job=job, options=opts).value == "Instagram"
    # nothing in the sheet or the bank: a recruiter (user, 2026-10-24)
    assert look("How did you hear about this opportunity?", bank_file, job={"company": "Z"}).value == "Recruiter"


def test_referral_default_picks_the_closest_option(bank_file):
    z = {"company": "Z"}
    assert look("How did you hear about us?", bank_file, job=z,
                options=["LinkedIn", "Recruiter contacted me", "Recruiter (agency)", "Other"]).value == "Recruiter contacted me"
    assert look("How did you hear about us?", bank_file, job=z,
                options=["Employee referral", "Agency Recruiter", "Other"]).value == "Agency Recruiter"
    assert look("How did you hear about us?", bank_file, job=z,
                options=["Indeed", "LinkedIn", "Other"]).value == "LinkedIn"
    assert look("How did you hear about us?", bank_file, job=z, options=["Friend", "Other"]).value == "Other"
    assert ab.lookup_key("referral_source", path=bank_file).value == "Recruiter"


def test_a_phone_extension_is_never_the_phone_number(bank_file):
    bank = ab.load(bank_file)
    for label in ("Phone Extension", "Ext.", "Country Phone Code", "Phone Device Type"):
        assert ab.match_field(bank, ab.normalize(label)).key != "phone", label
    assert ab.match_field(bank, ab.normalize("Phone Number")).key == "phone"


# ── match steps, logged ─────────────────────────────────────────────────────

def test_match_reports_the_step_that_matched(bank_file, caplog):
    bank = ab.load(bank_file)
    assert ab.match_field(bank, "school").how == "exact"
    assert ab.match_field(bank, "what should we call you").how == "alias"
    assert ab.match_field(bank, "what school are you currently attending").how == "synonym"
    with caplog.at_level(logging.INFO, logger="aria.features.answer_bank"):
        ab.lookup("Name of college or university", path=bank_file)
    assert any("via synonym" in r.message for r in caplog.records)


def test_llm_step_picks_a_bank_key_never_a_value(bank_file, monkeypatch):
    seen = {}

    def fake(task, messages, validate=None, **kw):
        seen["task"], seen["prompt"] = task, messages[0]["content"]
        return None, validate('{"key": "school"}')
    monkeypatch.setattr(ab, "_complete_checked", fake)
    got = look("Alma mater", bank_file, llm=True)
    assert got.value == "University at Buffalo, SUNY" and got.matched == "alma mater"
    assert seen["task"] == "form_field_match" and "University at Buffalo" not in seen["prompt"]   # keys only
    assert ab.match_field(ab.load(bank_file), "alma mater", llm=True).how == "llm"


def test_llm_step_rejects_unknown_keys_and_none(bank_file, monkeypatch):
    monkeypatch.setattr(ab, "_complete_checked", lambda task, messages, validate=None, **kw:
                        (None, validate('{"key": "favorite_color"}')))
    assert look("What archetype are you most closely aligned with?", bank_file, llm=True) is None
    monkeypatch.setattr(ab, "_complete_checked", lambda task, messages, validate=None, **kw:
                        (None, validate('{"key": "NONE"}')))
    assert ab.match_field(ab.load(bank_file), "favourite food", llm=True).key is None


def test_llm_step_off_by_default(bank_file, monkeypatch):
    monkeypatch.setattr(ab, "_complete_checked", lambda *a, **k: pytest.fail("LLM called"))
    assert look("Alma mater", bank_file) is None


# ── aliases learned from your answers ───────────────────────────────────────

def test_answering_a_new_wording_saves_it_as_an_alias(bank_file):
    ab.save_answer_for_question("Are you legally authorized to work in the United States for our Company?",
                                "Yes", key="work_authorization", path=bank_file)
    data = json.loads(bank_file.read_text())
    assert data["answers"]["work_authorization"] == "Yes"
    assert "are you legally authorized to work in the united states for our company" in data["aliases"]["work_authorization"]
    assert ab.match_field(ab.load(bank_file), "are you legally authorized to work in the united states for our company").how == "alias"


def test_answering_an_unknown_question_learns_it(bank_file):
    ab.save_answer_for_question("What archetype are you most closely aligned with?", "Full Stack", path=bank_file)
    assert look("What archetype are you most closely aligned with?", bank_file, options=["Full Stack", "Big Data"]).value \
        == "Full Stack"


def test_referral_source_falls_back_to_your_default(bank_file):
    ab.set_answer("referral_source", "LinkedIn", path=bank_file)
    assert look("How did you hear about us?", bank_file, job={"company": "Z"}).value == "LinkedIn"
    assert look("How did you hear about us?", bank_file, job={"company": "Z", "source": "Instagram"}).value == "Instagram"


def test_generic_filler_gets_derived_fields_and_the_source(bank_file):
    ctx = ab.fillable_answers(ab.load(bank_file), {"company": "Ookla", "role": "DE", "source": "Instagram"})
    assert ctx["major"] == "Computer Science" and ctx["grad_month"] == "May" and ctx["grad_year"] == "2026"
    assert ctx["how_did_you_hear_about_us"] == "Instagram"


# ── regressions from the live runs (2026-10-21) ─────────────────────────────

def test_options_by_year_unique_name_and_country_alias():
    assert ab.choose_option("05/18/2026", ["2024", "2025", "2026", "Other"]) == "2026"
    unis = ["State University of New York - Buffalo State College",
            "State University of New York - University at Buffalo", "University at Albany"]
    assert ab.choose_option("University at Buffalo, SUNY", unis) == unis[1]
    assert ab.choose_option("United States", ["US", "Canadian-French", "European Economic Area"]) == "US"
    assert ab.choose_option("Buffalo State", unis[1:]) is None


def test_pronounce_is_not_pronouns_and_high_school_is_not_your_university():
    bank = ab.load()
    assert ab.match_field(bank, ab.normalize("Name Pronunciation | How do you pronounce your name?")).key is None
    assert ab.match_field(bank, ab.normalize("Your pronouns")).key == "eeo.pronouns"
    assert ab.match_field(bank, ab.normalize("High School Name")).key is None
    assert ab.match_field(bank, ab.normalize("Year of High School Graduation")).key is None
    assert ab.match_field(bank, ab.normalize("Where did you attend university?")).key == "school"


def test_state_abbreviation_picks_the_full_name_and_back():
    # 2026-10-25: the bank says "NY", Meijer's Workday State list says "New York" → State failed
    states = ["Alabama", "New Jersey", "New York", "North Carolina", "Michigan", "Oregon"]
    assert ab.choose_option("NY", states) == "New York"
    assert ab.choose_option("mi", states) == "Michigan"
    assert ab.choose_option("New York", ["NJ", "NY", "NC"]) == "NY"
    assert ab.choose_option("OR", ["Yes", "No"]) is None       # never an unrelated option
