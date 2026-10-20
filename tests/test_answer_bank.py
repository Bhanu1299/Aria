"""tests/test_answer_bank.py — lookup order, never-invent, options, learning, salary rule."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import aria.features.answer_bank as ab
from aria.core import paths


@pytest.fixture
def bank_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "identity.json")  # no identity unless a test writes one
    path = tmp_path / "answer_bank.json"
    path.write_text(json.dumps({
        "answers": {
            "first_name": "Alex",
            "last_name": "Jordan",
            "email": "me@example.com",
            "phone": "<+1 555 555 5555>",            # placeholder → unset
            "sponsorship": "Yes",
            "work_authorization": "Yes",
            "relocation": "",                          # blank → unset
            "years_experience": "1",
            "eeo": {"gender": "Decline to self-identify", "veteran": "I am not a protected veteran"},
        },
        "aliases": {"start_date": ["when could you join us"]},
        "learned": {"why do you want to work here": "Because of the agent work."},
        "salary_rule": {"floor": 120000},
    }))
    return path


def look(label, bank_file, **kw):
    return ab.lookup(label, path=bank_file, **kw)


# ── lookup order ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("label,key,value", [
    ("First Name*", "first_name", "Alex"),
    ("Last name (required)", "last_name", "Jordan"),
    ("Email Address", "email", "me@example.com"),
    ("E-mail", "email", "me@example.com"),
])
def test_exact_alias(bank_file, label, key, value):
    a = look(label, bank_file)
    assert (a.key, a.value, a.source) == (key, value, "bank")


def test_exact_learned(bank_file):
    a = look("Why do you want to work here?", bank_file)
    assert (a.source, a.value) == ("learned", "Because of the agent work.")


@pytest.mark.parametrize("label,key", [
    ("Will you now or in the future require sponsorship for employment visa status (e.g., H-1B)?", "sponsorship"),
    ("Are you legally authorized to work in the United States?", "work_authorization"),
    ("How many years of professional experience do you have?", "years_experience"),
    ("Veteran Status", "eeo.veteran"),
    ("Gender", "eeo.gender"),
])
def test_patterns_map_real_phrasings_to_keys(bank_file, label, key):
    assert look(label, bank_file).key == key


def test_fuzzy_alias_at_threshold(bank_file):
    assert look("Frist Name", bank_file).value == "Alex"   # typo, ratio >= 90


def test_fuzzy_learned(bank_file):
    assert look("Why do you want to work here!!", bank_file).source == "learned"
    assert look("Why do you want to wrok here?", bank_file).source == "learned"


def test_file_aliases_extend_defaults(bank_file):
    # key exists but has no value → still None (never invented)
    assert look("When could you join us?", bank_file) is None


# ── never invent ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("label", [
    "Phone",                                   # placeholder value
    "Are you willing to relocate?",            # blank value
    "What is your favorite programming language?",  # unknown question
    "Disability Status",                       # eeo key not in file
    "Company name",                            # must not fuzzy-match "name"
    "",
])
def test_unknown_or_unset_returns_none(bank_file, label):
    assert look(label, bank_file) is None


def test_no_bank_file_and_no_identity_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "missing.json")
    assert ab.lookup("Will you require sponsorship?", path=tmp_path / "none.json") is None
    assert ab.lookup("First name", path=tmp_path / "none.json") is None


def test_identity_fills_contact_fields_only(tmp_path, monkeypatch):
    ident = tmp_path / "identity.json"
    ident.write_text(json.dumps({"name": "Alex Jordan V", "phone": "+1 555 555 0100", "location": "Springfield, IL"}))
    monkeypatch.setattr(paths, "IDENTITY_JSON", ident)
    empty = tmp_path / "bank.json"
    assert ab.lookup("Phone number", path=empty).value == "+1 555 555 0100"
    assert ab.lookup("Last Name", path=empty).value == "V"
    assert ab.lookup("Location (City)", path=empty).value == "Springfield, IL"
    assert ab.lookup("Do you require visa sponsorship?", path=empty) is None


def test_bank_value_beats_identity(bank_file, tmp_path):
    (tmp_path / "identity.json").write_text(json.dumps({"name": "Someone Else", "email": "other@x.com"}))
    assert look("Email", bank_file).value == "me@example.com"


# ── options ─────────────────────────────────────────────────────────────────

def test_yes_no_maps_onto_long_options(bank_file):
    opts = ["Select...", "Yes, I will require sponsorship", "No, I will not require sponsorship"]
    assert look("Do you require sponsorship?", bank_file, options=opts).value == "Yes, I will require sponsorship"


def test_value_not_among_options_returns_none(bank_file):
    assert look("Gender", bank_file, options=["Male", "Female", "Non-binary"]) is None


def test_exact_option_case_insensitive(bank_file):
    opts = ["I am a protected veteran", "I am not a protected veteran", "I don't wish to answer"]
    assert look("Veteran status", bank_file, options=opts).value == "I am not a protected veteran"


@pytest.mark.parametrize("value,options,expected", [
    ("Yes", ["Yes", "No"], "Yes"),
    ("no", ["Yes", "No"], "No"),
    ("Yes", ["Yes, absolutely", "Yes, partially", "No"], None),   # ambiguous yes
    ("Decline to self-identify", ["Male", "Female", "Decline To Self Identify"], "Decline To Self Identify"),
    ("Buffalo", ["Buffalo, NY", "Boston, MA"], "Buffalo, NY"),
    ("Remote", ["Yes", "No"], None),
])
def test_choose_option(value, options, expected):
    assert ab.choose_option(value, options) == expected


# ── learning ────────────────────────────────────────────────────────────────

def test_learn_then_lookup_and_persists(bank_file):
    assert ab.learn("Preferred pronouns?", "he/him", path=bank_file) is True
    assert look("Preferred pronouns", bank_file).value == "he/him"
    data = json.loads(bank_file.read_text())
    assert data["learned"]["preferred pronouns"] == "he/him"
    assert data["answers"]["first_name"] == "Alex"   # rest of the file untouched


def test_learn_creates_file(tmp_path):
    path = tmp_path / "sub" / "answer_bank.json"
    assert ab.learn("Favorite color", "blue", path=path)
    assert ab.lookup("favorite color", path=path).value == "blue"


def test_learn_ignores_empty(bank_file):
    assert ab.learn("Anything", "   ", path=bank_file) is False


def test_corrupt_file_degrades_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "none.json")
    path = tmp_path / "answer_bank.json"
    path.write_text("{not json")
    assert ab.lookup("First name", path=path) is None


# ── salary rule ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Base pay $128k–$256k plus equity", (128000, 256000)),
    ("$114.4k–$153k ($137.8k–$167.7k SF/NYC metro)", (114400, 153000)),
    ("The range is $122,400 - $176,000 per year.", (122400, 176000)),
    ("$100k to $140k", (100000, 140000)),
    ("$18–$28/hr", None),
    ("Competitive salary", None),
    ("", None),
])
def test_parse_salary_range(text, expected):
    assert ab.parse_salary_range(text) == expected


@pytest.mark.parametrize("rng,expected", [
    ((128000, 256000), "120000"),       # low end >= floor → floor
    ((120000, 150000), "120000"),       # boundary
    ((100000, 140000), "100000-140000"),  # below floor → posting's range
    (None, None),                       # no range → leave for review
])
def test_salary_answer(rng, expected):
    assert ab.salary_answer(rng, floor=120000) == expected


def test_salary_label_uses_job_description(bank_file):
    job = {"description": "Compensation: $130,000 - $160,000"}
    a = look("What are your salary expectations?", bank_file, job=job)
    assert (a.key, a.value) == ("salary_rule", "120000")


def test_salary_without_range_is_left_for_review(bank_file):
    assert look("Desired salary", bank_file, job={"description": "Great benefits"}) is None
    assert look("Desired salary", bank_file) is None


# ── LinkedIn no longer guesses ──────────────────────────────────────────────

def test_linkedin_has_no_hardcoded_answers():
    src = (Path(__file__).parent.parent / "aria" / "features" / "linkedin_applicator.py").read_text()
    assert "_match_field_label" not in src
    assert "answer_bank.lookup" in src


def test_linkedin_bank_answer_uses_lookup(monkeypatch):
    import aria.features.linkedin_applicator as li
    monkeypatch.setattr(li.answer_bank, "lookup", lambda label, job=None: None)
    assert li._bank_answer("Do you require sponsorship?", {}) is None
    monkeypatch.setattr(li.answer_bank, "lookup",
                        lambda label, job=None: ab.Answer("sponsorship", "Yes", "bank", "sponsor"))
    assert li._bank_answer("Do you require sponsorship?", {}) == "Yes"


# ── the user's own keys ─────────────────────────────────────────────────────

def test_answer_key_names_are_aliases_and_beat_cross_key_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "none.json")
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({
        "answers": {"city": "Springfield", "location": "Springfield, IL", "zip": "10001",
                    "school": "State University", "eeo": {"pronouns": "he/him"}},
        "aliases": {"location": ["city", "where are you based"]},
    }))
    assert ab.lookup("City", path=path).value == "Springfield"
    assert ab.lookup("Location (City)", path=path).value == "Springfield, IL"
    assert ab.lookup("Where are you based?", path=path).value == "Springfield, IL"
    assert ab.lookup("Zip / Postal Code", path=path).value == "10001"
    assert ab.lookup("University", path=path).value == "State University"
    assert ab.lookup("Pronouns", path=path).value == "he/him"


def test_null_answer_is_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "none.json")
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"answers": {"sponsorship": None, "salary_rule": "text rule"}}))
    assert ab.lookup("Will you now or in the future require sponsorship?", path=path) is None
    assert ab.lookup("Desired salary", path=path, job={"description": "$130k - $150k"}).value == "120000"


def test_name_falls_back_to_legal_name(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "none.json")
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"answers": {"legal_name": "Alex Jordan Rivera", "first_name": "Alex Jordan"}}))
    assert ab.lookup("Name", path=path).value == "Alex Jordan Rivera"
    assert ab.lookup("Preferred Name", path=path).value == "Alex Jordan"


def test_learned_answer_used_when_matched_key_is_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "none.json")
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"answers": {"sponsorship": None}}))
    q = "Will you now or in the future require sponsorship?"
    assert ab.lookup(q, path=path) is None
    ab.learn(q, "Yes", path=path)
    assert ab.lookup(q, path=path).value == "Yes"


def test_matched_key_ignores_values(bank_file):
    assert ab.matched_key("Are you willing to relocate?", path=bank_file) == "relocation"   # blank value
    assert ab.matched_key("Desired salary", path=bank_file) == "salary_rule"
    assert ab.matched_key("What's your favorite movie?", path=bank_file) is None
