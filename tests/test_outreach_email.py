"""tests/test_outreach_email.py — the cold email (the user's spec v2): Muse's research in,
the LLM writes the subject + three lines, code adds the frame and refuses what breaks the spec."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from aria.features.outreach import writer
from aria.features.outreach.writer import Brief

PROBLEM = "911 dispatch centers drown in non-emergency calls, so call-takers miss real emergencies."
ANGLE = ("Aria, a voice agent that turns speech into tool calls; the hard part was knowing when "
         "the caller had finished talking.")
SIGNAL = "MACC 911 moved its non-emergency line to AVA."
LINES = ("Dispatchers at places like MACC 911 lose minutes on non-emergency calls, time a real emergency needs.\n\n"
         "I built Aria, a voice agent that turns speech into tool calls. The hard part was telling "
         "when a caller had actually finished.\n\n"
         "Voice that has to be right the first time is the problem I keep coming back to.")
OUT = "SUBJECT: triaging non-emergency 911 calls\n\n" + LINES


def brief(**kw) -> Brief:
    base = dict(company="Aurelian (YC)", role="Software Engineer (New Grad)", problem=PROBLEM, angle=ANGLE,
                signal=SIGNAL, person="Max Keenan", title="CEO / co-founder")
    base.update(kw)
    return Brief(**base)


@pytest.fixture(autouse=True)
def profile(monkeypatch):
    bank = {"answers": {"first_name": "Bhanu Teja", "preferred_name": "Bhanu Teja", "last_name": "V",
                        "linkedin": "https://linkedin.com/in/bhanuteja1299", "github": "https://github.com/Bhanu1299",
                        "sponsorship": "Yes", "phone": "555"}}
    monkeypatch.setattr(writer.answer_bank, "load", lambda: bank)
    monkeypatch.setattr(writer, "_resume", lambda: "Aria: voice agent. LLM Gateway: model router.")


def _llm(monkeypatch, text):
    calls = []

    def complete_checked(task, messages, validate, system=None, max_tokens=1024):
        calls.append({"task": task, "content": messages[0]["content"], "system": system})
        return SimpleNamespace(text=text, stop_reason="end_turn"), validate(text)
    monkeypatch.setattr(writer.llm_client, "complete_checked", complete_checked)
    return calls


def test_the_email_is_the_spec_frame_around_the_llm_lines(monkeypatch):
    calls = _llm(monkeypatch, OUT)
    msg = writer.email(brief(), "max@aurelian.io")
    assert msg["to"] == "max@aurelian.io" and msg["subject"] == "triaging non-emergency 911 calls"
    assert msg["body"] == (
        "Hi Max,\n\n" + LINES + "\n\n"
        "I applied for the Software Engineer (New Grad) role. Resume attached. Would you be open to 15 minutes, "
        "or could you point me to the right person on the team?\n\n"
        "Thanks,\nBhanu\nlinkedin.com/in/bhanuteja1299 | github.com/Bhanu1299")
    c = calls[0]
    assert c["task"] == "outreach_email"
    assert f"Company problem: {PROBLEM}" in c["content"] and f"Signal: {SIGNAL}" in c["content"]
    assert ANGLE in c["content"] and "Company: Aurelian\n" in c["content"]
    assert "sponsorship" not in c["content"].lower() and "555" not in c["content"]


def test_generic_inbox_is_the_company_team_and_asks_to_pass_it_on(monkeypatch):
    _llm(monkeypatch, OUT)
    msg = writer.email(brief(person=""), "team@aurelian.io")
    assert msg["body"].startswith("Hi Aurelian team,\n\n")
    assert ("I applied for the Software Engineer (New Grad) role. Resume attached. Could you pass this to whoever "
            "is hiring for the Software Engineer (New Grad) role?\n\nThanks,") in msg["body"]


def test_a_recruiter_gets_the_team_problem_variant(monkeypatch):
    calls = _llm(monkeypatch, OUT)
    writer.email(brief(person="Jo Park", title="Technical Recruiter"), "jo@aurelian.io")
    assert "Jo Park, Technical Recruiter (recruiter / manager)" in calls[0]["content"]
    assert "team's problem, not the company mission" in calls[0]["system"]


@pytest.mark.parametrize("person,title,given,kind", [
    ("", "", "", writer.GENERIC), ("—", "", "", writer.GENERIC), ("Max Keenan", "CEO / co-founder", "", writer.ENGINEER),
    ("Jo Park", "Talent Partner", "", writer.HIRING), ("Ana Diaz", "Engineering Manager", "", writer.HIRING),
    ("Max Keenan", "CEO", "big company recruiter", writer.HIRING), ("Max", "", "generic inbox", writer.GENERIC),
])
def test_person_type(person, title, given, kind):
    assert writer.person_type(person, title, given) == kind


@pytest.mark.parametrize("missing", [{"problem": ""}, {"angle": "unknown"}, {"problem": "Unknown"}, {"angle": " "}])
def test_no_problem_or_angle_means_no_email_and_no_llm_call(monkeypatch, missing):
    calls = _llm(monkeypatch, OUT)
    assert writer.missing(brief(**missing)) and writer.email(brief(**missing), "max@aurelian.io") is None
    assert calls == []


@pytest.mark.parametrize("bad", [
    OUT.replace("SUBJECT: triaging non-emergency 911 calls", "SUBJECT: my application for the SWE role"),
    OUT.replace("SUBJECT: triaging non-emergency 911 calls", "SUBJECT: " + "x" * 45),
    OUT.replace("SUBJECT: triaging non-emergency 911 calls\n\n", ""),                         # no subject
    OUT.replace("\n\nVoice that has to be right the first time is the problem I keep coming back to.", ""),  # 2 lines
    OUT.replace("Dispatchers at", "Hi Max,\n\nDispatchers at"),                               # its own greeting
    OUT + "\n\nThanks,",                                                                       # its own sign-off
    OUT.replace("I keep coming back to", "I'm excited about"),                                # banned
    OUT.replace("I keep coming back to", "I'm passionate about"),
    OUT.replace("I built Aria", "I'm reaching out because I built Aria"),
    OUT.replace("had actually finished", "had finished, cutting latency 40%"),               # a metric
    OUT.replace("I built Aria, a voice agent", "I built Aria in 3 weeks, a voice agent"),     # a number
    OUT.replace("Dispatchers at places like MACC 911 lose minutes on non-emergency calls, time a real emergency needs.",
                "So 911 dispatch centers drown in non-emergency calls, so call-takers miss real emergencies."),  # copy
    OUT.replace("I built Aria, a voice agent that turns speech into tool calls. The hard part was telling "
                "when a caller had actually finished.",
                "I have built many systems in my career and enjoy hard problems of all kinds."),  # no Angle
    OUT.replace("Voice that has to be right", "I applied for the role. Voice that has to be right"),  # the ask is code's
    OUT.replace("Voice that has to be right the first time is the problem I keep coming back to.",
                " ".join(["Voice that has to be right the first time keeps pulling me back."] * 4)),  # over 100 words
])
def test_output_that_breaks_the_spec_is_dropped(monkeypatch, bad):
    _llm(monkeypatch, bad)
    assert writer.email(brief(), "max@aurelian.io") is None


def test_a_number_in_their_own_problem_or_name_is_allowed(monkeypatch):
    _llm(monkeypatch, OUT)                          # "911" is in the problem and signal: a name, not a metric
    assert writer.email(brief(), "max@aurelian.io") is not None


def test_signature_is_thanks_first_name_and_links():
    bank = {"answers": {"first_name": "Alex", "last_name": "Doe", "linkedin": "https://www.linkedin.com/in/me/",
                        "github": "https://github.com/alexdoe"}}
    assert writer.signature(bank) == "Thanks,\nAlex\nlinkedin.com/in/me | github.com/alexdoe"


def test_writer_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("gateway down")
    monkeypatch.setattr(writer.llm_client, "complete_checked", boom)
    assert writer.email(brief(), "max@aurelian.io") is None


@pytest.mark.parametrize("role,phrase", [("Software Engineer (New Grad)", "Software Engineer (New Grad) role"),
                                         ("MTS - Generalist role", "MTS - Generalist role"), ("", "open role")])
def test_role_phrase(role, phrase):
    assert writer.role_phrase(role) == phrase
