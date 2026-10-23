"""tests/test_understanding.py — any wording reaches an application action; risky ones wait for the
user's own yes; transcript lead-ins and a glued wake word don't break the exact phrases."""
from __future__ import annotations

import time

import pytest

from aria.dispatch.parse import normalize
from aria.features import applicator
from aria.features.applying import commands, tabs
from aria.integrations import sheets


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(tabs, "REGISTRY_PATH", tmp_path / "apply_tabs.json")
    commands._pending.clear()
    commands._asking.clear()
    calls = []
    monkeypatch.setattr(commands, "_submit", lambda e: calls.append(("submit", e["company"])) or f"Submitted {e['company']}.")
    monkeypatch.setattr(commands, "_close", lambda e: calls.append(("close", e["company"])) or f"Closed {e['company']}.")
    monkeypatch.setattr(applicator, "mark_submitted_by_hand",
                        lambda e: calls.append(("by_hand", e["company"])) or f"Got it: {e['company']} is submitted.")
    monkeypatch.setattr(applicator, "skip_entry", lambda e: calls.append(("skip", e["company"])) or f"Skipped {e['company']}.")
    monkeypatch.setattr(commands, "_write_mark",
                        lambda job, entry, st: calls.append(("mark", job.get("company"), st)) or f"Marked {job.get('company')} {st}.")
    monkeypatch.setattr(commands.agent_browser, "front_tabs", lambda ids: [])
    monkeypatch.setattr(commands, "_last_job", lambda: None)
    return calls


def _open(company="RTX", role="Software Engineer I (Onsite)", job_id="J0018", state=tabs.FILLED):
    e = tabs.add({"row": 18, "job_id": job_id, "company": company, "role": role, "url": "https://x.example/1"})
    return tabs.update(e["id"], state=state)


# ── transcript clean-up ─────────────────────────────────────────────────────

@pytest.mark.parametrize("said,clean", [
    ("Yeah, I submitted it.", "i submitted it"),
    ("AriaHeadSubmit", "head submit"),
    ("So, open my dashboard.", "open my dashboard"),
    ("Aria, yeah can you open the dashboard please", "open the dashboard"),
    ("yes", "yes"),                      # a bare yes still reaches the yes/no handlers
    ("Arial font size", "arial font size"),
])
def test_normalize_drops_lead_ins(said, clean):
    assert normalize(said) == clean


def test_yeah_i_submitted_it_is_the_exact_phrase_now(isolated):
    _open()
    assert commands.handle("Yeah, I submitted it.") == "Got it: RTX is submitted."
    assert isolated == [("by_hand", "RTX")]


# ── request(): any wording, via the agent ───────────────────────────────────

def test_submit_is_only_staged_until_the_users_yes(isolated):
    _open()
    out = commands.request("submit", "the RTX one")
    assert "Submit the Software Engineer I (Onsite) at RTX (J0018) application?" in out
    assert isolated == []
    assert commands.handle("yeah go ahead") == "Submitted RTX."
    assert isolated == [("submit", "RTX")]


@pytest.mark.parametrize("reply", ["no wait", "don't", "cancel that", "nope"])
def test_a_no_cancels(isolated, reply):
    _open()
    commands.request("submit")
    assert commands.handle(reply) == "OK, I won't."
    assert isolated == []
    assert commands.handle("yes") is None                    # nothing staged any more


def test_other_words_drop_the_staged_action_for_good(isolated):
    _open()
    commands.request("close")
    assert commands.handle("what's the weather") is None
    assert commands.handle("yes") is None
    assert isolated == []


def test_yes_with_a_negation_does_not_run(isolated):
    _open()
    commands.request("submit")
    commands.handle("yes but don't submit yet")
    assert isolated == []


def test_staged_action_expires(isolated, monkeypatch):
    _open()
    commands.request("submit")
    commands._pending["at"] = time.time() - commands.PENDING_SECS - 1
    assert commands.handle("yes") is None
    assert isolated == []


def test_i_submitted_and_mark_applied_are_confirmed_too(isolated):
    _open()
    assert "record it as submitted?" in commands.request("i_submitted")
    commands.handle("yes")
    assert "as Applied in the sheet?" in commands.request("mark", "", "applied")
    commands.handle("sure")
    assert isolated == [("by_hand", "RTX"), ("mark", "RTX", "Applied")]


def test_skip_and_reversible_marks_act_at_once(isolated):
    _open()
    assert commands.request("skip", "rtx") == "Skipped RTX."
    assert commands.request("mark", "", "not applied") == "Marked RTX Not applied."
    assert commands._pending == {}


def test_named_tab_among_several(isolated):
    _open()
    _open("John Deere", "IT Development Program", "J0004")
    assert "John Deere" in commands.request("submit", "the deere one")
    commands.handle("yes")
    assert isolated == [("submit", "John Deere")]


def test_several_tabs_and_no_name_asks_which(isolated):
    _open()
    _open("John Deere", "IT Development Program", "J0004")
    out = commands.request("submit")
    assert out.startswith("Which one?") and commands._pending == {}


def test_nothing_open_to_submit(isolated, monkeypatch):
    monkeypatch.setattr(sheets, "resolve_job", lambda q, t=None: {
        "kind": "one", "sure": True, "jobs": [{"row": 4, "job_id": "J0004", "company": "John Deere", "role": "IT"}]})
    out = commands.request("submit", "john deere")
    assert "no open application tab" in out and commands._pending == {}


def test_still_filling_is_never_staged(isolated):
    _open(state=tabs.FILLING)
    assert "still filling" in commands.request("submit")
    assert commands._pending == {}


def test_tab_closed_before_the_yes(isolated):
    e = _open()
    commands.request("submit")
    tabs.mark_closed(e["id"])
    assert "isn't open anymore" in commands.handle("yes")
    assert isolated == []


def test_unknown_action():
    assert commands.request("delete").startswith("action must be one of")
