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


# ── "yes, and …": the yes confirms, the rest is routed on its own ───────────

def test_yes_and_more_confirms_then_hands_the_rest_on(isolated):
    _open()
    commands.request("i_submitted")
    out = commands.handle("Yes, please, and start applying to next jobs as well")
    assert out == "Got it: RTX is submitted." and out.rest == "start applying to next jobs as well"
    assert isolated == [("by_hand", "RTX")]


def test_yes_go_ahead_and_submit_it_is_one_submit(isolated):
    _open()
    commands.request("submit")
    out = commands.handle("yes go ahead and submit it")
    assert out == "Submitted RTX." and out.rest == "" and isolated == [("submit", "RTX")]


def test_dispatcher_routes_the_rest_and_joins_the_answers():
    from aria.dispatch.dispatcher import Dispatcher, Partial

    class Agent:
        last_run_tools: list = []

        def run(self, text):
            return f"agent did: {text}"
    d = Dispatcher(registry=None, agent=Agent(), rules=None,
                   pre_handlers=[("p", lambda t: Partial("Done.", "start applying") if t.startswith("yes") else None)])
    r = d.handle("yes and start applying")
    assert r.answer == "Done. agent did: start applying" and r.route == "pre:p+agent"


# ── continue / start / Handshake step 2 ─────────────────────────────────────

def test_continue_refills_the_named_tab(isolated, monkeypatch):
    e = _open()
    seen = []
    monkeypatch.setattr(applicator, "run_application", lambda job, entry=None, **kw: seen.append(entry["id"]) or "filled")
    assert commands.continue_application("the RTX one") == "filled" and seen == [e["id"]]


def test_i_submitted_on_handshake_with_a_step_two_fills_step_two(isolated, monkeypatch):
    from aria.features.ats.adapters import handshake
    e = tabs.update(_open()["id"], ats="handshake")
    monkeypatch.setattr(handshake, "has_step_two", lambda tab_id: True)
    started = []
    monkeypatch.setattr(commands, "_handshake_step_two", lambda entry: started.append(entry["id"]))
    out = commands.handle("I submitted it")
    assert "step 2 on the employer's site" in out and started == [e["id"]]


def test_i_submitted_elsewhere_has_no_step_two(isolated, monkeypatch):
    from aria.features.ats.adapters import handshake
    _open()
    monkeypatch.setattr(handshake, "has_step_two", lambda tab_id: pytest.fail("checked a non-Handshake tab"))
    assert "step 2" not in commands.handle("I submitted it")
