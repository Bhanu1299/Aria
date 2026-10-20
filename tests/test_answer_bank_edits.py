"""Answer-bank edits from the dashboard: backup, history/undo, always-ask, job-only answers."""
from __future__ import annotations

import json

import pytest

from aria.features import answer_bank as ab
from aria.features.ats import greenhouse as gh


@pytest.fixture
def bank(tmp_path, monkeypatch):
    path = tmp_path / "answer_bank.json"
    path.write_text(json.dumps({"answers": {"first_name": "Sam", "eeo": {"gender": "<decline>"}},
                                "aliases": {"first_name": ["what should we call you"]}}))
    monkeypatch.setattr(ab, "BANK_PATH", path)
    monkeypatch.setattr(ab, "_identity_answers", lambda: {})
    return path


def test_first_write_backs_up_once_and_keeps_other_sections(bank):
    original = bank.read_text()
    ab.set_answer("phone", "555-0100")
    backup = ab.backup_path()
    assert backup is not None and backup.read_text() == original
    ab.set_answer("city", "Springfield")
    assert len(list(bank.parent.glob("answer_bank.backup-*.json"))) == 1
    data = json.loads(bank.read_text())
    assert data["answers"]["phone"] == "555-0100" and data["aliases"]["first_name"] == ["what should we call you"]
    assert ab.lookup("Phone number").value == "555-0100"


def test_history_and_undo(bank):
    cid = ab.set_answer("first_name", "Alex")
    assert ab.lookup("First name").value == "Alex"
    h = ab.history()[0]
    assert (h["target"], h["key"], json.loads(h["old"]), json.loads(h["new"])) == ("answer", "first_name", "Sam", "Alex")
    ab.undo(cid)
    assert ab.lookup("First name").value == "Sam"
    assert ab.history()[1]["undone_by"] is not None
    with pytest.raises(ValueError, match="already undone"):
        ab.undo(cid)


def test_undo_removes_a_new_key_and_nested_keys_work(bank):
    cid = ab.set_answer("eeo.veteran", "No")
    assert json.loads(bank.read_text())["answers"]["eeo"]["veteran"] == "No"
    ab.undo(cid)
    assert "veteran" not in json.loads(bank.read_text())["answers"]["eeo"]


def test_learned_set_forget_and_undo(bank):
    ab.set_learned("How did you hear about us?", "LinkedIn")
    assert ab.lookup("How did you hear about us").value == "LinkedIn"
    cid = ab.set_learned("How did you hear about us?", "")
    assert ab.lookup("How did you hear about us") is None
    ab.undo(cid)
    assert ab.lookup("How did you hear about us").value == "LinkedIn"


def test_always_ask_blocks_bank_and_agent_context(bank):
    ab.set_answer("phone", "555-0100")
    ab.set_always_ask("phone", True)
    assert ab.always_ask("Phone number") and ab.lookup("Phone number") is None
    assert "phone" not in ab.fillable_answers(ab.load())
    ab.set_always_ask("Why do you want to work here?", True)
    assert ab.always_ask("why do you want to work here")
    ab.set_always_ask("phone", False)
    assert ab.lookup("Phone number").value == "555-0100"


def test_job_only_answers(bank):
    acme = {"company": "Acme", "role": "SWE"}
    ab.set_job_answer(ab.job_key(acme), "Why Acme?", "Robots")
    assert ab.lookup("Why Acme?", job=acme).value == "Robots"
    assert ab.lookup("Why Acme?", job={"company": "Other", "role": "SWE"}) is None
    assert ab.lookup("Why Acme?") is None
    assert ab.fillable_answers(ab.load(), acme)["why acme"] == "Robots"


@pytest.mark.parametrize("call", [lambda: ab.set_answer("Bad Key!", "x"), lambda: ab.set_salary_floor(5),
                                  lambda: ab.set_job_answer("", "q", "a"), lambda: ab.set_learned("", "a")])
def test_bad_edits_rejected(bank, call):
    with pytest.raises(ValueError):
        call()


def test_salary_floor(bank):
    ab.set_salary_floor(130000)
    assert ab.load()["salary_rule"]["floor"] == 130000


def test_greenhouse_never_fills_or_guesses_always_ask(bank):
    ab.set_answer("phone", "555-0100")
    ab.set_always_ask("phone", True)
    ab.set_always_ask("What excites you about this role?", True)
    form = gh.ParsedForm(fields=[gh.FormField("phone", "Phone", "text", True),
                                 gh.FormField("q1", "What excites you about this role?", "text", True),
                                 gh.FormField("first", "First Name", "text", True)])
    guessed = []
    plans = {p.field.label: p for p in gh.plan(form, {}, "profile",
                                               guess_fn=lambda qs, prof: guessed.extend(qs) or {},
                                               draft_fn=lambda *a: "")}
    assert plans["Phone"].source == "needs_me" and "always answer" in plans["Phone"].note
    assert plans["What excites you about this role?"].source == "needs_me" and guessed == []
    assert plans["First Name"].value == "Sam"
