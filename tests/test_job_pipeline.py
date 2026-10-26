"""tests/test_job_pipeline.py — "run the job pipeline": only its own words start it; both parts run; one report."""
from __future__ import annotations

import threading

import pytest

from aria.features.applying import commands, job_pipeline, session, startup


@pytest.fixture
def parts(monkeypatch):
    """bring_up ok; the apply run and the contacts run are fakes that end when released."""
    got = {"said": [], "apply": None, "contacts": None, "started": []}
    release = {"apply": threading.Event(), "contacts": threading.Event()}
    monkeypatch.setattr(startup, "bring_up", lambda open_dashboard=True: (True, "up"))
    monkeypatch.setattr(session, "reattach", lambda: 0)
    monkeypatch.setattr(job_pipeline, "_current", None)

    def fake_start(say, voice_ask=None, limit=None, finish_fn=None):
        got["started"].append("apply")

        def work():
            release["apply"].wait(2)
            finish_fn("Done with 2 jobs, nothing submitted.")
        session._thread = threading.Thread(target=work, daemon=True)
        session._thread.start()
        return "Starting 2 applications, best fit first: Scale AI and Ookla."

    def fake_contacts(done_fn):
        got["started"].append("contacts")

        def work():
            release["contacts"].wait(2)
            done_fn("Contacts: 3 drafts in the jobs Gmail. Nothing was sent.")
        threading.Thread(target=work, daemon=True).start()
        return True

    monkeypatch.setattr(session, "start", fake_start)
    monkeypatch.setattr(job_pipeline, "_start_contacts", fake_contacts)
    monkeypatch.setattr(commands, "_say", lambda msg: got["said"].append(msg))
    yield got, release
    release["apply"].set()
    release["contacts"].set()


def _wait(pred, secs=2.0):
    ev = threading.Event()
    for _ in range(int(secs / 0.02)):
        if pred():
            return True
        ev.wait(0.02)
    return pred()


@pytest.mark.parametrize("said", [
    "run the pipeline", "run the job pipeline", "start the job pipeline", "Aria, run the job pipeline.",
    "can you run the whole pipeline", "kick off the job pipeline", "start the job pipelayer",
    "okay run my jobs pipeline now",
])
def test_pipeline_words_start_both_parts(parts, said):
    got, release = parts
    out = commands.handle(said)
    assert "Starting 2 applications" in out and "drafts" in out and "nothing gets sent" in out
    assert sorted(got["started"]) == ["apply", "contacts"]


@pytest.mark.parametrize("said", [
    "don't run the pipeline", "why did you run the pipeline", "what's in the pipeline",
    "start applying", "open my dashboard", "find emails for my contacts",
])
def test_other_words_never_start_the_pipeline(monkeypatch, said):
    monkeypatch.setattr(job_pipeline, "start", lambda *a, **k: pytest.fail("pipeline started"))
    monkeypatch.setattr(commands, "start_run", lambda limit=None: "apply run")
    monkeypatch.setattr(startup, "open_dashboard", lambda: "dashboard")
    out = commands.handle(said)
    if said == "start applying":
        assert out == "apply run"
    elif said == "open my dashboard":
        assert out == "dashboard"
    else:
        assert out is None


def test_speaks_once_when_both_are_done(parts):
    got, release = parts
    commands.handle("run the job pipeline")
    release["contacts"].set()
    _wait(lambda: False, 0.1)
    assert got["said"] == []                       # contacts done, applications not yet: quiet
    release["apply"].set()
    assert _wait(lambda: len(got["said"]) == 1)
    assert got["said"][0] == ("Done with 2 jobs, nothing submitted. "
                              "Contacts: 3 drafts in the jobs Gmail. Nothing was sent.")
    assert not job_pipeline.running()


def test_second_start_while_running_is_refused(parts):
    commands.handle("run the job pipeline")
    assert "already running" in commands.handle("run the pipeline")


def test_bring_up_failure_starts_nothing(parts, monkeypatch):
    got, _ = parts
    monkeypatch.setattr(startup, "bring_up", lambda open_dashboard=True: (False, "GateLLM isn't working: down. Stopping."))
    assert commands.handle("run the job pipeline") == "GateLLM isn't working: down. Stopping."
    assert got["started"] == [] and not job_pipeline.running()


def test_no_new_jobs_still_runs_contacts_and_reports_it(parts, monkeypatch):
    got, release = parts
    monkeypatch.setattr(session, "start", lambda say, voice_ask=None, limit=None, finish_fn=None:
                        "There are no new jobs in your sheet that aren't already open in a tab.")
    out = commands.handle("run the job pipeline")
    assert out.startswith("No applications this time") and "drafts" in out
    release["contacts"].set()
    assert _wait(lambda: got["said"] == ["Contacts: 3 drafts in the jobs Gmail. Nothing was sent."])


def test_contacts_already_running_waits_only_for_applications(parts, monkeypatch):
    got, release = parts
    monkeypatch.setattr(job_pipeline, "_start_contacts", lambda done_fn: False)
    out = commands.handle("run the job pipeline")
    assert "already being worked on" in out
    release["apply"].set()
    assert _wait(lambda: got["said"] == ["Done with 2 jobs, nothing submitted."])


def test_apply_run_hands_its_end_line_to_finish_fn(monkeypatch):
    """session._run: with finish_fn the end line goes there, not to the speaker."""
    said, ended = [], []
    monkeypatch.setattr(session.prevent_sleep, "start", lambda: None)
    monkeypatch.setattr(session.prevent_sleep, "stop", lambda: None)
    monkeypatch.setattr(session, "fill_one", lambda job, say, voice_ask=None, ctl=None, ids=None: "ok")
    monkeypatch.setattr(session, "run_report", lambda ids, done: f"Done with {done} jobs.")
    monkeypatch.setattr(session, "waiting_names", lambda: [])
    from aria.features.applying import control
    ctl = control.RunControl()
    monkeypatch.setattr(control, "current", ctl)
    session._run([{"company": "A"}], said.append, None, ctl, ended.append)
    assert ended == ["Done with 1 jobs."] and said == []
