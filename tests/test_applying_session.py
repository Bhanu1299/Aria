"""tests/test_applying_session.py — start applications, tab-targeted voice commands, applications-are-done."""
from __future__ import annotations

import pytest

import aria.web.agent_browser as agent_browser
from aria.features import applicator
from aria.features.applying import commands, control, session, tabs
from aria.features.ats import submit as ats_submit
from aria.integrations import sheets


def _row(row, company, fit, status="", url=None):
    return {"row": row, "company": company, "role": "SWE", "alignment": str(fit), "status": status,
            "url": url if url is not None else f"https://jobs.example/{company.lower().replace(' ', '')}",
            "location": "", "fields": {}}


ROWS = [
    _row(13, "Old Co", 5, "Applied"),
    _row(14, "Ookla", 3, "Not applied"),
    _row(15, "Scale AI", 4),
    _row(16, "SingleStore", 4),
    _row(17, "ZipRecruiter", 3),
    _row(18, "Reviewing Inc", 5, "In review"),
]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(tabs, "REGISTRY_PATH", tmp_path / "apply_tabs.json")
    said = []
    commands.configure(say=said.append, voice_ask=None)
    front = {"names": []}
    browser = {"closed": [], "fronted": [], "attached": None}
    monkeypatch.setattr(agent_browser, "front_tabs", lambda names: [n for n in names if n in front["names"]])
    monkeypatch.setattr(agent_browser, "close_tab", lambda name: browser["closed"].append(name) or True)
    monkeypatch.setattr(agent_browser, "bring_to_front", lambda name: browser["fronted"].append(name))
    monkeypatch.setattr(agent_browser, "navigate", lambda *a, **k: None)
    monkeypatch.setattr(session.walls, "check", lambda tab: None)
    monkeypatch.setattr(session.browser_profile, "cdp_up", lambda port=0: True)
    monkeypatch.setattr(session.prevent_sleep, "start", lambda: None)
    monkeypatch.setattr(session.prevent_sleep, "stop", lambda: None)
    sheet = []
    monkeypatch.setattr(sheets, "set_status", lambda row, status, note="", company=None, role=None:
                        sheet.append((row, status, note)) or row)
    monkeypatch.setattr(ats_submit, "page_confirms", lambda tab: False)
    monkeypatch.setattr(control, "current", None)
    commands._asking.clear()
    yield {"said": said, "front": front, "browser": browser, "sheet": sheet}
    if control.current is not None:
        control.current.stop()
    if session._thread is not None:
        session._thread.join(2)


def _filled(company, row, **kw):
    e = tabs.add(_row(row, company, 4))
    fields = dict(state=tabs.FILLED, form_url=e["url"], ats="greenhouse")
    fields.update(kw)
    return tabs.update(e["id"], **fields)


# ── picking jobs ────────────────────────────────────────────────────────────

def test_select_jobs_fit_order_and_skips_handled_and_open():
    tabs.add(_row(16, "SingleStore", 4))                        # already open in a tab
    picked = session.select_jobs(ROWS, limit=10)
    assert [j["company"] for j in picked] == ["Scale AI", "Ookla", "ZipRecruiter"]


def test_select_jobs_order_ties_by_row():
    picked = session.select_jobs(ROWS, limit=10)
    assert [j["company"] for j in picked] == ["Scale AI", "SingleStore", "Ookla", "ZipRecruiter"]


def test_select_jobs_caps_at_limit():
    assert [j["company"] for j in session.select_jobs(ROWS, limit=2)] == ["Scale AI", "SingleStore"]


def test_fit_parses_loose_values():
    assert session.fit({"alignment": "4/5"}) == 4.0 and session.fit({"alignment": ""}) == 0.0
    assert session.fit({"alignment": "n/a"}) == 0.0


# ── the run ─────────────────────────────────────────────────────────────────

def test_start_fills_one_at_a_time_in_order(monkeypatch, env):
    monkeypatch.setattr(sheets, "default_sheet", lambda: type("S", (), {"jobs": lambda self: list(ROWS)})())
    order = []
    monkeypatch.setattr(session, "fill_one", lambda job, say, voice_ask=None, ctl=None:
                        order.append(job["company"]) or f"{job['company']} filled.")
    out = session.start(env["said"].append)
    assert out.startswith("Starting 4 applications, best fit first: Scale AI, SingleStore, Ookla and ZipRecruiter.")
    session._thread.join(2)
    assert order == ["Scale AI", "SingleStore", "Ookla", "ZipRecruiter"]
    assert "Nothing was submitted" in env["said"][-1]
    assert control.current is None


def test_start_respects_ten_open_tabs(monkeypatch, env):
    for i in range(9):
        tabs.add(_row(100 + i, f"Co{i}", 3))
    monkeypatch.setattr(sheets, "default_sheet", lambda: type("S", (), {"jobs": lambda self: list(ROWS)})())
    monkeypatch.setattr(session, "fill_one", lambda job, say, voice_ask=None, ctl=None: "ok")
    assert session.start(env["said"].append).startswith("Starting 1 application, best fit first: Scale AI.")


def test_start_refuses_at_ten_open_tabs(env):
    for i in range(10):
        tabs.add(_row(100 + i, f"Co{i}", 3))
    assert "10 application tabs open" in session.start(env["said"].append)


def test_fill_one_opens_its_own_tab_and_fills_there(monkeypatch, env):
    seen = {}

    def run_application(job, progress_fn=None, voice_ask_fn=None, run=None, entry=None, gate=None):
        seen.update(entry=entry, gate=gate)
        tabs.update(entry["id"], state=tabs.FILLED)
        return "Greenhouse form filled."
    monkeypatch.setattr(applicator, "run_application", run_application)
    ctl = control.RunControl()
    out = session.fill_one(_row(15, "Scale AI", 4), env["said"].append, ctl=ctl)
    assert out == "Greenhouse form filled."
    assert seen["entry"]["company"] == "Scale AI" and seen["gate"] == ctl.checkpoint
    assert env["browser"]["fronted"] == [seen["entry"]["id"]]


def test_fill_one_waits_at_a_wall_then_continues(monkeypatch, env):
    checks = iter(["captcha", "captcha", None, None])
    monkeypatch.setattr(session.walls, "check", lambda tab: next(checks))
    monkeypatch.setattr(session, "WALL_POLL_SECS", 0.01)
    monkeypatch.setattr(applicator, "run_application", lambda job, **kw: "filled")
    assert session.fill_one(_row(14, "Ookla", 3), env["said"].append, ctl=control.RunControl()) == "filled"
    assert any("Ookla is showing a CAPTCHA" in s for s in env["said"])
    assert len(env["browser"]["fronted"]) >= 2                         # opened, then brought back for the wall


def test_fill_one_stop_leaves_tab_open(monkeypatch, env):
    def stopped(job, **kw):
        raise control.Stopped()
    monkeypatch.setattr(applicator, "run_application", stopped)
    with pytest.raises(control.Stopped):
        session.fill_one(_row(15, "Scale AI", 4), env["said"].append, ctl=control.RunControl())
    [e] = tabs.live()
    assert e["state"] in (tabs.FILLING, tabs.OPEN) and env["browser"]["closed"] == []


def test_reattach_binds_open_tabs_and_closes_missing(monkeypatch, env):
    a = _filled("Scale AI", 15)
    b = _filled("SingleStore", 16)
    c = tabs.update(tabs.add(_row(17, "ZipRecruiter", 3))["id"], state=tabs.FILLING)
    monkeypatch.setattr(agent_browser, "attach", lambda wanted: {a["id"], c["id"]} & set(wanted))
    assert session.reattach() == 2
    assert tabs.get(b["id"])["state"] == tabs.CLOSED
    assert tabs.get(c["id"])["state"] == tabs.OPEN                    # a crash mid-fill → open


def test_reattach_keeps_entries_when_chrome_unreachable(monkeypatch, env):
    a = _filled("Scale AI", 15)

    def boom(wanted):
        raise RuntimeError("Chrome down")
    monkeypatch.setattr(agent_browser, "attach", boom)
    assert session.reattach() == 0 and tabs.get(a["id"])["state"] == tabs.FILLED


# ── voice commands ──────────────────────────────────────────────────────────

def test_start_applications_brings_up_then_starts(monkeypatch, env):
    monkeypatch.setattr(commands.startup, "bring_up", lambda: (True, "All up."))
    monkeypatch.setattr(session, "reattach", lambda: 0)
    monkeypatch.setattr(session, "start", lambda say, voice_ask=None: "Starting 4 applications.")
    assert commands.handle("Aria, start applications.") == "Starting 4 applications."


def test_start_applications_stops_when_a_service_fails(monkeypatch, env):
    monkeypatch.setattr(commands.startup, "bring_up", lambda: (False, "GateLLM isn't working: down. Stopping."))
    monkeypatch.setattr(session, "start", lambda *a, **k: pytest.fail("started anyway"))
    assert commands.handle("start applications") == "GateLLM isn't working: down. Stopping."


@pytest.mark.parametrize("text", ["pause", "stop", "resume"])
def test_pause_not_claimed_without_run(env, text):
    assert commands.handle(text) is None


def test_pause_resume_stop_during_run(env):
    ctl = control.RunControl()
    control.current = ctl
    assert "Paused" in commands.handle("pause") and ctl.paused
    assert "Resuming" in commands.handle("resume") and not ctl.paused
    assert "tabs stay open" in commands.handle("stop applications") and ctl.stopped


def test_submit_it_uses_front_tab(monkeypatch, env):
    a, b = _filled("Scale AI", 15), _filled("SingleStore", 16)
    env["front"]["names"] = [b["id"]]
    got = []
    monkeypatch.setattr(applicator, "submit_entry", lambda e: got.append(e["id"]) or "Submitted.")
    assert commands.handle("submit it") == "Submitted." and got == [b["id"]]


def test_submit_it_asks_when_no_application_in_front(monkeypatch, env):
    _filled("Scale AI", 15), _filled("SingleStore", 16)
    monkeypatch.setattr(applicator, "submit_entry", lambda e: pytest.fail("submitted without a clear target"))
    out = commands.handle("submit it")
    assert out.startswith("Which one?") and "Scale AI" in out and "SingleStore" in out


def test_submit_company_targets_tab(monkeypatch, env):
    a, _b = _filled("Scale AI", 15), _filled("SingleStore", 16)
    got = []
    monkeypatch.setattr(applicator, "submit_entry", lambda e: got.append(e["id"]) or "Submitted.")
    assert commands.handle("submit scale ai") == "Submitted." and got == [a["id"]]


def test_submit_unrelated_text_falls_through(env):
    _filled("Scale AI", 15)
    assert commands.handle("submit the report to my boss") is None
    assert commands.handle("skip this song") is None


def test_submit_refuses_while_filling(monkeypatch, env):
    e = tabs.add(_row(15, "Scale AI", 4))
    monkeypatch.setattr(applicator, "submit_entry", lambda e: pytest.fail("submitted mid-fill"))
    assert "still filling" in commands.handle("submit scale ai")


def test_no_tabs_submit_it_falls_through(env):
    assert commands.handle("submit it") is None


def test_i_submitted_company(monkeypatch, env):
    _filled("Scale AI", 15)
    b = _filled("SingleStore", 16)
    got = []
    monkeypatch.setattr(applicator, "mark_submitted_by_hand", lambda e: got.append(e["id"]) or "Got it.")
    assert commands.handle("I submitted SingleStore") == "Got it." and got == [b["id"]]


def test_close_it_closes_front_tab(env):
    a, b = _filled("Scale AI", 15), _filled("SingleStore", 16)
    env["front"]["names"] = [a["id"]]
    assert "Closed" in commands.handle("close it")
    assert env["browser"]["closed"] == [a["id"]] and tabs.get(a["id"])["state"] == tabs.CLOSED
    assert tabs.get(b["id"])["state"] == tabs.FILLED


def test_close_all_closes_only_app_tabs(env):
    a, b = _filled("Scale AI", 15), _filled("SingleStore", 16)
    assert "Closed 2 application tabs" in commands.handle("close all")
    assert sorted(env["browser"]["closed"]) == sorted([a["id"], b["id"]])
    assert "dashboard" not in env["browser"]["closed"] and tabs.live() == []


def test_done_marks_submitted_and_asks_per_unconfirmed(env):
    a = _filled("Scale AI", 15, state=tabs.SUBMITTED, submitted_by="aria", note="confirmation seen")
    b = _filled("SingleStore", 16, state=tabs.SUBMITTED, submitted_by="user", note="you submitted it")
    c = _filled("Ookla", 14)
    d = _filled("ZipRecruiter", 17)
    out = commands.handle("applications are done")
    assert [(r, s) for r, s, _ in env["sheet"]] == [(15, sheets.STATUS_APPLIED), (16, sheets.STATUS_APPLIED)]
    assert "Scale AI" in out and "SingleStore" in out
    assert "SWE at Ookla is filled but I never saw a confirmation" in out and "applied or not applied" in out
    assert tabs.get(c["id"])["sheet_marked"] is False and env["browser"]["closed"] == []
    assert tabs.get(a["id"])["sheet_marked"] and tabs.get(b["id"])["sheet_marked"]


def test_done_counts_a_confirmation_page_seen_now(monkeypatch, env):
    c = _filled("Ookla", 14)
    monkeypatch.setattr(ats_submit, "page_confirms", lambda tab: tab == c["id"])
    out = commands.handle("applications are done")
    assert env["sheet"] and env["sheet"][0][:2] == (14, sheets.STATUS_APPLIED) and "confirmation" in env["sheet"][0][2]
    assert "never saw a confirmation" not in out


def test_done_answer_applied_marks_then_asks_next(env):
    c, d = _filled("Ookla", 14), _filled("ZipRecruiter", 17)
    commands.handle("applications are done")
    out = commands.handle("applied")
    assert env["sheet"][-1][:2] == (14, sheets.STATUS_APPLIED) and "you said you applied" in env["sheet"][-1][2]
    assert "ZipRecruiter" in out and "applied or not applied" in out
    out = commands.handle("not applied")
    assert len(env["sheet"]) == 1 and "tabs stay open" in out.lower()
    assert commands.handle("applied") is None                          # queue empty → not claimed


def test_done_with_nothing_open(env):
    assert "No application tabs" in commands.handle("applications are done")


# ── review fixes ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["send it", "send that", "yes send it", "send"])
def test_send_is_not_submit(monkeypatch, env, text):
    a = _filled("Scale AI", 15)
    env["front"]["names"] = [a["id"]]
    monkeypatch.setattr(applicator, "submit_entry", lambda e: pytest.fail("'send' submitted an application"))
    assert commands.handle(text) is None


@pytest.mark.parametrize("company,text", [("GitHub", "skip it"), ("Meta", "send me"), ("Adobe", "skip ad"),
                                          ("Citadel", "close it all"), ("Spotify", "close spotify")])
def test_everyday_phrases_never_hit_a_company_tab(monkeypatch, env, company, text):
    _filled(company, 15)
    monkeypatch.setattr(applicator, "skip_entry", lambda e: pytest.fail("skipped"))
    monkeypatch.setattr(applicator, "submit_entry", lambda e: pytest.fail("submitted"))
    assert commands.handle(text) is None and env["browser"]["closed"] == []


def test_skip_and_close_company_need_the_job_word(monkeypatch, env):
    a = _filled("Spotify", 15)
    got = []
    monkeypatch.setattr(applicator, "skip_entry", lambda e: got.append(e["id"]) or "Skipped.")
    assert commands.handle("skip the spotify application") == "Skipped." and got == [a["id"]]
    assert "Closed" in commands.handle("close the spotify tab")


def test_company_match_is_at_word_start(monkeypatch, env):
    _filled("GitHub", 15)
    monkeypatch.setattr(applicator, "submit_entry", lambda e: pytest.fail("submitted"))
    assert commands.handle("submit hub") is None


def test_done_marks_submitted_tabs_the_user_already_closed(env):
    a = _filled("Scale AI", 15, state=tabs.SUBMITTED, submitted_by="aria", note="confirmation seen")
    tabs.mark_closed(a["id"])
    out = commands.handle("applications are done")
    assert env["sheet"] and env["sheet"][0][:2] == (15, sheets.STATUS_APPLIED) and "Scale AI" in out


def test_unrelated_words_end_the_applied_question(env):
    _filled("Ookla", 14)
    commands.handle("applications are done")
    assert commands.handle("what's the weather") is None
    assert commands.handle("yes") is None and env["sheet"] == []


def test_reattach_does_nothing_when_chrome_is_not_running(monkeypatch, env):
    a = _filled("Scale AI", 15)
    monkeypatch.setattr(session.browser_profile, "cdp_up", lambda port=0: False)
    monkeypatch.setattr(agent_browser, "attach", lambda wanted: pytest.fail("launched Chrome to reattach"))
    assert session.reattach() == 0 and tabs.get(a["id"])["state"] == tabs.FILLED


def test_fill_one_error_leaves_entry_open_not_filling(monkeypatch, env):
    def broken(*a, **k):
        raise RuntimeError("Chrome went away")
    monkeypatch.setattr(agent_browser, "navigate", broken)
    with pytest.raises(RuntimeError):
        session.fill_one(_row(15, "Scale AI", 4), env["said"].append, ctl=control.RunControl())
    [e] = tabs.live()
    assert e["state"] == tabs.OPEN and "Chrome went away" in e["note"]
