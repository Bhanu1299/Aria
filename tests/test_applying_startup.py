from __future__ import annotations

import types

import pytest

import aria.web.agent_browser as agent_browser
from aria.features.applying import startup


@pytest.fixture
def up(monkeypatch):
    state = {"gateway": None, "coldstart": True, "dashboard": True, "chrome": True, "opened": [], "spawned": 0}
    monkeypatch.setattr(startup, "_gateway_problem", lambda: state["gateway"])

    def cold():
        if not state["coldstart"]:
            raise ImportError("No module named 'coldstart'")
        return types.SimpleNamespace(find_emails=lambda *a, **k: [])
    monkeypatch.setattr(startup, "_import_coldstart", cold)
    monkeypatch.setattr(startup, "dashboard_up", lambda: state["dashboard"])

    def spawn():
        state["spawned"] += 1
    monkeypatch.setattr(startup, "_spawn_dashboard", spawn)
    monkeypatch.setattr(startup, "DASHBOARD_WAIT_SECS", 0.05)

    def run(fn, tab=None):
        if not state["chrome"]:
            raise RuntimeError("Chrome didn't open its debug port within 15 s.")
        return "about:blank"
    monkeypatch.setattr(agent_browser, "run", run)
    monkeypatch.setattr(agent_browser, "navigate", lambda url, settle_secs=0, tab=None, **k: state["opened"].append((url, tab)))
    return state


def test_bring_up_all_ok_opens_live_tab(up):
    ok, msg = startup.bring_up()
    assert ok and "GateLLM" in msg and "Coldstart" in msg and "dashboard" in msg
    assert up["opened"] == [("http://127.0.0.1:8787/#live", "dashboard")]
    assert up["spawned"] == 0


def test_bring_up_stops_at_first_failure_names_it(up, monkeypatch):
    up["gateway"] = "connection refused"
    monkeypatch.setattr(startup, "_import_coldstart", lambda: pytest.fail("checked Coldstart after GateLLM failed"))
    ok, msg = startup.bring_up()
    assert not ok and msg == "GateLLM isn't working: connection refused. Stopping."
    assert up["opened"] == []


def test_bring_up_names_coldstart(up):
    up["coldstart"] = False
    ok, msg = startup.bring_up()
    assert not ok and msg.startswith("Coldstart isn't working") and up["opened"] == []


def test_bring_up_starts_dashboard_when_down(up):
    up["dashboard"] = False
    ok, msg = startup.bring_up()
    assert up["spawned"] == 1 and not ok and msg.startswith("The dashboard isn't working")


def test_bring_up_names_chrome(up):
    up["chrome"] = False
    ok, msg = startup.bring_up()
    assert not ok and msg.startswith("Chrome isn't working: Chrome didn't open")


def test_python_m_aria_start_only_runs_aria(monkeypatch):
    import aria.__main__ as cli
    brought_up = []
    monkeypatch.setattr(startup, "bring_up", lambda open_dashboard=True: brought_up.append(1) or (True, "up"))
    ran = []
    monkeypatch.setattr(cli, "_run_aria", lambda: ran.append(1) or 0)
    assert cli.main(["start"]) == 0 and ran == [1]
    assert brought_up == []                     # no dashboard, no Chrome, no run on launch


def test_launching_aria_never_starts_applications(monkeypatch):
    from aria import app
    from aria.features.applying import commands
    monkeypatch.setattr(app.applying_session, "reattach", lambda: 0)
    monkeypatch.setattr(commands, "handle", lambda text: (_ for _ in ()).throw(AssertionError(text)))
    app._reattach_tabs()


def test_python_m_aria_usage(capsys):
    import aria.__main__ as cli
    assert cli.main([]) == 2 and "python -m aria start" in capsys.readouterr().out


# ── 2026-10-22 live log: "open the dashboard" went to the agent, "apply it to" missed ──

def test_open_dashboard_by_voice_starts_and_shows_it_without_a_run(up, monkeypatch):
    from aria.features.applying import commands, session
    monkeypatch.setattr(session, "start", lambda *a, **k: pytest.fail("opening the dashboard must not start a run"))
    up["dashboard"] = False
    spawned = []
    monkeypatch.setattr(startup, "_spawn_dashboard", lambda: spawned.append(1) or up.update(dashboard=True))
    for said in ("Open the dashboard, please.", "Open my dashboard for job applications.", "show me the job dashboard"):
        assert commands.handle(said) == "Your job dashboard is open in a tab in Aria's Chrome.", said
    assert spawned == [1] and up["opened"][-1] == (startup.DASHBOARD_URL, startup.DASHBOARD_TAB)
    assert commands.handle("open the dashboard folder") is None              # not ours: normal routing


def test_whisper_it_after_apply_still_starts_and_negation_still_blocks():
    from aria.dispatch.parse import normalize
    from aria.features.applying import commands
    starts = lambda s: bool(commands._START_LEAD.fullmatch(normalize(s))) and not commands._NEGATION.search(normalize(s))
    assert starts("Arya, let's apply it to some jobs.") and starts("apply it to some jobs")
    assert not starts("Don't apply it to some jobs.")


def test_apply_job_tool_never_speaks_progress(monkeypatch, capsys):
    from aria.plugins import apply as apply_plugin
    apply_plugin._terminal_progress("Typing: someone@example.com")
    assert "[APPLY] Typing: someone@example.com" in capsys.readouterr().out
    import inspect
    assert "progress_fn=_terminal_progress" in inspect.getsource(apply_plugin.ApplyPlugin._apply_job_tool)
