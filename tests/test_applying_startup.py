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
    monkeypatch.setattr(agent_browser, "bring_to_front", lambda name: None)
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


def test_python_m_aria_start_stops_when_bring_up_fails(monkeypatch, capsys):
    import aria.__main__ as cli
    monkeypatch.setattr(startup, "bring_up", lambda open_dashboard=True: (False, "Coldstart isn't working: x. Stopping."))
    monkeypatch.setattr(cli, "_run_aria", lambda: pytest.fail("launched Aria anyway"))
    assert cli.main(["start"]) == 1
    assert "Coldstart isn't working" in capsys.readouterr().out


def test_python_m_aria_start_launches_aria_with_the_run_flag(monkeypatch):
    import os

    import aria.__main__ as cli
    monkeypatch.delenv("ARIA_START_APPLICATIONS", raising=False)
    monkeypatch.setattr(startup, "bring_up", lambda open_dashboard=True: (True, "up"))
    seen = []
    monkeypatch.setattr(cli, "_run_aria", lambda: seen.append(os.environ.get("ARIA_START_APPLICATIONS")) or 0)
    assert cli.main(["start"]) == 0 and seen == ["1"]


def test_python_m_aria_usage(capsys):
    import aria.__main__ as cli
    assert cli.main([]) == 2 and "python -m aria start" in capsys.readouterr().out
