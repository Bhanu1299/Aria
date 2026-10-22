"""tests/test_dispatcher.py — one path per command: pre-handlers → direct route → agent."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.core.tool import ToolDescriptor, ToolRegistry
from aria.dispatch import Dispatcher, RuleSet
from aria.plugins.system import SystemPlugin, reminder_script
from aria.plugins.system.timers import TimerManager

RULES = {
    "rules": [
        {"id": "pause", "regex": "pause", "tool": "music_pause", "params": {}},
        {"id": "timer", "regex": "timer for (?P<d>.+)", "tool": "timer_set",
         "params": {"seconds": "{d:duration}"}},
        {"id": "ghost", "regex": "ghost", "tool": "no_such_tool", "params": {}},
        {"id": "boom", "regex": "boom", "tool": "explodes", "params": {}},
    ]
}


class FakeAgent:
    def __init__(self):
        self.runs, self.notes = [], []
        self.last_run_tools = [("web_search", True)]

    def run(self, text):
        from aria.llm import meter
        self.runs.append(text)
        meter.add({"cost_usd": 0.002, "tokens_in": 900, "tokens_out": 40})  # one gateway call
        return "agent answer"

    def note_turn(self, text, answer):
        self.notes.append((text, answer))


def _registry(**tools):
    reg = ToolRegistry()
    for name, fn in tools.items():
        reg.register(ToolDescriptor(name, name, {"type": "object", "properties": {}}, fn))
    return reg


def _boom(params):
    raise RuntimeError("tool crashed")


def _dispatcher(pre=(), rules=RULES):
    agent = FakeAgent()
    reg = _registry(
        music_pause=lambda p: "Paused.",
        timer_set=lambda p: f"Timer {p['seconds']}s.",
        explodes=_boom,
    )
    return Dispatcher(reg, agent, RuleSet(rules) if rules else None, pre_handlers=pre), agent


def test_direct_route_runs_tool_without_agent():
    d, agent = _dispatcher()
    r = d.handle("Pause.")
    assert (r.answer, r.route) == ("Paused.", "direct:pause")
    assert r.tools == [("music_pause", True)]
    assert r.usage["tokens_in"] == 0
    assert agent.runs == []
    assert agent.notes == [("Pause.", "Paused.")]  # follow-ups keep context


def test_typed_slots_reach_the_tool():
    d, _ = _dispatcher()
    assert d.handle("timer for 5 minutes").answer == "Timer 300s."


def test_unmatched_command_goes_to_agent_with_usage():
    d, agent = _dispatcher()
    r = d.handle("what's the weather in paris")
    assert (r.answer, r.route, r.escalated_from) == ("agent answer", "agent", "")
    assert r.usage["tokens_in"] == 900 and r.usage["calls"] == 1
    assert r.usage["cost_usd"] == pytest.approx(0.002)
    assert r.tools == [("web_search", True)]
    assert agent.runs == ["what's the weather in paris"]


def test_unparseable_slot_escalates_to_agent():
    d, agent = _dispatcher()
    r = d.handle("timer for a while")
    assert (r.route, r.escalated_from) == ("agent", "timer")
    assert agent.runs == ["timer for a while"]


def test_missing_or_crashing_tool_falls_back_to_agent():
    d, agent = _dispatcher()
    assert d.handle("ghost").escalated_from == "ghost"
    assert d.handle("boom").escalated_from == "boom"
    assert agent.runs == ["ghost", "boom"]


def test_pre_handler_wins_and_failures_are_skipped():
    def crashing(text):
        raise ValueError("bad")

    d, agent = _dispatcher(pre=[("crash", crashing), ("jobs", lambda t: "job info" if "job" in t else None)])
    assert d.handle("tell me about the second job").route == "pre:jobs"
    assert d.handle("pause").route == "direct:pause"
    assert agent.runs == []


def test_fastpath_off_sends_everything_to_agent():
    d, agent = _dispatcher(rules=None)
    assert d.handle("pause").route == "agent"
    assert agent.runs == ["pause"]


# ---------------------------------------------------------------------------
# System plugin pieces
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_timer_manager_status_and_cancel():
    clock = FakeClock()
    tm = TimerManager(on_fire=lambda label: None, clock=clock)
    assert tm.start(300, "pasta") == "Pasta timer set for 5 minutes."
    clock.t += 61
    assert tm.status() == "About 4 minutes left on your pasta timer."
    assert tm.cancel("laundry") == "I couldn't find a laundry timer."
    assert tm.cancel("pasta") == "Canceled your pasta timer."
    assert tm.status() == "You don't have any timers running."


def test_timer_fires_callback():
    fired = []
    tm = TimerManager(on_fire=fired.append)
    tm.start(0.01, "tea")
    import time
    time.sleep(0.2)
    assert fired == ["tea"]
    assert tm.status() == "You don't have any timers running."


def test_reminder_script_builds_date_from_components():
    script = reminder_script('call "mom"', datetime(2026, 10, 19, 17, 30))
    assert 'name:"call \\"mom\\""' in script
    assert "set year of d to 2026" in script
    assert "set month of d to 10" in script
    assert "set day of d to 19" in script
    assert "set time of d to 63000" in script
    assert "due date:d, remind me date:d" in script
    assert "tell default list" in script


def test_reminder_script_without_due():
    script = reminder_script("buy milk", None)
    assert "due date" not in script and 'name:"buy milk"' in script


def test_system_plugin_registers_tools_and_calculates():
    reg = ToolRegistry()
    SystemPlugin(speaker=MagicMock()).register(reg)
    names = {t.name for t in reg.all_available()}
    assert {"system_control", "clock", "calculate", "timer_set", "timer_status",
            "timer_cancel", "reminder_add", "agenda"} <= names
    calc = reg.get("calculate")
    assert calc.execute({"expression": "15 x 8"}) == "The result is 120."
    assert calc.execute({"expression": "15% of 80"}) == "The result is 12."
    assert calc.execute({"expression": "1,200 divided by 4"}) == "The result is 300."


def test_timer_done_speaks_and_notifies(monkeypatch):
    sent = []
    monkeypatch.setattr("aria.ui.notifier.send_notification", lambda title, body: sent.append(body))
    speaker = MagicMock()
    SystemPlugin(speaker=speaker)._timer_done("pasta")
    speaker.say.assert_called_once_with("Your pasta timer is done.")
    assert sent == ["Your pasta timer is done."]


def test_every_command_sets_the_screen_consent_words():
    """Screen captures check the words of the command being handled — follow-ups included."""
    from aria.screen import consent
    d, _agent = _dispatcher()
    d.handle("what's on my screen")
    assert consent.asked()
    d.handle("what's going on")
    assert not consent.asked()
