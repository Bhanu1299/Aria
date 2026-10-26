from __future__ import annotations

from aria.core import agent


def test_background_reports_reach_the_agent_prompt(monkeypatch):
    monkeypatch.setattr(agent, "_background", [])
    assert "on your own recently" not in agent._build_system_prompt()
    agent.note_background("Done with 10 jobs, nothing submitted. Contacts: 2 drafts in the jobs Gmail.")
    prompt = agent._build_system_prompt()
    assert "on your own recently" in prompt and "2 drafts in the jobs Gmail" in prompt
    assert prompt.index("on your own recently") < prompt.index("# Right now")


def test_only_the_last_few_are_kept(monkeypatch):
    monkeypatch.setattr(agent, "_background", [])
    for i in range(agent._BACKGROUND_KEEP + 3):
        agent.note_background(f"report {i}")
    assert [x for _t, x in agent._background] == [f"report {i}" for i in range(3, agent._BACKGROUND_KEEP + 3)]
