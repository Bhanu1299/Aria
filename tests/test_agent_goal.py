"""tests/test_agent_goal.py — goal-directed agent loop: completion gate,
tool-result truncation, budget wrap-up reminder, subagent tool.

Patterns extracted from Claude Code (TodoWrite reminders, turn budget,
Task/AgentTool subagents) adapted to Aria's voice loop.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm.base import LLMResponse, ToolCall
from aria.core.tool import ToolDescriptor, ToolRegistry
from aria.core.agent import Agent, truncate_tool_result


def _ok(text: str = "Done.") -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                       provider_used="anthropic", model_used="claude-sonnet-4-6")


def _tool(name: str, tool_id: str, input: dict) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[ToolCall(id=tool_id, name=name, input=input)],
                       stop_reason="tool_use",
                       provider_used="anthropic", model_used="claude-sonnet-4-6")


def _make_tool(name: str, result: str = "tool result") -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        execute=lambda p: result,
    )


def _registry(*tools: ToolDescriptor) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_truncate_short_result_unchanged():
    assert truncate_tool_result("hello") == "hello"


def test_truncate_long_result_keeps_head_and_tail():
    text = "A" * 5000 + "MIDDLE" + "B" * 5000
    out = truncate_tool_result(text, limit=2000)
    assert len(out) < 3000
    assert out.startswith("A")
    assert out.rstrip().endswith("B")
    assert "truncated" in out


def test_giant_tool_output_truncated_in_messages():
    big_tool = _make_tool("dumper", result="X" * 50_000)
    agent = Agent(_registry(big_tool))
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs.get("messages", []))
        if len(seen) == 1:
            return _tool("dumper", "t1", {})
        return _ok("done")

    with patch("aria.llm.llm_client.complete", side_effect=capture):
        agent.run("dump it")

    second = seen[1]
    tool_result_blocks = [
        c for m in second if isinstance(m.get("content"), list)
        for c in m["content"] if c.get("type") == "tool_result"
    ]
    assert tool_result_blocks
    content = tool_result_blocks[0]["content"]
    assert len(content) < 20_000
    assert "truncated" in content


# ---------------------------------------------------------------------------
# update_tasks auto-registration + completion gate
# ---------------------------------------------------------------------------

def test_agent_registers_update_tasks_tool():
    reg = _registry()
    Agent(reg)
    assert reg.get("update_tasks") is not None


def test_completion_gate_nudges_on_unfinished_tasks():
    """end_turn with unfinished tasks -> reminder injected, loop continues once."""
    reg = _registry()
    agent = Agent(reg)
    seen = []

    responses = [
        _tool("update_tasks", "t1", {"tasks": [
            {"content": "step one", "status": "completed"},
            {"content": "step two", "status": "pending"},
        ]}),
        _ok("All done!"),          # premature — step two still pending
        _ok("Actually finished now."),
    ]

    def capture(*args, **kwargs):
        seen.append(kwargs.get("messages", []))
        return responses[len(seen) - 1]

    with patch("aria.llm.llm_client.complete", side_effect=capture):
        result = agent.run("do a two step thing")

    assert result == "Actually finished now."
    assert len(seen) == 3
    # the injected nudge mentions the unfinished task
    last_msgs = seen[2]
    flat = str(last_msgs)
    assert "step two" in flat
    assert "unfinished" in flat.lower() or "not completed" in flat.lower()


def test_completion_gate_fires_only_once_per_run():
    """If the model insists on ending with unfinished tasks, accept the second end_turn."""
    reg = _registry()
    agent = Agent(reg)

    responses = [
        _tool("update_tasks", "t1", {"tasks": [{"content": "x", "status": "pending"}]}),
        _ok("Done."),
        _ok("Still done."),
        _ok("Should never be called."),
    ]

    with patch("aria.llm.llm_client.complete", side_effect=responses) as mock:
        result = agent.run("do x")

    assert result == "Still done."
    assert mock.call_count == 3


def test_no_gate_when_tasks_all_completed():
    reg = _registry()
    agent = Agent(reg)

    responses = [
        _tool("update_tasks", "t1", {"tasks": [{"content": "x", "status": "completed"}]}),
        _ok("Done for real."),
    ]

    with patch("aria.llm.llm_client.complete", side_effect=responses) as mock:
        result = agent.run("do x")

    assert result == "Done for real."
    assert mock.call_count == 2


def test_task_list_cleared_between_completed_runs():
    """A finished run leaves no stale tasks to trip the gate on the next utterance."""
    reg = _registry()
    agent = Agent(reg)

    responses = [
        _tool("update_tasks", "t1", {"tasks": [{"content": "x", "status": "completed"}]}),
        _ok("Done."),
        _ok("Just chatting."),
    ]

    with patch("aria.llm.llm_client.complete", side_effect=responses) as mock:
        agent.run("do x")
        result = agent.run("hello")

    assert result == "Just chatting."
    assert mock.call_count == 3


# ---------------------------------------------------------------------------
# Budget wrap-up reminder
# ---------------------------------------------------------------------------

def test_default_budget_raised_to_25():
    agent = Agent(_registry())
    assert agent.max_tool_calls >= 25


def test_wrapup_reminder_injected_near_budget():
    tool = _make_tool("step", result="ok")
    agent = Agent(_registry(tool))
    agent.max_tool_calls = 4
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs.get("messages", []))
        return _tool("step", f"t{len(seen)}", {})

    with patch("aria.llm.llm_client.complete", side_effect=capture):
        result = agent.run("loop")

    flat = str(seen[-1])
    assert "wrap up" in flat.lower() or "step budget" in flat.lower()
    assert "step limit" in result.lower()


# ---------------------------------------------------------------------------
# Subagent tool
# ---------------------------------------------------------------------------

def test_agent_registers_run_subtask_tool():
    reg = _registry()
    Agent(reg)
    assert reg.get("run_subtask") is not None


def test_subagent_runs_isolated_and_returns_answer():
    """run_subtask spins a fresh agent whose tool list excludes run_subtask."""
    reg = _registry(_make_tool("web_search", result="found it"))
    agent = Agent(reg)
    tools_seen = []

    responses = [
        _tool("run_subtask", "t1", {"prompt": "research flights"}),   # main agent
        _ok("sub answer: cheapest is $220"),                          # subagent
        _ok("The cheapest flight is $220."),                          # main agent final
    ]
    idx = {"i": 0}

    def capture(*args, **kwargs):
        tools_seen.append([t["name"] for t in (kwargs.get("tools") or [])])
        r = responses[idx["i"]]
        idx["i"] += 1
        return r

    with patch("aria.llm.llm_client.complete", side_effect=capture):
        result = agent.run("find me cheap flights")

    assert result == "The cheapest flight is $220."
    # second LLM call is the subagent: it must not see run_subtask
    assert "run_subtask" not in tools_seen[1]
    assert "web_search" in tools_seen[1]


def test_subagent_never_raises_to_main_loop():
    reg = _registry()
    agent = Agent(reg)
    sub_tool = reg.get("run_subtask")

    with patch("aria.llm.llm_client.complete", side_effect=RuntimeError("api down")):
        result = sub_tool.execute({"prompt": "anything"})

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# System prompt mentions the new machinery
# ---------------------------------------------------------------------------

def test_system_prompt_mentions_task_tracking():
    from aria.core.agent import _build_system_prompt
    p = _build_system_prompt()
    assert "update_tasks" in p
    assert len(p.split()) < 600  # keep the per-utterance token cost bounded
