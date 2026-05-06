"""tests/test_agent.py — Agent tool-loop unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.base import LLMResponse, ToolCall
from tool import ToolDescriptor, ToolRegistry
from agent import Agent


def _ok_response(text: str = "Done.") -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                       provider_used="anthropic", model_used="claude-sonnet-4-6")


def _tool_response(name: str, tool_id: str, input: dict) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(id=tool_id, name=name, input=input)],
        stop_reason="tool_use",
        provider_used="anthropic",
        model_used="claude-sonnet-4-6",
    )


def _make_registry(*tools: ToolDescriptor) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _make_tool(name: str, result: str = "tool result") -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        execute=lambda p: result,
    )


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

def test_simple_text_response():
    """When LLM returns end_turn with text, agent returns that text."""
    reg = _make_registry()
    agent = Agent(reg)

    with patch("llm.llm_client.complete") as mock_complete:
        mock_complete.return_value = _ok_response("The sky is blue.")
        result = agent.run("What color is the sky?")

    assert result == "The sky is blue."


def test_single_tool_call_and_final_response():
    """LLM calls one tool, then returns final text."""
    search_tool = _make_tool("web_search", result="search results")
    reg = _make_registry(search_tool)
    agent = Agent(reg)

    responses = [
        _tool_response("web_search", "tc1", {"query": "python news"}),
        _ok_response("Here are the latest Python news."),
    ]

    with patch("llm.llm_client.complete") as mock_complete:
        mock_complete.side_effect = responses
        result = agent.run("What's in the Python news?")

    assert result == "Here are the latest Python news."
    assert mock_complete.call_count == 2


def test_tool_result_appended_to_messages():
    """Tool result is included in messages sent to LLM on the second call."""
    search_tool = _make_tool("web_search", result="found: Python 3.13")
    reg = _make_registry(search_tool)
    agent = Agent(reg)

    calls_received = []

    def capture_and_respond(*args, **kwargs):
        calls_received.append(kwargs.get("messages", []))
        if len(calls_received) == 1:
            return _tool_response("web_search", "tc1", {"query": "python"})
        return _ok_response("Python 3.13 released.")

    with patch("llm.llm_client.complete", side_effect=capture_and_respond):
        agent.run("Python news")

    assert len(calls_received) == 2
    # Second call's messages should include tool result
    second_msgs = calls_received[1]
    tool_result_found = any(
        isinstance(m.get("content"), list) and
        any(c.get("type") == "tool_result" for c in m["content"])
        for m in second_msgs
    )
    assert tool_result_found


def test_multi_step_chain():
    """Agent correctly chains two tool calls before final answer."""
    jobs_tool = _make_tool("jobs", result="Found 3 jobs")
    kb_tool = _make_tool("knowledge", result="San Francisco is expensive")
    reg = _make_registry(jobs_tool, kb_tool)
    agent = Agent(reg)

    responses = [
        _tool_response("jobs", "tc1", {"query": "ML engineer"}),
        _tool_response("knowledge", "tc2", {"query": "cost of living"}),
        _ok_response("Found 3 ML jobs. San Francisco is expensive."),
    ]

    with patch("llm.llm_client.complete") as mock_complete:
        mock_complete.side_effect = responses
        result = agent.run("Find ML jobs and what's the cost of living?")

    assert "ML jobs" in result
    assert mock_complete.call_count == 3


def test_tool_error_continues_loop():
    """When a tool raises, the error is reported as a tool result and the loop continues."""
    def bad_execute(params):
        raise ValueError("disk full")

    bad_tool = ToolDescriptor(
        name="broken",
        description="broken tool",
        input_schema={},
        execute=bad_execute,
    )
    reg = _make_registry(bad_tool)
    agent = Agent(reg)

    responses = [
        _tool_response("broken", "tc1", {}),
        _ok_response("I had an issue but recovered."),
    ]

    with patch("llm.llm_client.complete") as mock_complete:
        mock_complete.side_effect = responses
        result = agent.run("Do the broken thing")

    # Should complete without raising
    assert isinstance(result, str)
    assert mock_complete.call_count == 2


def test_unknown_tool_returns_error_result():
    """Calling an unregistered tool returns an error result, loop continues."""
    reg = _make_registry()  # empty registry
    agent = Agent(reg)

    responses = [
        _tool_response("ghost_tool", "tc1", {}),
        _ok_response("I tried something but it wasn't available."),
    ]

    with patch("llm.llm_client.complete") as mock_complete:
        mock_complete.side_effect = responses
        result = agent.run("Do the ghost thing")

    assert isinstance(result, str)


def test_max_tool_calls_returns_partial():
    """When max tool calls exceeded, agent returns partial + note."""
    tool = _make_tool("loop_tool", result="partial step")
    reg = _make_registry(tool)
    agent = Agent(reg)

    # Always return a tool call — force hitting the limit
    with patch("llm.llm_client.complete") as mock_complete:
        mock_complete.return_value = _tool_response("loop_tool", "tc1", {})
        result = agent.run("Loop forever")

    assert "step limit" in result.lower()


def test_empty_text_on_end_turn_returns_rephrase():
    """end_turn with empty text returns 'rephrase' fallback."""
    reg = _make_registry()
    agent = Agent(reg)

    with patch("llm.llm_client.complete") as mock_complete:
        mock_complete.return_value = LLMResponse(
            text="", tool_calls=[], stop_reason="end_turn",
            provider_used="anthropic", model_used="claude-sonnet-4-6"
        )
        result = agent.run("hmm")

    assert "rephrase" in result.lower()


def test_run_never_raises():
    """Agent.run() catches all exceptions and returns a string."""
    reg = _make_registry()
    agent = Agent(reg)

    with patch("llm.llm_client.complete", side_effect=RuntimeError("catastrophic failure")):
        result = agent.run("anything")

    assert isinstance(result, str)
    assert len(result) > 0
