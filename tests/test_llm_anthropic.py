"""tests/test_llm_anthropic.py — AnthropicProvider response normalization."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.base import RateLimitError, AuthError


def _make_anthropic_response(text="hello", stop_reason="end_turn", tool_blocks=None):
    blocks = []
    if text:
        tb = MagicMock()
        tb.type = "text"
        tb.text = text
        blocks.append(tb)
    for t in (tool_blocks or []):
        b = MagicMock()
        b.type = "tool_use"
        b.id = t["id"]
        b.name = t["name"]
        b.input = t["input"]
        blocks.append(b)
    resp = MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    return resp


def _make_provider():
    with patch("llm.providers.anthropic.config") as mock_cfg:
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        from llm.providers.anthropic import AnthropicProvider
        p = AnthropicProvider("claude-haiku-4-5-20251001")
        mock_client = MagicMock()
        p._client = mock_client
        return p, mock_client


def test_text_response_parsed():
    p, mock_client = _make_provider()
    mock_client.messages.create.return_value = _make_anthropic_response(text="hello world")

    result = p.complete([{"role": "user", "content": "hi"}])

    assert result.text == "hello world"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"
    assert result.provider_used == "anthropic"


def test_tool_call_parsed():
    p, mock_client = _make_provider()
    mock_client.messages.create.return_value = _make_anthropic_response(
        text="",
        stop_reason="tool_use",
        tool_blocks=[{"id": "tc1", "name": "bash", "input": {"command": "ls"}}],
    )

    result = p.complete([{"role": "user", "content": "list files"}])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].input == {"command": "ls"}
    assert result.stop_reason == "tool_use"


def test_rate_limit_raises():
    import anthropic as _ant
    p, mock_client = _make_provider()
    mock_client.messages.create.side_effect = _ant.RateLimitError(
        message="429", response=MagicMock(), body={}
    )

    try:
        p.complete([])
        assert False, "should have raised"
    except RateLimitError:
        pass


def test_system_and_tools_passed_through():
    p, mock_client = _make_provider()
    mock_client.messages.create.return_value = _make_anthropic_response()

    p.complete(
        messages=[{"role": "user", "content": "hi"}],
        system="You are Aria",
        tools=[{"name": "bash", "description": "run bash", "input_schema": {}}],
        max_tokens=512,
    )

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["system"] == "You are Aria"
    assert call_kwargs["tools"][0]["name"] == "bash"
    assert call_kwargs["max_tokens"] == 512
