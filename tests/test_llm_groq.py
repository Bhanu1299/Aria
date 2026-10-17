"""tests/test_llm_groq.py — GroqProvider response normalization."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm.base import RateLimitError, AuthError


def _make_groq_response(text="hello", tool_calls=None):
    msg = MagicMock()
    msg.content = text
    raw_tcs = []
    for tc in (tool_calls or []):
        t = MagicMock()
        t.id = tc["id"]
        t.function.name = tc["name"]
        t.function.arguments = json.dumps(tc["input"])
        raw_tcs.append(t)
    msg.tool_calls = raw_tcs if raw_tcs else None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _make_provider():
    with patch("aria.llm.providers.groq.config") as mock_cfg:
        mock_cfg.GROQ_API_KEY = "test-key"
        from aria.llm.providers.groq import GroqProvider
        p = GroqProvider("llama-3.3-70b-versatile")
        mock_client = MagicMock()
        p._client = mock_client
        return p, mock_client


def test_text_response_parsed():
    p, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _make_groq_response(text="hello world")

    result = p.complete([{"role": "user", "content": "hi"}])

    assert result.text == "hello world"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"
    assert result.provider_used == "groq"


def test_tool_call_parsed():
    p, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _make_groq_response(
        text="",
        tool_calls=[{"id": "tc1", "name": "bash", "input": {"command": "ls"}}],
    )

    result = p.complete([{"role": "user", "content": "list files"}])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].input == {"command": "ls"}
    assert result.stop_reason == "tool_use"


def test_system_prepended_as_message():
    p, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _make_groq_response()

    p.complete([{"role": "user", "content": "hi"}], system="Be concise")

    msgs = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs[0] == {"role": "system", "content": "Be concise"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_tools_converted_to_openai_format():
    p, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _make_groq_response()

    p.complete(
        [{"role": "user", "content": "hi"}],
        tools=[{"name": "bash", "description": "run bash", "input_schema": {"type": "object"}}],
    )

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["function"]["name"] == "bash"
    assert kwargs["tool_choice"] == "auto"


def test_anthropic_tool_history_converted_to_openai_format():
    """agent.py builds Anthropic-style history; Groq must get OpenAI-style or it 400s
    (live check 2026-10-14: 'messages.2.content must be a string')."""
    p, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _make_groq_response()

    p.complete([
        {"role": "user", "content": "search hotels"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Searching now."},
            {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"q": "hotels"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Hyatt $129"},
        ]},
    ])

    msgs = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs[0] == {"role": "user", "content": "search hotels"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Searching now."
    assert msgs[1]["tool_calls"] == [{
        "id": "t1", "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"q": "hotels"})},
    }]
    assert msgs[2] == {"role": "tool", "tool_call_id": "t1", "content": "Hyatt $129"}


def test_text_block_lists_flattened_to_string():
    p, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _make_groq_response()

    p.complete([{"role": "user", "content": [{"type": "text", "text": "hi"},
                                             {"type": "text", "text": "there"}]}])

    msgs = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs[0] == {"role": "user", "content": "hi\nthere"}


def test_gpt_oss_gets_low_reasoning_effort():
    """gpt-oss spends max_tokens on reasoning and returns empty content otherwise."""
    p, mock_client = _make_provider()
    p.model = "openai/gpt-oss-120b"
    mock_client.chat.completions.create.return_value = _make_groq_response()

    p.complete([{"role": "user", "content": "hi"}])

    assert mock_client.chat.completions.create.call_args[1]["reasoning_effort"] == "low"


def test_rate_limit_raises():
    import groq as _groq
    p, mock_client = _make_provider()
    mock_client.chat.completions.create.side_effect = _groq.RateLimitError(
        message="429", response=MagicMock(), body={}
    )

    try:
        p.complete([])
        assert False, "should have raised"
    except RateLimitError:
        pass
