"""tests/test_llm_gateway.py — GateLLM is Aria's only path to an LLM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm import meter
from aria.llm.base import GatewayError, GatewayUnavailable
from aria.llm.client import ProviderChain
from aria.llm.providers.gateway import GatewayChatClient, GatewayProvider

_HEADERS = {
    "x-gateway-model": "groq/gpt-oss-120b",
    "x-gateway-provider": "groq",
    "x-gateway-fallbacks": "0",
    "x-gateway-cost-usd": "0.000123",
}


def _completion(content="hello", tool_calls=None, prompt=10, completion=5):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "groq/gpt-oss-120b",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def _http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gw.test")


def _provider(handler, model="small"):
    return GatewayProvider(model, base_url="http://gw.test", client=_http(handler))


def _ok(request):
    return httpx.Response(200, json=_completion(), headers=_HEADERS)


# ---------------------------------------------------------------------------
# GatewayProvider (agent)
# ---------------------------------------------------------------------------

def test_sends_tier_converted_messages_and_tools():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return _ok(request)

    _provider(handler, model="frontier").complete(
        [{"role": "user", "content": "hi"}],
        tools=[{"name": "music_pause", "description": "Pause.", "input_schema": {"type": "object"}}],
        max_tokens=200,
        system="You are Aria.",
    )
    body = seen["body"]
    assert seen["path"] == "/v1/chat/completions"
    assert body["model"] == "frontier"
    assert body["max_tokens"] == 200
    assert body["messages"][0] == {"role": "system", "content": "You are Aria."}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["tools"][0]["function"]["name"] == "music_pause"
    assert body["tool_choice"] == "auto"
    # Model-specific params (gpt-oss reasoning_effort) are the gateway's job.
    assert "reasoning_effort" not in body


def test_converts_anthropic_tool_history_to_openai():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok(request)

    history = [
        {"role": "user", "content": "pause"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "music_pause", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Paused."},
        ]},
    ]
    _provider(handler).complete(history, max_tokens=50)
    msgs = seen["body"]["messages"]
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "music_pause"
    assert msgs[2] == {"role": "tool", "tool_call_id": "t1", "content": "Paused."}


def test_parses_text_tool_calls_and_gateway_meta():
    calls = [{
        "id": "c1", "type": "function",
        "function": {"name": "timer_set", "arguments": '{"seconds": 300}'},
    }]

    def handler(request):
        return httpx.Response(200, json=_completion(content=None, tool_calls=calls), headers=_HEADERS)

    resp = _provider(handler).complete([{"role": "user", "content": "timer"}], max_tokens=50)
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "timer_set"
    assert resp.tool_calls[0].input == {"seconds": 300}
    assert resp.provider_used == "gateway:groq"
    assert resp.model_used == "groq/gpt-oss-120b"
    assert resp.meta["cost_usd"] == pytest.approx(0.000123)
    assert resp.meta["tokens_in"] == 10 and resp.meta["tokens_out"] == 5


def test_malformed_tool_arguments_become_empty_dict():
    calls = [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{not json"}}]

    def handler(request):
        return httpx.Response(200, json=_completion(content=None, tool_calls=calls), headers=_HEADERS)

    resp = _provider(handler).complete([{"role": "user", "content": "x"}], max_tokens=50)
    assert resp.tool_calls[0].input == {}


@pytest.mark.parametrize("status", [401, 429, 502, 503])
def test_gateway_http_errors_raise_gateway_error(status):
    p = _provider(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(GatewayError):
        p.complete([{"role": "user", "content": "x"}], max_tokens=50)


def test_gateway_down_raises_unavailable():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(GatewayUnavailable):
        _provider(handler).complete([{"role": "user", "content": "x"}], max_tokens=50)


def test_chain_speaks_clearly_when_gateway_is_down():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    resp = ProviderChain([_provider(handler)], tier="smart").complete(
        [{"role": "user", "content": "x"}], None, 50, None)
    assert resp.stop_reason == "error"
    assert "gateway" in resp.text.lower()


def test_every_tier_is_gateway_only():
    from aria.llm import GATEWAY_TIERS, build_tier_chains

    chains = build_tier_chains("http://gw.test")
    for tier, gw_tier in GATEWAY_TIERS.items():
        providers = chains[tier].providers
        assert [p.name for p in providers] == ["gateway"], tier
        assert providers[0].model == gw_tier


# ---------------------------------------------------------------------------
# GatewayChatClient (modules: vision, classifiers, computer use)
# ---------------------------------------------------------------------------

def test_chat_client_passes_openai_messages_and_images_through():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion("A login page."), headers=_HEADERS)

    client = GatewayChatClient("http://gw.test", client=_http(handler))
    image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
    resp = client.chat.completions.create(
        model="vision",
        messages=[{"role": "user", "content": [{"type": "text", "text": "what is this"}, image]}],
        max_tokens=300,
        temperature=0.2,
    )
    assert resp.choices[0].message.content == "A login page."
    assert seen["body"]["model"] == "vision"
    assert seen["body"]["messages"][0]["content"][1] == image
    assert seen["body"]["temperature"] == 0.2


def test_every_gateway_call_is_metered():
    before = meter.snapshot()
    _provider(_ok).complete([{"role": "user", "content": "x"}], max_tokens=50)
    GatewayChatClient("http://gw.test", client=_http(_ok)).chat.completions.create(
        model="small", messages=[{"role": "user", "content": "y"}])
    used = meter.since(before)
    assert used["calls"] == 2
    assert used["tokens_in"] == 20
    assert used["cost_usd"] == pytest.approx(0.000246)


def test_no_module_calls_a_provider_sdk_directly():
    """Each and every LLM call goes through GateLLM — no Groq/Anthropic SDK use in aria/."""
    root = Path(__file__).parent.parent / "aria"
    offenders = [
        str(p.relative_to(root.parent))
        for p in root.rglob("*.py")
        for line in p.read_text().splitlines()
        if line.strip().startswith(("import groq", "from groq", "import anthropic", "from anthropic"))
    ]
    assert offenders == []
