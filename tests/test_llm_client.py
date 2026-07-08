"""tests/test_llm_client.py — LLMClient tier routing and failover."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm.base import LLMResponse, ToolCall, RateLimitError, AuthError
from aria.llm.client import LLMClient, ProviderChain


def _ok_response(provider="anthropic", model="claude-haiku-4-5-20251001"):
    return LLMResponse(
        text="ok", tool_calls=[], stop_reason="end_turn",
        provider_used=provider, model_used=model,
    )


def _mock_provider(name, response=None, side_effect=None):
    p = MagicMock()
    p.name = name
    p.disabled = False
    if side_effect:
        p.complete.side_effect = side_effect
    else:
        p.complete.return_value = response or _ok_response(name)
    return p


# ---------------------------------------------------------------------------
# Tier routing
# ---------------------------------------------------------------------------

def test_smart_tier_uses_first_provider():
    p1 = _mock_provider("anthropic")
    p2 = _mock_provider("groq")
    chain = ProviderChain([p1, p2], tier="smart")
    client = LLMClient({"smart": chain})

    result = client.complete([{"role": "user", "content": "hi"}], tier="smart")

    assert result.provider_used == "anthropic"
    p1.complete.assert_called_once()
    p2.complete.assert_not_called()


def test_fast_tier_uses_groq_first():
    groq_p = _mock_provider("groq", _ok_response("groq", "llama-3.3-70b-versatile"))
    ant_p = _mock_provider("anthropic")
    chain = ProviderChain([groq_p, ant_p], tier="fast")
    client = LLMClient({"fast": chain, "smart": ProviderChain([ant_p], tier="smart")})

    result = client.complete([], tier="fast")

    assert result.provider_used == "groq"
    ant_p.complete.assert_not_called()


def test_unknown_tier_falls_back_to_smart():
    p = _mock_provider("anthropic")
    client = LLMClient({"smart": ProviderChain([p], tier="smart")})

    result = client.complete([], tier="nonexistent")

    assert result.provider_used == "anthropic"


# ---------------------------------------------------------------------------
# Failover
# ---------------------------------------------------------------------------

def test_rate_limit_falls_over_to_next():
    p1 = _mock_provider("anthropic", side_effect=RateLimitError("429"))
    p2 = _mock_provider("groq", _ok_response("groq"))
    chain = ProviderChain([p1, p2], tier="smart")
    client = LLMClient({"smart": chain})

    result = client.complete([])

    assert result.provider_used == "groq"


def test_timeout_falls_over_to_next():
    p1 = _mock_provider("anthropic", side_effect=TimeoutError("timeout"))
    p2 = _mock_provider("groq", _ok_response("groq"))
    chain = ProviderChain([p1, p2], tier="smart")
    client = LLMClient({"smart": chain})

    result = client.complete([])

    assert result.provider_used == "groq"


def test_auth_error_disables_provider():
    p1 = _mock_provider("anthropic", side_effect=AuthError("401"))
    p2 = _mock_provider("groq", _ok_response("groq"))
    chain = ProviderChain([p1, p2], tier="smart")
    client = LLMClient({"smart": chain})

    result = client.complete([])

    assert result.provider_used == "groq"
    assert p1.disabled is True


def test_all_providers_fail_returns_error_response():
    p1 = _mock_provider("anthropic", side_effect=RateLimitError("429"))
    p2 = _mock_provider("groq", side_effect=RateLimitError("429"))
    chain = ProviderChain([p1, p2], tier="smart")
    client = LLMClient({"smart": chain})

    result = client.complete([])

    assert result.stop_reason == "error"
    assert result.provider_used == "none"
    assert "trouble" in result.text


def test_disabled_provider_is_skipped():
    p1 = _mock_provider("anthropic")
    p1.disabled = True
    p2 = _mock_provider("groq", _ok_response("groq"))
    chain = ProviderChain([p1, p2], tier="smart")
    client = LLMClient({"smart": chain})

    result = client.complete([])

    assert result.provider_used == "groq"
    p1.complete.assert_not_called()
