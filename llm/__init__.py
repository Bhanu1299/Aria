"""llm — LLM provider layer with tier-based failover.

Usage:
    from llm import llm_client
    resp = llm_client.complete(messages, tier="cheap", system="...", max_tokens=100)
    print(resp.text)
"""
from __future__ import annotations

from llm.base import LLMResponse, ToolCall, RateLimitError, AuthError
from llm.client import LLMClient, ProviderChain
from llm.providers.anthropic import AnthropicProvider
from llm.providers.groq import GroqProvider

_TIER_CHAINS: dict[str, ProviderChain] = {
    "smart": ProviderChain(
        [
            AnthropicProvider("claude-sonnet-4-6"),
            AnthropicProvider("claude-haiku-4-5-20251001"),
            GroqProvider("llama-3.3-70b-versatile"),
        ],
        tier="smart",
    ),
    "cheap": ProviderChain(
        [
            AnthropicProvider("claude-haiku-4-5-20251001"),
            GroqProvider("llama-3.3-70b-versatile"),
        ],
        tier="cheap",
    ),
    "fast": ProviderChain(
        [
            GroqProvider("llama-3.3-70b-versatile"),
            AnthropicProvider("claude-haiku-4-5-20251001"),
        ],
        tier="fast",
    ),
}

llm_client = LLMClient(_TIER_CHAINS)

__all__ = ["llm_client", "LLMResponse", "ToolCall", "RateLimitError", "AuthError"]
