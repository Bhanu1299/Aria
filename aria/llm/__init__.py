"""llm — Aria's LLM layer. Every call goes through GateLLM (ARIA_GATEWAY_URL).

There are no direct provider calls anywhere in Aria: GateLLM owns providers,
model choice, fallback across vendors, retries and pricing.

Usage:
    from aria.llm import llm_client                  # agent-style (Anthropic history, tools)
    resp = llm_client.complete(messages, tier="cheap", system="...", max_tokens=100)

    from aria.llm import gateway_client               # OpenAI-shaped, for vision/classifiers
    r = gateway_client().chat.completions.create(model="vision", messages=[...], max_tokens=300)
"""
from __future__ import annotations

import aria.core.config as config
from aria.llm.base import (
    AuthError, GatewayError, GatewayUnavailable, LLMResponse, RateLimitError, ToolCall,
)
from aria.llm.client import LLMClient, ProviderChain
from aria.llm.providers.gateway import GatewayProvider, check_health, gateway_client

# Aria tier → GateLLM tier.
GATEWAY_TIERS = {"smart": "frontier", "cheap": "small", "fast": "small"}


def build_tier_chains(gateway_url: str) -> dict[str, ProviderChain]:
    return {
        tier: ProviderChain([GatewayProvider(gw_tier, gateway_url)], tier=tier)
        for tier, gw_tier in GATEWAY_TIERS.items()
    }


llm_client = LLMClient(build_tier_chains(config.GATEWAY_URL))

__all__ = [
    "llm_client", "gateway_client", "build_tier_chains", "check_health", "GATEWAY_TIERS",
    "LLMResponse", "ToolCall", "RateLimitError", "AuthError", "GatewayError", "GatewayUnavailable",
]
