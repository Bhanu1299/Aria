"""llm/client.py — LLMClient and ProviderChain with tier-based failover."""
from __future__ import annotations

import logging
import time

from aria.llm.base import LLMResponse, ToolCall, Provider, RateLimitError, AuthError

logger = logging.getLogger(__name__)

_ERROR_RESPONSE = LLMResponse(
    text="I'm having trouble connecting right now. Please try again.",
    tool_calls=[],
    stop_reason="error",
    provider_used="none",
    model_used="none",
)

_TIMEOUT_SECONDS = 15.0


class ProviderChain:
    def __init__(self, providers: list[Provider], tier: str) -> None:
        self.providers = providers
        self.tier = tier

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        system: str | None,
    ) -> LLMResponse:
        for provider in self.providers:
            if provider.disabled:
                continue
            try:
                result = provider.complete(messages, tools, max_tokens, system)
                logger.debug(
                    "[LLM] tier=%s provider=%s model=%s stop=%s",
                    self.tier, result.provider_used, result.model_used, result.stop_reason,
                )
                return result
            except RateLimitError:
                logger.warning("[LLM] %s rate_limited → trying next", provider.name)
                continue
            except TimeoutError:
                logger.warning("[LLM] %s timed out → trying next", provider.name)
                continue
            except AuthError:
                logger.error("[LLM] %s auth failed — disabling for session", provider.name)
                provider.disabled = True
                continue
            except Exception as exc:
                logger.error("[LLM] %s unexpected error: %s", provider.name, exc)
                continue

        logger.error("[LLM] all providers failed for tier=%s", self.tier)
        return _ERROR_RESPONSE


class LLMClient:
    def __init__(self, tier_chains: dict[str, ProviderChain]) -> None:
        self._chains = tier_chains

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tier: str = "smart",
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        chain = self._chains.get(tier) or self._chains["smart"]
        return chain.complete(messages, tools, max_tokens, system)
