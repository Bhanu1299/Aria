"""llm/client.py — LLMClient and ProviderChain with tier-based failover."""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Optional

from aria.llm import meter
from aria.llm import context
from aria.llm.tasks import ESCALATE_TO, FRONTIER, tier_for

from aria.llm.base import (
    AuthError, GatewayError, GatewayUnavailable, LLMResponse, Provider, RateLimitError, ToolCall,
)

logger = logging.getLogger(__name__)

_ERROR_RESPONSE = LLMResponse(
    text="I'm having trouble connecting right now. Please try again.",
    tool_calls=[],
    stop_reason="error",
    provider_used="none",
    model_used="none",
)

_GATEWAY_DOWN_RESPONSE = LLMResponse(
    text="I can't reach the LLM gateway. Start GateLLM and try again.",
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
            except GatewayUnavailable as exc:
                logger.error("[LLM] tier=%s: %s", self.tier, exc)
                return _GATEWAY_DOWN_RESPONSE
            except GatewayError as exc:
                logger.error("[LLM] tier=%s gateway error: %s", self.tier, exc)
                return _ERROR_RESPONSE
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


# A GateLLM model id ("deepseek-v4-pro", "groq/gpt-oss-120b"), not a tier name.
_MODEL_ID = re.compile(r"^[a-z0-9][\w.-]*(?:[-/][\w.-]+)+$")


class LLMClient:
    def __init__(self, tier_chains: dict[str, ProviderChain],
                 model_chain: Optional[Callable[[str], ProviderChain]] = None) -> None:
        """model_chain(model_id) builds a chain for an explicit GateLLM model id (a task in
        TASK_TIERS may name one instead of a tier); without it an unknown name falls back to smart."""
        self._chains = tier_chains
        self._model_chain = model_chain

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tier: str | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
        task: str | None = None,
    ) -> LLMResponse:
        """task (preferred) looks the tier up in aria.llm.tasks.TASK_TIERS; tier is the raw override."""
        if task is not None:
            tier = tier_for(task)
        tier = tier or "smart"
        chain = self._chains.get(tier)
        if chain is None and self._model_chain is not None and _MODEL_ID.match(tier):
            chain = self._chains[tier] = self._model_chain(tier)      # an explicit model: its own chain
        if chain is None:
            logger.warning("[LLM] no chain for tier %r — using smart", tier)
            chain = self._chains["smart"]
        # An enclosing task label (e.g. complete_checked's) wins over the bare tier.
        label = task or (None if context.current().get("task") else f"tier:{tier}")
        with context.scope(task=label):
            return chain.complete(messages, tools, max_tokens, system)

    def complete_checked(
        self,
        task: str,
        messages: list[dict],
        validate: Callable[[str], Any],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> tuple[LLMResponse, Any]:
        """Run `task` on its tier; validate(text) returns the parsed value or None.

        Invalid output is retried ONCE on the next tier up (ESCALATE_TO: small → strong;
        strong / vision → frontier), counted in aria.llm.meter. Returns (last response, value | None).
        """
        tier = tier_for(task)
        with context.scope(task=task):
            resp = self.complete(messages, tier=tier, max_tokens=max_tokens, system=system)
        value = _validated(validate, resp)
        up = ESCALATE_TO.get(tier, FRONTIER)
        if value is not None or tier == FRONTIER:
            return resp, value
        why = "gateway error" if resp.stop_reason == "error" else "output failed validation"
        logger.warning("[LLM] task=%s: %s %s — escalating to %s", task, tier, why, up)
        meter.add_escalation()
        with context.scope(task=task, escalated_from=f"{tier}: {why}"):
            resp = self.complete(messages, tier=up, max_tokens=max_tokens, system=system)
        return resp, _validated(validate, resp)


def _validated(validate: Callable[[str], Any], resp: LLMResponse) -> Any:
    if resp.stop_reason == "error":
        return None
    try:
        return validate(resp.text or "")
    except Exception as exc:
        logger.debug("[LLM] validator raised: %s", exc)
        return None
