"""llm/base.py — shared types and exceptions for the LLM layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class RateLimitError(Exception):
    pass


class AuthError(Exception):
    pass


class GatewayUnavailable(Exception):
    """GateLLM isn't reachable (not running, or wrong ARIA_GATEWAY_URL)."""


class GatewayError(Exception):
    """GateLLM answered with an error (e.g. every upstream model in the tier failed)."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "error"
    provider_used: str
    model_used: str
    # Optional accounting (GatewayProvider fills it): cost_usd, tokens_in, tokens_out, fallbacks.
    meta: dict = field(default_factory=dict)


class Provider(Protocol):
    name: str
    disabled: bool

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        system: str | None,
    ) -> LLMResponse: ...
