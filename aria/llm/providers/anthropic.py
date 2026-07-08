"""llm/providers/anthropic.py — Anthropic Claude wrapper."""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import anthropic as _anthropic

import aria.core.config as config
from aria.llm.base import LLMResponse, ToolCall, Provider, RateLimitError, AuthError

logger = logging.getLogger(__name__)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str) -> None:
        self.model = model
        self.disabled = False
        self._client: _anthropic.Anthropic | None = None

    def _get_client(self) -> _anthropic.Anthropic:
        if self._client is None:
            if not config.ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return self._client

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        t0 = time.monotonic()
        try:
            resp = client.messages.create(**kwargs)
        except _anthropic.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except _anthropic.AuthenticationError as exc:
            raise AuthError(str(exc)) from exc
        except Exception:
            raise

        latency = time.monotonic() - t0
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        text = next((b.text for b in resp.content if b.type == "text"), "")
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "output_tokens", 0) if usage else 0

        logger.debug(
            "[LLM] tier=- provider=anthropic model=%s tokens_in=%d tokens_out=%d latency=%.2fs",
            self.model, tokens_in, tokens_out, latency,
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
            provider_used="anthropic",
            model_used=self.model,
        )
