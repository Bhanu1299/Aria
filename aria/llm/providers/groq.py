"""llm/providers/groq.py — Groq wrapper (OpenAI-compatible API)."""
from __future__ import annotations

import json
import logging
import time

import groq as _groq

import aria.core.config as config
from aria.llm.base import LLMResponse, ToolCall, Provider, RateLimitError, AuthError

logger = logging.getLogger(__name__)


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool definitions to OpenAI-compatible format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return result


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, model: str) -> None:
        self.model = model
        self.disabled = False
        self._client: _groq.Groq | None = None

    def _get_client(self) -> _groq.Groq:
        if self._client is None:
            if not config.GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY not set")
            self._client = _groq.Groq(api_key=config.GROQ_API_KEY)
        return self._client

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": msgs,
        }
        if tools:
            kwargs["tools"] = _convert_tools(tools)
            kwargs["tool_choice"] = "auto"

        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kwargs)
        except _groq.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except _groq.AuthenticationError as exc:
            raise AuthError(str(exc)) from exc
        except Exception:
            raise

        latency = time.monotonic() - t0
        choice = resp.choices[0]
        raw_tool_calls = choice.message.tool_calls or []
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments),
            )
            for tc in raw_tool_calls
        ]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

        logger.debug(
            "[LLM] tier=- provider=groq model=%s tokens_in=%d tokens_out=%d latency=%.2fs",
            self.model, tokens_in, tokens_out, latency,
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            provider_used="groq",
            model_used=self.model,
        )
