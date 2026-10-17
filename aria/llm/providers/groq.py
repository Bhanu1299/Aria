"""llm/providers/groq.py — Groq wrapper (OpenAI-compatible API)."""
from __future__ import annotations

import json
import logging
import time

import groq as _groq

import aria.core.config as config
from aria.llm.base import LLMResponse, ToolCall, Provider, RateLimitError, AuthError
from aria.llm.models import groq_params

logger = logging.getLogger(__name__)


def _block_text(content) -> str:
    """Flatten Anthropic content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)


def _convert_messages(messages: list[dict]) -> list[dict]:
    """Anthropic-style history (what agent.py builds) → OpenAI-style for Groq.

    assistant tool_use blocks → assistant.tool_calls
    user tool_result blocks   → one role="tool" message each
    lists of text blocks      → a single string
    """
    out: list[dict] = []
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        blocks = [b for b in (content or []) if isinstance(b, dict)]
        if role == "assistant":
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
                }
                for b in blocks if b.get("type") == "tool_use"
            ]
            converted: dict = {"role": "assistant", "content": _block_text(blocks) or None}
            if calls:
                converted["tool_calls"] = calls
            out.append(converted)
            continue
        results = [b for b in blocks if b.get("type") == "tool_result"]
        for b in results:
            out.append({
                "role": "tool",
                "tool_call_id": b["tool_use_id"],
                "content": _block_text(b.get("content", "")),
            })
        text = _block_text(blocks)
        if text or not results:
            out.append({"role": role, "content": text})
    return out


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
        msgs = _convert_messages(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": msgs,
            **groq_params(self.model),
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
