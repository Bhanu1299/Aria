"""llm/providers/gateway.py — GateLLM (~/GateLLM), Aria's only path to any LLM.

Every LLM call in Aria goes through here, over GateLLM's OpenAI-compatible HTTP
API. The gateway owns model choice, cross-provider fallback, retries and model
quirks (e.g. gpt-oss reasoning_effort). `model` is a gateway tier ("small",
"frontier", "vision") or an explicit gateway model id.

  GatewayProvider    — aria/llm Provider for the agent (Anthropic-style history in)
  gateway_client()   — OpenAI-SDK-shaped client (`.chat.completions.create`) for
                       modules that build OpenAI-format messages (vision, classifiers)

Errors: GatewayUnavailable (not running) / GatewayError (answered with an error).
Every successful call is tallied in aria.llm.meter.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx

from aria.llm import autostart, context, meter
from aria.llm.base import GatewayError, GatewayUnavailable, LLMResponse, Provider, ToolCall

logger = logging.getLogger(__name__)

# Connect fails fast so a stopped gateway is reported immediately; the read
# budget covers the gateway's own retries and fallbacks.
_TIMEOUT = httpx.Timeout(90.0, connect=1.0)


# ---------------------------------------------------------------------------
# Anthropic-style history (what agent.py builds) → OpenAI format
# ---------------------------------------------------------------------------

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
    """assistant tool_use blocks → assistant.tool_calls; user tool_result blocks →
    one role="tool" message each; lists of text blocks → a single string."""
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
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _float_header(headers: httpx.Headers, name: str) -> float:
    try:
        return float(headers.get(name, ""))
    except ValueError:
        return 0.0


def _send(client: httpx.Client, payload: dict) -> httpx.Response:
    """POST, starting GateLLM on demand if it's down (one retry)."""
    headers = context.headers()  # task / job / run labels for GateLLM's per-call log
    try:
        return client.post("/v1/chat/completions", json=payload, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        if not autostart.ensure_running(str(client.base_url)):
            raise GatewayUnavailable(f"GateLLM not reachable at {client.base_url}: {exc}") from exc
    try:
        return client.post("/v1/chat/completions", json=payload, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise GatewayUnavailable(f"GateLLM not reachable at {client.base_url}: {exc}") from exc


def _post(client: httpx.Client, payload: dict) -> tuple[dict, dict]:
    """POST one chat completion. → (response json, meta). Tallies the meter."""
    try:
        with autostart.in_flight():
            resp = _send(client, payload)
    except httpx.TransportError as exc:
        raise GatewayError(f"GateLLM transport error: {exc}") from exc
    if resp.status_code >= 400:
        raise GatewayError(f"GateLLM HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    usage = data.get("usage") or {}
    meta = {
        "cost_usd": _float_header(resp.headers, "x-gateway-cost-usd"),
        "tokens_in": int(usage.get("prompt_tokens") or 0),
        "tokens_out": int(usage.get("completion_tokens") or 0),
        "fallbacks": int(_float_header(resp.headers, "x-gateway-fallbacks")),
        "provider": resp.headers.get("x-gateway-provider", "unknown"),
        "model": resp.headers.get("x-gateway-model") or data.get("model", payload.get("model")),
    }
    meter.add(meta)
    logger.debug(
        "[LLM] gateway request=%s model=%s provider=%s cost=$%.6f",
        payload.get("model"), meta["model"], meta["provider"], meta["cost_usd"],
    )
    return data, meta


def _http_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=_TIMEOUT)


def check_health(base_url: str) -> str | None:
    """None if GateLLM answers /health, else a short reason. Never raises."""
    try:
        r = httpx.get(base_url.rstrip("/") + "/health", timeout=2.0)
        return None if r.status_code == 200 else f"HTTP {r.status_code}"
    except Exception as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# Agent provider
# ---------------------------------------------------------------------------

class GatewayProvider(Provider):
    name = "gateway"

    def __init__(self, model: str, base_url: str, client: httpx.Client | None = None) -> None:
        self.model = model
        self.disabled = False
        self._client = client or _http_client(base_url)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        msgs = _convert_messages(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        payload: dict = {"model": self.model, "max_tokens": max_tokens, "messages": msgs}
        if tools:
            payload["tools"] = _convert_tools(tools)
            payload["tool_choice"] = "auto"

        data, meta = _post(self._client, payload)
        message = data["choices"][0]["message"]
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], input=args))
        return LLMResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            provider_used=f"gateway:{meta['provider']}",
            model_used=meta["model"],
            meta={k: meta[k] for k in ("cost_usd", "tokens_in", "tokens_out", "fallbacks")},
        )


# ---------------------------------------------------------------------------
# OpenAI-SDK-shaped client for modules
# ---------------------------------------------------------------------------

class _Completions:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def create(self, *, model: str, messages: list[dict], task: str | None = None, **params) -> SimpleNamespace:
        """Same call shape as openai/groq `chat.completions.create`; returns an object
        with `.choices[0].message.content` and `.usage`. `task` labels the call in
        GateLLM's per-call log and is not sent to the model."""
        payload = {"model": model, "messages": messages,
                   **{k: v for k, v in params.items() if v is not None}}
        with context.scope(task=task or f"model:{model}"):
            data, meta = _post(self._client, payload)
        choices = [
            SimpleNamespace(
                index=c.get("index", 0),
                finish_reason=c.get("finish_reason"),
                message=SimpleNamespace(
                    role=c["message"].get("role", "assistant"),
                    content=c["message"].get("content"),
                    tool_calls=c["message"].get("tool_calls"),
                ),
            )
            for c in data.get("choices", [])
        ]
        usage = SimpleNamespace(prompt_tokens=meta["tokens_in"], completion_tokens=meta["tokens_out"])
        return SimpleNamespace(choices=choices, usage=usage, model=meta["model"], meta=meta)


class GatewayChatClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.chat = SimpleNamespace(completions=_Completions(client or _http_client(base_url)))


_SHARED: GatewayChatClient | None = None


def gateway_client() -> GatewayChatClient:
    """Process-wide client for ARIA_GATEWAY_URL."""
    global _SHARED
    if _SHARED is None:
        import aria.core.config as config
        _SHARED = GatewayChatClient(config.GATEWAY_URL)
    return _SHARED
