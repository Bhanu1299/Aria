# Phase 5F — LLM Provider Layer Design

**Date:** 2026-05-06  
**Status:** Approved  
**Depends on:** Phase 5A (core engine)  
**Unlocks:** Transparent failover, centralized model management, cost-aware routing

---

## Goal

Replace hardcoded `anthropic.Anthropic()` and `groq.Groq()` calls scattered across the codebase with a single `LLMClient`. Failover happens automatically when a provider is rate-limited or down. Calling code never changes when providers do.

---

## Architecture

### New file: `llm.py`

Single module at the root level — not a plugin, because the engine itself (`agent.py`) depends on it.

```
llm.py
  LLMClient          ← singleton, used everywhere
  ProviderChain      ← ordered list of providers with fallback logic
  providers/
    anthropic.py     ← Claude wrapper
    groq.py          ← Groq wrapper
```

Wait — providers live in `llm/` subdirectory:

```
llm/
  __init__.py        ← exposes llm_client singleton
  client.py          ← LLMClient, ProviderChain
  providers/
    anthropic.py
    groq.py
```

---

## Model tiers

Not every task needs the best model. Three tiers control which provider is tried first:

| Tier | First choice | Fallback | Used by |
|---|---|---|---|
| `smart` | Claude Sonnet | Claude Haiku → Groq Llama 70B | agent.py main loop, coder, computer_use |
| `cheap` | Claude Haiku | Groq Llama 70B | agent_summary, away_summary, memory_extractor, compact |
| `fast` | Groq Llama 70B | Claude Haiku | router decisions, tips, prompt_suggester |

Tier is passed as a hint: `llm_client.complete(messages, tier="smart")`. Default is `smart`.

---

## LLMClient interface

```python
class LLMClient:
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        tier: str = "smart",
        max_tokens: int = 1024,
        system: str = None,
    ) -> LLMResponse

class LLMResponse:
    text: str                    # final text content (may be empty if tools used)
    tool_calls: list[ToolCall]   # populated when model wants to use tools
    stop_reason: str             # "end_turn" | "tool_use" | "error"
    provider_used: str           # "anthropic" | "groq" (for logging)
    model_used: str              # exact model ID
```

`ToolCall`:
```python
class ToolCall:
    id: str
    name: str
    input: dict
```

---

## Provider chain and failover

```python
class ProviderChain:
    def __init__(self, providers: list[Provider]):
        self.providers = providers

    async def complete(self, messages, tools, max_tokens, system) -> LLMResponse:
        last_error = None
        for provider in self.providers:
            try:
                return await provider.complete(messages, tools, max_tokens, system)
            except RateLimitError:
                logger.warning(f"{provider.name} rate limited, trying next")
                last_error = "rate_limited"
                continue
            except TimeoutError:
                logger.warning(f"{provider.name} timed out, trying next")
                last_error = "timeout"
                continue
            except AuthError:
                logger.error(f"{provider.name} auth failed — skipping permanently")
                provider.disabled = True
                continue
            except Exception as e:
                logger.error(f"{provider.name} unexpected error: {e}")
                last_error = str(e)
                continue

        return LLMResponse(
            text="I'm having trouble connecting right now. Please try again.",
            tool_calls=[],
            stop_reason="error",
            provider_used="none",
            model_used="none"
        )
```

**Tier chains:**
```python
TIER_CHAINS = {
    "smart": [
        AnthropicProvider("claude-sonnet-4-6"),
        AnthropicProvider("claude-haiku-4-5-20251001"),
        GroqProvider("llama-3.3-70b-versatile"),
    ],
    "cheap": [
        AnthropicProvider("claude-haiku-4-5-20251001"),
        GroqProvider("llama-3.3-70b-versatile"),
    ],
    "fast": [
        GroqProvider("llama-3.3-70b-versatile"),
        AnthropicProvider("claude-haiku-4-5-20251001"),
    ],
}
```

---

## Provider implementations

### AnthropicProvider

Wraps existing `anthropic` SDK usage. Normalizes tool format to `LLMResponse`.

```python
class AnthropicProvider(Provider):
    name = "anthropic"

    async def complete(self, messages, tools, max_tokens, system) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = self._client.messages.create(**kwargs)

        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        text = next(
            (b.text for b in resp.content if b.type == "text"), ""
        )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
            provider_used="anthropic",
            model_used=self.model,
        )
```

Raises `RateLimitError` on HTTP 429, `AuthError` on HTTP 401/403, `TimeoutError` on request timeout.

### GroqProvider

Same pattern. Groq uses OpenAI-compatible API so tool format differs — normalize to `LLMResponse` the same way.

```python
class GroqProvider(Provider):
    name = "groq"

    async def complete(self, messages, tools, max_tokens, system) -> LLMResponse:
        msgs = messages.copy()
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": msgs,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments),
            )
            for tc in (choice.message.tool_calls or [])
        ]

        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            provider_used="groq",
            model_used=self.model,
        )
```

---

## What changes in existing files

Every file that calls Claude or Groq directly gets ~3 lines changed:

**Before:**
```python
client = anthropic.Anthropic()
resp = client.messages.create(model="claude-haiku-4-5", ...)
```

**After:**
```python
from llm import llm_client
resp = llm_client.complete(messages, tier="cheap")
```

Files affected: `coder.py`, `agent_summary.py`, `summarizer.py`, `computer_use.py`, `jobs.py`, `compact.py`, `briefing.py`, `memory_extractor.py`, `away_summary.py`, `auto_dream.py`, `prompt_suggester.py`.

---

## Singleton

`llm/__init__.py` exposes a module-level singleton:

```python
from llm.client import LLMClient, TIER_CHAINS
llm_client = LLMClient(TIER_CHAINS)
```

All imports: `from llm import llm_client`. No instantiation in calling code.

---

## Logging

Every `complete()` call logs at DEBUG level:
```
[LLM] tier=smart provider=anthropic model=claude-sonnet-4-6 tokens_in=842 tokens_out=156 latency=1.2s
```

On failover:
```
[LLM] anthropic rate_limited → trying groq
[LLM] tier=smart provider=groq model=llama-3.3-70b-versatile tokens_in=842 tokens_out=201 latency=0.4s
```

---

## Error handling

| Scenario | Behavior |
|---|---|
| Single provider rate limited | Transparent failover to next, no user impact |
| All providers rate limited | Speak: "I'm having trouble connecting right now" |
| Auth error on a provider | Disable that provider for session, log error, move on |
| Timeout (>15s) | Fail to next provider |
| Network offline | All providers fail → graceful error message |

---

## Testing

- `tests/test_llm_client.py` — tier routing, failover on rate limit, failover on timeout, auth error disables provider, all-fail graceful response
- `tests/test_llm_anthropic.py` — response normalization, tool call parsing
- `tests/test_llm_groq.py` — response normalization, tool format conversion

---

## Dependencies

```
# Already in requirements.txt:
anthropic>=0.25.0
groq>=0.5.0
```

No new dependencies.

---

## Definition of done

- [ ] `llm/client.py` with `LLMClient`, `ProviderChain`, `LLMResponse`, `ToolCall`
- [ ] `llm/providers/anthropic.py` normalizing Anthropic responses
- [ ] `llm/providers/groq.py` normalizing Groq responses
- [ ] `llm_client` singleton in `llm/__init__.py`
- [ ] All 11 affected files migrated to `llm_client.complete()`
- [ ] Tier routing tested: smart/cheap/fast chains correct
- [ ] Failover tested: rate limit, timeout, auth error, all-fail
- [ ] Existing tests still pass after migration
