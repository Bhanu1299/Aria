# Phase 5A — Core Engine Design

**Date:** 2026-05-06  
**Status:** Approved  
**Depends on:** Phase 4B (complete)  
**Unlocks:** All Phase 5B–5F subsystems

---

## Goal

Replace `main.py`'s `if/elif` intent chain with a proper tool-loop agent. Every capability — existing and future — registers as a `Tool`. The LLM decides which tools to call. `router.py` is retired. One architecture for everything.

---

## Architecture

### New files

**`tool.py`**  
Defines the plugin contract. Everything that Aria can do is a `Tool`.

```
ToolDescriptor
  name: str                    # "briefing", "imessage_send", etc.
  description: str             # what the LLM reads to decide when to use it
  input_schema: dict           # JSON Schema for parameters
  execute(params) -> str       # async, returns text result
  availability: Callable       # optional; returns bool at runtime

ToolRegistry
  register(tool: ToolDescriptor)
  get(name: str) -> ToolDescriptor
  all_available() -> list[ToolDescriptor]   # filters by availability()
```

**`agent.py`**  
The tool-loop. Called by `main.py` with transcribed text, returns final spoken response.

```
Agent
  registry: ToolRegistry
  llm: LLMClient               # Phase 5F; uses existing clients until then

  run(text: str) -> str
    1. Build tool list from registry.all_available()
    2. Call LLM with system prompt + user text + tool descriptors
    3. If stop_reason == "tool_use":
         execute tool → append result → loop
    4. Return final text response
```

Max tool calls per turn: 10 (config: `AGENT_MAX_TOOL_CALLS`, default 10). On exceed, return partial result with note.

**`plugin.py`**  
Base class every subsystem implements.

```
PluginBase (ABC)
  register(registry: ToolRegistry)   # called once at startup
```

---

## What happens to existing files

Every existing capability becomes a `Tool` implementation. The files themselves barely change — they get a thin wrapper.

| Existing file | Becomes |
|---|---|
| `briefing.py` | `tools/briefing_tool.py` wraps `briefing.run()` |
| `coder.py` | `tools/coder_tool.py` — coder loop becomes a tool action |
| `jobs.py` | `tools/jobs_tool.py` |
| `app_launcher.py` | `tools/app_tool.py` |
| `media.py` | `tools/media_tool.py` (until Phase 5E replaces it) |
| `computer_use.py` | `tools/computer_use_tool.py` |
| `mac_controller.py` | `tools/mac_tool.py` |
| `scene_executor.py` | `tools/scene_tool.py` |
| `summarizer.py` | `tools/summarize_tool.py` |
| `away_summary.py` | stays — lifecycle, not a tool |
| `auto_dream.py` | stays — lifecycle, not a tool |
| `session_notes.py` | stays — lifecycle, not a tool |
| `memory_extractor.py` | stays — lifecycle, not a tool |
| `router.py` | **retired** — LLM is the router now |
| `planner.py` | **retired** — multi-step handled by tool loop |
| `plan_context.py` | **retired** — agent loop handles context |

---

## `main.py` after the change

```python
# startup
registry = ToolRegistry()
for plugin in load_plugins():
    plugin.register(registry)

agent = Agent(registry)

# command loop (unchanged externally)
def handle_command(text: str) -> str:
    return agent.run(text)
```

All existing lifecycle hooks (sleep_guard, session_notes, auto_dream, prompt_suggester, notifier) stay wired exactly as they are — they wrap `handle_command`, not replace it.

---

## Plugin loading

Plugins are discovered from `plugins/` directory at startup. Each subfolder with an `__init__.py` that exports a `Plugin(PluginBase)` class is loaded automatically. Order: core tools first, then plugins alphabetically.

```
plugins/
  core/
    __init__.py    ← CorePlugin — registers all existing tool wrappers
  messaging/       ← Phase 5B
  productivity/    ← Phase 5C
  memory/          ← Phase 5D
  media/           ← Phase 5E
```

---

## System prompt

The agent uses a single system prompt defined in `agent.py`:

```
You are Aria, a personal voice assistant running on macOS.
The user speaks to you — your response will be read aloud.
Keep answers short and conversational. No markdown.
Use tools when needed. You may chain multiple tools.
Today is {date}. User: {identity.name}.
```

Memory context injection (Phase 5D) appends relevant facts below this.

---

## Error handling

- Tool raises exception → catch, log, return `"I ran into an issue with {tool_name}: {short error}"` as tool result, continue loop
- LLM returns no tool call and no text → return `"I didn't understand that. Could you rephrase?"`
- Max tool calls exceeded → return whatever partial result exists + `"I hit my step limit on that one"`

---

## Testing

- `tests/test_agent.py` — tool loop unit tests: single tool call, multi-step chain, error recovery, max calls limit
- `tests/test_tool_registry.py` — register, lookup, availability filtering
- All 33 existing tests pass unchanged (tool implementations are in same files)

---

## Dependencies

```
# No new pip dependencies for core engine
# existing: anthropic, groq (until Phase 5F unifies them)
```

---

## Definition of done

- [ ] `tool.py`, `agent.py`, `plugin.py` written and tested
- [ ] All existing capabilities wrapped as tools in `plugins/core/`
- [ ] `router.py`, `planner.py`, `plan_context.py` retired
- [ ] `main.py` simplified to plugin load + agent.run()
- [ ] All 33 existing tests still pass
- [ ] New tests: agent loop, registry, plugin loading
