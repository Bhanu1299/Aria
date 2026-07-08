# Aria Architecture

Aria is a **background voice agent for macOS**: hotkey or wake word → speech → a
tool-using LLM agent → spoken answer. The screen never moves, the browser never
steals focus, and the whole loop runs while you keep working.

Three ideas shape everything below:

1. **One brain, many hands.** A single tool-loop Agent decides; ~43 tools across
   7 drop-in plugins act. There is no intent classifier, no routing table —
   deleted, on purpose (see [Design invariants](#design-invariants)).
2. **The user is never left guessing.** Every state Aria can be in is visible
   (menubar + on-screen HUD) or audible (chime, no-speech cue, "one moment").
3. **It measures itself.** Every command and every wake-word event is logged
   locally; a daily checklist turns those logs into a per-capability
   pass/fail report. Reliability work is driven by data, not vibes.

---

## The command lifecycle

```mermaid
flowchart TB
    subgraph triggers [" triggers "]
        HK["⌥ Space hotkey<br/>(pynput)"]
        WW["'Aria' wake word<br/>(Porcupine / custom ONNX / openwakeword)"]
    end

    subgraph capture [" capture & understand "]
        VAD["mic + VAD<br/>(sounddevice)"]
        WH["Whisper base<br/>(faster-whisper, local,<br/>preloaded at startup)"]
    end

    subgraph brain [" the brain "]
        AG["Agent tool loop<br/>(agent.py)"]
        REG["ToolRegistry<br/>~43 tools from 7 plugins"]
        LLM["llm/ failover chains<br/>Claude ⇄ Groq"]
    end

    subgraph out [" respond "]
        TTS["speaker.say()<br/>(macOS say, interruptible)"]
        CONV["conversation.hold()<br/>follow-up window, no re-trigger"]
    end

    subgraph async [" async extractors (daemon threads) "]
        EX["session notes · memory facts ·<br/>dream consolidation · suggestions"]
    end

    HK --> VAD
    WW --> VAD
    VAD --> WH --> AG
    AG <--> REG
    AG <--> LLM
    AG --> TTS --> CONV
    CONV -->|"follow-up speech"| AG
    TTS -.-> EX
```

A follow-up spoken into the `conversation.hold()` window goes straight back
into the Agent **with full cross-turn history** — "what about tomorrow?" just
works, no hotkey, no wake word.

## The Agent: one loop, no routing

`agent.py` is ~180 lines and is the only decision-maker. Each turn:

```mermaid
sequenceDiagram
    participant U as user (voice)
    participant A as Agent
    participant L as llm/ chain
    participant T as tool

    U->>A: transcript
    A->>L: system + history + ~43 tool schemas
    loop until end_turn — capped at 10 tool calls
        L-->>A: tool_use(name, input)
        A->>T: execute(input)
        T-->>A: result string (never raises)
        A->>L: tool_result
    end
    L-->>A: final text
    A-->>U: spoken answer (1–3 sentences)
```

- **History compaction** keeps multi-turn sessions inside the context window.
- **Memory injection**: relevant facts from the vector store are prepended to
  the system prompt per query (`plugins/memory/context_injector.py`).
- **Failure discipline**: a tool that throws becomes a readable error string in
  the transcript; the model gets a chance to recover, and `Agent.run()` itself
  can never raise into the voice loop.

### LLM failover tiers (`llm/`)

Every completion names a tier, not a model. Each tier is a provider chain that
falls through on rate limits, auth errors, or outages:

| tier | chain | used for |
|---|---|---|
| `smart` | Claude Sonnet 4.6 → Claude Haiku 4.5 → Llama-3.3-70B (Groq) | the Agent loop |
| `cheap` | Claude Haiku 4.5 → Llama-3.3-70B | extractors, summaries |
| `fast` | Llama-3.3-70B → Claude Haiku 4.5 | latency-sensitive helpers |

One provider being down degrades quality, never availability.

## Plugins: drop a folder in, get capabilities

A plugin is a folder under `plugins/` whose `__init__.py` defines a
`PluginBase` subclass. **That's the entire contract.** `plugins.discover()`
finds it at startup — no imports to add, no registration list to edit, and a
broken plugin is skipped with a log line instead of taking Aria down.

```mermaid
flowchart LR
    D["plugins.discover()<br/>(pkgutil scan)"] --> P1

    subgraph P1 [" phase 1 — service plugins "]
        CORE["core · 10 tools<br/>web, apps, jobs, coder"]
        MSG["messaging · 12 tools<br/>iMessage, WhatsApp, Telegram…"]
        MED["media · 7 tools"]
        SCR["screen · 2 tools<br/>look / point"]
        MEM["memory · ChromaDB wiring"]
        HLT["health · self_report"]
    end

    P1 --> AGENT["Agent created<br/>over the registry"]
    AGENT --> P2

    subgraph P2 [" phase 2 — agent-powered plugins "]
        PROD["productivity · 11 tools<br/>gmail, calendar, cron<br/>(cron runs prompts through the live Agent)"]
    end
```

A minimal plugin:

```python
# plugins/hue/__init__.py
from plugin import PluginBase
from tool import ToolDescriptor, ToolRegistry

class HuePlugin(PluginBase):
    def register(self, registry: ToolRegistry) -> None:
        registry.register(ToolDescriptor(
            name="lights_set",
            description="Set the room lights: on, off, or a brightness 0-100.",
            input_schema={"type": "object",
                          "properties": {"level": {"type": "string"}},
                          "required": ["level"]},
            execute=lambda p: _set_lights(p["level"]),
        ))
```

Restart Aria; the Agent can now control your lights. Plugins that need shared
services (browser, speaker, the Agent itself) override `from_context()` and
declare `requires_agent = True` to load in phase 2.

## Memory: four layers, one assistant that remembers

```mermaid
flowchart TB
    Q["user query"] --> INJ["context_injector<br/>top-k relevant facts → system prompt"]
    subgraph stores [" stores "]
        S1["session KV (memory.py)<br/>last jobs, caches — TTL'd"]
        S2["SQLite (~/.aria/aria.db)<br/>applications, counters, notes"]
        S3["ChromaDB vector store<br/>+ sentence-transformers<br/>long-term facts"]
    end
    ANS["answer spoken"] --> EXT["memory_extractor (async)<br/>new facts worth keeping?"]
    EXT --> S3
    DREAM["auto_dream (async)<br/>periodic consolidation:<br/>merge, dedupe, decay"] --> S3
    S3 --> INJ
```

Tell Aria your favorite coffee once; it's embedded, consolidated during
"dreams," and injected into context whenever coffee comes up — this session or
next month.

## Feedback: nothing happens silently

| you experience | source |
|---|---|
| pulsing pill at screen bottom whenever the mic is open | `listening_indicator.py` (click-through, never takes focus) |
| Glass chime on wake, Basso thunk if it woke but heard nothing | `wake_word.py` |
| soft pop when the follow-up window opens, "Anytime." on thanks | `conversation.py` |
| "One moment." when a turn runs past 3.5 s | `speaker.ThinkingAck` |
| menubar ◉ → 🎙 → ⏳ → ✓ | `menubar.py` (rumps) |

## Observability: the assistant that files its own bug reports

Two local-only JSONL logs, one report tool:

```mermaid
flowchart LR
    CMD["every command"] --> FR["flight_recorder<br/>~/.aria/flight_log.jsonl<br/>transcript · tools · duration · failed?"]
    WK["every wake event"] --> WS["wake_stats<br/>~/.aria/wake_log.jsonl<br/>detections · near-misses · heartbeat · restarts"]
    FR --> DC["daily_check.py report<br/>per-capability PASS / FAIL<br/>vs tests/daily_questions.json"]
    WS --> DW["daily_check.py wake<br/>engine alive? threshold suggestion"]
    DC --> FIX["fix the top offender"]
    DW --> FIX
    FIX -.->|"weekly loop"| CMD
```

- `daily_check.py list --core` — a ~3-minute spoken checklist covering every
  capability.
- `daily_check.py report` — matches the flight log against that checklist:
  what passed, what failed, what the mic actually heard, which tool broke.
- `daily_check.py wake` — is the wake engine alive, how many near-misses, and
  a data-driven threshold suggestion when misses outnumber hits.
- Ask Aria herself: *"how have you been performing?"* (health plugin reads the
  same log).

The wake-word listener runs under a supervisor loop: crashes restart with
exponential backoff and are logged — it cannot die silently.

## Design invariants

These are enforced, not aspirational:

1. **Never steal focus.** All Playwright work runs through a single worker
   thread (`agent_browser.run()` / `navigate()`); on-screen surfaces
   (`overlay.py`, `listening_indicator.py`) are borderless, click-through, and
   shown with `orderFrontRegardless`.
2. **Never raise into the voice loop.** Every handler returns a graceful
   spoken fallback. `Agent.run()`, `conversation.hold()`, both loggers, and
   every plugin tool are exception-proof at their boundary.
3. **One brain.** The intent classifier / planner stack (~2,100 lines) was
   deleted once the tool-loop Agent proved strictly better. Dead paths drift;
   drift ships bugs.
4. **Local-first.** Whisper, wake models, ChromaDB, both logs — all on-device.
   Only LLM completions leave the machine.
5. **Voice is the UI.** Answers are 1–3 spoken sentences; no markdown, no URL
   dumps. Long work is acknowledged out loud rather than silently spinning.
6. **Python 3.9 discipline.** `from __future__ import annotations` everywhere;
   no `str | None` at runtime.

## Repository map

```
main.py                 wiring + command lifecycle (the only god allowed)
agent.py                the tool loop
tool.py                 ToolDescriptor / ToolRegistry
plugin.py               PluginBase + PluginContext (the plugin contract)
plugins/                drop-in capability packs (core, media, messaging,
                        memory, productivity, screen, health)
llm/                    provider chains: Anthropic + Groq, tiered failover
conversation.py         follow-up window (the "Jarvis feel")
speaker.py              interruptible TTS + ThinkingAck
wake_word.py            3-backend wake engine under a restart supervisor
listening_indicator.py  on-screen listening HUD
transcriber.py / voice_capture.py / hotkey.py / menubar.py
memory.py / memory_extractor.py / auto_dream.py     memory layers
flight_recorder.py / wake_stats.py / daily_check.py observability
screen_qa.py / selection.py / overlay.py / vision.py screen intelligence
browser.py / agent_browser.py / computer_use.py      background browsing
training/               custom wake-word model pipeline (record → train → onnx)
tests/                  448 passing; tests/daily_questions.json drives the
                        daily voice checklist
```

## Testing strategy

- **Unit**: pure logic (VAD thresholds, end-phrase detection, report building,
  discovery ordering) — fast, no hardware.
- **Contract**: every tool returns a string and never raises; plugins load
  from a bare `PluginContext`.
- **On-device**: `tests/checklist.md` + the daily voice checklist exercise the
  real mic, TTS, screen permissions, and focus behavior — the things mocks
  can't prove.
- **Continuous**: normal daily use *is* the test run; the flight recorder
  scores it and `daily_check.py report` grades it.
