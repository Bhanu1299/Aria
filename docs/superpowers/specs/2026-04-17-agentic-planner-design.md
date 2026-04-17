# Agentic Planner — Design Spec
**Date:** 2026-04-17
**Status:** Approved

---

## Overview

Aria currently handles one intent per voice command. This spec adds a planner layer that detects multi-step goals, generates a sequential plan, confirms it with the user, and executes each step — passing results forward as context, narrating progress, and recovering from failures before asking the user for help.

**Example:**
> "Book the cheapest flight to NYC next week and add it to my calendar"
→ Aria generates a 3-step plan, reads it out, waits for go-ahead, executes each step, narrates progress, speaks final summary.

---

## Architecture

Two new files. Zero changes to existing handlers.

```
main.py
  └── planner.is_multi_step(transcript)?
        YES → planner.run(transcript, speaker, voice_capture, transcriber)
        NO  → route() + _handle_intent()   ← existing flow, untouched
```

### New files
- **`planner.py`** — detection, plan generation, confirmation, execution loop
- **`plan_context.py`** — thin dataclass: step list + result dict + retry state

### Existing files changed
- **`main.py`** — one new check before routing (~10 lines)
- **`memory.py`** — store last plan for restart recovery (1 function)

### Untouched
`router.py`, `computer_use.py`, `jobs.py`, `applicator.py`, `browser.py`, `summarizer.py`, all handlers.

---

## Step Format

Each step is a dict:

```python
{
    "id": 1,
    "description": "Search Kayak for cheapest flight to NYC on April 25",
    "intent_type": "browser_task",     # must be a known Aria intent
    "params": {
        "browser_goal": "find cheapest flight to NYC April 25 on Kayak",
    },
    "result_key": "flight_result",     # key to store output in plan_context
    "depends_on": [],                  # step IDs whose results this step needs
}
```

`intent_type` always maps to an existing Aria intent: `browser_task`, `knowledge`, `web_search`, `jobs`, `apply`, `app_control`, `navigate`, `media`. The planner sequences existing capabilities — it never invents new ones.

---

## Plan Generation

**Groq call** with a tight prompt:
> *"Break this goal into 2–5 sequential steps. Each step must map to one of the known intent types. Return a JSON array only. No markdown."*

Hard constraints:
- **Min 2 steps** — single-step goals fall through to the existing router
- **Max 5 steps** — prevents runaway plans
- Any step with an unknown `intent_type` → reject entire plan, fall back to single-intent routing

---

## Plan Confirmation

After generation, Aria speaks the plan:
> *"Here's my plan: Step 1, search Kayak for the cheapest flight. Step 2, book it. Step 3, add it to your calendar. Should I go ahead?"*

User responses:
- **"Go ahead" / "yes"** → execute as-is
- **"Skip step 3"** → remove step 3, confirm revised plan, execute
- **"Use Google Flights instead of Kayak"** → Groq revises only the affected step, re-speaks that step, executes

Plan is only revised once before execution. Mid-execution changes happen via voice checkpoints (see below).

---

## Execution Loop

### Context passing
Each step's result is stored in `plan_context.results[result_key]`. Before each step runs, a lightweight Groq call injects relevant context into the step's params:

```
step 1 result → context["flight_result"] = "Delta $320, departs 8am Apr 25"
step 2 params → browser_goal = "book Delta flight $320 8am Apr 25 on Kayak" (context injected)
```

### Ambiguity rule
- **One clear result matching user's spec** → auto-advance, speak status only
- **Multiple options / something user didn't specify** → speak options, wait for voice pick before continuing

### Status narration
Between every step:
> *"Got it. Cheapest flight is Delta at $320. Moving to step 2, booking now…"*

Final summary after all steps:
> *"All done. Booked Delta for $320 on April 25, and added it to your calendar at 8am."*

### Retry logic
```
attempt 1 → fails
attempt 2 → Groq generates different approach (different site / rephrased query)
attempt 3 → Groq generates another different approach
all 3 fail → speak: "I tried 3 ways and got stuck on step 2. Should I skip it or try something else?"
```

Each retry passes the failure reason back to Groq so it generates a meaningfully different approach, not a blind repeat.

---

## `is_multi_step()` Detection

Two-layer, fast-first:

1. **Regex** — conjunction + multiple action verbs: `and then`, `after that`, `then`, `also`, `and add`, `and book`, `and send`. If matched AND >1 distinct action verb → multi-step.
2. **Borderline** → one cheap Groq call: *"Single task or multiple sequential tasks? Answer: single / multi"*

Single-intent commands never hit the planner.

---

## `plan_context.py`

```python
@dataclass
class PlanContext:
    goal: str
    steps: list[dict]
    results: dict[str, str]   # result_key → spoken result from that step
    current_step: int
    retry_count: int
```

Stored to `memory.py` (SQLite) so "what was I doing?" works across restarts.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Plan generation fails | Fall back to single-intent routing, no crash |
| Unknown intent_type in plan | Reject plan, fall back to single-intent |
| Step fails after 3 retries | Ask user: skip / try something else / abort |
| User says "stop" mid-plan | Abort immediately, speak what was completed. **No rollback** — completed steps (bookings, calendar entries, messages) are not undone. Aria warns before any irreversible step via the existing `confirm` checkpoint so the user has already opted in. Partial state (e.g. booked flight but no calendar entry) is the user's responsibility to resolve manually. |
| Ambiguous result | Voice checkpoint before advancing |
| Context injection fails | Use raw step params without injection, log warning |

---

## Files Changed Summary

| File | Change |
|---|---|
| `planner.py` | NEW — ~200 lines |
| `plan_context.py` | NEW — ~20 lines |
| `main.py` | +10 lines: multi-step check before routing |
| `memory.py` | +1 function: store/retrieve last plan |
