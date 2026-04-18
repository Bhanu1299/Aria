# Aria Project State

## Status: PHASE 3G COMPLETE — DOM-first browser redesign shipped
## Phase 3F: COMPLETE — Agentic planner wired into main.py
## Phase 3E: COMPLETE — general browser research (Groq-first + Claude fallback)
## Phase 3D: COMPLETE (Session 1 + Session 2) — memory, wake word, scenes
## Phase 3C: COMPLETE — visible browser + coordinate-based computer use + stealth

## NEXT: Manual smoke test
Say "Search for AirPods Pro on Amazon and add it to my cart."
Watch logs — should see `DOM step N/M` lines instead of vision calls.
Branch `feature/dom-first-browser` ready to merge into master.

## Phase 3G — What was added (2026-04-18)

### Modified files
- `dom_browser.py` — added `get_dom_snapshot() -> tuple[str, int]`
- `computer_use.py` — added `_dom_decide()`, `_dom_research_decide()`,
  `_CU_DOM_SYSTEM`, `_CU_DOM_GENERAL_SYSTEM`, `_VALID_DOM_FORM_ACTIONS`,
  `_VALID_DOM_RESEARCH_ACTIONS`, `_DOM_TEXT_MODEL`; updated `execute()` for
  selector-based click/click_text/type; wired DOM-first path into `run_loop()`
  and `research_loop()`

### New test files
- `tests/test_dom_snapshot.py` — 5 tests for get_dom_snapshot()
- `tests/test_computer_use_dom.py` — 22 tests for all new DOM-first code

### Key design
```
get_dom_snapshot() → Groq text model (llama-3.3-70b-versatile)  ← PRIMARY
       ↓ interactive_count < 5 (CAPTCHA, canvas, unrendered SPA)
take_screenshot() → Groq vision model                           ← FALLBACK
       ↓ 2 consecutive stucks, no data
take_screenshot() → Claude vision                               ← LAST RESORT
```

Actions now use `{"action": "click", "selector": "#btn"}` instead of
`{"action": "click", "x": 640, "y": 450}` on the DOM path.
Vision fallback still returns coordinates — execute() detects by key presence.

## Architecture (Phase 3G)
Voice → Whisper → handle_command() → planner.is_multi_step()?
  YES → planner.run() → generate_plan → confirm → execute_plan (step by step)
        each step calls research_loop() or run_loop() → DOM-first execution
  NO  → route() → single intent handler

## All files
- `requirements.txt`, `config.py`, `hotkey.py`, `voice_capture.py`, `transcriber.py`
- `router.py`, `browser.py`, `browser_profile.py`, `agent_browser.py`
- `computer_use.py` — DOM-first + vision fallback loops (Phase 3G)
- `dom_browser.py` — DOM snapshot helpers + fill/click/screenshot helpers
- `briefing.py`, `app_launcher.py`, `summarizer.py`, `speaker.py`, `menubar.py`, `main.py`
- `mac_controller.py`, `media.py`, `identity.json`, `memory.py`, `db.py`
- `jobs.py`, `applicator.py`, `linkedin_applicator.py`, `tracker.py`
- `wake_word.py`, `scenes.json`, `scene_executor.py`
- `planner.py`, `plan_context.py`
- `skills/__init__.py`, `skills/skill_loader.py`, `skills/apply_status/__init__.py`, `skills/calculate/__init__.py`

## Known issues / risks
- Playwright worker thread: all Playwright calls must go via agent_browser.run() or navigate()
- LinkedIn browser profile was cleared (2026-04-02) — need re-login before next run
- Python 3.9 on this machine — no bare `str | None` without `from __future__ import annotations`
- 2 pre-existing test failures in test_jobs.py (unrelated to Phase 3G)
- Branch feature/dom-first-browser not yet merged to master
