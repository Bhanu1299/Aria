# Aria Project State

## Status: PHASE 4 Tasks 1-8 COMPLETE — Tasks 9-14 PLANNED
## Branch: main (Tasks 1-8 merged)

## NEXT: Start Phase 4 Tasks 9-14
Run: `git checkout -b feature/phase4-awareness-speed`
Plan is at: docs/superpowers/plans/2026-04-30-phase4-awareness-speed.md

## Phase 4 — Awareness & Speed (Tasks 9-14, PLANNED)

### 🔲 Task 9: Session Notes Compaction (compact.py)
### 🔲 Task 10: AutoDream — Background Memory Consolidation (auto_dream.py)
### 🔲 Task 11: Notifier (notifier.py)
### 🔲 Task 12: Voice Keyterms — STT Accuracy (voice_keyterms.py)
### 🔲 Task 13: Prompt Suggestions (prompt_suggester.py)
### 🔲 Task 14: Numpy Transcription — Faster STT

---

## Phase 4 — Jarvis Upgrades (COMPLETE, merged to main)

### ✅ Task 1: SleepGuard (DONE)
- New file: `sleep_guard.py` — SleepGuard class, caffeinate -i -t 300, 4-min restart, ref-counted
- Modified: `main.py` — `sleep_guard.acquire()` inside try, `_processing.clear()` + `sleep_guard.release()` in finally
- Tests: `tests/test_sleep_guard.py` — 12 tests passing

### ✅ Task 2: Session Notes (DONE)
- New file: `session_notes.py` — extract() + extract_async(), Groq llama-3.3-70b-versatile, bullet summaries
- Modified: `memory.py` — store_session_notes() APPENDS (not overwrites), get_session_notes(), clear_session_notes()
- Modified: `main.py` — fires session_notes.extract_async(transcript, answer) after speaker.say()
- Tests: `tests/test_session_notes.py` — 4 tests passing

### 🔲 Task 3: Memory Extractor — START HERE
- Create: `memory_extractor.py` — background user model builder
- Modify: `identity.json` — add "learned_facts" array
- Modify: `summarizer.py` — inject learned facts into answer_knowledge() system prompt
- Modify: `main.py` — fire memory_extractor.extract_async(transcript, answer) after session_notes call
- Reference: `claude-code-main/src/services/extractMemories/extractMemories.ts` and `prompts.ts`
- Tests: `tests/test_memory_extractor.py`

### 🔲 Task 4: Away Summary
- Create: `away_summary.py`
- Modify: `main.py` — call away_summary.speak_greeting(speaker) after menubar starts, before hotkey starts

### 🔲 Task 5: Streaming Progress
- Modify: `computer_use.py` — add progress_fn callback to research_loop() and run_loop()
- Modify: `main.py` — pass speaker.say as progress_fn

### 🔲 Task 6: VAD Auto-Stop
- Modify: `voice_capture.py` — add auto_stop RMS silence detection
- Modify: `main.py` — pass auto_stop=True to start_recording()

### 🔲 Task 7: Query Recovery
- Modify: `computer_use.py` — _with_retry(), stuck circuit breaker, budget warning

### 🔲 Task 8: Parallel Planner
- Modify: `planner.py` — _classify_dependencies(), ThreadPoolExecutor parallel batches
- Modify: `plan_context.py` — add parallel_results dict

## Architecture (Phase 4 in progress)
Voice → Whisper → handle_command() →
  [sleep_guard.acquire()]
  → planner (multi-step) OR single intent handler
  → speaker.say(answer)
  → session_notes.extract_async()   ← NEW Phase 4
  [finally: _processing.clear(), sleep_guard.release()]

## All files
- `requirements.txt`, `config.py`, `hotkey.py`, `voice_capture.py`, `transcriber.py`
- `router.py`, `browser.py`, `browser_profile.py`, `agent_browser.py`
- `computer_use.py` — DOM-first + vision fallback loops (Phase 3G)
- `dom_browser.py` — DOM snapshot helpers
- `briefing.py`, `app_launcher.py`, `summarizer.py`, `speaker.py`, `menubar.py`, `main.py`
- `mac_controller.py`, `media.py`, `identity.json`, `memory.py`, `db.py`
- `jobs.py`, `applicator.py`, `linkedin_applicator.py`, `tracker.py`
- `wake_word.py`, `scenes.json`, `scene_executor.py`
- `planner.py`, `plan_context.py`
- `sleep_guard.py`         ← NEW Phase 4 Task 1
- `session_notes.py`       ← NEW Phase 4 Task 2
- `skills/__init__.py`, `skills/skill_loader.py`, `skills/apply_status/__init__.py`, `skills/calculate/__init__.py`

## Known issues / risks
- Playwright worker thread: all Playwright calls must go via agent_browser.run() or navigate()
- LinkedIn browser profile was cleared (2026-04-02) — need re-login before next run
- Python 3.9 on this machine — no bare `str | None` without `from __future__ import annotations`
- 2 pre-existing test failures in test_jobs.py (unrelated to Phase 4)
- datetime.utcnow() deprecation warnings in memory.py — pre-existing, not blocking
