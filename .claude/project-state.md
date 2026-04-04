# Aria Project State

## Status: PHASE 3D IN PROGRESS (Session 1 complete) — persistent memory + gap fixes
## Phase 3C: COMPLETE — visible browser + coordinate-based computer use + stealth

## Architecture (Phase 3C)
Voice → Whisper → router.py → handler → speaker.py

One persistent visible Chromium window for the whole session:
  agent_browser.py — singleton worker thread, opens on first job search, stays open
  computer_use.py  — coordinate-based see→think→act (like Claude computer use)

Handlers per intent:
- knowledge   → summarizer.answer_knowledge()
- web_search  → browser.extract_links() → browser.fetch() → summarizer.summarize()
- web_direct  → browser.fetch() → summarizer.summarize()
- navigate    → browser.goto() → speak "Opening [site]"
- app         → app_launcher.open_app() → speak result
- media       → media.handle_media_command()
- app_control → mac_controller.handle_app_command()
- briefing    → briefing.build_briefing()
- jobs        → jobs.search_jobs() →
                  agent_browser worker thread opens VISIBLE LinkedIn Jobs page
                  → computer_use.take_screenshot() → Groq vision extracts listings
                  → DOM query gets real LinkedIn job URLs
                  → memory.store_jobs() → speak results
                  → browser stays open
- apply       → applicator.run_application() →
                  agent_browser.navigate(job_url) — reuses open window
                  → computer_use.run_loop() — coordinate-based form fill
                  → voice confirmation → submit on yes
                  → tracker.log_application() → SQLite

## What exists (all files)
- `requirements.txt` — added playwright-stealth
- `config.py` — GROQ_API_KEY, CURRENT_LOCATION, BROWSER_TIMEOUT, check_permissions()
- `hotkey.py` — HotkeyListener
- `voice_capture.py` — VoiceCapture (16kHz sounddevice)
- `transcriber.py` — faster-whisper base, worker thread (ctranslate2 thread affinity)
- `router.py` — contact/apply/job pre-checks + Groq LLM classifier
- `browser.py` — headless BrowserExecutor for web_search/web_direct intents
- `browser_profile.py` — persistent Playwright profile at ~/.aria/browser_profile
- `agent_browser.py` — UPDATED: singleton visible browser on dedicated worker thread
                        (Playwright sync_api thread affinity fix), playwright-stealth applied
- `computer_use.py` — coordinate-based see→think→act; all calls via agent_browser.run()
- `briefing.py` — four concurrent fetchers + Groq assembler
- `app_launcher.py` — open macOS apps with optional contact
- `summarizer.py` — identity.json injected, hallucination guard
- `speaker.py` — macOS TTS
- `menubar.py` — rumps menubar status
- `main.py` — UPDATED: import agent_browser, close on shutdown
- `mac_controller.py` — 3-layer AppleScript/pyobjc/vision
- `media.py` — Apple Music + YouTube
- `identity.json` — name, email, phone, skills, experience, education
- `memory.py` — session dict; store_jobs(), get_job_by_index()
- `jobs.py` — REWRITTEN: visible browser via agent_browser, LinkedIn Jobs, DOM URL extraction
- `applicator.py` — REWRITTEN: computer_use.run_loop() coordinate-based, no DOM
- `tracker.py` — SQLite application log at ~/.aria/applications.db

## What to do next
1. Re-login to LinkedIn (profile was cleared): `python main.py --login linkedin`
2. Test: "Find me SWE jobs in New York" → LinkedIn opens visibly, 5 jobs with real URLs
3. Test: "Apply to the first one" → coordinate loop fills form
4. Say "yes" → check: `python3 -c "import tracker; print(tracker.get_applications())"`

## Known issues / risks
- Playwright worker thread: all Playwright calls must go via agent_browser.run() or navigate()
  Never pass a `page` object outside the worker thread
- LinkedIn browser profile was cleared (2026-04-02) — need re-login before next run
- Llama-4-Scout coordinate accuracy varies — small buttons may need retries
- LinkedIn Easy Apply is multi-step SPA — may need >30 steps for complex forms
- Do NOT make rapid repeated "find me jobs" requests — triggers LinkedIn rate limits
  Wait for voice readout to finish before saying next command

## Phase 3D — What was added (Session 1)

### New files
- `db.py` — centralised SQLite (aria.db). All DB access goes here.
- `skills/__init__.py` — package marker
- `skills/skill_loader.py` — auto-discover + match skills before Groq
- `skills/apply_status/__init__.py` — "what jobs have I applied to"
- `skills/calculate/__init__.py` — "calculate 5 times 8" (safe AST eval)

### Modified files
- `tracker.py` — now uses db.get_connection() (aria.db). Old applications.db deprecated.
- `memory.py` — threading.Lock + SQLite persistence. State survives restart.
- `jobs.py` — salary filter (f_SB2). "Find jobs paying 100k" works.
- `main.py` — store_last_search() called after job search; capability + skill handlers added.
- `router.py` — capability pre-check (no Groq); skill_loader check before Groq classifier.
- `summarizer.py` — full identity injection (name, email, location, skills, education).

### Known gaps fixed
- "What's my name?" → now uses identity.json (name, email, location, skills)
- "What can you do?" → hardcoded response, never hallucinates features
- Salary filter for job search ("find jobs paying 100k") added to jobs.py
- Last search query persists across restarts via memory.py → SQLite

### What to do next (Session 2)
- Phase 3D Session 2: Channel adapter layer (Telegram, iMessage, Discord)
  - Create channels/base.py, channels/telegram_channel.py, etc.
  - Create channels/channel_manager.py
  - Extract handle_command() from main.py _process_release() first

## Blocked on
Re-login to LinkedIn: `python main.py --login linkedin`
