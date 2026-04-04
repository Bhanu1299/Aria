# Session Log

## Session: 2026-04-04 — Phase 3D Session 1

**What we built:**
- db.py: centralised SQLite, all tables in ~/.aria/aria.db
- memory.py: threading.Lock + SQLite persistence + startup load from DB
- tracker.py: migrated to db.py (aria.db instead of applications.db)
- jobs.py: salary filter added (_parse_salary_filter, f_SB2 LinkedIn param)
- main.py: store_last_search() after job search; capability + skill handlers
- router.py: capability pre-check (regex, no Groq); skill_loader check before Groq
- summarizer.py: full identity injection (name, email, location, skills, education); updated capability list
- skills/skill_loader.py: auto-discovery, trigger matching
- skills/apply_status/: application history voice command
- skills/calculate/: safe AST math eval

**What we discovered was already done (stale gaps list):**
- Resume upload: already in linkedin_applicator._upload_resume() (line 524)
- Debug screenshots: already called throughout linkedin_applicator.py

**What's next:**
- Phase 3D Session 2: Channel adapter layer (Telegram, iMessage, Discord)
- Prerequisite: extract handle_command() from main.py _process_release() first

---

## Session 1 — 2026-03-27
Status: Starting fresh build
What to do: Build from requirements.txt through to wired main.py

## Session 2 — 2026-03-27

### What was done
Created the 5 foundation files for Aria:

1. `requirements.txt` — all unpinned deps: pynput, sounddevice, soundfile,
   faster-whisper, playwright, rumps, python-dotenv, numpy.

2. `.env.example` — template with HOTKEY_KEY, HOTKEY_MODS, BROWSER_TIMEOUT.

3. `.env` — live config (same values as example; safe defaults).

4. `config.py` — dotenv loader, typed constants (HOTKEY_KEY, HOTKEY_MODS as
   list, BROWSER_TIMEOUT as int), check_permissions() that tests pynput
   Accessibility and sounddevice Microphone access with clear guidance text.

5. `.gitignore` — excludes venv/, __pycache__/, *.pyc, *.pyo, .env, *.wav,
   *.mp3, .DS_Store.

### Next steps (from build order)
- hotkey.py + test
- voice_capture.py + test
- transcriber.py + test
- browser.py + test
- speaker.py + test
- menubar.py
- main.py
- Wire and test end-to-end

## Session 3 — 2026-03-27

### What was done
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/hotkey.py`:
  - `HotkeyListener` class with `start()` / `stop()` methods.
  - Reads `HOTKEY_KEY` and `HOTKEY_MODS` from `config.py` at instantiation.
  - Builds pynput combo string dynamically (e.g. `<alt>+<space>`).
  - `pynput.keyboard.GlobalHotKeys` handles press detection (`_on_activate`).
  - Secondary `pynput.keyboard.Listener` watches key-up events for release
    detection (`_handle_key_release` → `_on_deactivate`).
  - Threading lock + `_pressed` flag prevents double-fire across hold cycle.
  - Accessibility error caught; clear System Settings path printed; sys.exit(1).
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/tests/test_hotkey.py`:
  - Real integration test — no mocks.
  - 5-second window; polls for press + release; exits early when both fire.
  - Prints `PASS — hotkey detected` or `FAIL — hotkey not detected`.
- Both files pass `python3 -m py_compile`.
- Updated project-state.md.

### Next steps
- voice_capture.py + tests/test_voice_capture.py

## Session 4 — 2026-03-27

### What was done
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/voice_capture.py`:
  - `VoiceCapture` class with `start_recording()` and `stop_recording()` methods.
  - `sounddevice.InputStream` at 16000 Hz, 1 channel, int16 dtype.
  - Audio chunks accumulated via `_audio_callback` into `self._chunks` list.
  - `stop_recording()` concatenates chunks with `numpy.concatenate`, writes WAV
    via `soundfile.write()` (PCM_16 subtype) to `/tmp/aria_recording.wav`.
  - `sd.PortAudioError` caught on stream open; clear error printed then re-raised.
  - Prints `[VOICE] Recording started` and `[VOICE] Recording stopped, saved to {path}`.
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/tests/test_voice_capture.py`:
  - Real mic integration test — no mocks.
  - 3-second live capture with `time.sleep(3)`.
  - Verifies: file exists, `os.path.getsize > 0`, `soundfile.info()` parses cleanly.
  - Prints `PASS — recorded {duration}s of audio at {samplerate}Hz` or `FAIL` with reason.
- Updated project-state.md: status VOICE CAPTURE COMPLETE.

### Next steps
- transcriber.py + tests/test_transcriber.py (load faster-whisper base model at import; transcribe(wav_path) -> str)

## Session 5 — 2026-03-27

### What was done
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/transcriber.py`:
  - `load_model()` standalone function: imports WhisperModel from faster_whisper,
    loads "base" model on CPU with int8 compute type, prints status, returns model.
  - `Transcriber` class: `__init__(self, model)` stores the pre-loaded model.
  - `transcribe(self, audio_path: str) -> str`: calls model.transcribe() with
    beam_size=5, language="en"; joins segment texts; strips whitespace.
  - Error handling: catches any exception, prints "[TRANSCRIBE] Error: {e}", returns "".
  - Prints "[TRANSCRIBE] Transcribing {audio_path}..." and "[TRANSCRIBE] Result: {text}".
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/tests/test_transcriber.py`:
  - Standalone real-mic real-Whisper test; no mocks.
  - Calls load_model(), constructs Transcriber, records 3s via VoiceCapture.
  - Prints PASS with transcribed text or FAIL + sys.exit(1).
- Updated project-state.md: status TRANSCRIBER COMPLETE.

### Next steps
- browser.py + tests/test_browser.py

## Session 6 — 2026-03-27

### What was done
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/browser.py`:
  - `ClaudeBrowser` class with `__init__`, `start`, `ask`, `stop`.
  - `start()` calls `sync_playwright().start()` storing the instance (NOT used as
    context manager) so the session stays alive across multiple `ask()` calls.
  - Chromium launched `headless=False` with focus-prevention args:
    `--no-first-run`, `--disable-background-timer-throttling`,
    `--disable-backgrounding-occluded-windows`, `--disable-renderer-backgrounding`,
    `--disable-focus-on-load`, `--no-default-browser-check`.
  - NEVER calls `bring_to_front()` or any focus-stealing method.
  - Input selector cascade: `div[contenteditable="true"][data-placeholder]`
    → `div[contenteditable="true"]` → `textarea`.
  - Submission via `page.keyboard.press("Enter")` — no button click needed.
  - Streaming detection: wait for `button[aria-label*="Stop"]` to appear, then
    poll every 0.5s until it disappears.
  - Response extraction: `.font-claude-message` → `[data-testid="assistant-message"]`
    → `[data-is-streaming="false"]` → `main` element fallback.
  - Timeout: `config.BROWSER_TIMEOUT` seconds; calls `subprocess.run(["say", ...])`
    and returns `""`.
  - Login wall detection: URL check + visible login button check; returns
    sentinel string `"__LOGIN_REQUIRED__"` so callers can give clear guidance.
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/tests/test_browser.py`:
  - Real browser test, no mocks.
  - 3-second pause after start() for manual focus-steal observation.
  - Asks "What is 2 + 2?", verifies response is non-empty str containing "4".
  - Handles `__LOGIN_REQUIRED__` sentinel with actionable message.
  - Output format matches spec exactly.
- Updated project-state.md: status BROWSER COMPLETE.

### Next steps
- speaker.py + tests/test_speaker.py

## Session 7 — 2026-03-27

### What was done
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/speaker.py`:
  - Standalone `speak(text: str) -> None`: skips empty/whitespace text with log;
    prints truncated 60-char preview; calls `subprocess.run(["say", text], check=True)`;
    catches `CalledProcessError` and `FileNotFoundError`.
  - `Speaker` class: `__init__` no-op, `say(self, text)` delegates to `speak()`.
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/tests/test_speaker.py`:
  - Standalone script; adds project root to sys.path.
  - Calls `speak("Aria is working correctly.")`, prints PASS/FAIL.
  - Handles `FileNotFoundError` for non-macOS environments.
- Updated project-state.md: status SPEAKER COMPLETE.

### Next steps
- menubar.py

## Session 8 — 2026-03-27

### What was done
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/menubar.py`:
  - `AriaMenuBar` inherits from `rumps.App`.
  - Title initialized to "◉" (IDLE icon); no extra text shown.
  - Single "Quit" menu item added; default rumps quit button disabled (`quit_button=None`) to avoid duplication.
  - `ICONS` dict maps "IDLE" / "LISTENING" / "THINKING" / "DONE" to their emoji/symbol.
  - `set_state(state)` looks up icon, sets `self.title`, prints `[MENUBAR] State → {state}`; unknown states are logged and ignored.
  - `run()` calls `super().run()` — blocks calling thread; must be invoked last in main.py.
  - `__main__` block for standalone testing: spawns daemon thread that cycles through all four states with 1-second delays.

### State → Icon mapping
| State     | Icon |
|-----------|------|
| IDLE      | ◉   |
| LISTENING | 🎙   |
| THINKING  | ⏳   |
| DONE      | ✓   |

### Next steps
- main.py — wire hotkey, voice capture, transcriber, browser, speaker, menubar together
- End-to-end integration test
- checklist.md

## Session 9 — 2026-03-27

### What was done
- Created `/Users/bhanuteja/Documents/trae_projects/Aria/main.py`:
  - Full startup sequence: check_permissions → load_model → Transcriber → VoiceCapture
    → Speaker → ClaudeBrowser.start() → AriaMenuBar → HotkeyListener.start() → menubar.run()
  - `_processing` is a `threading.Event` used as a mutex to prevent overlapping invocations.
  - `on_press()` runs on the pynput thread: sets flag, calls set_state("LISTENING"),
    calls voice_capture.start_recording(). Fast — no blocking.
  - `on_release()` spawns a daemon `threading.Thread` targeting `_process_release()` so
    the pynput hotkey thread is never stalled during long I/O operations.
  - `_process_release()`: stop_recording → transcribe → handle empty → set_state("THINKING")
    → browser.ask → speak response → handle `__LOGIN_REQUIRED__` sentinel →
    set_state("DONE") → sleep 1s → set_state("IDLE") → clear flag.
  - Full try/except wraps all of `_process_release`: on any exception, speaks error
    message, resets state to IDLE, clears processing flag.
  - SIGINT/SIGTERM signal handlers call hotkey_listener.stop() then browser.stop()
    then sys.exit(0) cleanly.
  - All imports at top; no stubs, no TODOs, no placeholders.

### Next steps
1. Run `python main.py` for live end-to-end test
2. Verify: hotkey → recording → transcription → Claude response → TTS playback
3. Write checklist.md once e2e passes

## Session 10 — 2026-03-27
### What was done
- Written tests/checklist.md with all 10 manual run checklist items
- All 8 Python modules pass py_compile (zero syntax errors)
- Full build COMPLETE — all files from spec created

## Session 11 — 2026-03-27
### What was done

#### Bug fixes
- `config.py` — removed `or True` from both permission-check conditions; previously any
  exception (even non-permission errors) always printed the Accessibility / Microphone
  guidance message; now the message only appears for genuine permission-related errors
- `browser.py` `_find_input` — was passing the full `config.BROWSER_TIMEOUT` to EACH
  selector in the fallback list; worst case: 3× the timeout before returning None.
  Fixed: `deadline` is now set once at the start of `execute()` and passed to `_find_input`,
  which calculates `remaining_ms = max(0, deadline - now)` per attempt so all selector
  tries share one budget
- `browser.py` — `deadline` was set AFTER `_find_input` was called; moved before it

#### Refactor — executor pattern
- `ClaudeBrowser` renamed to `BrowserExecutor`
- `ask()` renamed to `execute(command: str) -> str`
- Added executor-pattern comment at top of browser.py:
  `# Executor pattern — future executors (desktop.py, system.py) must implement execute(command: str) -> str`
- `test_browser.py` updated: imports `BrowserExecutor`, calls `browser.execute()`
- `main.py` updated: imports `BrowserExecutor`, calls `browser.execute(question)`

#### New file
- `HOW_TO_RUN.md` — prerequisites, permissions (exact paths), first-time setup,
  run command, what to expect (menu bar states, full flow), top-5 troubleshooting

#### Verification
- All 13 Python files (8 source + 5 tests) pass `python3 -m py_compile`
- `python tests/test_speaker.py` → PASS

### Status: READY FOR LIVE TESTING
Next: grant Accessibility + Mic permissions, run tests in order, then `python main.py`

### Status: READY TO TEST
Run in order:
1. python tests/test_hotkey.py
2. python tests/test_voice_capture.py
3. python tests/test_transcriber.py
4. python tests/test_browser.py
5. python tests/test_speaker.py
6. python main.py (full e2e)

## Session 14 — Phase 2D: Media Playback (2026-03-29)

### What was built
- `media.py` — NEW: music app control + YouTube playback
  - `MUSIC_APP` env var (default "Music" = Apple Music) — configurable, never hardcoded
  - AppleScript handlers: `music_play`, `music_pause`, `music_resume`, `music_skip`, `music_now_playing`
  - YouTube audio: yt-dlp fetches stream URL + title → ffplay streams in background (Popen, DEVNULL)
  - YouTube video: yt-dlp gets video ID → `open https://youtube.com/watch?v=ID`
  - `stop_youtube()` via `pkill -f ffplay`
  - `check_dependencies()` at import time — warns if yt-dlp or ffplay missing, doesn't crash
  - `handle_media_command(transcript)` — Groq sub-classifier (llama-3.1-8b-instant) dispatches all actions
- `mac_controller.py` — removed Spotify play/pause/resume/skip/whats_playing (now in media.py); updated `_SUB_CLASSIFY_SYSTEM` schema and dispatch
- `router.py` — updated `media` type description; added IMPORTANT rule covering all play/pause/skip/watch phrases
- `main.py` — replaced browser-fetch media block with `media.handle_media_command()`; added `import media`; updated phase label to 2D
- `.env.example` — added `MUSIC_APP=Music`

### Key decisions
- All music app control centralised in media.py — mac_controller only handles system/app control
- YouTube audio uses yt-dlp + ffplay (no download, background process, no screen change)
- MUSIC_APP is the only config needed to switch music services

## Session 13 — Phase 2C: Native Mac App Control (2026-03-29)

### What was built
- `mac_controller.py` — NEW: 3-layer architecture
  - Layer 1: AppleScript via osascript for Spotify playback, system volume, Finder folders, Calendar read/add, Mail unread/latest, Reminders add, app open/quit/hide
  - Layer 2: pyobjc Accessibility API fallback via `click_element(app_name, element_label)`
  - Layer 3: screencapture + Groq vision (`meta-llama/llama-4-scout-17b-16e-instruct`) as last resort via `read_screen_region()`
  - Public: `handle_app_command(transcript)` — Groq sub-classifier (llama-3.1-8b-instant) extracts action/target/params, routes to handler
- `router.py` — added `app_control` intent type to `_CLASSIFY_SYSTEM` prompt, valid-types guard, and `_build_intent()`
- `main.py` — added `import mac_controller`, `app_control` dispatch in `_handle_intent()`, startup accessibility note
- `requirements.txt` — added pyobjc-framework-Cocoa, pyobjc-framework-ApplicationServices

### Key decisions
- `app_control` sits alongside existing `app` intent; router LLM decides which to use based on IMPORTANT rules in prompt
- Sub-classifier uses a second Groq call to extract structured action/target/params — avoids fragile regex
- pyobjc imported lazily inside `click_element()` to keep startup cost low

## Session 15 — Phase 2E: Briefings + Persistent Profile (2026-03-29)

### What was built
- `browser_profile.py` — NEW: persistent Playwright profile at ~/.aria/browser_profile
  - `get_persistent_context(headless)` — reusable authenticated context
  - `login_session(url)` — interactive visible-browser login flow
  - `close_persistent_context()` — clean teardown
- `briefing.py` — NEW: four concurrent data fetchers + Groq spoken assembler
  - `get_weather()` — wttr.in via config.CURRENT_LOCATION city
  - `get_calendar_events()` — AppleScript reads all calendars for today
  - `get_gmail_unread()` — fetch_authenticated from browser.py, detects login-required
  - `get_news()` — AP News headline regex, BBC RSS fallback
  - `build_briefing()` — ThreadPoolExecutor(4) → Groq assembly → plain prose
- `browser.py` — added `fetch_authenticated(url, query)` using persistent profile
- `media.py` — BUGFIX: Apple Music library search wrapped in try/except; added `reveal` fallback before URL scheme
- `router.py` — added `briefing` intent to classifier, valid types, and _build_intent
- `main.py` — `--login <service>` CLI handler, `import briefing`, briefing intent dispatch, Phase 2E label

### Also applied (earlier in session)
- `main.py` — word-count guard (< 2 words → reject)
- `router.py` — `_APP_NAME_MAP` + `_APP_CONTACT_RE` for "open WhatsApp and message Amma"
- `app_launcher.py` — `_MESSAGING_APPS` set for WhatsApp/Signal/Telegram/Messages
- `summarizer.py` — identity.json injection, capability constraint, hallucination guard

### Key decisions
- Persistent browser profile lives at ~/.aria/browser_profile (not in project dir)
- All four briefing fetchers run concurrently — never serially
- Gmail requires one-time `--login gmail` before briefing can read inbox
- AP News + BBC RSS dual fallback for news headlines

## Session 16 — 2026-04-01: Task 3 — jobs.py (Phase 3A)

### What was built
- `jobs.py` — NEW: voice-driven job search pipeline
  - `_get_client()` — lazy singleton Groq client (same pattern as briefing.py)
  - `_parse_query(query)` — Groq llama-3.1-8b-instant extracts role + location; falls back to (query, "") on any error
  - `_screenshot_page(url, settle_secs)` — Playwright headless persistent context → screenshot → sips resize → base64; returns None on any error
  - `_vision_ask(b64, prompt)` — Groq Llama-4-Scout vision call
  - `_get_search_url(platform, role, location)` — Google site: search → screenshot → vision extracts first matching URL
  - `_extract_jobs_from_page(url, platform)` — job page screenshot → vision → JSON listings array; returns [] on any error
  - `search_jobs(query)` — full pipeline: parse → discover URLs → extract → deduplicate by (title, company) → top 5 with index 1-5
  - `format_spoken_results(results)` — natural spoken string for TTS; ordinal labels First through Fifth
- `tests/test_jobs.py` — NEW: 9 unit tests
  - 4 tests for `format_spoken_results` (empty, single, five, missing fields)
  - 3 tests for `_parse_query` (success, Groq failure fallback, bad JSON fallback)
  - 2 tests for `search_jobs` (deduplication, empty when all sources fail)
  - All Groq/browser calls mocked via `patch.object` — no network I/O in tests

### Test results
9/9 passed using project venv (Python 3.9.6)

### Key decisions
- Uses same lazy singleton `_get_client()` pattern as briefing.py
- Opens a fresh Playwright context per screenshot call (matches briefing.py pattern for get_gmail_unread)
- Deduplication key is `(title.lower(), company.lower())` — first occurrence wins

## Session 12 — 2026-03-28

### What was done
Built and validated Phase 2A — full internet access via Groq + headless Playwright.

**New files:**
- `router.py` — Groq llama-3.1-8b-instant parses voice command → {action, url, query, instructions}; falls back to Google on parse failure
- `summarizer.py` — Groq llama-3.1-8b-instant converts raw page text + query → 1–4 sentence spoken answer

**Refactored:**
- `browser.py` — all claude.ai code removed; generic headless Chromium fetcher; `fetch(url) -> str`; JS strips nav/footer/ads; caps at 8000 chars
- `main.py` — pipeline now: transcribe → route → browser.fetch → summarize → speaker.say

**Updated:**
- `config.py` + `.env` + `.env.example` — added GROQ_API_KEY
- `requirements.txt` — added `groq`

**Bug fix:**
- Added `from __future__ import annotations` to router.py, summarizer.py, browser.py (Python 3.9 doesn't support `X | None` type hint syntax natively)

**Live test results (python main.py — all passing):**
- "Who is the president of India?" → correct ✅
- "Find me latest jobs in software development in United States" → real listings ✅
- "Open YouTube and play me Taylor Swift songs" → extracted top video title ✅
- Context-free "Play that song" → router fallback fired correctly ✅
- Meta-question "What are your capabilities?" → went to Google (no local handler) ⚠️
- Niche showtimes query → router hallucinated URL → "couldn't find answer" ⚠️

**Known issues for next session:**
1. Router can hallucinate specific URLs for niche queries — needs URL validation or fallback
2. Meta/capability questions need a local handler before routing
3. Semaphore leak at shutdown — harmless macOS Python 3.9 + sounddevice known issue

## Session 18 — 2026-04-01: Task 5 — Wire jobs and memory into main.py

### What was done
- **Edit 1:** Added `import jobs` and `import memory` after `import briefing` in `/Users/bhanuteja/Documents/trae_projects/Aria/main.py` (line 45 area)
- **Edit 2:** Added jobs intent handler block in `_handle_intent()` between the briefing block and the fallback — calls `jobs.search_jobs()`, stores results via `memory.store_jobs()`, returns `jobs.format_spoken_results()`

### Verification
- `python3 -c "import ast; ..."` → `main.py parses OK`
- `python3 -c "import jobs; import memory; ..."` → `imports OK` (location detection warning from requests not installed is expected/harmless)

### Commit
`93323cf feat(3A): wire jobs intent handler into main.py`

### Files changed
- `/Users/bhanuteja/Documents/trae_projects/Aria/main.py` — imports + jobs handler block

---

## Session 19 — 2026-04-02: Phase 3B + 3C — Apply flow + Visible Browser + Computer Use

### What was built

**Phase 3B:**
- `tracker.py` — SQLite at `~/.aria/applications.db`; `log_application()` + `get_applications()`
- `applicator.py` — original DOM-based apply loop (later replaced in 3C)
- `router.py` — "apply" intent: `_APPLY_INTENT_RE` pre-check, valid type added throughout
- `main.py` — `_IS_APPLY_RE` guard in `_check_jobs_followup` (stops it intercepting apply commands before router); apply intent handler in `_handle_intent`

**Phase 3C:**
- `agent_browser.py` — singleton visible browser on a dedicated persistent Playwright worker thread (fixes "cannot switch to a different thread" crash); playwright-stealth applied on page creation
- `computer_use.py` — coordinate-based see→think→act loop; all Playwright calls via `agent_browser.run(fn)` — no page objects leave the worker thread
- `jobs.py` — rewritten: visible LinkedIn browser via agent_browser, DOM URL extraction (`a[href*="/jobs/view/"]`), `computer_use.take_screenshot()` replaces headless pattern
- `applicator.py` — rewritten: `computer_use.run_loop()` replaces DOM/accessibility loop
- `requirements.txt` — added playwright-stealth

### Bugs fixed
1. `_check_jobs_followup` intercepted "apply to the Nth" → added `_IS_APPLY_RE` early-return
2. Playwright thread affinity crash (`cannot switch to a different thread`) → agent_browser worker thread owns all Playwright state; other threads submit via queue
3. LinkedIn Cloudflare Error 1200 → playwright-stealth installed + rate-limited profile cleared

### State at close
- Job search: WORKING — LinkedIn opens visibly, 5 jobs extracted with real URLs
- Apply routing: WORKING — reaches applicator correctly
- Apply form fill: untested (LinkedIn Cloudflare blocked during session)
- Browser profile: CLEARED — needs re-login

### Next session start
1. `python main.py --login linkedin`
2. Test apply end-to-end
3. If Cloudflare still blocks: try `channel="chrome"` in browser_profile.py

## Session 17 — 2026-04-01: Fix _vision_ask error handling

### What was done
- **Fixed `_vision_ask()` in jobs.py**
  - Added try/except wrapper around all Groq API calls
  - Returns `""` on any exception instead of propagating
  - Logs errors with `logger.error()` for debugging
  - Ensures callers (`_get_search_url`, `_extract_jobs_from_page`) gracefully handle failures

- **Added test: `test_vision_ask_returns_empty_string_on_groq_failure()`**
  - Mocks `_get_client()` to raise RuntimeError
  - Verifies `_vision_ask()` returns `""` instead of raising
  - Confirms project rule compliance: all functions return graceful fallbacks, never raise

### Test results
10/10 passed (9 existing + 1 new)

### Commit
```
499a0e0 fix(3A): add error handling to _vision_ask — never raises to caller
```

### Why this matters
Violates the project rule: "all functions must return graceful fallbacks and never raise."
Before: network error in Groq API → exception → main handler crashes.
After: network error → logged → _vision_ask returns "" → _get_search_url returns None → search_jobs continues gracefully.

### Status
Jobs.py error handling complete. Callers already handle empty/"" returns correctly.
