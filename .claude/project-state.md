# Aria Project State

## Status: PHASE 3A COMPLETE — voice-driven job search (Google→vision pipeline)

## Architecture (Phase 3A)
Voice → Whisper → router.py (contact pre-check + Groq) → handler → speaker.py

Handlers per intent:
- knowledge   → summarizer.answer_knowledge()
- web_search  → browser.extract_links() → browser.fetch() → summarizer.summarize()
- web_direct  → browser.fetch() → summarizer.summarize()
- navigate    → browser.goto() → speak "Opening [site]"
- app         → app_launcher.open_app() → speak result
- media       → media.handle_media_command() →
                  music_play/pause/skip/now_playing (AppleScript, MUSIC_APP configurable)
                  play_youtube_audio (yt-dlp + ffplay, background)
                  play_youtube_video (open browser)
- app_control → mac_controller.handle_app_command() →
                  Layer 1: osascript AppleScript
                  Layer 2: pyobjc Accessibility API (fallback)
                  Layer 3: screencapture + Groq vision (last resort)
- briefing    → briefing.build_briefing() →
                  weather (wttr.in) + calendar (AppleScript) + gmail (authenticated fetch) + news (AP/BBC)
                  All four run concurrently via ThreadPoolExecutor → Groq assembles spoken briefing
- jobs        → jobs.search_jobs() →
                  Groq parses role+location → Google site: search → Playwright headless screenshot
                  → Llama-4-Scout extracts job search URL → screenshot job page → Llama-4-Scout
                  extracts listings as JSON → deduplicate → memory.store_jobs() → speak top 5

## What exists (all files)
- `requirements.txt` — pynput, sounddevice, soundfile, faster-whisper, playwright, rumps, python-dotenv, numpy, groq, requests, pyobjc-*
- `config.py` — GROQ_API_KEY, CURRENT_LOCATION, BROWSER_TIMEOUT, check_permissions()
- `hotkey.py` — HotkeyListener (pynput global hotkeys)
- `voice_capture.py` — VoiceCapture (sounddevice 16kHz capture)
- `transcriber.py` — faster-whisper base, beam_size=5
- `router.py` — UPDATED: added briefing intent, _APP_NAME_MAP, _APP_CONTACT_RE
- `browser.py` — UPDATED: added fetch_authenticated() using persistent profile
- `browser_profile.py` — NEW: persistent Playwright profile at ~/.aria/browser_profile
- `briefing.py` — NEW: four concurrent fetchers + Groq assembler
- `app_launcher.py` — UPDATED: _MESSAGING_APPS set for WhatsApp/Signal/Telegram
- `summarizer.py` — UPDATED: identity.json injected, capability constraint, hallucination guard
- `speaker.py` — unchanged
- `menubar.py` — unchanged
- `main.py` — UPDATED: --login CLI, briefing handler, word-count guard, Phase 2E label
- `mac_controller.py` — handle_app_command() + 3-layer AppleScript/pyobjc/vision
- `media.py` — UPDATED: Apple Music library search try/except + reveal fallback
- `identity.json` — full profile (name, email, skills, experience, education) — injected into summarizer prompt
- `memory.py` — NEW: module-level session dict; store_jobs(), get_job_by_index() for Phase 3B
- `jobs.py` — NEW: Google+vision job search pipeline; search_jobs(), format_spoken_results()
- `HOW_TO_RUN.md` — needs updating for Phase 2E

## What to do next
1. Test: `python main.py --login linkedin` (one-time login if not done)
2. Test: "Find me software engineer jobs in New York" → hear 5 results
3. Test: "Find backend engineer jobs at fintech companies"
4. Test: "Any ML engineer roles remote"
5. After each: check `memory.session["last_jobs"]` has 5 results
6. Phase 3B options: "apply to the second one" (open job URL), "tell me more about the third one"

## Known risks
- Spotify AppleScript `search for` + `play` is unreliable in Spotify 1.2+
- Gmail page structure can change — fetch_authenticated may need selector updates
- AP News headline regex is fragile — BBC RSS is the more reliable fallback
- Groq vision model id `meta-llama/llama-4-scout-17b-16e-instruct` may change
- Persistent profile at ~/.aria/browser_profile — if Chromium updates break it, delete and re-login

## Blocked on
Nothing. Ready to test.
