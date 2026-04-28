# Architecture Decisions — Do Not Relitigate These

## Playwright over Claude API
Building an agent that controls apps invisibly.
API is a chatbot wrapper. Playwright proves the real engine.
Same engine later handles LinkedIn, job apps, any website.
LOCKED.

## faster-whisper over SFSpeechRecognizer
Python-only stack. Runs locally on Apple Silicon. No Swift.
Base model: accurate enough, fast enough.
LOCKED.

## pynput over NSEvent
No Swift. pynput handles global hotkeys in pure Python on macOS.
LOCKED.

## rumps for menu bar
Lightest Python menu bar library. No Swift, no Electron.
LOCKED.

## macOS say over ElevenLabs
Zero latency, zero cost, zero setup. ElevenLabs is Phase 2.
LOCKED.

## Wake word added (Phase 3D Session 2) — overrides prior push-to-talk-only decision
openwakeword chosen over Porcupine: fully open source, no API key, built-in models.
Models used as phonetic proxies (no exact "Aria" model exists):
  "Hey Aria" / "Yo Aria" / "Aria" / "Yo" → hey_jarvis (phonetically closest)
  "Wake up" / "Listen"                    → alexa (phonetically closest)
Threshold: 0.5 — tune down if too many false triggers, up if it misses.
PyAudio at 1280 frames / 16kHz (80ms chunks) — standard for oww, low CPU.
Hotkey (⌥ Space) remains as backup; both paths call the same handle_command().
CPU guard: WakeWordListener checks _processing event before firing.
LOCKED.

## Whisper preloaded at startup
Lazy init = silent delay on first keypress = feels broken.
Preload in main.py, pass instance to transcriber.
LOCKED.

## Music app is configurable via MUSIC_APP env var (Phase 2D)
All AppleScript music control in media.py uses `MUSIC_APP = os.getenv("MUSIC_APP", "Music")`.
Never hardcode "Music" or "Spotify" — change the env var to switch services.
Note: `search playlist "Library"` AppleScript works for Apple Music; Spotify and other
streaming apps require different AppleScript dictionaries and may need updating.
LOCKED.

## YouTube audio uses yt-dlp + ffplay streaming (Phase 2D)
No download to disk. yt-dlp fetches the stream URL; ffplay plays it via Popen with
stdout/stderr DEVNULL. No screen change, no focus steal.
`stop` kills ffplay via `pkill -f ffplay`.
System deps: `brew install yt-dlp` and `brew install ffmpeg`.
LOCKED.

## Persistent browser profile at ~/.aria (Phase 2E)
Persistent Playwright profile stored outside the project directory at ~/.aria/browser_profile.
One-time `python main.py --login gmail` opens visible browser for manual login.
Future headless runs reuse saved session cookies automatically.
LOCKED.

## Briefing fetchers run concurrently (Phase 2E)
All four briefing data sources (weather, calendar, gmail, news) run in a
ThreadPoolExecutor(max_workers=4). If any single fetcher fails, its result
is "[source] unavailable" — never crashes the whole briefing.
LOCKED.

## Scenes system (Phase 3D Session 2)
scenes.json in project root — human-editable. Loaded at router.py import via scene_executor.
Scene match is the highest priority in route() — runs before contact, skill, apply, Groq.
open_app + open_url actions in a scene run concurrently via ThreadPoolExecutor.
Other actions (briefing, speak, pause_music, lock_mac, play_hype_music) run sequentially after.
play_hype_music requires yt-dlp + ffplay: `brew install yt-dlp ffmpeg`.
LOCKED.

## Aria personality in all Groq prompts (Phase 3D Session 2)
All Groq system prompts now open with the Aria personality header.
JSON-output prompts (classifier, mac_controller sub-classifier, jobs parser) add personality
header but preserve "return JSON only, no markdown" instructions.
LOCKED.

## Job search cache (Phase 3D Session 2)
30-minute in-memory TTL cache in memory.py. Same normalized query within 1800s = instant reply.
Cache is session-only (not SQLite-persisted) since job listings change frequently.
LOCKED.

## AppleScript limitations (Phase 2C)
- Spotify `search for` via AppleScript does not reliably play the searched track in Spotify 1.2+;
  `activate` + `search for` + `delay 1` + `play` is the best available workaround without Spotify API.
- Calendar `tell calendar "Home"` may fail if user has no "Home" calendar — production should enumerate
  available calendar names and pick the first writable one.
- `open -a` used as secondary fallback in `_app_open()` because `tell application X to activate` fails
  for apps not currently running on some macOS versions.
- pyobjc Layer 2 is imported lazily inside `click_element()` to avoid startup cost.
- Layer 3 vision model id is `meta-llama/llama-4-scout-17b-16e-instruct` — update if Groq changes it.
LOCKED.
