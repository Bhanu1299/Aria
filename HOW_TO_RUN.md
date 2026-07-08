# Aria — How To Run

## Prerequisites

- macOS (Apple Silicon or Intel)
- Python 3.9 or later — check with `python3 --version` (the project venv runs 3.9)
- [Homebrew](https://brew.sh) installed
- `ffmpeg` installed (required by faster-whisper):
  ```bash
  brew install ffmpeg
  ```
- API keys: [Groq](https://console.groq.com) and [Anthropic](https://console.anthropic.com)

---

## Permissions

Grant these macOS permissions, then **restart your terminal**.

**1. Accessibility** (required)
For pynput to capture the global ⌥ Space hotkey from any app.

> System Settings → Privacy & Security → Accessibility
> → click `+` → add your terminal app (Terminal, iTerm2, etc.) → toggle ON

Without this: the hotkey does nothing. Aria appears to start normally but never responds to ⌥ Space.

**2. Microphone** (required)
For sounddevice to record your voice.

> System Settings → Privacy & Security → Microphone
> → click `+` → add your terminal app → toggle ON

Without this: recording fails immediately when you press the hotkey.

**3. Screen Recording** (optional)
For screen questions ("what's on my screen?", "show me where X is").

> System Settings → Privacy & Security → Screen Recording
> → click `+` → add your terminal app → toggle ON

Without this: screen tools return a graceful error; everything else works.

---

## First Time Setup

Run these commands once from the project directory:

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Chromium for Playwright (background browsing)
playwright install chromium

# 4. Configure API keys
cp .env.example .env
# edit .env and set GROQ_API_KEY and ANTHROPIC_API_KEY
```

Optional — log in once for authenticated background browsing (Gmail summaries, LinkedIn job search):

```bash
python main.py --login gmail      # also: google, linkedin
```

> **Note:** The first `python main.py` downloads the Whisper base model (~140 MB). It is cached at `~/.cache/huggingface/` after that.

---

## Running Aria

```bash
source venv/bin/activate
python main.py
```

Aria starts silently with a menu bar icon and prints:
```
Aria starting up...
Loading Whisper model (first run may download ~140 MB)...
[Aria] Plugin loaded: CorePlugin
[Aria] Plugin loaded: ... (7 plugins)
[Aria] Wake word active (...)
Aria ready. Hold ⌥ Space to ask a question.
```

To stop: press `Ctrl+C` in the terminal, or Quit from the menu bar icon.

---

## What To Expect

**Menu bar icon** (top-right of your screen):

| Icon | State | What it means |
|------|-------|---------------|
| ◉ | IDLE | Ready, waiting |
| 🎙 | LISTENING | Mic is open |
| ⏳ | THINKING | Agent is working |
| ✓ | DONE | Done, resetting to idle |

**The flow:**

1. Hold **⌥ Space** and speak — or just say **"Aria"** (wake word)
2. A pulsing pill appears at the bottom of the screen while the mic is open
3. Release (or stop talking) — the agent picks tools and works
4. If it takes more than a few seconds, Aria says "one moment"
5. The answer is spoken aloud; your screen never moves
6. The mic briefly re-opens (soft pop) — follow up naturally, or stay silent
7. Say "thanks" or nothing at all to end the conversation

**Daily reliability check:**

```bash
venv/bin/python daily_check.py list --core   # questions to read aloud (~3 min)
venv/bin/python daily_check.py report        # what passed / failed today
venv/bin/python daily_check.py wake          # wake word engine health
```

---

## Troubleshooting

**1. Hotkey does nothing**
→ Accessibility permission missing or terminal not added.
Fix: System Settings → Privacy & Security → Accessibility → add your terminal → toggle ON → **restart the terminal** → re-run.

**2. "ERROR: Microphone not available" when you press the hotkey**
→ Microphone permission missing.
Fix: System Settings → Privacy & Security → Microphone → add your terminal → toggle ON → **restart the terminal** → re-run.

**3. Aria answers "Something went wrong" on every command**
→ Usually missing/invalid API keys.
Fix: check `GROQ_API_KEY` and `ANTHROPIC_API_KEY` in `.env`. The console prints the failing provider. One valid key is enough to run (the tier chains fail over).

**4. Aria says "I didn't catch that, please try again."**
→ Whisper transcribed silence or too-short audio.
Fix: hold the hotkey, wait half a second, then speak clearly. Check your mic is the default input in System Settings → Sound → Input.

**5. Wake word never triggers (or triggers randomly)**
→ Check it's alive and see near-miss scores:
```bash
venv/bin/python daily_check.py wake
```
If near-misses outnumber detections, the report suggests a lower threshold (edit `_CUSTOM_THRESHOLD` / `_OWW_THRESHOLD` in `wake_word.py`). If the engine shows NOT RUNNING, check the startup console for a `[Aria] Wake word` line — missing deps or model fall back or disable it, and the hotkey always works regardless.

**6. Screen questions fail**
→ Screen Recording permission missing (see Permissions above). Screen tools need it; everything else runs without it.
