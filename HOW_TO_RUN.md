# Aria — How To Run

## Prerequisites

- macOS (Apple Silicon or Intel)
- Python 3.11 or later — check with `python3 --version`
- [Homebrew](https://brew.sh) installed
- `ffmpeg` installed (required by faster-whisper):
  ```bash
  brew install ffmpeg
  ```
- Internet connection (for Claude.ai — one-time browser login required)

---

## Permissions

Aria needs two macOS permissions before it will work. Grant both, then restart your terminal.

**1. Accessibility**
Required for pynput to capture the global ⌥ Space hotkey from any app.

> System Settings → Privacy & Security → Accessibility
> → click `+` → add your terminal app (Terminal, iTerm2, etc.) → toggle ON

Without this: the hotkey does nothing. Aria appears to start normally but never responds when you press ⌥ Space.

**2. Microphone**
Required for sounddevice to record your voice.

> System Settings → Privacy & Security → Microphone
> → click `+` → add your terminal app → toggle ON

Without this: recording fails immediately when you press the hotkey. Aria prints an error and resets to IDLE.

---

## First Time Setup

Run these commands once from the project directory:

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Chromium for Playwright
playwright install chromium

# 4. Log in to Claude.ai in Playwright's Chromium (one time only)
#
#    Aria uses its own Playwright Chromium — separate from your normal browser.
#    You must log in through Playwright's browser window once so the session persists.
#
python3 - <<'EOF'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto("https://claude.ai")
    input("Log in to claude.ai in the browser window, then press Enter here...")
    ctx.storage_state(path="claude_auth.json")
    browser.close()
print("Login saved.")
EOF
```

> **Note:** The first time you run `python main.py`, Aria downloads the Whisper base model (~140 MB). This only happens once — it is cached at `~/.cache/huggingface/` after that.

---

## Running Aria

```bash
source venv/bin/activate
python main.py
```

Aria prints:
```
Aria starting up...
Loading Whisper model (first run may take a moment)...
[TRANSCRIBE] Whisper model loaded.
[BROWSER] Starting Chromium...
[BROWSER] Navigated to claude.ai
[Aria] Hotkey listener active — combo: <alt>+<space>
Aria ready. Hold ⌥ Space to ask a question.
```

To stop: press `Ctrl+C` in the terminal.

---

## What To Expect

**Menu bar icon** (top-right of your screen):

| Icon | State | What it means |
|------|-------|---------------|
| ◉ | IDLE | Ready, waiting for the hotkey |
| 🎙 | LISTENING | Recording your voice |
| ⏳ | THINKING | Fetching Claude's answer |
| ✓ | DONE | Done, resetting to idle |

**The flow:**

1. Hold **⌥ Space** (Option + Space) — icon switches to 🎙
2. Speak your question clearly
3. Release the key — icon switches to ⏳
4. Aria types your question into Claude.ai in a background browser window
5. Claude's answer is read from the page
6. macOS **say** speaks the answer aloud through your speakers
7. Icon returns to ◉

Your screen never moves. You stay in whatever app you were using.

---

## Troubleshooting

**1. Hotkey does nothing**
→ Accessibility permission is missing or the terminal app was not added correctly.
Fix: System Settings → Privacy & Security → Accessibility → add your terminal → toggle ON → **restart the terminal** → re-run `python main.py`.

**2. "ERROR: Microphone not available" when you press the hotkey**
→ Microphone permission is missing.
Fix: System Settings → Privacy & Security → Microphone → add your terminal → toggle ON → **restart the terminal** → re-run.

**3. "Login wall detected — claude.ai requires authentication"**
→ The Playwright Chromium session is not logged in to Claude.ai.
Fix: Re-run the login snippet from the First Time Setup section above, then restart Aria.

**4. Aria says "I didn't catch that, please try again."**
→ Whisper transcribed silence or too-short audio.
Fix: Hold the hotkey, wait half a second, then speak clearly. Release after you finish. Check your microphone is the default input in System Settings → Sound → Input.

**5. Aria says "Aria timed out, please try again."**
→ Claude.ai did not respond within the configured timeout (default 30 seconds).
Fix: Try again. If this happens consistently, increase the timeout in `.env`:
```
BROWSER_TIMEOUT=60
```
Then restart Aria.
