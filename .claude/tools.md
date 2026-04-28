# Aria — Tools & Libraries

## Python
Version: 3.11+
Install: brew install python

## pynput
Purpose: Global hotkey listener
Install: pip install pynput
Docs: https://pynput.readthedocs.io

## sounddevice + soundfile
Purpose: Mic recording
Install: pip install sounddevice soundfile

## faster-whisper
Purpose: Local on-device transcription
Model: base
Install: pip install faster-whisper
Note: First run downloads ~140MB model. Warn user.

## Playwright
Purpose: Background browser control
Install: pip install playwright && playwright install chromium
Docs: https://playwright.dev/python

## rumps
Purpose: macOS menu bar icon
Install: pip install rumps

## python-dotenv
Purpose: Load .env config
Install: pip install python-dotenv

## ffmpeg (system dependency for faster-whisper)
Install: brew install ffmpeg

## yt-dlp (system dependency)
Purpose: Fetch YouTube stream URLs and video IDs without downloading
Install: brew install yt-dlp
Used in: media.py — play_youtube_audio(), play_youtube_video()
Note: Used via subprocess, not imported as a Python package

## ffplay (system dependency, part of ffmpeg)
Purpose: Stream YouTube audio in background (no screen, no download)
Install: brew install ffmpeg
Used in: media.py — play_youtube_audio() via Popen with -nodisp -autoexit
Note: Killed via `pkill -f ffplay` when stop command issued

## MUSIC_APP config
Purpose: Controls which music app Aria uses for play/pause/skip commands
Set in: .env as MUSIC_APP=Music (default) or MUSIC_APP=Spotify, MUSIC_APP=Tidal, etc.
Used in: media.py — all AppleScript music control uses this variable
Note: App name must match exactly what appears in /Applications

## pyobjc-framework-Cocoa + ApplicationServices
Purpose: Accessibility API fallback (Layer 2 in mac_controller.py) — find and click UI elements when AppleScript fails
Install: pip install pyobjc-framework-Cocoa pyobjc-framework-ApplicationServices
Note: Imported lazily inside click_element() — only loaded on first fallback invocation. AppleScript is primary.
