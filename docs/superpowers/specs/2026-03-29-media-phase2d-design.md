# Phase 2D — Media Playback Design

**Date:** 2026-03-29
**Status:** Approved

---

## Context

Aria can already control Spotify via AppleScript in `mac_controller.py` (app_control intent). That code is Spotify-specific and mixed in with unrelated system controls. Phase 2D replaces it with a dedicated `media.py` module that:

- Controls any music app via a configurable `MUSIC_APP` env var (default: "Music" = Apple Music)
- Adds real YouTube playback: audio-only via yt-dlp+ffplay (background, no screen) and video via browser
- Cleans up `mac_controller.py` by removing the Spotify-specific play/pause/skip/whats_playing actions

---

## Architecture

```
Voice → router.py (media intent) → main.py → media.handle_media_command()
                                                 ├── Groq sub-classifier
                                                 ├── music_play / music_pause / music_skip / music_now_playing
                                                 ├── play_youtube_audio  (yt-dlp + ffplay, background)
                                                 ├── play_youtube_video  (open browser)
                                                 └── stop_youtube        (pkill ffplay)
```

---

## New File: `media.py`

### Configuration
```python
MUSIC_APP = os.getenv("MUSIC_APP", "Music")  # "Music" = Apple Music
```
All AppleScript calls use `MUSIC_APP` — never hardcode "Music" or "Spotify".

### Music App Control (AppleScript)
- `music_play(query)` — search library, play first result; returns now-playing string
- `music_pause()` — pause
- `music_resume()` — resume
- `music_skip()` — next track; returns now-playing string
- `music_now_playing()` — returns "Now playing X by Y" or "Nothing is playing right now."

All use `subprocess.run(['osascript', '-e', script])`.

### YouTube Playback (yt-dlp + ffplay)
- `play_youtube_audio(query)` — yt-dlp fetches stream URL + title, ffplay streams in background (Popen, DEVNULL, no display). Returns "Playing {title} from YouTube."
- `play_youtube_video(query)` — yt-dlp gets video ID, `open https://youtube.com/watch?v=ID`. Returns "Opening YouTube for {query}."
- `stop_youtube()` — `pkill -f ffplay`

### Dependency Check
`check_dependencies()` runs at import time. Prints warnings for missing `yt-dlp` or `ffplay` — does not crash.

### Intent Handler
`handle_media_command(transcript)` — one Groq call (llama-3.1-8b-instant) classifies into:

| action | platform | routes to |
|--------|----------|-----------|
| play_music | music_app / auto | `music_play(query)` |
| play_youtube_audio | youtube | `play_youtube_audio(query)` |
| play_youtube_video | youtube | `play_youtube_video(query)` |
| pause | music_app | `music_pause()` |
| resume | music_app | `music_resume()` |
| skip | music_app | `music_skip()` |
| now_playing | any | `music_now_playing()` |
| stop | any | `stop_youtube()` + music app pause |

---

## Changes to Existing Files

### `mac_controller.py`
Remove actions: `play`, `pause`, `resume`, `skip`, `whats_playing` — these are now handled by `media.py`. Remove them from `_SUB_CLASSIFY_SYSTEM` rules and from the `handle_app_command` dispatch block.

### `router.py` — `_CLASSIFY_SYSTEM`
Add to `media` type definition trigger phrases:
- "play", "pause", "skip", "next song", "what's playing", "now playing"
- "watch", "stop music", "stop playback"
- "play X on YouTube", "play X on [MUSIC_APP]"

Add IMPORTANT rule: Use `media` (not `app_control`) for music playback, YouTube audio/video, pause/skip/now-playing.

### `main.py`
- `import media` at top
- Replace existing `if intent_type == "media": browser.fetch(...)` block with:
  ```python
  if intent_type == "media":
      return media.handle_media_command(original_question)
  ```

### `.env` / `.env.example`
Add: `MUSIC_APP=Music`

### `requirements.txt`
No new Python packages. yt-dlp and ffplay are system deps (brew).

---

## Files Modified
- `media.py` — NEW
- `mac_controller.py` — remove Spotify play/pause/skip/whats_playing
- `router.py` — update media intent trigger phrases
- `main.py` — replace media handler block, add import
- `.env.example` — add MUSIC_APP
- `.claude/project-state.md` — mark Phase 2D complete
- `.claude/session-log.md` — log session
- `.claude/decisions.md` — log MUSIC_APP decision, YouTube audio strategy
- `.claude/tools.md` — add yt-dlp, ffplay entries

---

## Verification

1. `python main.py`
2. Say "Play Taylor Swift" → Apple Music opens, plays
3. Say "What's playing" → speaks track + artist
4. Say "Skip" → next track, speaks new track name
5. Say "Pause" → music pauses
6. Say "Play lo-fi beats on YouTube" → ffplay streams audio in background, no screen change
7. Say "Watch how to make pasta on YouTube" → browser opens YouTube video
8. Say "Stop" → ffplay killed, music paused
