# Phase 5E — Media Control Design

**Date:** 2026-05-06  
**Status:** Approved  
**Depends on:** Phase 5A (core engine)  
**Unlocks:** Voice-controlled Spotify and Music.app playback

---

## Goal

Aria controls music playback by voice. Works with Spotify (if configured) and falls back to Music.app automatically.

---

## Voice examples

- "Play something chill"
- "Play Kendrick Lamar"
- "Pause the music"
- "Skip this"
- "Turn it up"
- "What's playing?"
- "Play my workout playlist"

---

## Architecture

### Plugin structure

```
plugins/
  media/
    __init__.py          ← MediaPlugin(PluginBase)
    spotify.py           ← SpotifyClient (spotipy wrapper)
    music_app.py         ← MusicAppClient (AppleScript)
    router.py            ← MediaRouter (picks active backend)
```

`media.py` in root is superseded by this plugin. The old `media.py` tool wrapper in `plugins/core/` is retired when this plugin loads.

### Backend selection

```python
class MediaRouter:
    def get_backend(self) -> "MediaBackend":
        backend = config.get("MUSIC_BACKEND", "auto")
        if backend == "spotify" or (backend == "auto" and SpotifyClient.is_configured()):
            return SpotifyClient.get()
        return MusicAppClient.get()
```

`MUSIC_BACKEND` in `.env`: `spotify` | `music_app` | `auto` (default).

---

## Spotify

### Mechanism

`spotipy` library with Authorization Code flow. Requires Spotify Premium for playback control (Spotify API restriction — free accounts cannot control playback).

### Setup (one-time)

```bash
python -m aria.setup spotify
```

1. Prompts for Spotify Client ID + Secret (from developer.spotify.com — 2-minute setup)
2. Opens browser to Spotify OAuth consent screen
3. Token saved to `~/.aria/credentials/spotify.json`
4. Refresh token auto-renews silently

### SpotifyClient interface

```python
class SpotifyClient:
    def play(self, query: str = None)    # search + play, or resume if no query
    def pause(self)
    def resume(self)
    def skip(self)
    def previous(self)
    def set_volume(self, level: int)     # 0–100
    def now_playing(self) -> dict        # {track, artist, album, progress_ms}
    def search(self, query: str) -> list[dict]
    def get_devices(self) -> list[dict]  # active playback devices
```

### Device handling

Spotify requires an active device. If no device is active (Spotify not open on any device), `play()` returns: `"Spotify isn't active on any device. Open Spotify on your Mac or phone first."` — does not attempt to launch Spotify silently.

---

## Music.app

### Mechanism

AppleScript via `osascript`. No auth, no setup. Works with Apple Music subscription or local library. Always available as fallback.

### MusicAppClient interface

```python
class MusicAppClient:
    def play(self, query: str = None)    # search library/Apple Music, or resume
    def pause(self)
    def resume(self)
    def skip(self)
    def previous(self)
    def set_volume(self, level: int)     # 0–100
    def now_playing(self) -> dict        # {track, artist, album}
```

### AppleScript snippets used

```applescript
-- play/pause
tell application "Music" to playpause

-- skip
tell application "Music" to next track

-- volume
tell application "Music" to set sound volume to {level}

-- now playing
tell application "Music"
  set t to name of current track
  set a to artist of current track
  return t & " by " & a
end tell

-- search and play
tell application "Music"
  play (search playlist "Library" for "{query}")
end tell
```

Music.app is launched silently if not running — it does not steal focus (launched with `open -jg` flag first, then AppleScript takes over).

---

## Unified tools registered

Both backends share the same tool interface. Callers never know which backend responded.

**`music_play`**
```
params:
  query: str      # optional — "Kendrick Lamar", "chill playlist", etc.
returns:
  "Playing {track} by {artist}." or "Resuming." or error
```

**`music_pause`**
```
params: none
returns: "Paused."
```

**`music_resume`**
```
params: none
returns: "Resuming."
```

**`music_skip`**
```
params: none
returns: "Skipped."
```

**`music_previous`**
```
params: none
returns: "Going back."
```

**`music_volume`**
```
params:
  level: int      # 0–100, OR
  direction: str  # "up" | "down" (adjusts by 10)
returns:
  "Volume at {level}%."
```

**`music_now_playing`**
```
params: none
returns:
  "{track} by {artist}" or "Nothing is playing."
```

---

## Not-yet-setup behavior

If Spotify is not configured and Music.app is unavailable (non-macOS — not applicable here), `MediaPlugin` logs a warning but does not crash. All tools register and return `"No music backend is configured."`.

---

## Error handling

| Error | Response |
|---|---|
| Spotify token expired | Auto-refresh. If fails: "Spotify auth expired. Run: python -m aria.setup spotify" |
| Spotify no active device | "Open Spotify on your Mac or phone first." |
| Spotify Premium required | "Playback control requires Spotify Premium." |
| AppleScript timeout | "Music app isn't responding." |
| Nothing playing (skip/pause) | "Nothing is currently playing." |

---

## Testing

- `tests/test_spotify.py` — play, pause, skip, volume, now_playing with mocked spotipy
- `tests/test_music_app.py` — AppleScript calls mocked via subprocess mock
- `tests/test_media_router.py` — backend selection: auto/explicit/fallback

---

## Dependencies

```
spotipy>=2.23.0
```

Music.app requires no new dependencies (subprocess is stdlib).

---

## Definition of done

- [ ] `SpotifyClient` with all 7 operations + OAuth setup
- [ ] `MusicAppClient` with all 6 operations via AppleScript
- [ ] `MediaRouter` backend selection logic
- [ ] `MediaPlugin` registers all 7 unified tools
- [ ] Setup script `python -m aria.setup spotify`
- [ ] Old `media.py` tool wrapper retired from `plugins/core/`
- [ ] Tests passing for Spotify, Music.app, router
