"""plugins/media/music_app.py — MusicAppClient via AppleScript."""
from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_TIMEOUT_MSG = "Music app isn't responding."
_NOTHING_PLAYING_MSG = "Nothing is currently playing."


class MusicAppClient:
    _instance: "MusicAppClient | None" = None

    @classmethod
    def get(cls) -> "MusicAppClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _run(self, script: str) -> tuple[bool, str]:
        """Run an AppleScript snippet. Returns (ok, output)."""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=8,
            )
            if result.returncode != 0:
                err = result.stderr.strip()
                if "not running" in err.lower() or "can't get" in err.lower():
                    return False, _NOTHING_PLAYING_MSG
                return False, err or "AppleScript error"
            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, _TIMEOUT_MSG
        except Exception as exc:
            return False, str(exc)

    def _ensure_running(self) -> None:
        """Launch Music.app in background without stealing focus."""
        subprocess.run(["open", "-jg", "-a", "Music"],
                       capture_output=True, timeout=5)

    def play(self, query: str = "") -> str:
        self._ensure_running()
        if query:
            escaped = query.replace('"', '\\"')
            script = f"""
tell application "Music"
    set results to search playlist "Library" for "{escaped}"
    if results is not {{}} then
        play (item 1 of results)
    else
        play
    end if
end tell
"""
            ok, out = self._run(script)
            if not ok:
                return out
            info = self.now_playing()
            if isinstance(info, dict):
                return f"Playing {info['track']} by {info['artist']}."
            return f"Playing {query}."
        else:
            ok, out = self._run('tell application "Music" to play')
            return "Resuming." if ok else out

    def pause(self) -> str:
        ok, out = self._run('tell application "Music" to pause')
        return "Paused." if ok else out

    def resume(self) -> str:
        return self.play()

    def skip(self) -> str:
        ok, out = self._run('tell application "Music" to next track')
        return "Skipped." if ok else out

    def previous(self) -> str:
        ok, out = self._run('tell application "Music" to back track')
        return "Going back." if ok else out

    def set_volume(self, level: int) -> str:
        level = max(0, min(100, level))
        ok, out = self._run(f'tell application "Music" to set sound volume to {level}')
        return f"Volume at {level}%." if ok else out

    def now_playing(self) -> dict | str:
        script = """
tell application "Music"
    if player state is stopped then
        return "nothing"
    end if
    set t to name of current track
    set a to artist of current track
    set al to album of current track
    return t & "|||" & a & "|||" & al
end tell
"""
        ok, out = self._run(script)
        if not ok or out == "nothing":
            return _NOTHING_PLAYING_MSG
        parts = out.split("|||")
        return {
            "track": parts[0] if len(parts) > 0 else "",
            "artist": parts[1] if len(parts) > 1 else "",
            "album": parts[2] if len(parts) > 2 else "",
        }
