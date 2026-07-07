"""plugins/media/spotify.py — SpotifyClient via spotipy."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CREDS_PATH = Path.home() / ".aria" / "credentials" / "spotify.json"
_NOT_SETUP_MSG = "Spotify isn't set up yet. Run: python -m aria.setup spotify"
_AUTH_EXPIRED_MSG = "Spotify auth expired. Run: python -m aria.setup spotify"
_NO_DEVICE_MSG = "Spotify isn't active on any device. Open Spotify on your Mac or phone first."
_PREMIUM_MSG = "Playback control requires Spotify Premium."
_SCOPE = (
    "user-read-playback-state user-modify-playback-state "
    "user-read-currently-playing streaming"
)


class SpotifyClient:
    _instance: "SpotifyClient | None" = None

    @classmethod
    def get(cls) -> "SpotifyClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._sp = None

    @staticmethod
    def is_configured() -> bool:
        return _CREDS_PATH.exists()

    def _get_sp(self):
        if self._sp is not None:
            return self._sp
        if not self.is_configured():
            return None
        try:
            import json
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth

            raw = json.loads(_CREDS_PATH.read_text())
            client_id = raw.get("client_id", "")
            client_secret = raw.get("client_secret", "")
            refresh_token = raw.get("refresh_token", "")

            auth = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri="http://localhost:8888/callback",
                scope=_SCOPE,
                cache_path=str(_CREDS_PATH),
            )
            self._sp = spotipy.Spotify(auth_manager=auth)
            return self._sp
        except Exception as exc:
            logger.warning("spotify: init failed: %s", exc)
            return None

    def _active_device_id(self) -> str | None:
        sp = self._get_sp()
        if sp is None:
            return None
        try:
            devices = sp.devices()
            for d in devices.get("devices", []):
                if d.get("is_active"):
                    return d["id"]
            devs = devices.get("devices", [])
            return devs[0]["id"] if devs else None
        except Exception:
            return None

    def play(self, query: str = "") -> str:
        sp = self._get_sp()
        if sp is None:
            return _NOT_SETUP_MSG if not self.is_configured() else _AUTH_EXPIRED_MSG
        device_id = self._active_device_id()
        if device_id is None:
            return _NO_DEVICE_MSG
        try:
            if query:
                results = sp.search(q=query, type="track,playlist,album", limit=1)
                uri = _first_uri(results)
                if uri:
                    if "playlist" in uri or "album" in uri:
                        sp.start_playback(device_id=device_id, context_uri=uri)
                    else:
                        sp.start_playback(device_id=device_id, uris=[uri])
                    info = self.now_playing()
                    if isinstance(info, dict):
                        return f"Playing {info['track']} by {info['artist']}."
                return f"Playing {query}."
            else:
                sp.start_playback(device_id=device_id)
                return "Resuming."
        except Exception as exc:
            return _handle_error(exc)

    def pause(self) -> str:
        sp = self._get_sp()
        if sp is None:
            return _NOT_SETUP_MSG if not self.is_configured() else _AUTH_EXPIRED_MSG
        try:
            sp.pause_playback()
            return "Paused."
        except Exception as exc:
            return _handle_error(exc)

    def resume(self) -> str:
        return self.play()

    def skip(self) -> str:
        sp = self._get_sp()
        if sp is None:
            return _NOT_SETUP_MSG if not self.is_configured() else _AUTH_EXPIRED_MSG
        try:
            sp.next_track()
            return "Skipped."
        except Exception as exc:
            return _handle_error(exc)

    def previous(self) -> str:
        sp = self._get_sp()
        if sp is None:
            return _NOT_SETUP_MSG if not self.is_configured() else _AUTH_EXPIRED_MSG
        try:
            sp.previous_track()
            return "Going back."
        except Exception as exc:
            return _handle_error(exc)

    def set_volume(self, level: int) -> str:
        sp = self._get_sp()
        if sp is None:
            return _NOT_SETUP_MSG if not self.is_configured() else _AUTH_EXPIRED_MSG
        level = max(0, min(100, level))
        try:
            sp.volume(level)
            return f"Volume at {level}%."
        except Exception as exc:
            return _handle_error(exc)

    def now_playing(self) -> dict | str:
        sp = self._get_sp()
        if sp is None:
            return _NOT_SETUP_MSG if not self.is_configured() else _AUTH_EXPIRED_MSG
        try:
            current = sp.current_playback()
            if not current or not current.get("item"):
                return "Nothing is playing."
            item = current["item"]
            return {
                "track": item.get("name", ""),
                "artist": ", ".join(a["name"] for a in item.get("artists", [])),
                "album": item.get("album", {}).get("name", ""),
                "progress_ms": current.get("progress_ms", 0),
            }
        except Exception as exc:
            return _handle_error(exc)

    def get_devices(self) -> list | str:
        sp = self._get_sp()
        if sp is None:
            return []
        try:
            return sp.devices().get("devices", [])
        except Exception:
            return []


def _first_uri(results: dict) -> str | None:
    for kind in ("tracks", "playlists", "albums"):
        items = results.get(kind, {}).get("items", [])
        if items:
            return items[0].get("uri")
    return None


def _handle_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "premium" in msg or "403" in msg:
        return _PREMIUM_MSG
    if "no active device" in msg or "device" in msg:
        return _NO_DEVICE_MSG
    if "token" in msg or "auth" in msg or "401" in msg:
        return _AUTH_EXPIRED_MSG
    if "not playing" in msg or "204" in msg:
        return "Nothing is currently playing."
    return f"Spotify error: {exc}"
