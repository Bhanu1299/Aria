"""plugins/media/router.py — MediaRouter: selects active backend."""
from __future__ import annotations

import os


class MediaRouter:
    def get_backend(self):
        """Return the active media backend based on config and availability."""
        preference = os.getenv("MUSIC_BACKEND", "auto").lower()
        from plugins.media.spotify import SpotifyClient
        from plugins.media.music_app import MusicAppClient

        if preference == "spotify":
            return SpotifyClient.get()
        if preference == "music_app":
            return MusicAppClient.get()
        # auto: prefer Spotify if configured, else Music.app
        if SpotifyClient.is_configured():
            return SpotifyClient.get()
        return MusicAppClient.get()
