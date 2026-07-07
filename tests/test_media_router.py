"""tests/test_media_router.py — MediaRouter backend selection tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_auto_selects_spotify_when_configured():
    from plugins.media.router import MediaRouter
    from plugins.media.spotify import SpotifyClient
    from plugins.media.music_app import MusicAppClient
    SpotifyClient._instance = None
    MusicAppClient._instance = None

    mock_spotify = MagicMock()
    with patch("plugins.media.spotify.SpotifyClient.is_configured", return_value=True), \
         patch("plugins.media.spotify.SpotifyClient.get", return_value=mock_spotify), \
         patch.dict("os.environ", {"MUSIC_BACKEND": "auto"}):
        router = MediaRouter()
        backend = router.get_backend()
    assert backend is mock_spotify


def test_auto_falls_back_to_music_app_when_spotify_not_configured():
    from plugins.media.router import MediaRouter
    from plugins.media.spotify import SpotifyClient
    from plugins.media.music_app import MusicAppClient
    SpotifyClient._instance = None
    MusicAppClient._instance = None

    mock_music = MagicMock()
    with patch("plugins.media.spotify.SpotifyClient.is_configured", return_value=False), \
         patch("plugins.media.music_app.MusicAppClient.get", return_value=mock_music), \
         patch.dict("os.environ", {"MUSIC_BACKEND": "auto"}):
        router = MediaRouter()
        backend = router.get_backend()
    assert backend is mock_music


def test_explicit_spotify_ignores_auto():
    from plugins.media.router import MediaRouter
    from plugins.media.spotify import SpotifyClient
    SpotifyClient._instance = None

    mock_spotify = MagicMock()
    with patch("plugins.media.spotify.SpotifyClient.get", return_value=mock_spotify), \
         patch.dict("os.environ", {"MUSIC_BACKEND": "spotify"}):
        router = MediaRouter()
        backend = router.get_backend()
    assert backend is mock_spotify


def test_explicit_music_app_ignores_spotify():
    from plugins.media.router import MediaRouter
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None

    mock_music = MagicMock()
    with patch("plugins.media.music_app.MusicAppClient.get", return_value=mock_music), \
         patch.dict("os.environ", {"MUSIC_BACKEND": "music_app"}):
        router = MediaRouter()
        backend = router.get_backend()
    assert backend is mock_music


def test_media_plugin_registers_all_tools():
    from plugins.media import MediaPlugin
    from tool import ToolRegistry
    from plugins.media.router import MediaRouter
    from plugins.media.music_app import MusicAppClient

    MusicAppClient._instance = None
    mock_music = MagicMock()
    mock_music.play.return_value = "Resuming."

    with patch("plugins.media.spotify.SpotifyClient.is_configured", return_value=False), \
         patch("plugins.media.music_app.MusicAppClient.get", return_value=mock_music), \
         patch.dict("os.environ", {"MUSIC_BACKEND": "auto"}):
        registry = ToolRegistry()
        MediaPlugin().register(registry)
        available = registry.all_available()

    names = {t.name for t in available}
    expected = {"music_play", "music_pause", "music_resume", "music_skip",
                "music_previous", "music_volume", "music_now_playing"}
    assert expected == names
