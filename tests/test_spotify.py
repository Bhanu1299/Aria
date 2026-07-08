"""tests/test_spotify.py — SpotifyClient unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

_NOT_SETUP = "Spotify isn't set up yet. Run: python -m aria.setup spotify"
_NO_DEVICE = "Spotify isn't active on any device. Open Spotify on your Mac or phone first."
_PREMIUM = "Playback control requires Spotify Premium."


def _client_with_sp(mock_sp, configured=True):
    from aria.plugins.media.spotify import SpotifyClient
    SpotifyClient._instance = None
    client = SpotifyClient.__new__(SpotifyClient)
    client._sp = mock_sp
    client._configured = configured
    client.is_configured = staticmethod(lambda: configured)
    return client


def _make_sp():
    return MagicMock()


def test_play_returns_not_setup_when_unconfigured():
    from aria.plugins.media.spotify import SpotifyClient
    SpotifyClient._instance = None
    client = SpotifyClient.__new__(SpotifyClient)
    client._sp = None
    with patch.object(SpotifyClient, "is_configured", return_value=False):
        result = client.play("Kendrick")
    assert result == _NOT_SETUP


def test_play_returns_no_device_when_no_active_device():
    sp = _make_sp()
    sp.devices.return_value = {"devices": []}
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.play("Kendrick")
    assert result == _NO_DEVICE


def test_play_with_query_calls_search_and_start():
    sp = _make_sp()
    sp.devices.return_value = {"devices": [{"id": "dev1", "is_active": True}]}
    sp.search.return_value = {
        "tracks": {"items": [{"uri": "spotify:track:abc"}]},
        "playlists": {"items": []},
        "albums": {"items": []},
    }
    sp.current_playback.return_value = {
        "item": {
            "name": "HUMBLE.",
            "artists": [{"name": "Kendrick Lamar"}],
            "album": {"name": "DAMN."},
        },
        "progress_ms": 0,
    }
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.play("Kendrick")
    assert "HUMBLE" in result or "Kendrick" in result
    sp.start_playback.assert_called_once()


def test_pause_calls_sp():
    sp = _make_sp()
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.pause()
    assert result == "Paused."
    sp.pause_playback.assert_called_once()


def test_skip_calls_next_track():
    sp = _make_sp()
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.skip()
    assert result == "Skipped."
    sp.next_track.assert_called_once()


def test_previous_calls_previous_track():
    sp = _make_sp()
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.previous()
    assert result == "Going back."
    sp.previous_track.assert_called_once()


def test_set_volume_clamps_and_calls():
    sp = _make_sp()
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.set_volume(150)  # should clamp to 100
    assert "100%" in result
    sp.volume.assert_called_once_with(100)


def test_now_playing_returns_dict():
    sp = _make_sp()
    sp.current_playback.return_value = {
        "item": {
            "name": "Money Trees",
            "artists": [{"name": "Kendrick Lamar"}],
            "album": {"name": "good kid, m.A.A.d city"},
        },
        "progress_ms": 45000,
    }
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.now_playing()
    assert isinstance(result, dict)
    assert result["track"] == "Money Trees"
    assert result["artist"] == "Kendrick Lamar"


def test_now_playing_returns_nothing_when_stopped():
    sp = _make_sp()
    sp.current_playback.return_value = {"item": None}
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.now_playing()
    assert result == "Nothing is playing."


def test_premium_error_returns_friendly_msg():
    sp = _make_sp()
    sp.devices.return_value = {"devices": [{"id": "dev1", "is_active": True}]}
    sp.search.return_value = {"tracks": {"items": [{"uri": "spotify:track:x"}]},
                              "playlists": {"items": []}, "albums": {"items": []}}
    sp.start_playback.side_effect = Exception("403 premium required")
    client = _client_with_sp(sp)
    with patch.object(client, "is_configured", return_value=True):
        result = client.play("anything")
    assert "Premium" in result
