"""tests/test_music_app.py — MusicAppClient unit tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _run_ok(output: str = ""):
    m = MagicMock()
    m.returncode = 0
    m.stdout = output
    m.stderr = ""
    return m


def _run_fail(stderr: str = "error"):
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    m.stderr = stderr
    return m


def test_pause_returns_paused():
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None
    client = MusicAppClient()
    with patch("subprocess.run", return_value=_run_ok()):
        result = client.pause()
    assert result == "Paused."


def test_skip_returns_skipped():
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None
    client = MusicAppClient()
    with patch("subprocess.run", return_value=_run_ok()):
        result = client.skip()
    assert result == "Skipped."


def test_previous_returns_going_back():
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None
    client = MusicAppClient()
    with patch("subprocess.run", return_value=_run_ok()):
        result = client.previous()
    assert result == "Going back."


def test_set_volume_returns_level():
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None
    client = MusicAppClient()
    with patch("subprocess.run", return_value=_run_ok()):
        result = client.set_volume(70)
    assert "70%" in result


def test_set_volume_clamps():
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None
    client = MusicAppClient()
    with patch("subprocess.run", return_value=_run_ok()):
        result = client.set_volume(999)
    assert "100%" in result


def test_now_playing_returns_dict():
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None
    client = MusicAppClient()
    with patch("subprocess.run", return_value=_run_ok("Alright|||Kendrick Lamar|||To Pimp A Butterfly")):
        result = client.now_playing()
    assert isinstance(result, dict)
    assert result["track"] == "Alright"
    assert result["artist"] == "Kendrick Lamar"


def test_now_playing_returns_nothing_when_stopped():
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None
    client = MusicAppClient()
    with patch("subprocess.run", return_value=_run_ok("nothing")):
        result = client.now_playing()
    assert result == "Nothing is currently playing."


def test_play_with_query_searches_library():
    from plugins.media.music_app import MusicAppClient
    MusicAppClient._instance = None
    client = MusicAppClient()
    responses = [
        _run_ok(),  # ensure_running
        _run_ok(),  # play script
        _run_ok("Money Trees|||Kendrick Lamar|||good kid"),  # now_playing
    ]
    with patch("subprocess.run", side_effect=responses):
        result = client.play("Kendrick")
    assert "Money Trees" in result or "Kendrick" in result


def test_timeout_returns_not_responding():
    import subprocess
    from plugins.media.music_app import MusicAppClient, _TIMEOUT_MSG
    MusicAppClient._instance = None
    client = MusicAppClient()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("osascript", 8)):
        result = client.pause()
    assert result == _TIMEOUT_MSG
