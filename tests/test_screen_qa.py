"""Tests for screen_qa.py — answer questions about the user's current screen."""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import screen_qa


def _fake_vision_client(reply_text: str) -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = reply_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def test_answer_returns_vision_text(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=_fake_vision_client("Your screen shows a Python traceback.")):
        result = screen_qa.answer("what does this error mean")
    assert result == "Your screen shows a Python traceback."


def test_answer_returns_error_when_capture_fails():
    with patch.object(screen_qa.subprocess, "run", side_effect=OSError("no permission")):
        result = screen_qa.answer("what's on my screen")
    assert isinstance(result, str)
    assert result  # non-empty spoken fallback
    assert "screen" in result.lower()


def test_answer_never_raises_on_groq_failure(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = RuntimeError("api down")
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=failing_client):
        result = screen_qa.answer("what am i looking at")
    assert isinstance(result, str)
    assert result


def test_capture_returns_none_when_file_missing():
    ok = types.SimpleNamespace(returncode=0)
    with patch.object(screen_qa.subprocess, "run", return_value=ok):
        # screencapture "succeeded" but produced no file at the target path
        assert screen_qa._capture("/tmp/definitely_missing_aria_qa.jpg") is None


# ---------------------------------------------------------------------------
# Router pre-check
# ---------------------------------------------------------------------------
import router


@pytest.mark.parametrize("command", [
    "what's on my screen",
    "What is on my screen right now",
    "look at my screen and tell me what this says",
    "read my screen",
    "what am I looking at",
    "check my screen, what does this error mean",
])
def test_screen_qa_precheck_skips_classifier(command):
    with patch.object(router, "_classify", side_effect=AssertionError("classifier should not be called")):
        intent = router.route(command)
    assert intent["type"] == "screen_qa"
    assert intent["query"] == command.strip()


def test_non_screen_command_does_not_match_precheck():
    knowledge = {
        "type": "knowledge", "query": "play some music", "url": "",
        "instructions": "", "app_name": "", "contact": "", "site_name": "",
    }
    with patch.object(router, "_classify", return_value=knowledge) as mock_classify:
        intent = router.route("play some music")
    assert intent["type"] != "screen_qa"
