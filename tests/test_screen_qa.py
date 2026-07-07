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


def test_answer_includes_highlighted_text_in_prompt(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    client = _fake_vision_client("That function defines foo.")
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=client), \
         patch.object(screen_qa.selection, "get_selected_text", return_value="def foo(): pass"):
        result = screen_qa.answer("explain me what is highlighted")
    assert result == "That function defines foo."
    sent = client.chat.completions.create.call_args
    text_part = sent[1]["messages"][1]["content"][0]["text"]
    assert "def foo(): pass" in text_part


def test_answer_skips_selection_lookup_for_plain_screen_question(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=_fake_vision_client("A browser window.")), \
         patch.object(screen_qa.selection, "get_selected_text") as mock_sel:
        screen_qa.answer("what's on my screen")
    mock_sel.assert_not_called()


def test_answer_survives_selection_failure(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=_fake_vision_client("Answer.")), \
         patch.object(screen_qa.selection, "get_selected_text", side_effect=RuntimeError("boom")):
        assert screen_qa.answer("explain this") == "Answer."


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
# explain_visual — spoken answer + on-screen boxes
# ---------------------------------------------------------------------------

def test_explain_visual_speaks_answer_and_draws_scaled_regions(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    payload = ('{"answer": "The save button is in the top right.", '
               '"regions": [{"x": 900, "y": 50, "w": 80, "h": 40, "label": "Save"}]}')
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=_fake_vision_client(payload)), \
         patch.object(screen_qa.overlay, "screen_size", return_value=(1000.0, 1000.0)), \
         patch.object(screen_qa.overlay, "draw_boxes", return_value=True) as mock_draw:
        result = screen_qa.explain_visual("show me where the save button is")
    assert result == "The save button is in the top right."
    drawn = mock_draw.call_args[0][0]
    assert drawn[0]["x"] == 900.0 and drawn[0]["label"] == "Save"


def test_explain_visual_handles_markdown_fenced_json(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    payload = '```json\n{"answer": "Here.", "regions": []}\n```'
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=_fake_vision_client(payload)), \
         patch.object(screen_qa.overlay, "draw_boxes") as mock_draw:
        result = screen_qa.explain_visual("show me where it is")
    assert result == "Here."
    mock_draw.assert_not_called()


def test_explain_visual_speaks_plain_text_when_json_invalid(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=_fake_vision_client("The button is top right, not JSON")), \
         patch.object(screen_qa.overlay, "draw_boxes") as mock_draw:
        result = screen_qa.explain_visual("point out the button")
    assert result == "The button is top right, not JSON"
    mock_draw.assert_not_called()


def test_explain_visual_answer_survives_overlay_failure(tmp_path):
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    payload = '{"answer": "Top right.", "regions": [{"x": 1, "y": 1, "w": 10, "h": 10, "label": "b"}]}'
    with patch.object(screen_qa, "_capture", return_value=str(shot)), \
         patch.object(screen_qa, "_get_client", return_value=_fake_vision_client(payload)), \
         patch.object(screen_qa.overlay, "screen_size", return_value=(1000.0, 1000.0)), \
         patch.object(screen_qa.overlay, "draw_boxes", side_effect=RuntimeError("no gui")):
        result = screen_qa.explain_visual("show me where")
    assert result == "Top right."


def test_explain_visual_never_raises_on_capture_failure():
    with patch.object(screen_qa, "_capture", return_value=None):
        result = screen_qa.explain_visual("show me where the error is")
    assert isinstance(result, str) and result
