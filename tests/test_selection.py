"""Tests for selection.py — read the user's currently highlighted text."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aria.screen.selection as selection


def test_returns_ax_selected_text_when_available():
    with patch.object(selection, "_ax_selected_text", return_value="def foo(): pass"):
        assert selection.get_selected_text() == "def foo(): pass"


def test_falls_back_to_clipboard_when_ax_empty():
    with patch.object(selection, "_ax_selected_text", return_value=""), \
         patch.object(selection, "_clipboard_selected_text", return_value="hello world"):
        assert selection.get_selected_text() == "hello world"


def test_returns_empty_string_when_nothing_selected():
    with patch.object(selection, "_ax_selected_text", return_value=""), \
         patch.object(selection, "_clipboard_selected_text", return_value=""):
        assert selection.get_selected_text() == ""


def test_never_raises_when_both_paths_explode():
    with patch.object(selection, "_ax_selected_text", side_effect=RuntimeError("no AX")), \
         patch.object(selection, "_clipboard_selected_text", side_effect=OSError("no pbpaste")):
        assert selection.get_selected_text() == ""


def test_truncates_huge_selection():
    huge = "x" * 20000
    with patch.object(selection, "_ax_selected_text", return_value=huge):
        result = selection.get_selected_text()
    assert len(result) <= selection._MAX_CHARS


def test_clipboard_fallback_restores_original_clipboard():
    """The cmd+C trick must never clobber what the user had copied."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = MagicMock()
        out.returncode = 0
        if cmd[0] == "pbpaste":
            # first pbpaste = original clipboard save, second = captured selection
            out.stdout = b"ORIGINAL" if len([c for c in calls if c[0] == "pbpaste"]) == 1 else b"SELECTED"
        else:
            out.stdout = b""
        return out

    with patch.object(selection.subprocess, "run", side_effect=fake_run), \
         patch.object(selection.time, "sleep"):
        text = selection._clipboard_selected_text()

    assert text == "SELECTED"
    # original clipboard must be written back via pbcopy
    pbcopy_calls = [c for c in calls if c[0] == "pbcopy"]
    assert pbcopy_calls, "clipboard was not restored"
