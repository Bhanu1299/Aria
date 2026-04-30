"""
tests/test_compact.py — Unit tests for compact.py session notes compaction.

Tests:
  test_compress_returns_shorter_string                  — mock Groq, verify returned
                                                          string is different/shorter
  test_compress_graceful_on_groq_failure                — when Groq raises, returns
                                                          original notes unchanged
  test_store_session_notes_compacts_when_over_threshold — when combined notes > 3000
                                                          chars, compress() is called
  test_store_session_notes_does_not_compact_when_under_threshold — under 3000 chars,
                                                          compress() not called
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_groq_response(content: str):
    """Build a minimal mock that looks like a groq ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# test_compress_returns_shorter_string
# ---------------------------------------------------------------------------

def test_compress_returns_shorter_string():
    """compress() must return a shorter (or different) string than input when Groq succeeds."""
    long_notes = "- User asked about Python jobs\n" * 100  # ~3100 chars
    compressed = "- User asked about Python jobs\n- Aria returned results"

    fake_resp = _make_fake_groq_response(compressed)

    with patch("compact._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_resp
        mock_get_client.return_value = mock_client

        import compact
        result = compact.compress(long_notes)

    assert isinstance(result, str), "compress() must return str"
    assert result == compressed.strip(), f"Expected compressed notes, got {result!r}"
    assert len(result) < len(long_notes), "compress() result should be shorter than input"


# ---------------------------------------------------------------------------
# test_compress_graceful_on_groq_failure
# ---------------------------------------------------------------------------

def test_compress_graceful_on_groq_failure():
    """When Groq raises, compress() returns the original notes unchanged."""
    original_notes = "- User asked about Python jobs\n" * 100

    with patch("compact._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Groq API error")
        mock_get_client.return_value = mock_client

        import compact
        result = compact.compress(original_notes)

    assert result == original_notes, (
        f"compress() should return original notes on failure, got {result!r}"
    )


# ---------------------------------------------------------------------------
# test_store_session_notes_compacts_when_over_threshold
# ---------------------------------------------------------------------------

def test_store_session_notes_compacts_when_over_threshold():
    """When combined notes exceed 3000 chars, compact.compress() is called."""
    # Build notes that will exceed 3000 chars when combined
    existing_notes = "- Existing note\n" * 100   # ~1600 chars
    new_notes = "- New note\n" * 100              # ~1100 chars — total will be >3000

    with patch("memory.session", {}) as mock_session, \
         patch("memory._save"), \
         patch("compact.needs_compaction", return_value=True) as mock_needs, \
         patch("compact.compress", return_value="- Compacted summary") as mock_compress:

        import memory
        memory.session["session_notes"] = existing_notes

        memory.store_session_notes(new_notes)

        mock_compress.assert_called_once()
        assert memory.session[memory._SESSION_NOTES_KEY] == mock_compress.return_value


# ---------------------------------------------------------------------------
# test_store_session_notes_does_not_compact_when_under_threshold
# ---------------------------------------------------------------------------

def test_store_session_notes_does_not_compact_when_under_threshold():
    """When combined notes are under 3000 chars, compact.compress() is NOT called."""
    existing_notes = "- Short existing note"
    new_notes = "- Short new note"

    with patch("memory.session", {}) as mock_session, \
         patch("memory._save"), \
         patch("compact.needs_compaction", return_value=False) as mock_needs, \
         patch("compact.compress") as mock_compress:

        import memory
        memory.session["session_notes"] = existing_notes

        memory.store_session_notes(new_notes)

    mock_compress.assert_not_called()
