"""
tests/test_away_summary.py — Unit tests for away_summary.py

Tests:
  test_greeting_uses_session_notes          — generate() calls Groq when notes non-empty
  test_greeting_falls_back_to_ready_when_no_history — generate("","") skips Groq
  test_speak_greeting_calls_speaker         — speak_greeting() calls speaker.say()
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
# test_greeting_uses_session_notes
# ---------------------------------------------------------------------------

def test_greeting_uses_session_notes():
    """generate() must call Groq when session_notes is non-empty and return the result."""
    fake_greeting = "Welcome back. Last session you were looking at Python roles at Stripe."
    fake_resp = _make_fake_groq_response(fake_greeting)

    with patch("away_summary._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_resp
        mock_get_client.return_value = mock_client

        import away_summary
        result = away_summary.generate(
            session_notes="- Searched Python engineer roles\n- Looked at Stripe posting",
            last_search="Python engineer San Francisco",
        )

    assert result == fake_greeting.strip()
    mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# test_greeting_falls_back_to_ready_when_no_history
# ---------------------------------------------------------------------------

def test_greeting_falls_back_to_ready_when_no_history():
    """generate('', '') must return 'Ready when you are.' without calling Groq."""
    with patch("away_summary._get_client") as mock_get_client:
        import away_summary
        result = away_summary.generate(session_notes="", last_search="")

    assert result == "Ready when you are."
    mock_get_client.assert_not_called()


# ---------------------------------------------------------------------------
# test_speak_greeting_calls_speaker
# ---------------------------------------------------------------------------

def test_speak_greeting_calls_speaker():
    """speak_greeting() must load data, generate, and call speaker.say()."""
    fake_notes = "- Looked at ML jobs\n- Found Stripe posting"
    fake_search = "ML engineer roles"
    fake_greeting = "Welcome back. You were browsing ML engineer roles last session."
    fake_resp = _make_fake_groq_response(fake_greeting)

    mock_speaker = MagicMock()

    with patch("away_summary._get_client") as mock_get_client, \
         patch("away_summary.memory.get_session_notes", return_value=fake_notes), \
         patch("away_summary.memory.get_last_search", return_value=fake_search):

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_resp
        mock_get_client.return_value = mock_client

        import away_summary
        away_summary.speak_greeting(mock_speaker)

    mock_speaker.say.assert_called_once_with(fake_greeting.strip())
