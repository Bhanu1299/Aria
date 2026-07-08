"""
tests/test_away_summary.py — Unit tests for away_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm.base import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                       provider_used="groq", model_used="llama-3.3-70b-versatile")


def test_greeting_uses_session_notes():
    """generate() must call LLM when session_notes is non-empty and return the result."""
    fake_greeting = "Welcome back. Last session you were looking at Python roles at Stripe."

    with patch("aria.llm.llm_client.complete") as mock_complete:
        mock_complete.return_value = _make_llm_response(fake_greeting)

        import aria.state.away_summary as away_summary
        result = away_summary.generate(
            session_notes="- Searched Python engineer roles\n- Looked at Stripe posting",
            last_search="Python engineer San Francisco",
        )

    assert result == fake_greeting.strip()
    mock_complete.assert_called_once()


def test_greeting_falls_back_to_ready_when_no_history():
    """generate('', '') must return 'Ready when you are.' without calling LLM."""
    with patch("aria.llm.llm_client.complete") as mock_complete:
        import aria.state.away_summary as away_summary
        result = away_summary.generate(session_notes="", last_search="")

    assert result == "Ready when you are."
    mock_complete.assert_not_called()


def test_speak_greeting_calls_speaker():
    """speak_greeting() must load data, generate, and call speaker.say()."""
    fake_notes = "- Looked at ML jobs\n- Found Stripe posting"
    fake_search = "ML engineer roles"
    fake_greeting = "Welcome back. You were browsing ML engineer roles last session."

    mock_speaker = MagicMock()

    with patch("aria.llm.llm_client.complete") as mock_complete, \
         patch("aria.state.away_summary.memory.get_session_notes", return_value=fake_notes), \
         patch("aria.state.away_summary.memory.get_last_search", return_value=fake_search):
        mock_complete.return_value = _make_llm_response(fake_greeting)

        import aria.state.away_summary as away_summary
        away_summary.speak_greeting(mock_speaker)

    mock_speaker.say.assert_called_once_with(fake_greeting.strip())
