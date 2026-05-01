"""
tests/test_prompt_suggester.py — Unit tests for prompt_suggester.py

Tests:
  test_suggest_returns_string_for_browser_task  — mock Groq, browser_task intent, long answer → "Also…"
  test_suggest_returns_empty_for_short_answer   — < 20 words → ""
  test_suggest_returns_empty_for_excluded_intent — intent="weather" → ""
  test_suggest_async_does_not_block             — suggest_async returns quickly
  test_suggest_graceful_on_groq_failure         — Groq raises → "" not raised
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

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


_LONG_ANSWER = (
    "I found three senior Python engineer roles on LinkedIn. "
    "The first is at Stripe paying one hundred and sixty thousand dollars, "
    "the second is at Airbnb with remote options available."
)


# ---------------------------------------------------------------------------
# test_suggest_returns_string_for_browser_task
# ---------------------------------------------------------------------------

def test_suggest_returns_string_for_browser_task():
    """
    With a mocked Groq client returning 'Also — want me to apply to any of those?',
    suggest() must return that non-empty string starting with 'Also'.
    """
    fake_reply = "Also — want me to apply to any of those?"
    fake_resp = _make_fake_groq_response(fake_reply)

    with patch("prompt_suggester._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_resp
        mock_get_client.return_value = mock_client

        import prompt_suggester
        result = prompt_suggester.suggest(
            intent_type="browser_task",
            answer=_LONG_ANSWER,
        )

    assert isinstance(result, str), "suggest() must return str"
    assert result != "", "suggest() must return non-empty string for valid input"
    assert result.lower().startswith("also"), (
        f"Result must start with 'Also', got: {result!r}"
    )


# ---------------------------------------------------------------------------
# test_suggest_returns_empty_for_short_answer
# ---------------------------------------------------------------------------

def test_suggest_returns_empty_for_short_answer():
    """
    If answer has fewer than 20 words, suggest() must return '' without
    calling Groq at all.
    """
    short_answer = "I found two jobs."  # well under 20 words

    with patch("prompt_suggester._get_client") as mock_get_client:
        import prompt_suggester
        result = prompt_suggester.suggest(
            intent_type="browser_task",
            answer=short_answer,
        )
        # Groq should never be contacted for a short answer
        mock_get_client.assert_not_called()

    assert result == "", f"Expected '' for short answer, got {result!r}"


# ---------------------------------------------------------------------------
# test_suggest_returns_empty_for_excluded_intent
# ---------------------------------------------------------------------------

def test_suggest_returns_empty_for_excluded_intent():
    """
    For intents not in _TRIGGER_INTENTS (e.g. 'weather'), suggest() returns ''
    regardless of answer length.
    """
    with patch("prompt_suggester._get_client") as mock_get_client:
        import prompt_suggester
        result = prompt_suggester.suggest(
            intent_type="weather",
            answer=_LONG_ANSWER,
        )
        mock_get_client.assert_not_called()

    assert result == "", f"Expected '' for excluded intent, got {result!r}"


# ---------------------------------------------------------------------------
# test_suggest_async_does_not_block
# ---------------------------------------------------------------------------

def test_suggest_async_does_not_block():
    """
    suggest_async() must return to the caller in under 0.5 s even when the
    underlying suggest() call would take several seconds.
    """
    block_event = threading.Event()

    def slow_suggest(intent_type: str, answer: str) -> str:
        block_event.wait()   # block until test releases it
        return "Also — slow suggestion."

    mock_speaker = MagicMock()

    with patch("prompt_suggester.suggest", side_effect=slow_suggest):
        import prompt_suggester
        t0 = time.monotonic()
        prompt_suggester.suggest_async(
            intent_type="browser_task",
            answer=_LONG_ANSWER,
            speaker=mock_speaker,
        )
        elapsed = time.monotonic() - t0

    # Release the blocked thread so the process doesn't hang
    block_event.set()

    assert elapsed < 0.5, (
        f"suggest_async() blocked for {elapsed:.3f}s — expected < 0.5s"
    )


# ---------------------------------------------------------------------------
# test_suggest_graceful_on_groq_failure
# ---------------------------------------------------------------------------

def test_suggest_graceful_on_groq_failure():
    """
    When Groq raises an exception, suggest() must return '' — never re-raise.
    """
    with patch("prompt_suggester._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Groq down")
        mock_get_client.return_value = mock_client

        import prompt_suggester
        result = prompt_suggester.suggest(
            intent_type="jobs",
            answer=_LONG_ANSWER,
        )

    assert result == "", f"Expected '' on Groq failure, got {result!r}"
