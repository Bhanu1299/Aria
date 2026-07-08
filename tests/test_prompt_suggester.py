"""
tests/test_prompt_suggester.py — Unit tests for prompt_suggester.py
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm.base import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                       provider_used="groq", model_used="llama-3.3-70b-versatile")


_LONG_ANSWER = (
    "I found three senior Python engineer roles on LinkedIn. "
    "The first is at Stripe paying one hundred and sixty thousand dollars, "
    "the second is at Airbnb with remote options available."
)


def test_suggest_returns_string_for_browser_task():
    """suggest() must return non-empty string starting with 'Also' for valid input."""
    fake_reply = "Also — want me to apply to any of those?"

    with patch("aria.llm.llm_client.complete") as mock_complete:
        mock_complete.return_value = _make_llm_response(fake_reply)

        import aria.features.prompt_suggester as prompt_suggester
        result = prompt_suggester.suggest(intent_type="browser_task", answer=_LONG_ANSWER)

    assert isinstance(result, str)
    assert result != ""
    assert result.lower().startswith("also"), f"Result must start with 'Also', got: {result!r}"


def test_suggest_returns_empty_for_short_answer():
    """If answer has fewer than 20 words, suggest() must return '' without calling LLM."""
    short_answer = "I found two jobs."

    with patch("aria.llm.llm_client.complete") as mock_complete:
        import aria.features.prompt_suggester as prompt_suggester
        result = prompt_suggester.suggest(intent_type="browser_task", answer=short_answer)
        mock_complete.assert_not_called()

    assert result == ""


def test_suggest_returns_empty_for_excluded_intent():
    """For intents not in _TRIGGER_INTENTS, suggest() returns '' without calling LLM."""
    with patch("aria.llm.llm_client.complete") as mock_complete:
        import aria.features.prompt_suggester as prompt_suggester
        result = prompt_suggester.suggest(intent_type="weather", answer=_LONG_ANSWER)
        mock_complete.assert_not_called()

    assert result == ""


def test_suggest_async_does_not_block():
    """suggest_async() must return in under 0.5s even when suggest() is slow."""
    block_event = threading.Event()

    def slow_suggest(intent_type, answer):
        block_event.wait()
        return "Also — slow suggestion."

    mock_speaker = MagicMock()

    with patch("aria.features.prompt_suggester.suggest", side_effect=slow_suggest):
        import aria.features.prompt_suggester as prompt_suggester
        t0 = time.monotonic()
        prompt_suggester.suggest_async("browser_task", _LONG_ANSWER, mock_speaker)
        elapsed = time.monotonic() - t0

    block_event.set()
    assert elapsed < 0.5


def test_suggest_graceful_on_groq_failure():
    """When LLM raises, suggest() must return '' without re-raising."""
    with patch("aria.llm.llm_client.complete", side_effect=RuntimeError("LLM down")):
        import aria.features.prompt_suggester as prompt_suggester
        result = prompt_suggester.suggest(intent_type="jobs", answer=_LONG_ANSWER)

    assert result == ""
