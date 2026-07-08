"""
tests/test_compact.py — Unit tests for compact.py session notes compaction.
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


def test_compress_returns_shorter_string():
    """compress() must return a shorter string when LLM succeeds."""
    long_notes = "- User asked about Python jobs\n" * 100
    compressed = "- User asked about Python jobs\n- Aria returned results"

    with patch("aria.llm.llm_client.complete") as mock_complete:
        mock_complete.return_value = _make_llm_response(compressed)

        import aria.core.compact as compact
        result = compact.compress(long_notes)

    assert isinstance(result, str)
    assert result == compressed.strip()
    assert len(result) < len(long_notes)


def test_compress_graceful_on_groq_failure():
    """When LLM raises, compress() returns the original notes unchanged."""
    original_notes = "- User asked about Python jobs\n" * 100

    with patch("aria.llm.llm_client.complete", side_effect=RuntimeError("LLM error")):
        import aria.core.compact as compact
        result = compact.compress(original_notes)

    assert result == original_notes


def test_store_session_notes_compacts_when_over_threshold():
    """needs_compaction() returns True when notes exceed 3000 chars."""
    import aria.core.compact as compact
    assert compact.needs_compaction("x" * 3001)


def test_store_session_notes_does_not_compact_when_under_threshold():
    """needs_compaction() returns False for notes under 3000 chars."""
    import aria.core.compact as compact
    assert not compact.needs_compaction("x" * 100)
