"""
tests/test_compact_messages.py — Unit tests for compact.py message-history compaction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm.base import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                       provider_used="groq", model_used="llama-3.3-70b-versatile")


def _make_messages(n_pairs: int) -> list[dict]:
    """Build n_pairs of user+assistant messages."""
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"Question {i}"})
        msgs.append({"role": "assistant", "content": f"Answer {i}"})
    return msgs


# --- should_compact_messages ---

def test_should_compact_messages_false_for_short_history():
    """Short conversation is well under 80% of context — no compaction needed."""
    import aria.core.compact as compact
    messages = _make_messages(5)
    assert not compact.should_compact_messages(messages)


def test_should_compact_messages_true_when_over_threshold():
    """Messages that sum to > 80% of 200k context tokens trigger compaction."""
    import aria.core.compact as compact
    # 80% of 200k = 160k tokens ≈ 640k chars. One huge message exceeds this.
    huge_messages = [
        {"role": "user", "content": "x" * 700_000},
    ]
    assert compact.should_compact_messages(huge_messages)


def test_should_compact_messages_uses_json_size_for_estimation():
    """Token estimate is based on total JSON length / 4, not just content chars."""
    import aria.core.compact as compact
    # Construct messages just under and just over 640k chars total JSON
    threshold_chars = int(0.8 * 200_000 * 4)
    under = [{"role": "user", "content": "x" * (threshold_chars - 100)}]
    over = [{"role": "user", "content": "x" * (threshold_chars + 100)}]
    assert not compact.should_compact_messages(under)
    assert compact.should_compact_messages(over)


# --- compact_messages ---

def test_compact_messages_unchanged_when_20_or_fewer_messages():
    """10 or fewer turn pairs (≤20 messages) are returned as-is."""
    import aria.core.compact as compact
    messages = _make_messages(10)
    result = compact.compact_messages(messages)
    assert result == messages


def test_compact_messages_keeps_last_20_messages():
    """15 turn pairs → 5 older pairs summarized, last 10 pairs (20 msgs) kept verbatim."""
    import aria.core.compact as compact
    messages = _make_messages(15)
    with patch("aria.llm.llm_client.complete") as mock:
        mock.return_value = _make_llm_response("Prior context summary")
        result = compact.compact_messages(messages)

    # 1 summary msg + 20 recent messages = 21
    assert len(result) == 21
    # First message is the summary
    assert result[0]["role"] == "user"
    assert "Prior context summary" in result[0]["content"]
    # Last 20 messages preserved verbatim
    assert result[1:] == messages[-20:]


def test_compact_messages_summary_wraps_prior_content():
    """Summary message contains a clear label so the LLM knows it's prior context."""
    import aria.core.compact as compact
    messages = _make_messages(12)
    with patch("aria.llm.llm_client.complete") as mock:
        mock.return_value = _make_llm_response("User asked about jobs")
        result = compact.compact_messages(messages)

    summary_content = result[0]["content"]
    assert "prior context" in summary_content.lower() or "Prior context" in summary_content


def test_compact_messages_graceful_on_llm_failure():
    """When LLM fails, compact_messages returns the original messages unchanged."""
    import aria.core.compact as compact
    messages = _make_messages(15)
    with patch("aria.llm.llm_client.complete", side_effect=RuntimeError("LLM down")):
        result = compact.compact_messages(messages)

    assert result == messages


def test_compact_messages_logs_compaction_stats(caplog):
    """compact_messages logs how many turns were compacted and token delta."""
    import logging
    import aria.core.compact as compact

    messages = _make_messages(15)
    with patch("aria.llm.llm_client.complete") as mock:
        mock.return_value = _make_llm_response("Summary")
        with caplog.at_level(logging.DEBUG, logger="aria.core.compact"):
            compact.compact_messages(messages)

    assert any("Compacted" in r.message for r in caplog.records)
