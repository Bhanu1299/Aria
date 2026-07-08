"""Tests for agent_summary.py — post-task spoken summary."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.llm.base import LLMResponse


def _make_llm_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                       provider_used="anthropic", model_used="claude-haiku-4-5-20251001")


class TestAgentSummary(unittest.TestCase):

    def test_summarize_returns_string(self):
        import aria.features.agent_summary as agent_summary
        with patch("aria.llm.llm_client.complete") as mock_complete:
            mock_complete.return_value = _make_llm_response(
                "Done — created app.py and ran it successfully."
            )
            result = agent_summary.summarize("code", "Created app.py with Flask hello world.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_summarize_returns_fallback_on_api_error(self):
        import aria.features.agent_summary as agent_summary
        with patch("aria.llm.llm_client.complete", side_effect=Exception("API down")):
            result = agent_summary.summarize("code", "some answer")
        self.assertEqual(result, "some answer")

    def test_summarize_async_calls_speaker_in_thread(self):
        import aria.features.agent_summary as agent_summary
        speaker = MagicMock()
        with patch("aria.llm.llm_client.complete") as mock_complete:
            mock_complete.return_value = _make_llm_response("Done.")
            agent_summary.summarize_async("knowledge", "The answer is 42.", speaker)
            time.sleep(0.3)
        speaker.say.assert_called_once()

    def test_summarize_async_does_not_block(self):
        import aria.features.agent_summary as agent_summary
        speaker = MagicMock()
        start = time.time()

        def slow(*args, **kwargs):
            time.sleep(1)
            return _make_llm_response("Done.")

        with patch("aria.llm.llm_client.complete", side_effect=slow):
            agent_summary.summarize_async("knowledge", "answer", speaker)
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.5)

    def test_summarize_empty_answer_returns_fallback(self):
        import aria.features.agent_summary as agent_summary
        result = agent_summary.summarize("code", "")
        self.assertEqual(result, "")
