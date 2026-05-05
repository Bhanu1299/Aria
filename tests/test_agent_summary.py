"""Tests for agent_summary.py — post-task spoken summary."""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch


class TestAgentSummary(unittest.TestCase):

    def _make_response(self, text: str):
        mock_content = MagicMock()
        mock_content.text = text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    def test_summarize_returns_string(self):
        import agent_summary
        with patch("agent_summary._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = self._make_response(
                "Done — created app.py and ran it successfully."
            )
            result = agent_summary.summarize("code", "Created app.py with Flask hello world.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_summarize_returns_fallback_on_api_error(self):
        import agent_summary
        with patch("agent_summary._get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API down")
            result = agent_summary.summarize("code", "some answer")
        self.assertEqual(result, "some answer")

    def test_summarize_async_calls_speaker_in_thread(self):
        import agent_summary
        speaker = MagicMock()
        with patch("agent_summary._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = self._make_response("Done.")
            agent_summary.summarize_async("knowledge", "The answer is 42.", speaker)
            time.sleep(0.2)
        speaker.say.assert_called_once()

    def test_summarize_async_does_not_block(self):
        import agent_summary
        speaker = MagicMock()
        start = time.time()
        with patch("agent_summary._get_client") as mock_client:
            def slow(**kw):
                time.sleep(1)
                return self._make_response("Done.")
            mock_client.return_value.messages.create.side_effect = slow
            agent_summary.summarize_async("knowledge", "answer", speaker)
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.5)

    def test_summarize_empty_answer_returns_fallback(self):
        import agent_summary
        result = agent_summary.summarize("code", "")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
