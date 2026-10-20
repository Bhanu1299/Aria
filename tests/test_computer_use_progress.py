from __future__ import annotations

"""
tests/test_computer_use_progress.py — TDD tests for progress_fn callback
in computer_use.research_loop() and run_loop().
"""

import unittest
from unittest.mock import MagicMock, patch, call


import aria.web.computer_use as computer_use


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DONE_ACTION = {"action": "done", "summary": "test done"}
_STUCK_ACTION = {"action": "stuck", "reason": "test stuck"}
_NAV_ACTION = {"action": "navigate", "url": "https://example.com/path", "reason": "go"}
_CLICK_ACTION = {"action": "click", "reason": "press button", "x": 10, "y": 20}


# ---------------------------------------------------------------------------
# Test 1: progress_fn is called once per executable step, not for terminal actions
# ---------------------------------------------------------------------------

class TestProgressFnCalledOnEachStep(unittest.TestCase):

    def test_progress_fn_called_on_each_step(self) -> None:
        """progress_fn must be called for each executable step but NOT for 'done'."""
        call_count = 0
        messages: list[str] = []

        def fake_progress(msg: str) -> None:
            nonlocal call_count
            call_count += 1
            messages.append(msg)

        # First call → navigate (executable), second call → done (terminal)
        side_effects = [_NAV_ACTION, _DONE_ACTION]
        decide_mock = MagicMock(side_effect=side_effects)

        with (
            patch.object(computer_use, "_dom_research_decide", decide_mock),
            patch.object(computer_use, "execute", MagicMock()),
            patch.object(computer_use, "_human_sleep", MagicMock()),
            patch.object(computer_use, "agent_browser") as mock_ab,
            patch.object(computer_use.dom_browser, "get_dom_snapshot",
                         return_value=("PAGE TEXT: hi", 10)),
        ):
            mock_ab.navigate = MagicMock()
            result = computer_use.research_loop(
                goal="test goal",
                max_steps=10,
                progress_fn=fake_progress,
            )

        # navigate step should trigger progress_fn once; done should not
        self.assertEqual(call_count, 1)
        self.assertIn("example.com", messages[0])
        self.assertEqual(result, "test done")

    def test_progress_fn_not_called_for_stuck(self) -> None:
        """progress_fn must NOT be called for stuck terminal actions."""
        call_count = 0

        def fake_progress(msg: str) -> None:
            nonlocal call_count
            call_count += 1

        # Two stucks with no data escalate research_loop to the Claude decider,
        # which then navigates (one progress message) and finishes.
        decide_mock = MagicMock(side_effect=[_STUCK_ACTION, _STUCK_ACTION])
        claude_mock = MagicMock(side_effect=[_NAV_ACTION, _DONE_ACTION])

        with (
            patch.object(computer_use, "_dom_research_decide", decide_mock),
            patch.object(computer_use, "_claude_research_decide", claude_mock),
            patch.object(computer_use, "take_screenshot", MagicMock(return_value="b64")),
            patch.object(computer_use, "execute", MagicMock()),
            patch.object(computer_use, "_human_sleep", MagicMock()),
            patch.object(computer_use, "agent_browser") as mock_ab,
            patch.object(computer_use.dom_browser, "get_dom_snapshot",
                         return_value=("PAGE TEXT: hi", 10)),
        ):
            mock_ab.navigate = MagicMock()
            computer_use.research_loop(
                goal="test goal",
                max_steps=10,
                progress_fn=fake_progress,
            )

        # Stuck actions must not trigger progress_fn; only navigate should
        self.assertEqual(call_count, 1)


# ---------------------------------------------------------------------------
# Test 2: progress_fn=None does not crash
# ---------------------------------------------------------------------------

class TestProgressFnNoneDoesNotCrash(unittest.TestCase):

    def test_progress_fn_none_does_not_crash(self) -> None:
        """research_loop must not raise when progress_fn is omitted (None)."""
        side_effects = [_NAV_ACTION, _DONE_ACTION]
        decide_mock = MagicMock(side_effect=side_effects)

        with (
            patch.object(computer_use, "_dom_research_decide", decide_mock),
            patch.object(computer_use, "execute", MagicMock()),
            patch.object(computer_use, "_human_sleep", MagicMock()),
            patch.object(computer_use, "agent_browser") as mock_ab,
            patch.object(computer_use.dom_browser, "get_dom_snapshot",
                         return_value=("PAGE TEXT: hi", 10)),
        ):
            mock_ab.navigate = MagicMock()
            # Must not raise
            result = computer_use.research_loop(
                goal="test goal",
                max_steps=10,
                # progress_fn intentionally omitted
            )

        self.assertEqual(result, "test done")

    def test_progress_fn_exception_does_not_crash_loop(self) -> None:
        """An exception inside progress_fn must be swallowed, loop continues."""
        def bad_progress(msg: str) -> None:
            raise RuntimeError("boom")

        side_effects = [_NAV_ACTION, _DONE_ACTION]
        decide_mock = MagicMock(side_effect=side_effects)

        with (
            patch.object(computer_use, "_dom_research_decide", decide_mock),
            patch.object(computer_use, "execute", MagicMock()),
            patch.object(computer_use, "_human_sleep", MagicMock()),
            patch.object(computer_use, "agent_browser") as mock_ab,
            patch.object(computer_use.dom_browser, "get_dom_snapshot",
                         return_value=("PAGE TEXT: hi", 10)),
        ):
            mock_ab.navigate = MagicMock()
            result = computer_use.research_loop(
                goal="test goal",
                max_steps=10,
                progress_fn=bad_progress,
            )

        self.assertEqual(result, "test done")


# ---------------------------------------------------------------------------
# Test 3: dedup logic in _on_progress — same message not repeated
# ---------------------------------------------------------------------------

class TestDuplicateMessagesNotRepeated(unittest.TestCase):

    def test_duplicate_messages_not_repeated(self) -> None:
        """_on_progress should call speaker.say only when message differs."""
        mock_say = MagicMock()

        # Replicate the exact _on_progress pattern from main.py
        _last_progress: list[str] = [""]

        def _on_progress(msg: str) -> None:
            if msg and msg != _last_progress[0]:
                _last_progress[0] = msg
                mock_say(msg)

        _on_progress("Navigating to google.com")
        _on_progress("Navigating to google.com")  # duplicate — should NOT call say again
        _on_progress("Typing: hello")

        self.assertEqual(mock_say.call_count, 2)
        mock_say.assert_any_call("Navigating to google.com")
        mock_say.assert_any_call("Typing: hello")

    def test_empty_message_not_spoken(self) -> None:
        """_on_progress must skip empty strings."""
        mock_say = MagicMock()
        _last_progress: list[str] = [""]

        def _on_progress(msg: str) -> None:
            if msg and msg != _last_progress[0]:
                _last_progress[0] = msg
                mock_say(msg)

        _on_progress("")
        _on_progress("  ")  # not empty string but truthy — still spoken
        _on_progress("")

        # Only "  " (truthy) should be spoken
        self.assertEqual(mock_say.call_count, 1)


# ---------------------------------------------------------------------------
# Test 4: _format_progress helper
# ---------------------------------------------------------------------------

class TestFormatProgress(unittest.TestCase):

    def test_navigate_extracts_domain(self) -> None:
        result = computer_use._format_progress(
            {"action": "navigate", "url": "https://google.com/search?q=test"}
        )
        self.assertIn("google.com", result)

    def test_type_truncates_long_text(self) -> None:
        long_text = "a" * 50
        result = computer_use._format_progress({"action": "type", "text": long_text})
        self.assertIn("Typing", result)
        self.assertIn("...", result)

    def test_type_short_text_no_ellipsis(self) -> None:
        result = computer_use._format_progress({"action": "type", "text": "hello"})
        self.assertIn("hello", result)
        self.assertNotIn("...", result)

    def test_click_with_reason(self) -> None:
        result = computer_use._format_progress({"action": "click", "reason": "press submit"})
        self.assertIn("Clicking", result)
        self.assertIn("press submit", result)

    def test_click_without_reason(self) -> None:
        result = computer_use._format_progress({"action": "click"})
        self.assertEqual(result, "Clicking")

    def test_scroll(self) -> None:
        result = computer_use._format_progress({"action": "scroll", "direction": "up"})
        self.assertIn("up", result)

    def test_search(self) -> None:
        result = computer_use._format_progress({"action": "search", "text": "python jobs"})
        self.assertIn("python jobs", result)

    def test_unknown_action_with_reason(self) -> None:
        result = computer_use._format_progress({"action": "unknown_op", "reason": "get the title"})
        self.assertIn("get the title", result)

    def test_empty_action_fallback(self) -> None:
        result = computer_use._format_progress({})
        self.assertEqual(result, "Working...")


if __name__ == "__main__":
    unittest.main()
