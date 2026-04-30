from __future__ import annotations

"""
tests/test_computer_use_progress.py — TDD tests for progress_fn callback
in computer_use.research_loop() and run_loop().
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Minimal stubs so that importing computer_use doesn't need real dependencies
# ---------------------------------------------------------------------------

def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Playwright / browser stubs
_playwright_stub = _stub_module("playwright")
_playwright_sync = _stub_module("playwright.sync_api", sync_playwright=MagicMock())
sys.modules.setdefault("playwright", _playwright_stub)
sys.modules.setdefault("playwright.sync_api", _playwright_sync)

# sounddevice stub
sys.modules.setdefault("sounddevice", _stub_module("sounddevice"))

# agent_browser stub
_ab = _stub_module("agent_browser", run=MagicMock(), navigate=MagicMock())
sys.modules.setdefault("agent_browser", _ab)

# dom_browser stub — get_dom_snapshot returns minimal html + interactive_count > 5
_dom_browser = _stub_module(
    "dom_browser",
    get_dom_snapshot=MagicMock(return_value=("<html>PAGE TEXT: hi</html>", 10)),
    save_debug_screenshot=MagicMock(),
)
sys.modules.setdefault("dom_browser", _dom_browser)

# groq stub
sys.modules.setdefault("groq", _stub_module("groq", Groq=MagicMock()))

# anthropic stub
sys.modules.setdefault("anthropic", _stub_module("anthropic", Anthropic=MagicMock()))

# config stub
_config = _stub_module(
    "config",
    GROQ_API_KEY="test-key",
    ANTHROPIC_API_KEY="",
    SCREENSHOT_QUALITY=75,
    SCREENSHOT_MAX_WIDTH=1280,
)
sys.modules.setdefault("config", _config)

# Now import the module under test
import computer_use  # noqa: E402


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

        # research_loop treats first 'stuck' as continue (groq_consecutive_stucks<2),
        # so we need a navigate action after to trigger the done path.
        # Simpler: provide stuck x2 then done — loop returns after 2nd stuck with partial data.
        side_effects = [_STUCK_ACTION, _STUCK_ACTION, _NAV_ACTION, _DONE_ACTION]
        decide_mock = MagicMock(side_effect=side_effects)

        with (
            patch.object(computer_use, "_dom_research_decide", decide_mock),
            patch.object(computer_use, "execute", MagicMock()),
            patch.object(computer_use, "_human_sleep", MagicMock()),
            patch.object(computer_use, "agent_browser") as mock_ab,
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
