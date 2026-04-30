from __future__ import annotations

"""
tests/test_computer_use_recovery.py — TDD tests for retry logic, budget warnings,
and stuck-detection in computer_use.py (Phase 4, Task 7).
"""

import sys
import time
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

# groq stub — define a minimal RateLimitError compatible with tests
class _FakeRateLimitError(Exception):
    """Minimal stand-in for groq.RateLimitError with 'rate limit' in message."""
    pass

_groq_stub = _stub_module(
    "groq",
    Groq=MagicMock(),
    RateLimitError=_FakeRateLimitError,
)
sys.modules["groq"] = _groq_stub

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
from computer_use import _with_retry  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DONE_ACTION = {"action": "done", "summary": "test done"}
_STUCK_ACTION = {"action": "stuck", "reason": "test stuck"}
_NAV_ACTION = {"action": "navigate", "url": "https://example.com/path", "reason": "go"}


# ---------------------------------------------------------------------------
# Test 1: _with_retry retries on RateLimitError and succeeds on 3rd call
# ---------------------------------------------------------------------------

class TestRetryOnRateLimit(unittest.TestCase):

    def test_retry_on_rate_limit(self) -> None:
        """
        _with_retry should call fn up to max_retries+1 times.
        If fn raises RateLimitError on first 2 calls and succeeds on 3rd,
        the result should be the success value and 3 total calls made.
        time.sleep is mocked to avoid actual delays.
        """
        # Build a minimal mock RateLimitError — groq.RateLimitError needs an httpx.Response
        # so we use a simpler generic exception with "rate limit" in the message.
        class FakeRateLimitError(Exception):
            pass

        success_value = {"action": "done", "summary": "ok"}
        call_count = 0

        def fn_that_fails_twice(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeRateLimitError("rate limit exceeded")
            return success_value

        with patch("time.sleep") as mock_sleep:
            result = _with_retry(fn_that_fails_twice, max_retries=3, base_delay=1.0)

        self.assertEqual(result, success_value)
        self.assertEqual(call_count, 3)
        # time.sleep should have been called twice (once per failed attempt before success)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_retry_uses_exponential_backoff(self) -> None:
        """
        _with_retry should sleep with exponential backoff: 1.0, 2.0, 4.0 for base_delay=1.0.
        """
        call_count = 0

        def always_fails(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        with patch("time.sleep") as mock_sleep:
            with self.assertRaises(ValueError):
                _with_retry(always_fails, max_retries=3, base_delay=1.0)

        # 3 retries means 3 sleep calls: 1.0, 2.0, 4.0
        self.assertEqual(mock_sleep.call_count, 3)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        self.assertEqual(sleep_calls, [1.0, 2.0, 4.0])


# ---------------------------------------------------------------------------
# Test 2: _with_retry gives up after max_retries and re-raises
# ---------------------------------------------------------------------------

class TestRetryGivesUpAfter3Attempts(unittest.TestCase):

    def test_retry_gives_up_after_3_attempts(self) -> None:
        """
        If fn always raises, _with_retry should re-raise after exhausting
        max_retries=3 (i.e. 4 total attempts: 1 original + 3 retries).
        time.sleep should be called 3 times (before each retry).
        """
        class AlwaysFailsError(Exception):
            pass

        call_count = 0

        def always_fails(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise AlwaysFailsError("permanent failure")

        with patch("time.sleep") as mock_sleep:
            with self.assertRaises(AlwaysFailsError):
                _with_retry(always_fails, max_retries=3, base_delay=1.0)

        # 4 total attempts: attempt 0,1,2 raise and sleep, attempt 3 raises and is re-raised
        self.assertEqual(call_count, 4)
        # sleep called 3 times (before retries 1, 2, 3)
        self.assertEqual(mock_sleep.call_count, 3)


# ---------------------------------------------------------------------------
# Test 3: Stuck circuit breaker fires after 3 identical repeats
# ---------------------------------------------------------------------------

class TestStuckCircuitBreakerFiresAfter3Repeats(unittest.TestCase):

    def test_stuck_circuit_breaker_fires_after_3_repeats(self) -> None:
        """
        research_loop should break out early if the same (action, url) tuple
        repeats 3 times without any state change. The loop must not run forever.
        After the circuit-breaker fires, it may attempt a recovery scroll and then
        abort, returning a non-empty string.
        """
        same_nav = {"action": "navigate", "url": "https://example.com/stuck", "reason": "try again"}

        # Always return the same navigate action — if circuit breaker is working,
        # this should not loop indefinitely and must return within max_steps
        decide_mock = MagicMock(return_value=same_nav)

        with (
            patch.object(computer_use, "_dom_research_decide", decide_mock),
            patch.object(computer_use, "execute", MagicMock()),
            patch.object(computer_use, "_human_sleep", MagicMock()),
            patch.object(computer_use, "agent_browser") as mock_ab,
        ):
            mock_ab.navigate = MagicMock()
            result = computer_use.research_loop(
                goal="test stuck detection",
                max_steps=20,
            )

        # The loop must have terminated before running all 20 steps
        total_calls = decide_mock.call_count
        self.assertLess(total_calls, 20,
            f"Circuit breaker did not fire — decide was called {total_calls} times (max 20)")
        # Must return a non-empty string
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


# ---------------------------------------------------------------------------
# Test 4: Budget warning fires at 80% of max_steps
# ---------------------------------------------------------------------------

class TestBudgetWarningNearStepLimit(unittest.TestCase):

    def test_budget_warning_near_step_limit(self) -> None:
        """
        research_loop should call progress_fn with a budget warning message
        when step == int(max_steps * 0.8).
        With max_steps=5, threshold = int(5 * 0.8) = 4.
        The loop runs: step 1 → navigate, step 2 → navigate, step 3 → navigate,
        step 4 → navigate (budget warning fires here), step 5 → done.
        """
        max_steps = 5
        budget_threshold = int(max_steps * 0.8)  # 4

        progress_messages: list[str] = []

        def fake_progress(msg: str) -> None:
            progress_messages.append(msg)

        # step 1-4: navigate actions, step 5: done
        nav = {"action": "navigate", "url": "https://example.com/page", "reason": "search"}
        side_effects = [nav] * (budget_threshold - 1) + [nav, _DONE_ACTION]

        decide_mock = MagicMock(side_effect=side_effects)

        with (
            patch.object(computer_use, "_dom_research_decide", decide_mock),
            patch.object(computer_use, "execute", MagicMock()),
            patch.object(computer_use, "_human_sleep", MagicMock()),
            patch.object(computer_use, "agent_browser") as mock_ab,
        ):
            mock_ab.navigate = MagicMock()
            result = computer_use.research_loop(
                goal="test budget warning",
                max_steps=max_steps,
                progress_fn=fake_progress,
            )

        self.assertEqual(result, "test done")

        # Find the budget warning message among all progress messages
        budget_msgs = [m for m in progress_messages if "limit" in m.lower() or "budget" in m.lower() or "wrapping" in m.lower()]
        self.assertTrue(
            len(budget_msgs) >= 1,
            f"Expected at least one budget warning in progress messages, got: {progress_messages}"
        )


if __name__ == "__main__":
    unittest.main()
