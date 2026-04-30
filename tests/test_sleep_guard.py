"""
Tests for sleep_guard.py — SleepGuard keeps Mac awake during long tasks.

Strategy:
- All subprocess.Popen calls are mocked so tests pass in CI (non-darwin too).
- threading.Timer is mocked to prevent real timers firing during tests.
- Platform detection is patched for the no-op test.
"""
from __future__ import annotations

import sys
import os
import threading
import importlib
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_proc() -> MagicMock:
    """Return a fake Popen process that is still running."""
    proc = MagicMock()
    proc.poll.return_value = None   # process still running
    proc.pid = 12345
    return proc


def _fresh_guard(monkeypatch, mock_proc=None, platform="darwin"):
    """
    Import a completely fresh SleepGuard instance with mocked subprocess and
    mocked platform.  Returns (guard_instance, MockPopen).
    """
    # Remove cached module so module-level state resets
    for key in list(sys.modules.keys()):
        if "sleep_guard" in key:
            del sys.modules[key]

    if mock_proc is None:
        mock_proc = _make_mock_proc()

    mock_popen_cls = MagicMock(return_value=mock_proc)

    with patch("sys.platform", platform), \
         patch("subprocess.Popen", mock_popen_cls):
        import sleep_guard as sg
        # Also patch sys.platform inside the loaded module
        monkeypatch.setattr(sg, "_PLATFORM", platform)
        guard = sg.SleepGuard()

    return guard, mock_popen_cls, mock_proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAcquireStartsCaffeinate:
    """acquire() should spawn caffeinate -i -t 300 on macOS."""

    def test_acquire_starts_caffeinate(self, monkeypatch):
        mock_proc = _make_mock_proc()
        mock_popen = MagicMock(return_value=mock_proc)

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()
            guard.acquire()

        mock_popen.assert_called_once_with(
            ["caffeinate", "-i", "-t", "300"],
            stdout=subprocess_DEVNULL(),
            stderr=subprocess_DEVNULL(),
        )

    def test_acquire_increments_ref_count(self, monkeypatch):
        mock_popen = MagicMock(return_value=_make_mock_proc())

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()
            guard.acquire()
            guard.acquire()

        assert guard._ref_count == 2

    def test_acquire_only_spawns_once_for_multiple_calls(self, monkeypatch):
        """Multiple acquire() calls should only spawn one caffeinate process."""
        mock_popen = MagicMock(return_value=_make_mock_proc())

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()
            guard.acquire()
            guard.acquire()
            guard.acquire()

        mock_popen.assert_called_once()


class TestReleaseKillsProcess:
    """release() should kill the caffeinate process when ref count hits 0."""

    def test_release_kills_process(self, monkeypatch):
        mock_proc = _make_mock_proc()
        mock_popen = MagicMock(return_value=mock_proc)

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()
            guard.acquire()
            guard.release()

        mock_proc.kill.assert_called_once()

    def test_release_resets_ref_count_to_zero(self, monkeypatch):
        mock_proc = _make_mock_proc()
        mock_popen = MagicMock(return_value=mock_proc)

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()
            guard.acquire()
            guard.release()

        assert guard._ref_count == 0

    def test_release_without_acquire_is_safe(self, monkeypatch):
        """release() with no prior acquire should not raise."""
        mock_popen = MagicMock(return_value=_make_mock_proc())

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()

        # Should not raise
        guard.release()
        assert guard._ref_count == 0


class TestNestedAcquire:
    """Nested callers: only kill caffeinate when the LAST release() fires."""

    def test_nested_acquire_only_kills_on_last_release(self, monkeypatch):
        mock_proc = _make_mock_proc()
        mock_popen = MagicMock(return_value=mock_proc)

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()

            guard.acquire()
            guard.acquire()
            guard.acquire()

            guard.release()
            # Still 2 refs — process should NOT be killed yet
            mock_proc.kill.assert_not_called()

            guard.release()
            # Still 1 ref — process should NOT be killed yet
            mock_proc.kill.assert_not_called()

            guard.release()
            # Ref count is now 0 — process MUST be killed
            mock_proc.kill.assert_called_once()

    def test_reacquire_after_full_release_spawns_new_process(self, monkeypatch):
        """After a full release cycle, acquire() should spawn a fresh process."""
        proc1 = _make_mock_proc()
        proc2 = _make_mock_proc()
        mock_popen = MagicMock(side_effect=[proc1, proc2])

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()

            guard.acquire()
            guard.release()
            guard.acquire()  # second cycle

        assert mock_popen.call_count == 2


class TestNoopOnNonMacos:
    """On non-darwin platforms, acquire/release should be silent no-ops."""

    def test_noop_on_non_macos(self, monkeypatch):
        mock_popen = MagicMock(return_value=_make_mock_proc())

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "linux")
            guard = sg.SleepGuard()
            guard.acquire()
            guard.release()

        mock_popen.assert_not_called()

    def test_noop_ref_count_unchanged_on_non_macos(self, monkeypatch):
        """On non-macos, ref count should still be managed correctly."""
        mock_popen = MagicMock(return_value=_make_mock_proc())

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "linux")
            guard = sg.SleepGuard()
            guard.acquire()
            assert guard._ref_count == 1
            guard.release()
            assert guard._ref_count == 0


class TestRestartTimer:
    """SleepGuard must restart caffeinate before the 5-min timeout (at 4 min)."""

    def test_restart_timer_is_scheduled_on_acquire(self, monkeypatch):
        """A timer should be created when caffeinate is first spawned."""
        mock_popen = MagicMock(return_value=_make_mock_proc())
        mock_timer = MagicMock()
        mock_timer_cls = MagicMock(return_value=mock_timer)

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen), \
             patch("threading.Timer", mock_timer_cls):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()
            guard.acquire()

        # Timer should be created with 4-minute (240s) interval
        mock_timer_cls.assert_called_once()
        delay_arg = mock_timer_cls.call_args[0][0]
        assert delay_arg == 240, f"Expected 240s restart interval, got {delay_arg}s"
        mock_timer.start.assert_called_once()

    def test_restart_timer_cancelled_on_release(self, monkeypatch):
        """Timer must be cancelled when caffeinate is killed."""
        mock_popen = MagicMock(return_value=_make_mock_proc())
        mock_timer = MagicMock()
        mock_timer_cls = MagicMock(return_value=mock_timer)

        for key in list(sys.modules.keys()):
            if "sleep_guard" in key:
                del sys.modules[key]

        with patch("subprocess.Popen", mock_popen), \
             patch("threading.Timer", mock_timer_cls):
            import sleep_guard as sg
            monkeypatch.setattr(sg, "_PLATFORM", "darwin")
            guard = sg.SleepGuard()
            guard.acquire()
            guard.release()

        mock_timer.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# Helper used inline above — avoid importing subprocess at module level so
# the mock patch target is unambiguous.
# ---------------------------------------------------------------------------

def subprocess_DEVNULL():
    """Return subprocess.DEVNULL for use in assert_called_once_with."""
    import subprocess
    return subprocess.DEVNULL
