"""Tests for prevent_sleep.py — caffeinate wrapper."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch


class TestPreventSleep(unittest.TestCase):

    def setUp(self):
        # Reset module state between tests
        import importlib
        if "prevent_sleep" in sys.modules:
            del sys.modules["prevent_sleep"]

    def test_start_spawns_caffeinate_on_macos(self):
        with patch("sys.platform", "darwin"), \
             patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            import aria.system.prevent_sleep as prevent_sleep
            prevent_sleep.start()
            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            self.assertIn("caffeinate", args)
            prevent_sleep._ref_count = 0
            prevent_sleep._process = None

    def test_stop_kills_process(self):
        with patch("sys.platform", "darwin"), \
             patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            import aria.system.prevent_sleep as prevent_sleep
            prevent_sleep.start()
            prevent_sleep.stop()
            mock_proc.kill.assert_called_once()

    def test_reference_counting_keeps_process_alive(self):
        with patch("sys.platform", "darwin"), \
             patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            import aria.system.prevent_sleep as prevent_sleep
            prevent_sleep.start()
            prevent_sleep.start()
            prevent_sleep.stop()
            mock_proc.kill.assert_not_called()
            prevent_sleep.stop()
            mock_proc.kill.assert_called_once()

    def test_no_op_on_non_macos(self):
        with patch("sys.platform", "linux"), \
             patch("subprocess.Popen") as mock_popen:
            import aria.system.prevent_sleep as prevent_sleep
            prevent_sleep.start()
            prevent_sleep.stop()
            mock_popen.assert_not_called()

    def test_stop_without_start_does_not_crash(self):
        import aria.system.prevent_sleep as prevent_sleep
        try:
            prevent_sleep.stop()
        except Exception as e:
            self.fail(f"stop() raised unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
