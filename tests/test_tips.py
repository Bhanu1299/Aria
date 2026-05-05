"""Tests for tips.py — periodic tip scheduler."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestTips(unittest.TestCase):

    def test_no_tip_on_non_multiple_of_10(self):
        import tips
        speaker = MagicMock()
        for count in (1, 3, 7, 9, 11, 19):
            tips.maybe_speak_tip(count, speaker)
        speaker.say.assert_not_called()

    def test_speaks_tip_on_multiple_of_10(self):
        import tips
        speaker = MagicMock()
        tips.maybe_speak_tip(10, speaker)
        speaker.say.assert_called_once()
        spoken = speaker.say.call_args[0][0]
        self.assertIsInstance(spoken, str)
        self.assertGreater(len(spoken), 5)

    def test_cycles_through_tips(self):
        import tips
        speaker = MagicMock()
        tips.maybe_speak_tip(10, speaker)
        tips.maybe_speak_tip(20, speaker)
        tips.maybe_speak_tip(30, speaker)
        self.assertEqual(speaker.say.call_count, 3)
        calls = [c[0][0] for c in speaker.say.call_args_list]
        # At least two distinct tips spoken (list has > 1 tip)
        self.assertGreater(len(set(calls)), 1)

    def test_speaks_on_every_multiple_of_10(self):
        import tips
        speaker = MagicMock()
        for count in (10, 20, 30, 40, 50):
            tips.maybe_speak_tip(count, speaker)
        self.assertEqual(speaker.say.call_count, 5)

    def test_tips_list_is_non_empty(self):
        import tips
        self.assertGreater(len(tips.TIPS), 3)


if __name__ == "__main__":
    unittest.main()
