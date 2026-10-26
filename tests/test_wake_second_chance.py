from __future__ import annotations

from aria.voice import wake_word as ww
from aria.voice.wake_word import SecondChance

NORMAL = 0.7


def test_a_near_miss_lowers_the_threshold_for_a_few_seconds():
    c = SecondChance()
    assert c.threshold(0.0, NORMAL) == NORMAL
    c.near_miss(0.0, 0.58)
    assert c.threshold(1.0, NORMAL) == ww._SECOND_CHANCE_THRESHOLD
    assert c.threshold(ww._SECOND_CHANCE_SECS + 0.1, NORMAL) == NORMAL


def test_a_reject_at_the_normal_threshold_opens_a_second_chance():
    c = SecondChance()
    c.rejected(0.0, lowered=False)
    assert c.threshold(1.0, NORMAL) == ww._SECOND_CHANCE_THRESHOLD


def test_noise_cannot_keep_the_window_open():
    """Room noise at 0.60–0.69: each lowered fire is rejected; the window must close and rest
    instead of reopening forever (the 2026-10-26 live run: ~200 rejected fires)."""
    c = SecondChance()
    c.near_miss(0.0, 0.6)
    c.rejected(1.0, lowered=True)
    assert c.threshold(1.5, NORMAL) == NORMAL
    c.near_miss(3.0, 0.62)                        # still resting
    c.rejected(4.0, lowered=False)                # even a full-threshold reject waits out the rest
    assert c.threshold(5.0, NORMAL) == NORMAL
    c.near_miss(1.0 + ww._SECOND_CHANCE_REST_SECS + 0.1, 0.6)
    assert c.threshold(1.0 + ww._SECOND_CHANCE_REST_SECS + 0.5, NORMAL) == ww._SECOND_CHANCE_THRESHOLD


def test_waking_closes_the_window():
    c = SecondChance()
    c.near_miss(0.0, 0.6)
    c.woke()
    assert c.threshold(1.0, NORMAL) == NORMAL
