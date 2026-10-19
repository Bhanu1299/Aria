"""tests/test_dispatch_parse.py — deterministic parsing for direct routes."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.dispatch.parse import (
    SlotError,
    humanize_seconds,
    normalize,
    parse_duration,
    parse_when,
    split_task_when,
    spoken_when,
    words_to_digits,
)

# Sunday 2026-10-18, 10:30 AM
NOW = datetime(2026, 10, 18, 10, 30)


@pytest.mark.parametrize("raw,expected", [
    ("Set a timer for 5 minutes.", "set a timer for 5 minutes"),
    ("Hey Aria, pause the music", "pause the music"),
    ("Aria pause", "pause"),
    ("Can you please lock my Mac?", "lock my mac"),
    ("Remind me at 5 p.m. to call Mom", "remind me at 5 pm to call mom"),
    ("what's 2.5 times 4", "what's 2.5 times 4"),
    ("Open Safari for me", "open safari"),
    ("What’s the time", "what's the time"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_words_to_digits():
    assert words_to_digits("twenty five minutes") == "25 minutes"
    assert words_to_digits("twenty-five") == "25"
    assert words_to_digits("call mom") == "call mom"


@pytest.mark.parametrize("text,seconds", [
    ("5 minutes", 300),
    ("five minutes", 300),
    ("1 hour and 30 minutes", 5400),
    ("an hour and a half", 5400),
    ("half an hour", 1800),
    ("90 seconds", 90),
    ("2 and a half minutes", 150),
    ("1.5 hours", 5400),
    ("ten-minute", 600),
    ("20 min", 1200),
    ("a minute", 60),
])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["pasta", "5", "minutes", "25 hours", "0 minutes", "5 bananas"])
def test_parse_duration_rejects(text):
    with pytest.raises(SlotError):
        parse_duration(text)


@pytest.mark.parametrize("text,expected", [
    ("at 5pm", datetime(2026, 10, 18, 17, 0)),
    ("at 5:30 pm", datetime(2026, 10, 18, 17, 30)),
    ("at 17:00", datetime(2026, 10, 18, 17, 0)),
    ("at 5", datetime(2026, 10, 18, 17, 0)),            # next 5 o'clock is 5 PM
    ("at 11", datetime(2026, 10, 18, 11, 0)),           # 11 AM is still ahead
    ("at 9", datetime(2026, 10, 18, 21, 0)),            # 9 AM passed → 9 PM
    ("at noon", datetime(2026, 10, 18, 12, 0)),
    ("in 10 minutes", datetime(2026, 10, 18, 10, 40)),
    ("in an hour", datetime(2026, 10, 18, 11, 30)),
    ("tomorrow", datetime(2026, 10, 19, 9, 0)),
    ("tomorrow at 7", datetime(2026, 10, 19, 7, 0)),
    ("tomorrow at 5pm", datetime(2026, 10, 19, 17, 0)),
    ("at 5pm tomorrow", datetime(2026, 10, 19, 17, 0)),
    ("tomorrow morning", datetime(2026, 10, 19, 9, 0)),
    ("tonight", datetime(2026, 10, 18, 20, 0)),
    ("tonight at 8", datetime(2026, 10, 18, 20, 0)),
    ("this evening", datetime(2026, 10, 18, 18, 0)),
    ("on friday at 3", datetime(2026, 10, 23, 15, 0)),
    ("monday", datetime(2026, 10, 19, 9, 0)),
    ("next sunday", datetime(2026, 10, 25, 9, 0)),      # today is Sunday → a week out
    ("at five pm", datetime(2026, 10, 18, 17, 0)),
])
def test_parse_when(text, expected):
    assert parse_when(text, NOW) == expected


@pytest.mark.parametrize("text", ["in the dryer", "at the store", "today", "whenever", "at 25"])
def test_parse_when_rejects(text):
    with pytest.raises(SlotError):
        parse_when(text, NOW)


def test_split_task_when_with_time():
    assert split_task_when("call mom at 5pm", NOW) == ("call mom", datetime(2026, 10, 18, 17, 0))
    assert split_task_when("submit the report tomorrow morning", NOW) == (
        "submit the report", datetime(2026, 10, 19, 9, 0))
    assert split_task_when("pay rent in 2 hours", NOW) == ("pay rent", datetime(2026, 10, 18, 12, 30))


def test_split_task_when_without_time_keeps_whole_task():
    assert split_task_when("buy milk", NOW) == ("buy milk", None)
    # keyword tails that aren't times stay part of the task
    assert split_task_when("put clothes in the dryer", NOW) == ("put clothes in the dryer", None)
    assert split_task_when("check on the build", NOW) == ("check on the build", None)


def test_split_task_when_unreadable_time_escalates():
    with pytest.raises(SlotError):
        split_task_when("call mom at around half past seven-ish", NOW)


def test_humanize_seconds():
    assert humanize_seconds(300) == "5 minutes"
    assert humanize_seconds(5400) == "1 hour and 30 minutes"
    assert humanize_seconds(3661) == "1 hour, 1 minute and 1 second"


def test_spoken_when():
    assert spoken_when(datetime(2026, 10, 18, 17, 0), NOW) == "at 5 PM"
    assert spoken_when(datetime(2026, 10, 18, 17, 30), NOW) == "at 5:30 PM"
    assert spoken_when(datetime(2026, 10, 19, 9, 0), NOW) == "tomorrow at 9 AM"
    assert spoken_when(datetime(2026, 10, 18, 10, 40), NOW) == "in 10 minutes"
    assert spoken_when(datetime(2026, 10, 23, 15, 0), NOW) == "on Friday at 3 PM"
