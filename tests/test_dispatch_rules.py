"""tests/test_dispatch_rules.py — shipped patterns.json: what goes direct, what goes to the agent."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.dispatch.rules import PATTERNS_PATH, Escalate, RuleMatch, RuleSet

NOW = datetime(2026, 10, 18, 10, 30)  # Sunday


@pytest.fixture(scope="module")
def rules():
    with open(PATTERNS_PATH) as f:
        return RuleSet(json.load(f), now=lambda: NOW)


@pytest.mark.parametrize("utterance,tool,params", [
    ("Pause the music.", "music_pause", {}),
    ("pause", "music_pause", {}),
    ("Hey Aria, stop the music", "music_pause", {}),
    ("resume", "music_resume", {}),
    ("play music", "music_resume", {}),
    ("skip this song", "music_skip", {}),
    ("next track", "music_skip", {}),
    ("previous song", "music_previous", {}),
    ("what's playing", "music_now_playing", {}),
    ("play Taylor Swift", "music_play", {"query": "taylor swift"}),
    ("play lofi beats on spotify", "music_play", {"query": "lofi beats"}),
    ("turn it up", "system_control", {"action": "volume_up"}),
    ("volume down", "system_control", {"action": "volume_down"}),
    ("set volume to 40%", "system_control", {"action": "volume_set", "level": 40}),
    ("mute", "system_control", {"action": "mute"}),
    ("brightness up", "system_control", {"action": "brightness_up"}),
    ("turn on dark mode", "system_control", {"action": "dark_mode_on"}),
    ("take a screenshot", "system_control", {"action": "screenshot"}),
    ("lock my mac", "system_control", {"action": "lock_screen"}),
    ("turn off wifi", "system_control", {"action": "wifi_off"}),
    ("open Safari", "app", {"app_name": "safari"}),
    ("launch visual studio code", "app", {"app_name": "visual studio code"}),
    ("quit Spotify", "system_control", {"action": "quit_app", "app_name": "spotify"}),
    ("what time is it", "clock", {"what": "time"}),
    ("what's the date today", "clock", {"what": "date"}),
    ("what's 15 times 8", "calculate", {"expression": "15 times 8"}),
    ("what is 2.5 x 4", "calculate", {"expression": "2.5 x 4"}),
    ("what's 15% of 80", "calculate", {"expression": "15% of 80"}),
    ("What's 1,200 divided by 4?", "calculate", {"expression": "1,200 divided by 4"}),
    ("set a timer for 5 minutes", "timer_set", {"seconds": 300}),
    ("Set a timer for ten minutes for pasta.", "timer_set", {"seconds": 600, "label": "pasta"}),
    ("start a 20 minute timer", "timer_set", {"seconds": 1200}),
    ("timer for an hour and a half", "timer_set", {"seconds": 5400}),
    ("how much time is left", "timer_status", {}),
    ("cancel the timer", "timer_cancel", {}),
    ("stop the pasta timer", "timer_cancel", {"label": "pasta"}),
    ("remind me to call mom at 5pm", "reminder_add", {"task": "call mom", "due": "2026-10-18T17:00"}),
    ("remind me to buy milk", "reminder_add", {"task": "buy milk"}),
    ("remind me tomorrow at 9 to submit the report", "reminder_add",
     {"task": "submit the report", "due": "2026-10-19T09:00"}),
    ("remind me in 10 minutes to check the oven", "reminder_add",
     {"task": "check the oven", "due": "2026-10-18T10:40"}),
    ("add a reminder to water the plants tonight", "reminder_add",
     {"task": "water the plants", "due": "2026-10-18T20:00"}),
    ("add eggs to my reminders", "reminder_add", {"task": "eggs"}),
    ("what's on my calendar today", "agenda", {"day": "today"}),
    ("what's my schedule for tomorrow", "agenda", {"day": "tomorrow"}),
    ("what do i have tomorrow", "agenda", {"day": "tomorrow"}),
    ("Aria, apply to the next job from Muse", "apply_job", {}),
    ("apply to the next job", "apply_job", {}),
    ("apply for the next role in my sheet", "apply_job", {}),
    ("apply to row 7", "apply_job", {"row": 7}),
    ("apply to sheet row 12", "apply_job", {"row": 12}),
    ("next job from muse", "job_from_sheet", {}),
    ("what's the next job on my list", "job_from_sheet", {}),
    ("do my outreach", "job_outreach", {}),
    ("Aria, run the job outreach for my jobs", "job_outreach", {}),
    ("find recruiters for my jobs", "job_outreach", {}),
    ("dry run my outreach", "job_outreach", {"dry_run": True}),
    ("do a dry run of the job outreach", "job_outreach", {"dry_run": True}),
    ("run my outreach as a dry run", "job_outreach", {"dry_run": True}),
    ("preview outreach", "job_outreach", {"dry_run": True}),
])
def test_direct_routes(rules, utterance, tool, params):
    m = rules.match(utterance)
    assert isinstance(m, RuleMatch), f"{utterance!r} → {m!r}"
    assert (m.tool, m.params) == (tool, params)


@pytest.mark.parametrize("utterance", [
    "what's the weather in paris",
    "text mom that i'm running late",
    "open my email",
    "open safari and go to github",
    "open google.com",
    "play the latest mkbhd video on youtube",
    "play a game",
    "close this window",
    "exit full screen",
    "pause the music and then open safari",
    "summarize my unread emails",
    "stop",
    "what's five times eight plus the square root of nine",
    "apply to the next job on linkedin",
    "find me the next job in new york",
    "submit it",
    "skip this job",
])
def test_open_ended_commands_go_to_agent(rules, utterance):
    assert rules.match(utterance) is None, utterance


@pytest.mark.parametrize("utterance", [
    "set a timer for a while",
    "remind me to call mom at around half past seven-ish",
    "start a timer",
])
def test_unparseable_slots_escalate(rules, utterance):
    assert isinstance(rules.match(utterance), Escalate), utterance


def test_every_rule_targets_a_known_param_shape():
    data = json.loads(PATTERNS_PATH.read_text())
    ids = [r["id"] for r in data["rules"]]
    assert len(ids) == len(set(ids)), "duplicate rule ids"
    for r in data["rules"]:
        assert r["tool"] and r["regex"], r["id"]
