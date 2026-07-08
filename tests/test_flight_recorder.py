"""Tests for flight_recorder.py — command telemetry that drives weekly fixes."""
from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aria.observability.flight_recorder as flight_recorder


@pytest.fixture(autouse=True)
def _tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(flight_recorder, "_LOG_PATH", str(tmp_path / "flight_log.jsonl"))
    yield


def test_record_appends_jsonl_entry():
    flight_recorder.record("what time is it", "It is 3pm.", 1.2,
                           tools=[("knowledge", True)])
    with open(flight_recorder._LOG_PATH) as f:
        lines = f.readlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["transcript"] == "what time is it"
    assert entry["failed"] is False
    assert entry["tools"] == [["knowledge", True]]
    assert "ts" in entry


def test_failure_detected_from_fallback_answer():
    assert flight_recorder.is_failure("Something went wrong. Please try again.", None)
    assert flight_recorder.is_failure("I didn't understand that. Could you rephrase?", None)
    assert flight_recorder.is_failure("I couldn't capture your screen.", None)
    assert flight_recorder.is_failure("", None)


def test_failure_detected_from_error():
    assert flight_recorder.is_failure("fine answer", "RuntimeError: boom")


def test_success_not_flagged():
    assert not flight_recorder.is_failure("The weather is sunny and 24 degrees.", None)


def test_record_never_raises_on_bad_path(monkeypatch):
    monkeypatch.setattr(flight_recorder, "_LOG_PATH", "/nonexistent-dir/x/y.jsonl")
    flight_recorder.record("q", "a", 0.1)  # must not raise


def test_read_recent_skips_corrupt_lines():
    flight_recorder.record("good one", "answer", 0.5)
    with open(flight_recorder._LOG_PATH, "a") as f:
        f.write("{corrupt json\n")
    flight_recorder.record("good two", "answer", 0.5)
    entries = flight_recorder.read_recent(days=7)
    assert len(entries) == 2


def test_read_recent_filters_by_age():
    old = {"ts": time.time() - 30 * 86400, "transcript": "old", "answer": "x",
           "failed": False, "duration": 1, "tools": [], "error": None}
    with open(flight_recorder._LOG_PATH, "w") as f:
        f.write(json.dumps(old) + "\n")
    flight_recorder.record("new", "answer", 0.5)
    entries = flight_recorder.read_recent(days=7)
    assert len(entries) == 1
    assert entries[0]["transcript"] == "new"


def test_spoken_report_summarizes_counts_and_top_failing_tool():
    flight_recorder.record("q1", "Fine.", 1.0, tools=[("web_search", True)])
    flight_recorder.record("q2", "Something went wrong. Please try again.", 1.0,
                           tools=[("web_search", False)])
    flight_recorder.record("q3", "Something went wrong. Please try again.", 1.0,
                           tools=[("web_search", False)])
    report = flight_recorder.spoken_report(days=7)
    assert "3" in report            # total commands
    assert "2" in report            # failures
    assert "web_search" in report   # top failing tool


def test_spoken_report_handles_empty_log():
    report = flight_recorder.spoken_report(days=7)
    assert isinstance(report, str) and report
