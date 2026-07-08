"""Tests for wake_stats.py and the daily_check wake report."""
from __future__ import annotations

import json
import time

import daily_check
import aria.observability.wake_stats as wake_stats


def _use_log(monkeypatch, tmp_path):
    log = tmp_path / "wake_log.jsonl"
    monkeypatch.setattr(wake_stats, "_LOG_PATH", str(log))
    monkeypatch.setattr(wake_stats, "_last_heartbeat", 0.0)
    return log


# ---------------------------------------------------------------------------
# log_event / heartbeat
# ---------------------------------------------------------------------------

def test_log_event_writes_entry(monkeypatch, tmp_path):
    log = _use_log(monkeypatch, tmp_path)
    wake_stats.log_event("detection", "custom_onnx", 0.83)
    entry = json.loads(log.read_text().strip())
    assert entry["kind"] == "detection"
    assert entry["backend"] == "custom_onnx"
    assert entry["score"] == 0.83


def test_heartbeat_is_throttled(monkeypatch, tmp_path):
    log = _use_log(monkeypatch, tmp_path)
    for _ in range(10):
        wake_stats.heartbeat("porcupine")
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1  # only the first within the interval is written


def test_log_event_never_raises_on_bad_path(monkeypatch):
    monkeypatch.setattr(wake_stats, "_LOG_PATH", "/dev/null/impossible/x.jsonl")
    wake_stats.log_event("detection", "porcupine")  # must not raise


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_empty(monkeypatch, tmp_path):
    _use_log(monkeypatch, tmp_path)
    s = wake_stats.summary(1)
    assert s["total_entries"] == 0
    assert s["alive_secs_ago"] is None


def test_summary_counts_and_threshold_suggestion(monkeypatch, tmp_path):
    _use_log(monkeypatch, tmp_path)
    wake_stats.log_event("detection", "custom_onnx", 0.72)
    for score in (0.55, 0.60, 0.65):
        wake_stats.log_event("near_miss", "custom_onnx", score)
    wake_stats.log_event("no_speech", "custom_onnx")
    wake_stats.log_event("restart", "custom_onnx")

    s = wake_stats.summary(1)
    assert s["detections"] == 1
    assert s["near_misses"] == 3
    assert s["no_speech"] == 1
    assert s["restarts"] == 1
    assert s["backend"] == "custom_onnx"
    assert s["alive_secs_ago"] < 5
    # near-misses (3) > detections (1) → suggest median (0.60) - 0.02
    assert s["suggested_threshold"] == 0.58


def test_summary_no_suggestion_when_detections_dominate(monkeypatch, tmp_path):
    _use_log(monkeypatch, tmp_path)
    for _ in range(5):
        wake_stats.log_event("detection", "custom_onnx", 0.8)
    wake_stats.log_event("near_miss", "custom_onnx", 0.6)
    s = wake_stats.summary(1)
    assert s["suggested_threshold"] is None


def test_summary_suggestion_floors_at_half(monkeypatch, tmp_path):
    _use_log(monkeypatch, tmp_path)
    for score in (0.41, 0.42, 0.43):
        wake_stats.log_event("near_miss", "custom_onnx", score)
    s = wake_stats.summary(1)
    assert s["suggested_threshold"] == 0.5


def test_read_recent_skips_corrupt_lines(monkeypatch, tmp_path):
    log = _use_log(monkeypatch, tmp_path)
    log.write_text(
        json.dumps({"ts": time.time(), "kind": "detection", "backend": "x"})
        + "\nnot json at all\n"
    )
    assert len(wake_stats.read_recent(1)) == 1


# ---------------------------------------------------------------------------
# daily_check wake CLI
# ---------------------------------------------------------------------------

def test_cli_wake_alive(monkeypatch, tmp_path, capsys):
    log = _use_log(monkeypatch, tmp_path)
    wake_stats.heartbeat("custom_onnx")
    wake_stats.log_event("detection", "custom_onnx", 0.75)
    for score in (0.55, 0.62):
        wake_stats.log_event("near_miss", "custom_onnx", score)

    code = daily_check.main(["wake", "--log", str(log)])
    out = capsys.readouterr().out
    assert code == 0
    assert "ALIVE" in out
    assert "Detections: 1" in out
    assert "Near-misses" in out


def test_cli_wake_empty_log(monkeypatch, tmp_path, capsys):
    log = _use_log(monkeypatch, tmp_path)
    code = daily_check.main(["wake", "--log", str(log)])
    out = capsys.readouterr().out
    assert code == 1
    assert "No wake log entries" in out


def test_cli_wake_dead_engine(monkeypatch, tmp_path, capsys):
    log = _use_log(monkeypatch, tmp_path)
    old = {"ts": time.time() - 7200, "kind": "heartbeat", "backend": "porcupine"}
    log.write_text(json.dumps(old) + "\n")
    daily_check.main(["wake", "--log", str(log)])
    out = capsys.readouterr().out
    assert "NOT RUNNING" in out


# ---------------------------------------------------------------------------
# listening indicator — safe no-op behavior without a GUI
# ---------------------------------------------------------------------------

def test_indicator_hide_without_show_is_safe():
    import aria.ui.listening_indicator as listening_indicator
    assert listening_indicator.hide() is False  # nothing shown, no crash


def test_indicator_disabled_via_env(monkeypatch):
    import aria.ui.listening_indicator as listening_indicator
    monkeypatch.setenv("ARIA_LISTENING_HUD", "0")
    assert listening_indicator.show("x") is False
