"""tests/test_cron.py — CronScheduler unit tests."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _scheduler(tmpdir: str, agent=None, speaker=None):
    from plugins.productivity.cron import CronScheduler
    with patch("plugins.productivity.cron._JOBS_DIR", Path(tmpdir)):
        sched = CronScheduler(agent=agent, speaker=speaker)
        sched._jobs_dir = Path(tmpdir)
    # Override _JOBS_DIR inside the instance
    import plugins.productivity.cron as _cron_mod
    _cron_mod._JOBS_DIR = Path(tmpdir)
    sched._lock = __import__("threading").Lock()
    return sched


def test_parse_schedule_every_hour():
    from plugins.productivity.cron import parse_schedule
    assert parse_schedule("every hour") == "0 * * * *"


def test_parse_schedule_every_morning():
    from plugins.productivity.cron import parse_schedule
    assert parse_schedule("every morning") == "0 9 * * *"


def test_parse_schedule_every_day_at_9am():
    from plugins.productivity.cron import parse_schedule
    expr = parse_schedule("every day at 9am")
    assert expr == "0 9 * * *"


def test_parse_schedule_every_30_minutes():
    from plugins.productivity.cron import parse_schedule
    assert parse_schedule("every 30 minutes") == "*/30 * * * *"


def test_parse_schedule_passthrough_valid_cron():
    from plugins.productivity.cron import parse_schedule
    assert parse_schedule("0 9 * * 1") == "0 9 * * 1"


def test_parse_schedule_invalid_raises():
    from plugins.productivity.cron import parse_schedule
    try:
        parse_schedule("do it whenever you feel like it")
        assert False, "Should have raised"
    except ValueError as e:
        assert "didn't understand" in str(e).lower()


def test_create_job_saves_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        import plugins.productivity.cron as _cron_mod
        orig = _cron_mod._JOBS_DIR
        _cron_mod._JOBS_DIR = Path(tmpdir)
        try:
            sched = _cron_mod.CronScheduler()
            result = sched.create("morning-emails", "Check my emails", "every morning at 9")
            assert "Scheduled" in result
            assert "morning-emails" in result
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1
        finally:
            _cron_mod._JOBS_DIR = orig


def test_list_jobs_returns_jobs():
    with tempfile.TemporaryDirectory() as tmpdir:
        import plugins.productivity.cron as _cron_mod
        orig = _cron_mod._JOBS_DIR
        _cron_mod._JOBS_DIR = Path(tmpdir)
        try:
            sched = _cron_mod.CronScheduler()
            sched.create("job1", "Do something", "0 9 * * *")
            jobs = sched.list_jobs()
            assert len(jobs) == 1
            assert jobs[0]["id"] == "job1"
            assert "next_run" in jobs[0]
        finally:
            _cron_mod._JOBS_DIR = orig


def test_delete_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        import plugins.productivity.cron as _cron_mod
        orig = _cron_mod._JOBS_DIR
        _cron_mod._JOBS_DIR = Path(tmpdir)
        try:
            sched = _cron_mod.CronScheduler()
            sched.create("temp-job", "Do thing", "0 9 * * *")
            result = sched.delete("temp-job")
            assert result == "Deleted."
            assert list(Path(tmpdir).glob("*.json")) == []
        finally:
            _cron_mod._JOBS_DIR = orig


def test_delete_nonexistent_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        import plugins.productivity.cron as _cron_mod
        orig = _cron_mod._JOBS_DIR
        _cron_mod._JOBS_DIR = Path(tmpdir)
        try:
            sched = _cron_mod.CronScheduler()
            result = sched.delete("ghost-job")
            assert "No job found" in result
        finally:
            _cron_mod._JOBS_DIR = orig


def test_toggle_pauses_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        import plugins.productivity.cron as _cron_mod
        orig = _cron_mod._JOBS_DIR
        _cron_mod._JOBS_DIR = Path(tmpdir)
        try:
            sched = _cron_mod.CronScheduler()
            sched.create("pausable", "Do stuff", "0 9 * * *")
            result = sched.toggle("pausable", False)
            assert result == "Paused."
            jobs = sched.list_jobs()
            assert jobs[0]["enabled"] is False
        finally:
            _cron_mod._JOBS_DIR = orig


def test_cron_job_delivers_via_speaker():
    with tempfile.TemporaryDirectory() as tmpdir:
        import plugins.productivity.cron as _cron_mod
        orig = _cron_mod._JOBS_DIR
        _cron_mod._JOBS_DIR = Path(tmpdir)
        try:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "You have 3 unread emails."
            mock_speaker = MagicMock()
            sched = _cron_mod.CronScheduler(agent=mock_agent, speaker=mock_speaker)
            from plugins.productivity.cron import CronJob
            job = CronJob(id="test-job", prompt="Check emails", schedule="0 9 * * *",
                          delivery="speak")
            sched._run_job(job)
            mock_speaker.say.assert_called_once_with("You have 3 unread emails.")
        finally:
            _cron_mod._JOBS_DIR = orig


def test_cron_job_silent_delivery_does_not_speak():
    with tempfile.TemporaryDirectory() as tmpdir:
        import plugins.productivity.cron as _cron_mod
        orig = _cron_mod._JOBS_DIR
        _cron_mod._JOBS_DIR = Path(tmpdir)
        try:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "done"
            mock_speaker = MagicMock()
            sched = _cron_mod.CronScheduler(agent=mock_agent, speaker=mock_speaker)
            from plugins.productivity.cron import CronJob
            job = CronJob(id="silent-job", prompt="maintenance", schedule="0 9 * * *",
                          delivery="silent")
            sched._run_job(job)
            mock_speaker.say.assert_not_called()
        finally:
            _cron_mod._JOBS_DIR = orig
