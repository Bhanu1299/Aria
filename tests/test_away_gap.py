"""Tests for away_summary gap-detection additions (check_and_speak, update_last_active)."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


def _make_identity(last_active_at: str | None, last_task_summary: str = "") -> dict:
    d = {"name": "Test"}
    if last_active_at is not None:
        d["last_active_at"] = last_active_at
    if last_task_summary:
        d["last_task_summary"] = last_task_summary
    return d


class TestAwayGap(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.path = self.tmp.name
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.path)

    def _write_identity(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f)

    def test_speaks_recap_after_30_min_gap(self):
        import away_summary
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
        self._write_identity(_make_identity(old_ts, "Found 3 Python jobs on LinkedIn."))
        speaker = MagicMock()
        with patch("away_summary._IDENTITY_PATH", self.path):
            away_summary.check_and_speak(speaker)
        speaker.say.assert_called_once()

    def test_no_recap_within_30_min(self):
        import away_summary
        recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        self._write_identity(_make_identity(recent_ts, "Found jobs."))
        speaker = MagicMock()
        with patch("away_summary._IDENTITY_PATH", self.path):
            away_summary.check_and_speak(speaker)
        speaker.say.assert_not_called()

    def test_no_recap_when_no_last_active(self):
        import away_summary
        self._write_identity(_make_identity(None))
        speaker = MagicMock()
        with patch("away_summary._IDENTITY_PATH", self.path):
            away_summary.check_and_speak(speaker)
        speaker.say.assert_not_called()

    def test_update_last_active_writes_timestamp(self):
        import away_summary
        self._write_identity({"name": "Test"})
        with patch("away_summary._IDENTITY_PATH", self.path):
            away_summary.update_last_active("Searched for jobs.")
        with open(self.path) as f:
            data = json.load(f)
        self.assertIn("last_active_at", data)
        self.assertIn("last_task_summary", data)
        self.assertEqual(data["last_task_summary"], "Searched for jobs.")

    def test_check_and_speak_never_raises_on_bad_file(self):
        import away_summary
        speaker = MagicMock()
        with patch("away_summary._IDENTITY_PATH", "/nonexistent/path.json"):
            try:
                away_summary.check_and_speak(speaker)
            except Exception as e:
                self.fail(f"check_and_speak raised: {e}")


if __name__ == "__main__":
    unittest.main()
