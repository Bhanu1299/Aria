"""Tests for seed skills — apply_status and calculate."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestApplyStatus:
    def test_no_applications(self, monkeypatch):
        import skills.apply_status as skill
        import tracker
        monkeypatch.setattr(tracker, "get_applications", lambda: [])
        result = skill.handle("what jobs have I applied to")
        assert "haven't applied" in result

    def test_with_applications(self, monkeypatch):
        import skills.apply_status as skill
        import tracker
        monkeypatch.setattr(tracker, "get_applications", lambda: [
            {"role": "SWE", "company": "Acme", "applied_at": "2026-04-01 10:00:00", "platform": "LinkedIn", "url": ""},
            {"role": "Backend Engineer", "company": "Globex", "applied_at": "2026-03-28 09:00:00", "platform": "Indeed", "url": ""},
        ])
        result = skill.handle("application history")
        assert "Acme" in result
        assert "Globex" in result
        assert "2" in result  # total count

    def test_caps_at_five(self, monkeypatch):
        import skills.apply_status as skill
        import tracker
        apps = [
            {"role": f"Role {i}", "company": f"Co {i}", "applied_at": "2026-04-01", "platform": "", "url": ""}
            for i in range(10)
        ]
        monkeypatch.setattr(tracker, "get_applications", lambda: apps)
        result = skill.handle("show my applications")
        # Should mention total=10 but only show 5
        assert "10" in result
        assert result.count("Role") <= 5
