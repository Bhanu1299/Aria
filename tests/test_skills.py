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


class TestCalculate:
    def _handle(self, cmd):
        import skills.calculate as skill
        return skill.handle(cmd)

    def test_multiplication(self):
        assert "40" in self._handle("calculate 5 times 8")

    def test_division(self):
        result = self._handle("how much is 120 divided by 4")
        assert "30" in result

    def test_addition(self):
        assert "15" in self._handle("calculate 7 plus 8")

    def test_subtraction(self):
        assert "3" in self._handle("calculate 10 minus 7")

    def test_power(self):
        assert "1024" in self._handle("compute 2 to the power of 10")

    def test_no_expression(self):
        result = self._handle("calculate")
        assert "couldn't find" in result.lower()

    def test_division_by_zero(self):
        result = self._handle("calculate 5 divided by 0")
        assert "couldn't calculate" in result.lower()

    def test_whole_number_result_no_decimal(self):
        result = self._handle("calculate 10 divided by 2")
        assert "5" in result
        assert "5.0" not in result  # should not show trailing .0
