"""tests/test_apply_corrections.py — the apply failure log: rows, snapshots, report, never raises."""
from __future__ import annotations

import stat

import pytest

import aria.state.db as db
from aria.features import ats
from aria.features.ats import corrections


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "aria.db"))
    monkeypatch.setattr(corrections, "CORRECTIONS_DIR", tmp_path / "corrections")
    return tmp_path


class FakePage:
    url = "https://careers.acme.com/apply/2"

    def evaluate(self, js):
        assert js is corrections.SNAPSHOT_JS
        return "<html><body>form</body></html>"

    def screenshot(self, timeout=0):
        return b"\x89PNG fake"


def run(fn):
    return fn(FakePage())


JOB = {"url": "https://careers.acme.com/job/1", "company": "Acme Corp", "row": 7}


def test_record_writes_a_row_with_every_column():
    assert corrections.record(JOB, "unmapped_field", "fill", "Desired start date", ats="workday", page_url="https://x/2")
    (row,) = corrections.recent(1)
    assert {k: row[k] for k in ("job_url", "company", "row", "ats", "step", "kind", "detail", "page_url", "snapshot")} == {
        "job_url": JOB["url"], "company": "Acme Corp", "row": 7, "ats": "workday", "step": "fill",
        "kind": "unmapped_field", "detail": "Desired start date", "page_url": "https://x/2", "snapshot": ""}


def test_snapshot_saves_png_and_html_owner_only(store):
    corrections.record(JOB, "captcha", "wall", "a CAPTCHA", run=run)
    (row,) = corrections.recent(1)
    assert row["page_url"] == FakePage.url
    for ext in (".png", ".html"):
        f = corrections.CORRECTIONS_DIR / (row["snapshot"].rsplit("/", 1)[1] + ext)
        assert f.exists() and stat.S_IMODE(f.stat().st_mode) == 0o600
    assert "acme-corp_captcha" in row["snapshot"]


def test_record_never_raises(monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", "/nonexistent-dir/\0bad")

    def broken(fn):
        raise RuntimeError("tab gone")
    assert corrections.record(JOB, "timeout", "wall", "x", run=broken) == 0


def test_report_groups_by_kind():
    corrections.record(JOB, "unknown_ats", "resolve", "acme.com")
    corrections.record({"url": "https://y.example/j"}, "unknown_ats", "resolve", "y.example")
    corrections.record(JOB, "login_failed", "login", "wrong password")
    out = corrections.report(corrections.recent(1))
    assert "unknown_ats (2)" in out and "login_failed (1)" in out and "y.example" in out
    assert corrections.report([]) == "No apply failures recorded."


def test_fill_corrections_log_unmapped_failed_and_errors_with_one_snapshot_each():
    rows = [{"label": "Pronouns", "source": "needs_me"}, {"label": "Start date", "source": "needs_me"},
            {"label": "Phone", "source": "failed", "note": "didn't stick after 2 retries"},
            {"label": "Email", "source": "bank"}]
    ats.log_corrections(JOB, "lever", rows, "The form fill failed: boom", run)
    got = corrections.recent(1)
    assert sorted((r["kind"], r["detail"]) for r in got) == [
        ("failed_field", "Phone — didn't stick after 2 retries"), ("fill_error", "The form fill failed: boom"),
        ("unmapped_field", "Pronouns"), ("unmapped_field", "Start date")]
    assert sum(1 for r in got if r["kind"] == "unmapped_field" and r["snapshot"]) == 1


def test_cli_prints_the_report(capsys):
    corrections.record(JOB, "captcha", "wall", "a CAPTCHA")
    assert corrections.main(["--days", "1"]) == 0
    assert "captcha (1)" in capsys.readouterr().out
