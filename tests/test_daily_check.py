"""Tests for daily_check.py — daily voice-test report."""
from __future__ import annotations

import json
import time

import daily_check


# ---------------------------------------------------------------------------
# Question file integrity
# ---------------------------------------------------------------------------

def test_questions_file_loads_and_is_well_formed():
    cats = daily_check.load_categories()
    assert cats, "tests/daily_questions.json must load"
    for cat in cats:
        assert cat.get("id") and cat.get("name")
        assert cat.get("questions"), f"category {cat['id']} has no questions"
        for q in cat["questions"]:
            assert q.get("q"), f"empty question in {cat['id']}"
            assert q.get("match"), f"question {q['q']!r} has no match keywords"
            assert q.get("expect"), f"question {q['q']!r} has no expectation"


def test_core_set_is_small_enough_for_daily_use():
    cats = daily_check.load_categories()
    core = [q for c in cats for q in c["questions"] if q.get("core")]
    assert 8 <= len(core) <= 16, f"core set should stay ~3 minutes, got {len(core)}"


def test_load_categories_missing_file_returns_empty(tmp_path):
    assert daily_check.load_categories(str(tmp_path / "nope.json")) == []


def test_load_categories_bad_json_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert daily_check.load_categories(str(p)) == []


# ---------------------------------------------------------------------------
# Transcript matching
# ---------------------------------------------------------------------------

def test_matches_is_case_and_punctuation_tolerant():
    assert daily_check.matches("Who wrote The Great Gatsby?!", ["great gatsby"])
    assert daily_check.matches("what's the WEATHER like", ["weather"])


def test_matches_requires_all_keywords():
    assert not daily_check.matches("what's my favorite color", ["favorite", "coffee"])
    assert daily_check.matches("what is my favorite coffee", ["favorite", "coffee"])


def test_matches_sentinel_never_matches():
    assert not daily_check.matches("anything at all", ["__silence__"])


def test_matches_empty_keywords_never_match():
    assert not daily_check.matches("anything", [])


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

_CATS = [
    {
        "id": "knowledge", "name": "Knowledge",
        "questions": [
            {"q": "Who wrote The Great Gatsby?", "match": ["great gatsby"],
             "expect": "x", "core": True},
        ],
    },
    {
        "id": "media", "name": "Media",
        "questions": [
            {"q": "Pause", "match": ["pause"], "expect": "x", "core": False},
        ],
    },
]


def _entry(transcript, failed=False, **kw):
    e = {"ts": time.time(), "transcript": transcript, "answer": "ok",
         "duration": 1.0, "tools": [], "error": None, "failed": failed}
    e.update(kw)
    return e


def test_build_report_pass_fail_not_asked():
    entries = [
        _entry("who wrote the great gatsby"),
        # media question never asked
    ]
    rows = daily_check.build_report(entries, _CATS)
    by_q = {r["question"]: r["status"] for r in rows}
    assert by_q["Who wrote The Great Gatsby?"] == "pass"
    assert by_q["Pause"] == "not_asked"


def test_build_report_latest_matching_entry_wins():
    entries = [
        _entry("pause", failed=True),
        _entry("pause the music please", failed=False),
    ]
    rows = daily_check.build_report(entries, _CATS)
    by_q = {r["question"]: r["status"] for r in rows}
    assert by_q["Pause"] == "pass"


def test_build_report_failure_reported():
    entries = [_entry("who wrote the great gatsby",
                      failed=True, answer="Something went wrong")]
    rows = daily_check.build_report(entries, _CATS)
    row = [r for r in rows if r["question"] == "Who wrote The Great Gatsby?"][0]
    assert row["status"] == "fail"
    assert row["entry"]["answer"] == "Something went wrong"


def test_build_report_core_only_filters():
    rows = daily_check.build_report([], _CATS, core_only=True)
    assert [r["question"] for r in rows] == ["Who wrote The Great Gatsby?"]


# ---------------------------------------------------------------------------
# CLI end-to-end against a synthetic log
# ---------------------------------------------------------------------------

def test_cli_report_with_synthetic_log(tmp_path, capsys):
    log = tmp_path / "flight_log.jsonl"
    lines = [
        _entry("who wrote the great gatsby"),
        _entry("pause", failed=True, answer="I ran into an issue with media_pause",
               tools=[["media_pause", False]]),
        _entry("random organic command", failed=True, answer="something went wrong"),
    ]
    log.write_text("\n".join(json.dumps(e) for e in lines) + "\n")

    code = daily_check.main(["report", "--days", "1", "--log", str(log)])
    out = capsys.readouterr().out

    assert code == 1  # a checklist question failed
    assert "[PASS] Who wrote The Great Gatsby?" in out
    assert "failed tools: media_pause" in out
    assert "Other failed commands" in out


def test_cli_list_runs(capsys):
    assert daily_check.main(["list", "--core"]) == 0
    out = capsys.readouterr().out
    assert "Knowledge" in out


def test_cli_list_md_runs(capsys):
    assert daily_check.main(["list", "--md"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Aria — Daily Voice Test Questions")
