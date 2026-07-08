#!/usr/bin/env python3
"""
daily_check.py — Aria's daily voice-test report.

Workflow:
  1. Read the questions aloud to Aria:   venv/bin/python daily_check.py list --core
  2. See what actually failed:           venv/bin/python daily_check.py report

`report` cross-references tests/daily_questions.json against the flight
recorder log (~/.aria/flight_log.jsonl): each question carries keywords that
are matched against logged transcripts, so imperfect Whisper transcription
still counts. A question is PASS if its latest matching entry succeeded,
FAIL if it failed, and "not asked" if nothing matched today.

Commands:
  list   [--core] [--md]        print the question sheet (--md = markdown)
  report [--days N] [--core] [--log PATH]   pass/fail table + failure detail
  wake   [--days N]             wake word health: alive?, detections,
                                near-misses, threshold suggestion

Regenerate the markdown sheet after editing the JSON:
  venv/bin/python daily_check.py list --md > tests/daily_questions.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

_QUESTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tests", "daily_questions.json"
)

# Questions whose match list contains a sentinel are verified by ear only.
_UNMATCHABLE = "__silence__"


# ---------------------------------------------------------------------------
# Loading + matching
# ---------------------------------------------------------------------------

def load_categories(path: str = _QUESTIONS_PATH) -> list:
    """Load the question set. Returns [] and prints the reason on any error."""
    try:
        with open(path) as f:
            data = json.load(f)
        cats = data.get("categories", [])
        if not isinstance(cats, list):
            print(f"[daily_check] Bad format in {path}: 'categories' must be a list")
            return []
        return cats
    except FileNotFoundError:
        print(f"[daily_check] Question file not found: {path}")
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[daily_check] Could not parse {path}: {exc}")
        return []


def _norm(text: str) -> str:
    """Lowercase and collapse everything non-alphanumeric to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def matches(transcript: str, keywords: list) -> bool:
    """True if every keyword appears in the normalized transcript."""
    if not keywords or _UNMATCHABLE in keywords:
        return False
    norm = f" {_norm(transcript)} "
    return all(f" {_norm(kw)} " in norm or _norm(kw) in norm for kw in keywords)


# ---------------------------------------------------------------------------
# Report building (pure — unit-testable)
# ---------------------------------------------------------------------------

def build_report(entries: list, categories: list, core_only: bool = False) -> list:
    """
    Match log entries to questions. Returns one row per question:
      {"category", "question", "status": "pass"|"fail"|"not_asked",
       "entry": <latest matching log entry or None>}
    """
    rows = []
    for cat in categories:
        for item in cat.get("questions", []):
            if core_only and not item.get("core"):
                continue
            keywords = item.get("match", [])
            hit = None
            for e in entries:  # oldest → newest; keep the newest match
                if matches(e.get("transcript", ""), keywords):
                    hit = e
            if hit is None:
                status = "not_asked"
            else:
                status = "fail" if hit.get("failed") else "pass"
            rows.append({
                "category": cat.get("name", cat.get("id", "?")),
                "question": item.get("q", ""),
                "status": status,
                "entry": hit,
            })
    return rows


def _matched_transcripts(rows: list) -> set:
    return {
        r["entry"].get("transcript", "")
        for r in rows if r["entry"] is not None
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_STATUS_TAG = {"pass": "[PASS]", "fail": "[FAIL]", "not_asked": "[ -- ]"}


def print_report(rows: list, entries: list, days: int) -> int:
    """Print the report. Returns exit code: 1 if anything failed, else 0."""
    print(f"Aria Daily Check — last {days} day(s), {len(entries)} logged commands")
    print()

    current_cat = None
    for r in rows:
        if r["category"] != current_cat:
            current_cat = r["category"]
            print(current_cat)
        tag = _STATUS_TAG[r["status"]]
        line = f"  {tag} {r['question'][:70]}"
        e = r["entry"]
        if e is not None:
            line += f"   ({e.get('duration', 0):.1f}s)"
        print(line)
    print()

    passed = sum(1 for r in rows if r["status"] == "pass")
    failed = [r for r in rows if r["status"] == "fail"]
    not_asked = sum(1 for r in rows if r["status"] == "not_asked")
    print(
        f"Summary: {passed} passed, {len(failed)} failed, "
        f"{not_asked} not asked (of {len(rows)} questions)."
    )

    if failed:
        print("\nFailure detail:")
        for r in failed:
            e = r["entry"]
            print(f"  Q: {r['question']}")
            print(f"     heard:  {e.get('transcript', '')!r}")
            print(f"     answer: {e.get('answer', '')[:120]!r}")
            if e.get("error"):
                print(f"     error:  {e['error'][:160]}")
            bad_tools = [t[0] for t in (e.get("tools") or []) if not t[1]]
            if bad_tools:
                print(f"     failed tools: {', '.join(bad_tools)}")

    # Organic (non-checklist) commands that failed are worth seeing too.
    matched = _matched_transcripts(rows)
    other_fails = [
        e for e in entries
        if e.get("failed") and e.get("transcript", "") not in matched
    ]
    if other_fails:
        print(f"\nOther failed commands (not on the checklist): {len(other_fails)}")
        for e in other_fails[-5:]:
            print(f"  {e.get('transcript', '')!r} → {e.get('answer', '')[:80]!r}")

    return 1 if failed else 0


def print_wake_report(days: float) -> int:
    """Wake word health from ~/.aria/wake_log.jsonl. Returns exit code."""
    import wake_stats

    s = wake_stats.summary(days)
    print(f"Wake Word Health — last {days:g} day(s)")
    print()

    if s["total_entries"] == 0:
        print("  No wake log entries. Either Aria hasn't run since this logging")
        print("  was added, or the wake listener never started. Start Aria and")
        print("  check the console for a '[Aria] Wake word active' line.")
        return 1

    ago = s["alive_secs_ago"]
    if ago is not None and ago < 180:
        print(f"  Engine: ALIVE ({s['backend']}, last signal {ago:.0f}s ago)")
    else:
        mins = (ago or 0) / 60
        print(f"  Engine: NOT RUNNING (last signal {mins:.0f} min ago, "
              f"backend was {s['backend']})")

    print(f"  Detections: {s['detections']}")
    print(f"  Woke but heard no speech: {s['no_speech']}")
    print(f"  Near-misses (said it, score too low?): {s['near_misses']}")
    print(f"  Listener restarts (crashes recovered): {s['restarts']}")

    if s["near_miss_scores"]:
        lo, hi = s["near_miss_scores"][0], s["near_miss_scores"][-1]
        print(f"  Near-miss scores ranged {lo:.2f}-{hi:.2f}")
    if s["suggested_threshold"] is not None:
        print(f"\n  Suggestion: near-misses outnumber detections — consider "
              f"lowering the threshold to ~{s['suggested_threshold']:.2f} "
              f"(wake_word.py: _CUSTOM_THRESHOLD / _OWW_THRESHOLD).")
    return 0


def print_list(categories: list, core_only: bool, as_md: bool) -> None:
    if as_md:
        print("# Aria — Daily Voice Test Questions")
        print()
        print("<!-- GENERATED from tests/daily_questions.json — edit the JSON, then run:")
        print("     venv/bin/python daily_check.py list --md > tests/daily_questions.md -->")
        print()
        print("Read these aloud to Aria (hotkey or wake word), then run "
              "`venv/bin/python daily_check.py report` to see what failed.")
        print("**Bold** questions are the ~3-minute daily core set; "
              "the rest are the weekly sweep.")
    else:
        scope = "core (daily)" if core_only else "all"
        print(f"Aria daily test questions — {scope}:")

    for cat in categories:
        items = [
            it for it in cat.get("questions", [])
            if not core_only or it.get("core")
        ]
        if not items:
            continue
        if as_md:
            print(f"\n## {cat.get('name', cat.get('id'))}\n")
        else:
            print(f"\n{cat.get('name', cat.get('id'))}")
        for it in items:
            q = it.get("q", "")
            expect = it.get("expect", "")
            if as_md:
                q_fmt = f"**{q}**" if it.get("core") else q
                print(f"- {q_fmt}")
                if expect:
                    print(f"  - Expected: {expect}")
            else:
                marker = "*" if it.get("core") else " "
                print(f" {marker} {q}")
                if expect:
                    print(f"      → {expect}")
    if not as_md and not core_only:
        print("\n(* = core daily set. Add --core to show only those.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aria daily voice-test checklist and flight-log report."
    )
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="print the question sheet")
    p_list.add_argument("--core", action="store_true", help="core daily set only")
    p_list.add_argument("--md", action="store_true", help="markdown output")

    p_rep = sub.add_parser("report", help="pass/fail report from the flight log")
    p_rep.add_argument("--days", type=int, default=1, help="lookback window (default 1)")
    p_rep.add_argument("--core", action="store_true", help="core daily set only")
    p_rep.add_argument("--log", default=None, help="alternate flight log path (for testing)")

    p_wake = sub.add_parser("wake", help="wake word health report")
    p_wake.add_argument("--days", type=float, default=1, help="lookback window (default 1)")
    p_wake.add_argument("--log", default=None, help="alternate wake log path (for testing)")

    args = parser.parse_args(argv)

    if args.cmd == "wake":
        if args.log:
            import wake_stats
            wake_stats._LOG_PATH = args.log
        return print_wake_report(days=args.days)

    categories = load_categories()
    if not categories:
        return 2

    if args.cmd == "list":
        print_list(categories, core_only=args.core, as_md=args.md)
        return 0

    if args.cmd == "report":
        import flight_recorder
        if args.log:
            flight_recorder._LOG_PATH = args.log
        entries = flight_recorder.read_recent(days=args.days)
        rows = build_report(entries, categories, core_only=args.core)
        return print_report(rows, entries, days=args.days)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
