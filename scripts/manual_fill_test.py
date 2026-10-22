"""scripts/manual_fill_test.py — the form-filling acceptance test on real postings. NEVER submits.

    venv/bin/python scripts/manual_fill_test.py [URL ...]

Each URL is filled in its own visible tab of Aria's Chrome (tabs stay open for you to
look at; nothing touches the sheet or the tab registry). Review + log JSON go to
~/.aria/fill_tests/. A run PASSES when the form was filled to its last page (or stopped at a
wall only the user can clear, or at facts missing from the bank) and no open-text question
was left empty — missing bank facts may go to the user, open questions never.
Default URLs: five different ATSes from past runs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.features import applicator, ats  # noqa: E402

OUT = Path.home() / ".aria" / "fill_tests"
DEFAULT = [
    ("Vals AI", "Member of Technical Staff", "https://jobs.ashbyhq.com/vals-ai/c39f98f0-8286-4313-9189-a917f5027b7b"),
    ("Palantir", "Software Engineer, New Grad", "https://jobs.lever.co/palantir/90329118-d9b7-4b96-acdf-13377e3196ac/apply"),
    ("Ookla", "Data Engineer (Entry Level)", "https://jobs.jobvite.com/ookla/job/oAaCAfwG/apply"),
    ("ZipRecruiter", "Software Engineer, New Grad", "https://job-boards.greenhouse.io/ziprecruiter/jobs/8127108"),
    ("Foundation AI", "Software Engineer", "https://foundationai.applytojob.com/apply/V9u3fTjcNX/Software-Engineer"),
]


def main(argv: list) -> int:
    jobs = [("", "", u) for u in argv] if argv else DEFAULT
    OUT.mkdir(parents=True, exist_ok=True)
    passed = 0
    table = []
    for i, (company, role, url) in enumerate(jobs, 1):
        job = {"company": company, "role": role, "url": url}
        tab = f"filltest-{int(time.time())}-{i}"
        print(f"\n=== {i}. {company or url}")
        t0 = time.time()
        result = ats.fill_application(job, progress_fn=lambda m: print("   ", m), tab=tab)
        result.tab = tab
        stem = applicator._slug(company or url, 30)
        log = applicator.build_log(job, result, company, role)
        (OUT / f"{stem}.log.json").write_text(json.dumps(log, indent=1))
        (OUT / f"{stem}.md").write_text(applicator.build_summary(job, result, company, role))
        rows = result.review
        # The spec's bar: open questions are never left for the user (missing facts may be).
        open_to_user = [r["label"] for r in rows if r.get("category") == "open_text" and r["source"] in ("needs_me", "failed")]
        unknown = [r["label"] for r in rows if r["source"] == "needs_me"]
        failed = [(r["label"], r["note"]) for r in rows if r["source"] == "failed"]
        stop = getattr(result, "stopped_at", "")
        reached = result.ats == "greenhouse" or stop in ("final page", "needs you") or stop.startswith("blocked by")
        ok = bool(rows) and reached and not open_to_user and not result.error
        passed += ok
        table.append((company or url[:40], "yes" if reached and not stop.startswith("blocked by") else
                      ("wall: " + stop[len("blocked by "):] if stop.startswith("blocked by") else "no"),
                      len(unknown), len(failed), len(open_to_user), "PASS" if ok else "FAIL"))
        print(f"   {'PASS' if ok else 'FAIL'} in {time.time() - t0:.0f}s · ats={result.ats}/{result.identified} · "
              f"stopped_at={getattr(result, 'stopped_at', '')} · {log['summary']}")
        for r in rows:
            print(f"     [{r['source']:9}] {r.get('category', ''):9} {r['label'][:70]!r} = {str(r['value'])[:60]!r} {r['note']}")
        if result.error:
            print("   ERROR:", result.error)
        if failed:
            print("   FAILED:", failed)
        if unknown:
            print("   UNKNOWN:", unknown)
    print(f"\n{'site':<16} {'reached review':<44} {'UNKNOWN':>7} {'FAILED':>6} {'open Qs to you':>14}  result")
    for row in table:
        print(f"{row[0]:<16} {row[1]:<44} {row[2]:>7} {row[3]:>6} {row[4]:>14}  {row[5]}")
    print(f"\n{passed}/{len(jobs)} passed")
    return 0 if passed >= max(1, len(jobs) - 1) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
