"""ATS acceptance test, by hand: every JD in tests/data/jds × a base resume → tailor → render →
strict ATS score, printed as a table (docs/job-prep-design.md §5).

    python scripts/manual_ats_test.py                    # fictional fixture, code-only path
    python scripts/manual_ats_test.py --llm              # + the strong-tier bullet rewrite (GateLLM up)
    python scripts/manual_ats_test.py --resume ~/.aria/resume_base.json --out /tmp/ats
    python scripts/manual_ats_test.py --resume-pdf ~/Downloads/resume.pdf   # builds the base from a PDF

PDFs and tailored resume.json files land in --out (default: a temp folder) for a look.
Exit code 0 = the gate passed (median ≥ 95, ≥ 80% ≥ 95, none < 85, one page, facts unchanged).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aria.features.resume import acceptance, model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resume", default=str(ROOT / "tests" / "data" / "resume_fixture.json"),
                    help="base resume.json (default: the fictional fixture)")
    ap.add_argument("--resume-pdf", help="build the base from this PDF instead (pdftotext + resume parser)")
    ap.add_argument("--jds", default=str(ROOT / "tests" / "data" / "jds"))
    ap.add_argument("--out", default="")
    ap.add_argument("--llm", action="store_true", help="use GateLLM for the bullet rewrite + skill extraction")
    args = ap.parse_args()

    if args.resume_pdf:
        from aria.features.ats import resume_text
        base = model.build_from_text(resume_text.text(args.resume_pdf))
    else:
        base = model.load(args.resume)
    if not base or not (base["experience"] or base["projects"]):
        print("Couldn't read a base resume.", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="aria-ats-"))
    report = acceptance.run(base, acceptance.load_jds(args.jds), out, use_llm=args.llm)
    print(report.table())
    print(f"\nPDFs: {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
