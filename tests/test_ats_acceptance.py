"""The ATS acceptance gate (docs/job-prep-design.md §5): 10+ real-style JDs × the fictional
fixture resumes → tailor → render → strict ATS score. Median ≥ 95, ≥ 80% of JDs ≥ 95, none
< 85, every PDF one page, every employer / date / degree / number unchanged.

Runs the code-only path (the LLM is blocked in tests): the worst case — with GateLLM up the
bullet rewrite only adds to it. scripts/manual_ats_test.py prints the same table (and can use
the live LLM and your real base resume)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aria.features.resume import acceptance

DATA = Path(__file__).parent / "data"
pytestmark = pytest.mark.skipif(shutil.which("pdftotext") is None or shutil.which("pdfinfo") is None,
                                reason="needs poppler (pdftotext, pdfinfo): brew install poppler")


@pytest.mark.parametrize("fixture", ["resume_fixture.json", "resume_fixture_dense.json"])
def test_acceptance_gate(fixture, tmp_path):
    base = json.loads((DATA / fixture).read_text())
    jds = acceptance.load_jds(DATA / "jds")
    assert len(jds) >= 10
    report = acceptance.run(base, jds, tmp_path)
    print(f"\n{fixture}\n{report.table()}")
    assert report.passed, report.failures
    assert report.median >= 95
    assert report.share_95 >= 0.80
    assert min(report.scores) >= 85
    assert all(r.one_page for r in report.rows)
    assert all(r.facts_ok for r in report.rows)


def test_jd_fixtures_are_well_formed():
    kinds = set()
    for p in sorted((DATA / "jds").glob("*.json")):
        d = json.loads(p.read_text())
        assert d["title"] and d["company"] and len(d["text"]) > 400, p.name
        kinds.add(d["kind"])
    assert {"backend", "ml", "fullstack", "data", "infra"} <= kinds
