"""features/resume/acceptance.py — the ATS acceptance gate, shared by the pytest version
(tests/test_ats_acceptance.py) and scripts/manual_ats_test.py.

For every JD fixture: base resume → tailor → render → strict ATS score, and the checks the
design requires: median ≥ 95, ≥ 80% of JDs ≥ 95, none < 85, every PDF one page, every
employer / date / degree / number unchanged.

    run(base, jds, out_dir, use_llm=False) → Report
    Report.table() → markdown table;  Report.passed → bool;  Report.failures → [why]
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from aria.features.resume import ats_score, model, render, tailor

MEDIAN_MIN, SHARE_95_MIN, FLOOR = 95, 0.80, 85


@dataclass
class Row:
    name: str
    title: str
    jd_skills: int
    before: int
    after: int
    one_page: bool
    facts_ok: bool
    added: int
    woven_llm: int
    woven_code: int
    dropped: int
    parts: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)


@dataclass
class Report:
    rows: list
    use_llm: bool = False

    @property
    def scores(self) -> list:
        return [r.after for r in self.rows]

    @property
    def median(self) -> float:
        return statistics.median(self.scores) if self.rows else 0.0

    @property
    def share_95(self) -> float:
        return sum(1 for s in self.scores if s >= 95) / len(self.rows) if self.rows else 0.0

    @property
    def failures(self) -> list:
        out = []
        if len(self.rows) < 10:
            out.append(f"only {len(self.rows)} JDs (need 10+)")
        if self.median < MEDIAN_MIN:
            out.append(f"median {self.median} < {MEDIAN_MIN}")
        if self.share_95 < SHARE_95_MIN:
            out.append(f"{self.share_95:.0%} of JDs ≥ 95 (< {SHARE_95_MIN:.0%})")
        out += [f"{r.name}: {r.after} < {FLOOR}" for r in self.rows if r.after < FLOOR]
        out += [f"{r.name}: not one page" for r in self.rows if not r.one_page]
        out += [f"{r.name}: {', '.join(r.problems)}" for r in self.rows if not r.facts_ok]
        return out

    @property
    def passed(self) -> bool:
        return not self.failures

    def table(self) -> str:
        head = ("| JD | Title | JD skills | Base | Tailored | Hard | Ctx | Title | Parse | Fmt | 1 page | Facts |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|")
        lines = [head]
        for r in self.rows:
            p = r.parts
            pts = lambda k: f"{p[k]['points']:g}/{p[k]['max']}"   # noqa: E731
            lines.append(f"| {r.name} | {r.title} | {r.jd_skills} | {r.before} | **{r.after}** | {pts('hard_skills')} | "
                         f"{pts('context')} | {pts('title')} | {pts('parse')} | {pts('format')} | "
                         f"{'yes' if r.one_page else 'NO'} | {'ok' if r.facts_ok else 'CHANGED'} |")
        lines.append("")
        lines.append(f"Median {self.median:g} · {self.share_95:.0%} of {len(self.rows)} JDs ≥ 95 · "
                     f"min {min(self.scores) if self.rows else 0} · LLM rewrite {'on' if self.use_llm else 'off (code-only path)'} · "
                     f"{'PASS' if self.passed else 'FAIL: ' + '; '.join(self.failures)}")
        return "\n".join(lines)


def load_jds(folder) -> list:
    out = []
    for p in sorted(Path(folder).glob("*.json")):
        d = json.loads(p.read_text())
        out.append((p.stem, d["title"], d["text"]))
    return out


def run(base: dict, jds: list, out_dir, use_llm: bool = False) -> Report:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = model.normalize(base)
    base_pdf = out_dir / "base.pdf"
    render.render(base, base_pdf)
    rows = []
    for name, title, text in jds:
        before = ats_score.score(base_pdf, base, text, title)
        t = tailor.tailor(base, text, title, use_llm=use_llm)
        pdf = out_dir / f"{name}.pdf"
        lay = render.render(t.resume, pdf)
        after = ats_score.score(pdf, t.resume, text, title)
        problems = tailor.verify(base, t.resume)
        pages = ats_score._run("pdfinfo", [str(pdf)])
        one_page = lay.one_page and bool(re.search(r"(?m)^Pages:\s+1$", pages))
        rows.append(Row(name=name, title=title, jd_skills=len(after["jd_skills"]), before=before["score"],
                        after=after["score"], one_page=one_page, facts_ok=not problems,
                        added=len(t.added_skills), woven_llm=sum(1 for c in t.changes if "llm" in c["how"]),
                        woven_code=sum(1 for c in t.changes if c["how"] == "code"), dropped=len(t.dropped),
                        parts=after["parts"], problems=problems))
        (out_dir / f"{name}.json").write_text(json.dumps(model.strip_ids(t.resume), indent=1))
    return Report(rows, use_llm)
