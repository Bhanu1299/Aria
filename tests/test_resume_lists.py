"""Your jobs / schools / projects as lists: read from the resume until you save them in the
dashboard (2026-10-25: Workday's own resume parse put projects in as jobs, and the bank held one
school — the bachelor's was nowhere)."""
from __future__ import annotations

import json

import pytest

from aria.features import answer_bank as ab
from aria.features.ats import resume_sections

RESUME = """Sam Lee
sam@example.com | github.com/sam
TECHNICAL SKILLS
• Languages: Python, Go
EXPERIENCE
Globex Jan 2025 - May 2025
Software Engineer Intern
• Built the billing service in Go, cutting invoice latency from 4 s
to 300 ms
• Wrote the CI pipeline
Initech | Austin, TX Sep 2023 - Present
Data Engineer
• Spark pipelines
PROJECTS
Router - Cost-Aware Model Router Sep 2026 - Present
• Routes prompts
EDUCATION
M.S., Computer Science Aug 2024 - May 2026
State University of Somewhere
GPA: 3.8
B.Tech., Electrical Engineering Sep 2020 - May 2024
Some Institute of Technology
CERTIFICATIONS
• AWS Certified Developer
"""


def test_resume_entries_are_read_by_code():
    r = resume_sections.parse(RESUME)
    assert [(e["company"], e["title"], e["start"], e["end"]) for e in r["experience"]] == [
        ("Globex", "Software Engineer Intern", "Jan 2025", "May 2025"),
        ("Initech", "Data Engineer", "Sep 2023", "Present")]
    assert r["experience"][1]["location"] == "Austin, TX"
    assert "4 s to 300 ms" in r["experience"][0]["description"]            # a wrapped bullet stays whole
    assert r["experience"][0]["description"].count("•") == 2
    assert [(e["school"], e["degree"], e["field"]) for e in r["education"]] == [
        ("State University of Somewhere", "M.S.", "Computer Science"),
        ("Some Institute of Technology", "B.Tech.", "Electrical Engineering")]
    assert r["education"][0]["gpa"] == "3.8"
    assert [p["name"] for p in r["projects"]] == ["Router - Cost-Aware Model Router"]     # never a job
    assert r["certifications"] == [{"name": "AWS Certified Developer", "issuer": "", "date": ""}]
    assert resume_sections.parse("") == {"experience": [], "education": [], "projects": [], "certifications": []}


@pytest.fixture
def bank(tmp_path, monkeypatch):
    path = tmp_path / "answer_bank.json"
    path.write_text(json.dumps({"answers": {"first_name": "Sam"}}))
    monkeypatch.setattr(ab, "BANK_PATH", path)
    monkeypatch.setattr(ab, "_resume_lists", lambda: resume_sections.parse(RESUME))
    return path


def test_lists_come_from_the_resume_until_saved(bank):
    entries, saved = ab.list_state("education")
    assert not saved and [e["degree"] for e in entries] == ["M.S.", "B.Tech."]
    entries.append({"school": "Night School", "degree": "Certificate", "nonsense": "dropped"})
    cid = ab.set_list("education", entries)
    entries2, saved2 = ab.list_state("education")
    assert saved2 and [e["school"] for e in entries2][-1] == "Night School"
    assert "nonsense" not in entries2[-1] and set(entries2[-1]) == {f for f, _l, _k in ab.LIST_FIELDS["education"]}
    assert json.loads(bank.read_text())["answers"]["first_name"] == "Sam"     # nothing else touched
    ab.undo(cid)
    assert ab.list_state("education")[1] is False                            # back to the resume


def test_an_emptied_list_stays_empty_and_blank_entries_are_dropped(bank):
    ab.set_list("experience", [{"title": "", "company": ""}])
    assert ab.get_list("experience") == [] and ab.list_state("experience")[1] is True


@pytest.mark.parametrize("section,entries", [("hobbies", []), ("experience", "x"), ("experience", ["x"]),
                                             ("experience", [{}] * 31)])
def test_bad_lists_are_refused(bank, section, entries):
    with pytest.raises(ValueError):
        ab.set_list(section, entries)
