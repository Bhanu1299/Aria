"""features/ats/resume_sections.py — the resume's entries as lists: jobs, schools, projects,
certifications. Pure code, no LLM: section headings (EXPERIENCE, EDUCATION …), an entry line
that ends in a date range ("Northwind Jan 2025 - May 2025"), the line under it (the title, or the
school), then its • bullets.

parse(text) → {"experience": [...], "education": [...], "projects": [...], "certifications": [...]}
Entries use answer_bank.LIST_FIELDS keys. Never raises; what it can't read is simply missing
(the user fixes it in the dashboard).
"""
from __future__ import annotations

import re

_HEADINGS = {
    "experience": r"(?:work |professional |relevant )?experience|employment(?: history)?|work history",
    "education": r"education(?: and training)?|academics?",
    "projects": r"(?:personal |selected |academic )?projects",
    "certifications": r"certifications?(?: and licenses)?|licenses?(?: and certifications)?",
}
_HEADING = re.compile(r"^\s*(" + "|".join(f"(?P<{k}>{v})" for k, v in _HEADINGS.items()) + r")\s*:?\s*$", re.I)
_OTHER_HEADING = re.compile(r"^[A-Z][A-Z &/,-]{2,40}$")             # SKILLS, TECHNICAL SKILLS, AWARDS …
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
_END = rf"(?:{_DATE}|present|current|now|ongoing)"
_RANGE = re.compile(rf"^(?P<head>.*?)\s*(?P<start>{_DATE})\s*(?:-|–|—|to)\s*(?P<end>{_END})\s*$", re.I)
_SINGLE = re.compile(rf"^(?P<head>.*?)\s+(?P<end>(?:expected\s+)?{_DATE})\s*$", re.I)
_BULLET = re.compile(r"^\s*[•●▪◦\-*]\s+")
_DEGREE = re.compile(r"^(?:m\.?\s?s\.?|b\.?\s?s\.?|b\.?\s?a\.?|m\.?\s?a\.?|b\.?\s?tech\.?|m\.?\s?tech\.?|b\.?\s?e\.?|"
                     r"m\.?\s?e\.?|ph\.?\s?d\.?|mba|bachelor|master|doctor|associate|diploma)\b", re.I)


def _title_case_month(d: str) -> str:
    """'jan 2025' / 'January 2025' → 'Jan 2025'; '05/2025' and '2025' stay; 'present' → 'Present'."""
    d = d.strip().rstrip(".")
    if re.fullmatch(r"(?i)present|current|now|ongoing", d):
        return "Present"
    m = re.match(rf"(?i)({_MONTH})\s+(\d{{4}})", d)
    if m:
        return f"{m.group(1)[:3].capitalize()} {m.group(2)}"
    return d


def _sections(text: str) -> dict:
    out: dict = {}
    current = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _HEADING.match(stripped)
        if m:
            current = next(k for k in _HEADINGS if m.group(k))
            out.setdefault(current, [])
            continue
        if _OTHER_HEADING.match(stripped) and not _RANGE.match(stripped) and stripped.upper() == stripped:
            current = None
            continue
        if current:
            out[current].append(stripped)
    return out


def _entries(lines: list) -> list:
    """[(head, start, end, [lines under it], [bullets])] — one per dated entry line."""
    found: list = []
    for line in lines:
        m = _RANGE.match(line) if not _BULLET.match(line) else None
        if m and m.group("head").strip():
            found.append([m.group("head").strip(" ,|-–—"), _title_case_month(m.group("start")),
                          _title_case_month(m.group("end")), [], []])
            continue
        if not found:
            continue
        if _BULLET.match(line):
            found[-1][4].append(_BULLET.sub("", line).strip())
        elif found[-1][4]:
            found[-1][4][-1] += " " + line                  # a bullet wrapped onto the next line
        else:
            found[-1][3].append(line)
    return found


def _description(bullets: list) -> str:
    return "\n".join(f"• {b}" for b in bullets)


def _experience(lines: list) -> list:
    out = []
    for head, start, end, under, bullets in _entries(lines):
        title = under[0] if under else ""
        company, location = head, ""
        if " | " in head:
            company, location = [p.strip() for p in head.split(" | ", 1)]
        out.append({"title": title, "company": company, "location": location, "start": start, "end": end,
                    "description": _description(bullets)})
    return out


def _split_degree(text: str) -> tuple:
    """'M.S., Computer Science' → ('M.S.', 'Computer Science'); 'Bachelor of Science in Physics'
    → ('Bachelor of Science', 'Physics')."""
    m = re.match(r"^(.*?),\s*(.+)$", text)
    if m and _DEGREE.match(m.group(1)):
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"^(.*?)\s+in\s+(.+)$", text, re.I)
    if m and _DEGREE.match(m.group(1)):
        return m.group(1).strip(), m.group(2).strip()
    return text.strip(), ""


def _education(lines: list) -> list:
    out = []
    for head, start, end, under, _bullets in _entries(lines):
        other = under[0] if under else ""
        degree_line, school = (head, other) if _DEGREE.match(head) or not _DEGREE.match(other) else (other, head)
        degree, field = _split_degree(degree_line)
        gpa = ""
        for extra in under[1:]:
            g = re.search(r"(?i)\bgpa\b\s*:?\s*([0-4](?:\.\d{1,2})?)", extra)
            if g:
                gpa = g.group(1)
        out.append({"school": school, "degree": degree, "field": field, "start": start, "end": end, "gpa": gpa})
    return out


def _projects(lines: list) -> list:
    return [{"name": head, "start": start, "end": end, "link": "", "description": _description(bullets)}
            for head, start, end, _under, bullets in _entries(lines)]


def _certifications(lines: list) -> list:
    out = []
    for line in lines:
        name = _BULLET.sub("", line).strip()
        if not name:
            continue
        m = _SINGLE.match(name)
        date = ""
        if m and m.group("head").strip():
            name, date = m.group("head").strip(" ,|-–—"), _title_case_month(m.group("end"))
        out.append({"name": name, "issuer": "", "date": date})
    return out


def parse(text: str) -> dict:
    try:
        sec = _sections(text)
        return {"experience": _experience(sec.get("experience", [])),
                "education": _education(sec.get("education", [])),
                "projects": _projects(sec.get("projects", [])),
                "certifications": _certifications(sec.get("certifications", []))}
    except Exception:
        return {"experience": [], "education": [], "projects": [], "certifications": []}
