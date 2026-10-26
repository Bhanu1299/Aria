"""features/resume/model.py — the resume as structured data (resume.json), and the base one.

Shape (every list keeps the order it is printed in):
    {"name": str,
     "contact": [{"text": str, "url": str}],              url "" = plain text
     "headline": str,
     "skills": [{"label": str, "items": [str]}],
     "experience": [{"company", "location", "dates", "role", "bullets": [str]}],
     "projects": [{"title", "dates", "link", "bullets": [str]}],
     "education": [{"school", "location", "dates", "degree", "detail"}],
     "certifications": [str]}

Bullet ids are positional in the BASE resume: "e<i>.<j>" (experience i, bullet j) and
"p<i>.<j>"; tailoring carries them in `_ids` so a reordered bullet keeps its id.

    BASE_PATH                       ~/.aria/resume_base.json (personal: never committed)
    load_base() / save_base(d)
    build_from_text(text)           pdftotext output of the user's PDF → resume dict
    ensure_base(pdf_path)           load_base(), else build it from the PDF once
    plain_text(d)                   the resume's words in print order (what pdftotext should read)
    numbers(text)                   every number / percentage in a string
    facts(d)                        the facts tailoring must never change

Never raises except save_base (an OSError the caller logs).
"""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Optional

from aria.features.ats import resume_sections

logger = logging.getLogger(__name__)

BASE_PATH = Path.home() / ".aria" / "resume_base.json"
SECTIONS = ("TECHNICAL SKILLS", "EXPERIENCE", "PROJECTS", "EDUCATION", "CERTIFICATIONS")

_NUMBER = re.compile(r"(?<![A-Za-z])[$€£]?\d[\d,]*(?:\.\d+)?\s*(?:%|[kKmMbB]\b|x\b|\+)?")
_EMAIL = re.compile(r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_URLISH = re.compile(r"(?i)^(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:/\S*)?$")


def empty() -> dict:
    return {"name": "", "contact": [], "headline": "", "skills": [], "experience": [], "projects": [],
            "education": [], "certifications": []}


def normalize(d: dict) -> dict:
    """A resume dict with every key present and the right types (missing → empty)."""
    out = empty()
    if not isinstance(d, dict):
        return out
    out["name"] = str(d.get("name") or "")
    out["headline"] = str(d.get("headline") or "")
    out["contact"] = [{"text": str(c.get("text", "")), "url": str(c.get("url", ""))}
                      for c in d.get("contact") or [] if isinstance(c, dict) and c.get("text")]
    out["skills"] = [{"label": str(s.get("label", "")), "items": [str(i) for i in s.get("items") or []]}
                     for s in d.get("skills") or [] if isinstance(s, dict)]
    for e in d.get("experience") or []:
        row = {k: str(e.get(k) or "") for k in ("company", "location", "dates", "role")}
        row["bullets"] = [str(b) for b in e.get("bullets") or []]
        if e.get("_ids"):
            row["_ids"] = list(e["_ids"])
        out["experience"].append(row)
    for p in d.get("projects") or []:
        row = {k: str(p.get(k) or "") for k in ("title", "dates", "link")}
        row["bullets"] = [str(b) for b in p.get("bullets") or []]
        if p.get("_ids"):
            row["_ids"] = list(p["_ids"])
        out["projects"].append(row)
    out["education"] = [{k: str(e.get(k) or "") for k in ("school", "location", "dates", "degree", "detail")}
                        for e in d.get("education") or [] if isinstance(e, dict)]
    out["certifications"] = [str(c) for c in d.get("certifications") or [] if str(c).strip()]
    return out


def with_ids(d: dict) -> dict:
    """A copy whose entries carry `_ids` (positional ids from this resume) where missing."""
    out = copy.deepcopy(normalize(d))
    for prefix, key in (("e", "experience"), ("p", "projects")):
        for i, entry in enumerate(out[key]):
            if not entry.get("_ids") or len(entry["_ids"]) != len(entry["bullets"]):
                entry["_ids"] = [f"{prefix}{i}.{j}" for j in range(len(entry["bullets"]))]
    return out


def bullets(d: dict) -> dict:
    """{bullet id: text} over experience and projects (ids from with_ids)."""
    out = {}
    for key in ("experience", "projects"):
        for entry in with_ids(d)[key]:
            out.update(zip(entry["_ids"], entry["bullets"]))
    return out


def strip_ids(d: dict) -> dict:
    out = copy.deepcopy(d)
    for key in ("experience", "projects"):
        for entry in out.get(key) or []:
            entry.pop("_ids", None)
    return out


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def load(path) -> Optional[dict]:
    try:
        return normalize(json.loads(Path(path).expanduser().read_text()))
    except (OSError, ValueError) as exc:
        logger.debug("resume.model: can't read %s: %s", path, exc)
        return None


def save(d: dict, path) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(normalize(d), indent=2, ensure_ascii=False))


def load_base() -> Optional[dict]:
    d = load(BASE_PATH)
    return d if d and (d["experience"] or d["projects"]) else None


def save_base(d: dict) -> None:
    save(d, BASE_PATH)


def ensure_base(pdf_path: Optional[str]) -> Optional[dict]:
    """The base resume: ~/.aria/resume_base.json, else built once from the PDF (and saved)."""
    d = load_base()
    if d is not None:
        return d
    if not pdf_path:
        return None
    try:
        from aria.features.ats import resume_text
        built = build_from_text(resume_text.text(pdf_path))
        if built["experience"] or built["projects"]:
            save_base(built)
            return built
    except Exception as exc:
        logger.warning("resume.model: building the base resume from %s failed: %s", pdf_path, exc)
    return None


# ---------------------------------------------------------------------------
# PDF text → resume dict
# ---------------------------------------------------------------------------

def _contact_item(text: str) -> dict:
    t = text.strip()
    if _EMAIL.fullmatch(t):
        return {"text": t, "url": f"mailto:{t}"}
    if _URLISH.match(t) and not _PHONE.fullmatch(t):
        return {"text": t, "url": t if t.lower().startswith("http") else f"https://{t}"}
    return {"text": t, "url": ""}


def _section_lines(text: str, heading: str) -> list:
    out, inside = [], False
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r"[A-Z][A-Z &/,-]{2,40}", s):
            inside = bool(re.fullmatch(heading, s, re.I))
            continue
        if inside:
            out.append(s)
    return out


def _skills(text: str) -> list:
    rows: list = []
    for line in _section_lines(text, r"(?:technical )?skills(?: (?:and|&) tools)?"):
        s = re.sub(r"^[•●▪◦\-*]\s*", "", line)
        m = re.match(r"^([^:]{2,40}):\s*(.+)$", s)
        if m:
            rows.append({"label": m.group(1).strip(), "items": [i.strip() for i in re.split(r",\s*", m.group(2))
                                                                   if i.strip()]})
        elif rows:                                  # a skill line wrapped onto the next line
            rows[-1]["items"] += [i.strip() for i in re.split(r",\s*", s) if i.strip()]
    return rows


def _split_link(head: str) -> tuple:
    if " | " in head:
        title, rest = head.split(" | ", 1)
        if _URLISH.match(rest.strip()):
            return title.strip(), rest.strip()
    return head.strip(), ""


def build_from_text(text: str) -> dict:
    """The base resume from its pdftotext text: header lines, skills, and the dated entries
    resume_sections already reads. What it can't read stays empty (fixed in the dashboard)."""
    d = empty()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return d
    d["name"] = lines[0]
    for line in lines[1:3]:
        if re.fullmatch(r"[A-Z][A-Z &/,-]{2,40}", line):
            break
        if _EMAIL.search(line) or _PHONE.search(line) or "|" in line and not d["contact"]:
            d["contact"] = [_contact_item(p) for p in re.split(r"\s+\|\s+|\s{3,}", line) if p.strip()]
        elif not d["headline"]:
            d["headline"] = line
    d["skills"] = _skills(text)
    parsed = resume_sections.parse(text)
    for e in parsed["experience"]:
        d["experience"].append({"company": e["company"], "location": e["location"],
                                "dates": f"{e['start']} - {e['end']}", "role": e["title"],
                                "bullets": [b[2:] for b in e["description"].splitlines() if b.startswith("• ")]})
    for p in parsed["projects"]:
        title, link = _split_link(p["name"])
        d["projects"].append({"title": title, "dates": f"{p['start']} - {p['end']}", "link": link,
                              "bullets": [b[2:] for b in p["description"].splitlines() if b.startswith("• ")]})
    for e in parsed["education"]:
        degree = ", ".join(x for x in (e["degree"], e["field"]) if x)
        d["education"].append({"school": e["school"], "location": "", "dates": f"{e['start']} - {e['end']}",
                               "degree": degree, "detail": f"GPA: {e['gpa']}" if e["gpa"] else ""})
    d["certifications"] = [c["name"] + (f" ({c['date']})" if c["date"] else "") for c in parsed["certifications"]]
    return d


# ---------------------------------------------------------------------------
# text and facts
# ---------------------------------------------------------------------------

def contact_line(d: dict) -> str:
    return " | ".join(c["text"] for c in d.get("contact") or [])


def plain_text(d: dict) -> str:
    """Every word of the resume in print order (no layout)."""
    d = normalize(d)
    out = [d["name"], contact_line(d), d["headline"]]
    if d["skills"]:
        out.append("TECHNICAL SKILLS")
        out += [f"{s['label']}: {', '.join(s['items'])}" for s in d["skills"]]
    if d["experience"]:
        out.append("EXPERIENCE")
        for e in d["experience"]:
            out += [" ".join(x for x in (e["company"], e["location"], e["dates"]) if x), e["role"]] + e["bullets"]
    if d["projects"]:
        out.append("PROJECTS")
        for p in d["projects"]:
            out += [" ".join(x for x in (p["title"], p["link"], p["dates"]) if x)] + p["bullets"]
    if d["education"]:
        out.append("EDUCATION")
        for e in d["education"]:
            out += [" ".join(x for x in (e["school"], e["location"], e["dates"]) if x), e["degree"], e["detail"]]
    if d["certifications"]:
        out.append("CERTIFICATIONS")
        out += d["certifications"]
    return "\n".join(x for x in out if x)


def bullet_text(d: dict) -> str:
    d = normalize(d)
    return "\n".join(b for key in ("experience", "projects") for e in d[key] for b in e["bullets"])


def numbers(text: str) -> list:
    """Numbers as written ("40%", "$1.2M", "3x", "10,000"), normalised for comparison."""
    out = []
    for m in _NUMBER.finditer(text or ""):
        n = re.sub(r"\s+", "", m.group(0)).lower().rstrip(",.")
        if n and any(c.isdigit() for c in n):
            out.append(n)
    return out


def facts(d: dict) -> dict:
    """What tailoring never changes: employers, roles, dates, schools, degrees, certifications."""
    d = normalize(d)
    return {"experience": [(e["company"], e["role"], e["dates"]) for e in d["experience"]],
            "projects": [(p["title"], p["dates"], p["link"]) for p in d["projects"]],
            "education": [(e["school"], e["degree"], e["dates"], e["detail"]) for e in d["education"]],
            "certifications": list(d["certifications"]),
            "name": d["name"], "contact": contact_line(d)}
