"""features/resume/ats_score.py — the strict ATS check: a resume PDF against a job description.

Pure code, deterministic, 0–100, modeled on Jobscan-style scoring:
  parse        25  pdftotext reads it in order; name, email, phone found; standard section
                   headers; every date parses (Mon YYYY - Mon YYYY|Present); no text lost
                   against resume.json
  hard_skills  40  share of the JD's hard skills (skills.extract) found anywhere in the PDF
                   text, any alias ("JS" = "JavaScript", "k8s" = "Kubernetes")
  title        10  share of the JD title's words in the headline / most recent role / a
                   project header
  context      15  share of the JD's hard skills that appear in a bullet, not only the
                   skills list
  format       10  one page, no images, no tables, standard fonts, file < 1 MB, every bullet
                   at most 2 lines

    score(pdf_path, resume, jd_text, jd_title) → dict
        {"score": int, "parts": {name: {"points", "max", "detail"}}, "missing": [skill],
         "not_in_bullets": [skill], "jd_skills": [skill]}

Never raises: a check that can't run scores 0 for its part and says why in "detail".
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from aria.features.resume import model, render, skills

logger = logging.getLogger(__name__)

WEIGHTS = {"parse": 25, "hard_skills": 40, "title": 10, "context": 15, "format": 10}
MAX_BYTES = 1024 * 1024
STANDARD_FONTS = {"Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic", "Helvetica", "Helvetica-Bold",
                  "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier", "Courier-Bold", "Symbol",
                  "ZapfDingbats", "Arial", "Calibri", "Cambria", "Georgia", "Garamond", "TimesNewRoman"}
STANDARD_HEADERS = {"TECHNICAL SKILLS", "SKILLS", "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE",
                    "PROJECTS", "EDUCATION", "CERTIFICATIONS", "SUMMARY"}
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
DATE_RANGE = re.compile(rf"^{_MONTH} \d{{4}} - (?:{_MONTH} \d{{4}}|Present)$")
_TITLE_STOP = {"senior", "sr", "junior", "jr", "staff", "principal", "lead", "i", "ii", "iii", "iv", "new", "grad",
               "graduate", "early", "career", "entry", "level", "the", "of", "and", "for", "a", "an", "to", "in",
               "remote", "hybrid", "onsite", "intern", "contract", "full", "time", "fulltime", "us", "usa", "2026",
               "2027", "with", "at", "&", "-", "–", "—", "/"}
_EMAIL = re.compile(r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\d)\+?\(?\d(?:[\s().-]{0,2}\d){9,14}(?!\d)")


def _tool(name: str) -> Optional[str]:
    return shutil.which(name) or (f"/opt/homebrew/bin/{name}" if Path(f"/opt/homebrew/bin/{name}").exists() else None)


def _run(name: str, args: list) -> str:
    exe = _tool(name)
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, *args], capture_output=True, timeout=30)
        return out.stdout.decode("utf-8", errors="replace") if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ats_score: %s failed: %s", name, exc)
        return ""


def pdf_text(pdf_path) -> str:
    """pdftotext in reading order (no -layout: what a parser that ignores columns reads)."""
    return _run("pdftotext", ["-enc", "UTF-8", str(pdf_path), "-"])


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def title_words(title: str) -> list:
    return [w for w in _words(title) if w not in _TITLE_STOP and len(w) > 1 or w in ("ml", "ai", "qa", "ui", "ux")]


def _part(points: float, maxi: int, detail: str) -> dict:
    return {"points": round(max(0.0, min(points, maxi)), 2), "max": maxi, "detail": detail}


def _parse_part(text: str, resume: dict) -> dict:
    checks, notes = 0.0, []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 1. order: name first, sections in the resume's order
    heads = [ln for ln in lines if ln.upper() == ln and ln in STANDARD_HEADERS]
    expected = [h for h, key in zip(model.SECTIONS, ("skills", "experience", "projects", "education",
                                                      "certifications")) if resume.get(key)]
    in_order = bool(lines) and lines[0] == resume["name"] and heads == expected
    checks += 5 if in_order else 0
    if not in_order:
        notes.append("text doesn't read in order")
    # 2. name, email, phone
    found = sum([bool(resume["name"]) and resume["name"] in text, bool(_EMAIL.search(text)),
                 bool(_PHONE.search(text))])
    checks += 5 * found / 3
    if found < 3:
        notes.append("name / email / phone not all found")
    # 3. standard headers only
    odd = [ln for ln in lines if re.fullmatch(r"[A-Z][A-Z &/]{3,40}", ln) and ln not in STANDARD_HEADERS
           and ln != resume["name"].upper()]
    checks += 5 if heads and not odd else 0
    if odd:
        notes.append(f"non-standard headers: {', '.join(odd[:3])}")
    # 4. dates parse
    dates = [e["dates"] for k in ("experience", "projects", "education") for e in resume[k] if e.get("dates")]
    bad = [dt for dt in dates if not DATE_RANGE.match(dt)]
    readable = [dt for dt in dates if dt in text]
    checks += 5 * (len(dates) - len(bad)) / len(dates) * (len(readable) / len(dates)) if dates else 5
    if bad:
        notes.append(f"dates that don't parse: {', '.join(bad[:3])}")
    # 5. nothing lost
    want = _words(model.plain_text(resume))
    have = set(_words(text))
    lost = [w for w in dict.fromkeys(want) if w not in have]
    share = 1 - len(lost) / max(1, len(set(want)))
    checks += 5 if not lost else 5 * share * 0.5
    if lost:
        notes.append(f"text lost: {', '.join(lost[:5])}")
    return _part(checks, WEIGHTS["parse"], "; ".join(notes) or "reads cleanly")


def _format_part(pdf_path: Path, resume: dict) -> dict:
    points, notes = 0.0, []
    info = _run("pdfinfo", [str(pdf_path)])
    pages = re.search(r"(?m)^Pages:\s+(\d+)", info)
    if pages and pages.group(1) == "1":
        points += 3
    else:
        notes.append(f"{pages.group(1) if pages else '?'} pages")
    images = _run("pdfimages", ["-list", str(pdf_path)])
    if images and len(images.strip().splitlines()) <= 2:
        points += 2
    else:
        notes.append("has images" if images else "couldn't list images")
    fonts = _run("pdffonts", [str(pdf_path)]).strip().splitlines()[2:]
    names = {re.sub(r"^[A-Z]{6}\+", "", f.split()[0]) for f in fonts if f.split()}
    if names and all(n in STANDARD_FONTS for n in names):
        points += 2
    else:
        notes.append(f"non-standard fonts: {', '.join(sorted(n for n in names if n not in STANDARD_FONTS))[:80]}"
                     if names else "couldn't read fonts")
    layout = _run("pdftotext", ["-layout", "-enc", "UTF-8", str(pdf_path), "-"])
    tables = [ln for ln in layout.splitlines() if len(re.findall(r"\S\s{3,}\S", ln.strip())) >= 2]
    if not tables:
        points += 1
    else:
        notes.append("looks like a table")
    try:
        small = pdf_path.stat().st_size < MAX_BYTES
    except OSError:
        small = False
    points += 1 if small else 0
    if not small:
        notes.append("file ≥ 1 MB")
    long = [b for e in resume["experience"] + resume["projects"] for b in e["bullets"] if render.bullet_lines(b) > 2]
    points += 1 if not long else 0
    if long:
        notes.append(f"{len(long)} bullet(s) over 2 lines")
    return _part(points, WEIGHTS["format"], "; ".join(notes) or "clean")


def score(pdf_path, resume: dict, jd_text: str, jd_title: str = "") -> dict:
    resume = model.normalize(resume)
    pdf_path = Path(pdf_path)
    text = pdf_text(pdf_path) if pdf_path.is_file() else ""
    parts: dict = {}
    if not text:
        parts["parse"] = _part(0, WEIGHTS["parse"], "pdftotext read nothing")
    else:
        parts["parse"] = _parse_part(text, resume)

    jd_skills = skills.extract(jd_text)
    missing = [s.name for s in jd_skills if not skills.present(s, text)]
    bullets_txt = model.bullet_text(resume)
    not_in_bullets = [s.name for s in jd_skills if not skills.present(s, bullets_txt)]
    if jd_skills:
        hard = (len(jd_skills) - len(missing)) / len(jd_skills)
        ctx = (len(jd_skills) - len(not_in_bullets)) / len(jd_skills)
    else:
        hard = ctx = 1.0
    parts["hard_skills"] = _part(WEIGHTS["hard_skills"] * hard, WEIGHTS["hard_skills"],
                                 f"{len(jd_skills) - len(missing)}/{len(jd_skills)} JD skills found"
                                 + (f"; missing {', '.join(missing[:6])}" if missing else ""))
    parts["context"] = _part(WEIGHTS["context"] * ctx, WEIGHTS["context"],
                             f"{len(jd_skills) - len(not_in_bullets)}/{len(jd_skills)} JD skills used in a bullet")

    want = title_words(jd_title)
    places = [resume["headline"]] + [e["role"] for e in resume["experience"][:1]] + \
        [p["title"] for p in resume["projects"]]
    got = set(_words(" ".join(places)))
    hit = [w for w in want if w in got]
    parts["title"] = _part(WEIGHTS["title"] * (len(hit) / len(want) if want else 1.0), WEIGHTS["title"],
                           f"title words {len(hit)}/{len(want)}" + (f"; missing {', '.join(w for w in want if w not in got)}"
                                                                    if len(hit) < len(want) else ""))
    parts["format"] = _format_part(pdf_path, resume) if text else _part(0, WEIGHTS["format"], "no PDF text")
    total = sum(p["points"] for p in parts.values())
    return {"score": int(round(total)), "parts": parts, "missing": missing, "not_in_bullets": not_in_bullets,
            "jd_skills": [s.name for s in jd_skills]}
