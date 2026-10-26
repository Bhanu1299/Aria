"""features/resume/render.py — resume.json → the one-page PDF, in the base resume's exact layout.

Measured from the user's PDF (US letter 612×792; y = points from the top, baselines):
  name       Times-Bold 17, centered, y=40
  contact    Times-Roman 9.5, centered, y=58, " | " separators, links underlined 0.4pt + linked
  headline   Times-Bold 9.5, centered, y=70
  section    Times-Bold 11.5 at x=42, 0.7pt rule x=42..570 under it
  entry      Times-Bold 9.5 at x=42, dates right-aligned at x=570; role Times-Italic 9.5
  bullets    "•" Helvetica 10 at x=46, text Times-Roman 9.5 at x=56 wrapped to x=570, leading 11.2
  skills     bullets of bold "Label: " + roman items
  project    bold title + "  |  github.com/…" roman, underlined, linked
The layout never changes: what doesn't fit is dropped by the caller (fit.py), never shrunk.

    layout(d)                → Layout  (lines placed, bottom y, lines per bullet) — no file written
    render(d, path)          → Layout  writes the PDF
    fits(d)                  → bool    one page
    bullet_lines(text)       → int     lines a bullet wraps to
    render_letter(...)       → None    cover letter, same fonts, one page

Pure code (ReportLab). render() raises on an unwritable path; the rest never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as rl_canvas

from aria.features.resume import model

PAGE_W, PAGE_H = letter                      # 612 × 792
LEFT, RIGHT = 42.0, 570.0
BULLET_X, TEXT_X = 46.0, 56.0
BOTTOM = 756.0                               # last baseline allowed (36pt bottom margin)
LEADING = 11.2
BODY = ("Times-Roman", 9.5)
BOLD = ("Times-Bold", 9.5)
ITALIC = ("Times-Italic", 9.5)
NAME = ("Times-Bold", 17.0)
SECTION = ("Times-Bold", 11.5)
BULLET_GLYPH = ("Helvetica", 10.0)
NAME_Y, CONTACT_Y, HEADLINE_Y = 40.0, 58.0, 70.0
SECTION_GAP = 7.0                            # space above a section title
RULE_DROP = 2.6                              # rule below the section title's baseline
AFTER_RULE = 11.0                            # first line baseline below the title baseline
ENTRY_GAP = 2.4                              # extra space between entries


@dataclass
class Run:
    text: str
    font: tuple
    url: str = ""


@dataclass
class Line:
    y: float                                 # baseline, points from the top
    runs: list                               # [(x, Run)]
    right: str = ""                          # right-aligned text at x=570 (dates)
    center: bool = False
    rule: bool = False
    glyph: bool = False                      # a "•" at BULLET_X


@dataclass
class Layout:
    lines: list = field(default_factory=list)
    bottom: float = 0.0
    bullet_lines: dict = field(default_factory=dict)   # bullet id → lines it wrapped to

    @property
    def one_page(self) -> bool:
        return self.bottom <= BOTTOM


def width(text: str, font: tuple) -> float:
    return stringWidth(text, font[0], font[1])


def _wrap(runs: list, x0: float, x1: float) -> list:
    """Word-wrap styled runs into lines of [(x, Run)] between x0 and x1."""
    words: list = []                                  # (word, font, url, space_before)
    pending = False
    for r in runs:
        for piece in re.findall(r"\s+|\S+", r.text):
            if piece.isspace():
                pending = True
            else:
                words.append((piece, r.font, r.url, pending))
                pending = False
    lines: list = [[]]
    x = x0
    for w, font, url, space in words:
        sp = width(" ", font) if space and lines[-1] else 0.0
        ww = width(w, font)
        if lines[-1] and x + sp + ww > x1 + 0.01:
            lines.append([])
            x, sp = x0, 0.0
        x += sp
        lines[-1].append((x, Run(w, font, url)))
        x += ww
    return [ln for ln in lines if ln] or [[]]


def bullet_lines(text: str) -> int:
    return len(_wrap([Run(text, BODY)], TEXT_X, RIGHT))


def _center_runs(runs: list) -> list:
    total = sum(width(r.text, r.font) for r in runs)
    x = (PAGE_W - total) / 2
    out = []
    for r in runs:
        out.append((x, r))
        x += width(r.text, r.font)
    return out


def layout(d: dict) -> Layout:
    d = model.with_ids(d)
    out = Layout()
    put = out.lines.append
    put(Line(NAME_Y, _center_runs([Run(d["name"], NAME)]), center=True))
    contact: list = []
    for i, c in enumerate(d["contact"]):
        if i:
            contact.append(Run(" | ", BODY))
        contact.append(Run(c["text"], BODY, c["url"]))
    y = NAME_Y
    if contact:
        y = CONTACT_Y
        put(Line(y, _center_runs(contact), center=True))
    if d["headline"]:
        y = HEADLINE_Y
        put(Line(y, _center_runs([Run(d["headline"], BOLD)]), center=True))

    def section(title: str) -> None:
        nonlocal y
        y += SECTION_GAP + 11.5
        put(Line(y, [(LEFT, Run(title, SECTION))], rule=True))
        y += AFTER_RULE - LEADING

    def bullet(runs: list, bid: str = "") -> None:
        nonlocal y
        wrapped = _wrap(runs, TEXT_X, RIGHT)
        for k, ln in enumerate(wrapped):
            y += LEADING
            put(Line(y, ln, glyph=(k == 0)))
        if bid:
            out.bullet_lines[bid] = len(wrapped)

    def header(runs: list, right: str) -> None:
        nonlocal y
        y += LEADING
        put(Line(y, [(LEFT, r) for r in runs] if len(runs) == 1 else _place(runs, LEFT), right=right))

    if d["skills"]:
        section("TECHNICAL SKILLS")
        for s in d["skills"]:
            bullet([Run(f"{s['label']}: ", BOLD), Run(", ".join(s["items"]), BODY)])
    if d["experience"]:
        section("EXPERIENCE")
        for n, e in enumerate(d["experience"]):
            if n:
                y += ENTRY_GAP
            head = e["company"] + (f", {e['location']}" if e["location"] else "")
            header([Run(head, BOLD)], e["dates"])
            if e["role"]:
                y += LEADING
                put(Line(y, [(LEFT, Run(e["role"], ITALIC))]))
            for bid, b in zip(e["_ids"], e["bullets"]):
                bullet([Run(b, BODY)], bid)
    if d["projects"]:
        section("PROJECTS")
        for n, p in enumerate(d["projects"]):
            if n:
                y += ENTRY_GAP
            runs = [Run(p["title"], BOLD)]
            if p["link"]:
                url = p["link"] if p["link"].startswith("http") else f"https://{p['link']}"
                runs += [Run("  |  ", BODY), Run(p["link"], BODY, url)]
            header(runs, p["dates"])
            for bid, b in zip(p["_ids"], p["bullets"]):
                bullet([Run(b, BODY)], bid)
    if d["education"]:
        section("EDUCATION")
        for n, e in enumerate(d["education"]):
            if n:
                y += ENTRY_GAP
            head = e["school"] + (f", {e['location']}" if e["location"] else "")
            header([Run(head, BOLD)], e["dates"])
            line = e["degree"] + (f"; {e['detail']}" if e["detail"] else "")
            if line:
                y += LEADING
                put(Line(y, [(LEFT, Run(line, ITALIC))]))
    if d["certifications"]:
        section("CERTIFICATIONS")
        for c in d["certifications"]:
            bullet([Run(c, BODY)])
    out.bottom = y
    return out


def _place(runs: list, x: float) -> list:
    out = []
    for r in runs:
        out.append((x, r))
        x += width(r.text, r.font)
    return out


def fits(d: dict) -> bool:
    return layout(d).one_page


def _draw(c, lay: Layout) -> None:
    for ln in lay.lines:
        base = PAGE_H - ln.y
        if ln.glyph:
            c.setFont(*BULLET_GLYPH)
            c.drawString(BULLET_X, base, "•")
        for x, r in ln.runs:
            c.setFont(*r.font)
            c.drawString(x, base, r.text)
            if r.url:
                w = width(r.text, r.font)
                c.setLineWidth(0.4)
                c.line(x, base - 1.2, x + w, base - 1.2)
                c.linkURL(r.url, (x, base - 2, x + w, base + r.font[1] - 1), relative=0)
        if ln.right:
            c.setFont(*BOLD)
            c.drawRightString(RIGHT, base, ln.right)
        if ln.rule:
            c.setLineWidth(0.7)
            c.line(LEFT, base - RULE_DROP, RIGHT, base - RULE_DROP)


def render(d: dict, path) -> Layout:
    lay = layout(d)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = rl_canvas.Canvas(str(p), pagesize=letter, pageCompression=1)
    c.setTitle(f"{model.normalize(d)['name']} — Resume")
    c.setAuthor(model.normalize(d)["name"])
    _draw(c, lay)
    c.showPage()
    c.save()
    return lay


def render_letter(d: dict, paragraphs: list, path, date_line: str = "", greeting: str = "Dear Hiring Team,",
                  closing: str = "Sincerely,") -> bool:
    """Cover letter in the resume's fonts: the same name / contact header, then the letter.
    One page: paragraphs that don't fit are dropped from the end (the caller keeps it short).
    Returns False when nothing fit."""
    d = model.normalize(d)
    lay = Layout()
    put = lay.lines.append
    put(Line(NAME_Y, _center_runs([Run(d["name"], NAME)]), center=True))
    contact: list = []
    for i, c in enumerate(d["contact"]):
        if i:
            contact.append(Run(" | ", BODY))
        contact.append(Run(c["text"], BODY, c["url"]))
    if contact:
        put(Line(CONTACT_Y, _center_runs(contact), center=True))
    y = HEADLINE_Y + 20
    left, right = 72.0, 540.0
    if date_line:
        put(Line(y, [(left, Run(date_line, BODY))]))
        y += LEADING * 2
    put(Line(y, [(left, Run(greeting, BODY))]))
    y += LEADING
    used = 0
    for para in paragraphs:
        wrapped = _wrap([Run(para, BODY)], left, right)
        if y + LEADING * (len(wrapped) + 4) > BOTTOM:
            break
        y += LEADING * 0.8
        for ln in wrapped:
            y += LEADING
            put(Line(y, ln))
        used += 1
    if not used:
        return False
    y += LEADING * 1.8
    put(Line(y, [(left, Run(closing, BODY))]))
    y += LEADING
    put(Line(y, [(left, Run(d["name"], BODY))]))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = rl_canvas.Canvas(str(p), pagesize=letter, pageCompression=1)
    c.setTitle(f"{d['name']} — Cover Letter")
    c.setAuthor(d["name"])
    _draw(c, lay)
    c.showPage()
    c.save()
    return True
