"""features/resume/skills.py — hard skills in a job description and in a resume. Pure code.

The lexicon (skills_lexicon.json, in this folder) maps category → canonical skill → aliases.
Matching is case- and punctuation-insensitive ("Node.js" = "NodeJS" = "node js"), aliases
starting with "=" are case-sensitive as written ("Go", "R", "Rust"), and every alias of a
skill counts as that skill ("k8s" = "Kubernetes", "Postgres" = "PostgreSQL").

    extract(jd_text)            → [Skill]  the JD's skills, in order of first mention, each with
                                           the wording the JD used ("k8s" stays "k8s")
    present(skill, text)        → bool     the skill under any of its aliases
    canonical(term)             → Skill | None
    category_of(name)           → "Cloud & DevOps" | ...

Never raises on bad input; a missing lexicon file is a packaging error and does raise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

LEXICON_PATH = Path(__file__).with_name("skills_lexicon.json")

_SEP = r"[\s.\-/_]*"
_EDGE_L = r"(?<![A-Za-z0-9+#])"
_EDGE_R = r"(?![A-Za-z0-9+#]|\.[A-Za-z0-9])"


@dataclass(frozen=True)
class Skill:
    name: str                  # canonical name ("PostgreSQL")
    category: str              # lexicon category ("Databases")
    wording: str = ""          # how the JD wrote it ("Postgres"); "" when not from a JD

    @property
    def text(self) -> str:
        """What goes on the resume: the JD's wording when there is one."""
        return self.wording or self.name


def _alias_regex(alias: str) -> re.Pattern:
    strict = alias.startswith("=")
    raw = alias[1:] if strict else alias
    parts = [p for p in re.split(r"[\s.\-/_]+", raw) if p]
    lead = r"\.?" if raw.startswith(".") else ""
    body = _SEP.join(re.escape(p) for p in parts)
    if strict and len(raw) == 1:            # "C" / "R": never "C-suite", "R&D", "R's"
        return re.compile(r"(?<![A-Za-z0-9+#\-&'’])" + body + r"(?![A-Za-z0-9+#\-&'’]|\.[A-Za-z0-9])")
    return re.compile(_EDGE_L + lead + body + _EDGE_R, 0 if strict else re.I)


@lru_cache(maxsize=1)
def _lexicon() -> tuple:
    """((Skill, [regex, ...]), ...) — canonical name and its own spelling count as aliases."""
    data = json.loads(LEXICON_PATH.read_text())
    out = []
    for category, skills in data.items():
        if category.startswith("_"):
            continue
        for name, aliases in skills.items():
            forms = list(aliases)
            if not any(a.lstrip("=").lower() == name.lower() for a in forms):
                forms.append(name if not name.isupper() or len(name) > 3 else "=" + name)
            out.append((Skill(name, category), tuple(_alias_regex(a) for a in forms)))
    return tuple(out)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", "", (s or "").lower())


def canonical(term: str) -> Optional[Skill]:
    """The lexicon skill `term` names (any alias), else None."""
    t = (term or "").strip()
    if not t:
        return None
    for skill, regexes in _lexicon():
        if _norm(skill.name) == _norm(t):
            return skill
        for rx in regexes:
            m = rx.fullmatch(t)
            if m:
                return skill
    return None


def category_of(name: str) -> str:
    s = canonical(name)
    return s.category if s else ""


def present(skill, text: str) -> bool:
    """`skill` (a Skill or a name) appears in `text` under any alias. A skill outside the
    lexicon (an LLM-found term) is matched on its own wording."""
    if not text:
        return False
    name = skill.name if isinstance(skill, Skill) else str(skill)
    for s, regexes in _lexicon():
        if s.name == name:
            return any(rx.search(text) for rx in regexes)
    wording = skill.text if isinstance(skill, Skill) else name
    return bool(_alias_regex(wording).search(text))


def extract(jd_text: str) -> list:
    """The JD's lexicon skills in order of first mention, with the JD's own wording.
    A skill whose only hit is inside a longer skill's hit ("React" in "React Native",
    "Kafka" in "Kafka Streams") still counts — both are named."""
    text = jd_text or ""
    found = []
    for skill, regexes in _lexicon():
        first = None
        for rx in regexes:
            m = rx.search(text)
            if m and (first is None or m.start() < first.start()):
                first = m
        if first is not None:
            found.append((first.start(), Skill(skill.name, skill.category, _wording(first.group(0), skill))))
    found.sort(key=lambda x: x[0])
    return [s for _pos, s in found]


def _wording(hit: str, skill: Skill) -> str:
    """The JD's spelling, tidied: all-lowercase hits ("kubernetes", "postgres") take the
    canonical's capitalisation when they are the same word, else title case for plain words."""
    hit = " ".join(hit.split())
    if _norm(hit) == _norm(skill.name):
        return skill.name
    if hit.islower() and len(hit) <= 4:
        return hit.upper() if hit.isalpha() and len(hit) <= 3 else hit
    if hit.islower():
        return " ".join(w if any(c in w for c in "./+#") else w.capitalize() for w in hit.split())
    return hit


def all_categories() -> list:
    seen = []
    for skill, _ in _lexicon():
        if skill.category not in seen:
            seen.append(skill.category)
    return seen
