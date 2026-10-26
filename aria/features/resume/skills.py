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
    close(skill, have)          → bool     shares a skill family with one of `have` (names)
    mentions(skill, text)       → int      how many times the text names it
    core(skill, jd_text, title) → bool     the job can't do without it: in the title, or named
                                           2+ times with one in the required part of the JD

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
        """What goes in a bullet: the JD's wording when there is one."""
        return self.wording or self.name

    @property
    def label(self) -> str:
        """What goes on a skill line: the JD's wording, a plain lowercase phrase capitalised
        ("distributed systems" → "Distributed Systems")."""
        w = self.text
        if self.category and _norm(w) == _norm(self.name):         # "pgvector", "kubernetes" → as the lexicon spells it
            return self.name
        if w.islower() and not any(c in w for c in "./+#") and len(w) > 4:
            small = {"and", "or", "of", "as", "for", "to", "in", "on", "with", "the", "a"}
            return " ".join(x if k and x in small else x.capitalize() for k, x in enumerate(w.split()))
        if w.islower() and len(w) <= 4 and w.isalpha():
            return self.name
        return w


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
    """The JD's spelling with whitespace tidied; a lowercase hit of a proper name
    ("kubernetes", "postgresql") takes the canonical's capitalisation."""
    hit = " ".join(hit.split())
    generic = " " in skill.name and all(w[:1].isupper() and w[1:].islower() for w in skill.name.split())
    if _norm(hit) == _norm(skill.name) and hit.islower() and not skill.name.islower() \
            and skill.category != "Practices" and not generic:
        return skill.name
    return hit


def all_categories() -> list:
    seen = []
    for skill, _ in _lexicon():
        if skill.category not in seen:
            seen.append(skill.category)
    return seen


@lru_cache(maxsize=1)
def _families() -> tuple:
    """((family, frozenset(names)), ...) from the lexicon's "_families"; "*" = close for anyone."""
    data = json.loads(LEXICON_PATH.read_text()).get("_families", {})
    return tuple((k, frozenset(v)) for k, v in data.items() if not k.startswith("_"))


def close(skill, have) -> bool:
    """`skill` is the same as, or shares a family with, one of `have` (skill names)."""
    name = skill.name if isinstance(skill, Skill) else str(skill)
    have = set(have)
    if name in have:
        return True
    for fam, names in _families():
        if name in names and (fam == "*" or names & have):
            return True
    return False


def mentions(skill, text: str) -> int:
    if not text:
        return 0
    name = skill.name if isinstance(skill, Skill) else str(skill)
    for s, regexes in _lexicon():
        if s.name == name:
            spans = {m.span() for rx in regexes for m in rx.finditer(text)}
            return len(spans)
    wording = skill.text if isinstance(skill, Skill) else name
    return len(_alias_regex(wording).findall(text))


_OPTIONAL_HEAD = re.compile(r"(?i)\b(?:nice[- ]to[- ]haves?|preferred|bonus|pluses|a plus|good to have|"
                            r"extra credit|not required)\b")
_REQUIRED_HEAD = re.compile(r"(?i)\b(?:requirements?|qualifications|must[- ]haves?|what you(?:'|’)?ll (?:need|bring|do)|"
                            r"you (?:have|bring|will)|responsibilities|about you|who you are|the role)\b")


def required_text(jd_text: str) -> str:
    """The JD without its nice-to-have / preferred parts (a heading line switches parts;
    a line that says "a plus" / "preferred" is optional on its own)."""
    keep, optional = [], False
    for line in (jd_text or "").splitlines():
        short = len(line.strip()) <= 60
        if short and _OPTIONAL_HEAD.search(line):
            optional = True
            continue
        if short and _REQUIRED_HEAD.search(line):
            optional = False
            continue
        if not optional and not _OPTIONAL_HEAD.search(line):
            keep.append(line)
    return "\n".join(keep)


def core(skill, jd_text: str, title: str = "") -> bool:
    if title and present(skill, title):
        return True
    return mentions(skill, jd_text) >= 2 and mentions(skill, required_text(jd_text)) >= 1
