"""features/resume/tailor.py — the base resume tailored to one job description. Code first.

The user's rules (docs/job-prep-design.md §5, not negotiable):
  * EVERY skill / tool / technology / methodology the JD names goes into this job's resume:
    merged into the matching skill line (a new line when none fits), and worked into bullets
    in the JD's wording where it fits.
  * Headline = the job's title framing. Projects and bullets reordered by relevance.
  * Employers, titles, dates, degrees, certifications and existing numbers are NEVER changed;
    no new numbers. Non-skill requirements (years, degrees, clearance, citizenship) are not
    added — they aren't in the skills lexicon, so they can't be.
  * One page always: drop least-relevant bullets, then the least-relevant project; never
    shrink fonts or margins.

Steps of tailor(base, jd_text, jd_title):
  1. jd skills    skills.extract (lexicon) + one small-tier LLM pass for terms the lexicon
                  misses (task resume_skills_extract), each validated to appear in the JD
  2. skill lines  merge every JD skill into the matching line
  3. headline     the JD title + the two JD skills the base already shows most
  4. reorder      bullets within each entry, and projects, by JD-skill relevance
                  (employers stay in date order)
  5. rewrite      one strong-tier LLM call (task resume_tailor) returning {bullet id: text},
                  each rewrite validated in code: same numbers, same names, ≤ 2 lines
  6. weave        skills still in no bullet are appended in code to the bullet that fits
                  them best (" using Kafka and Airflow"), while it stays ≤ 2 lines
  7. one page     drop least-relevant bullets, then projects, until it fits

    tailor(base, jd_text, jd_title, use_llm=True) → Tailored
    verify(base, tailored_resume) → [problem, ...]       [] = no fact changed

Never raises: an LLM failure skips that step (the code steps still run).
"""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from aria.features.ats import untrusted
from aria.features.resume import model, render, skills
from aria.features.resume.skills import Skill

logger = logging.getLogger(__name__)

MAX_WOVEN_PER_BULLET = 3
MAX_LLM_EXTRAS = 10

# Which existing skill-line label a lexicon category belongs on, by label words.
_LINE_WORDS = {
    "Languages": ("language",),
    "Frameworks": ("framework", "backend", "web", "frontend", "front-end", "full", "librar"),
    "AI/ML": ("ml", "ai", "machine", "learning", "llm", "data science"),
    "Data": ("data", "analytics", "pipeline"),
    "Databases": ("database", "data", "storage", "backend"),
    "Cloud & DevOps": ("cloud", "devops", "infra", "platform", "ops"),
    "Practices": ("practice", "method", "concept", "engineering"),
    "Tools": ("tool",),
}
_NEW_LINE_LABEL = {"Practices": "Engineering Practices", "Frameworks": "Frameworks & APIs"}
_NOT_A_SKILL = re.compile(
    r"(?i)\b(?:years?|degree|bachelor|master|ph\.?d|b\.?s\.?|m\.?s\.?|clearance|citizen|visa|sponsorship|"
    r"certif|license|communication|team ?player|self-starter|passion|ownership|mindset|english|travel)\b")


@dataclass
class Tailored:
    resume: dict
    jd_skills: list = field(default_factory=list)           # [Skill]
    added_skills: list = field(default_factory=list)        # names added to skill lines
    changes: list = field(default_factory=list)             # [{"id", "old", "new", "how"}]
    dropped: list = field(default_factory=list)             # ["bullet e1.3: …", "project Shelfie"]
    headline_before: str = ""
    llm_used: bool = False
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. JD skills
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You list hard skills from a job description: programming languages, frameworks, libraries, "
    "databases, cloud services, tools, platforms and named engineering methodologies. Never list years "
    "of experience, degrees, certifications, clearances, soft skills or job duties. Copy each term exactly "
    "as the posting writes it. Reply with a JSON array of strings only. " + untrusted.RULE
)


def llm_extra_skills(jd_text: str, known: list) -> list:
    """Hard skills the lexicon missed (small tier, task resume_skills_extract). Every term must
    appear in the JD as written. Returns [Skill] with category ""."""
    from aria.llm import llm_client
    block, _flags = untrusted.guard(jd_text[:6000], "Job description")
    user = (f"Already found (do not repeat): {', '.join(known) or '(none)'}\n\n{block}\n\n"
            "List the other hard skills it names.")
    jd_low = jd_text.lower()

    def validate(raw: str):
        m = re.search(r"\[.*\]", raw or "", re.S)
        if not m:
            return None
        got = json.loads(m.group(0))
        if not isinstance(got, list):
            return None
        out = []
        for term in got:
            t = " ".join(str(term).split()).strip(" .,;:")
            if not t or len(t) > 40 or len(t.split()) > 4 or t.lower() not in jd_low or _NOT_A_SKILL.search(t):
                continue
            if skills.canonical(t) or any(t.lower() == k.lower() for k in known) or t in out:
                continue
            out.append(t)
        return out[:MAX_LLM_EXTRAS]

    try:
        _resp, extra = llm_client.complete_checked("resume_skills_extract", [{"role": "user", "content": user}],
                                                   validate=validate, system=_EXTRACT_SYSTEM, max_tokens=300)
    except Exception as exc:
        logger.warning("tailor: skills extract failed: %s", exc)
        return []
    return [Skill(t, "", t) for t in (extra or [])]


def jd_skills(jd_text: str, use_llm: bool = True) -> list:
    found = skills.extract(jd_text)
    if use_llm:
        found += llm_extra_skills(jd_text, [s.name for s in found])
    return found


# ---------------------------------------------------------------------------
# 2. skill lines
# ---------------------------------------------------------------------------

def _line_for(lines: list, skill: Skill) -> Optional[int]:
    """Index of the skill line `skill` belongs on, or None (→ a new line)."""
    if not skill.category:                          # an LLM-found term: the Tools line, if any
        for i, ln in enumerate(lines):
            if any(w in ln["label"].lower() for w in _LINE_WORDS["Tools"]):
                return i
        return None
    for i, ln in enumerate(lines):
        label = ln["label"].lower()
        if any(re.search(r"(?<![a-z])" + re.escape(w), label) for w in _LINE_WORDS.get(skill.category, ())):
            return i
    best, best_n = None, 1
    for i, ln in enumerate(lines):
        n = sum(1 for it in ln["items"] if skills.category_of(it) == skill.category)
        if n > best_n:
            best, best_n = i, n
    return best


def merge_skills(resume: dict, wanted: list) -> list:
    """Adds every wanted skill missing from the skills section, in place. Returns names added."""
    lines = resume["skills"]
    section = "\n".join(", ".join(ln["items"]) for ln in lines)
    added = []
    for s in wanted:
        if skills.present(s, section):
            continue
        i = _line_for(lines, s)
        if i is None:
            label = _NEW_LINE_LABEL.get(s.category, s.category or "Tools")
            existing = next((k for k, ln in enumerate(lines) if ln["label"] == label), None)
            if existing is None:
                lines.append({"label": label, "items": []})
                existing = len(lines) - 1
            i = existing
        lines[i]["items"].append(s.label)
        section += "\n" + s.label
        added.append(s.label)
    return added


# ---------------------------------------------------------------------------
# 3. headline
# ---------------------------------------------------------------------------

def clean_title(title: str) -> str:
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", title or "")
    t = re.split(r"\s+[-–—|]\s+(?=(?:remote|hybrid|onsite|on-site|[A-Z][a-z]+,?\s+[A-Z]{2}\b|new york|san francisco|"
                 r"us\b|usa\b))", t, flags=re.I)[0]
    t = re.sub(r",\s*(?:remote|hybrid|us|usa)\s*$", "", t, flags=re.I)
    return " ".join(t.split()).strip(" ,-–—|")


def headline(base: dict, title: str, wanted: list) -> str:
    t = clean_title(title)
    if not t:
        return base["headline"]
    text = model.plain_text(base)
    have = [s.text for s in wanted if s.category and skills.present(s, text)]
    parts = [t] + have[:2]
    out = " | ".join(parts)
    while render.width(out, render.BOLD) > render.RIGHT - render.LEFT and len(parts) > 1:
        parts.pop()
        out = " | ".join(parts)
    return out


# ---------------------------------------------------------------------------
# 4. relevance + reorder
# ---------------------------------------------------------------------------

def relevance(text: str, wanted: list) -> int:
    return sum(1 for s in wanted if skills.present(s, text))


def reorder(resume: dict, wanted: list) -> None:
    for key in ("experience", "projects"):
        for e in resume[key]:
            order = sorted(range(len(e["bullets"])), key=lambda j: (-relevance(e["bullets"][j], wanted), j))
            e["bullets"] = [e["bullets"][j] for j in order]
            e["_ids"] = [e["_ids"][j] for j in order]
    resume["projects"].sort(key=lambda p: -relevance(" ".join([p["title"]] + p["bullets"]), wanted))


# ---------------------------------------------------------------------------
# 5. LLM rewrite, validated
# ---------------------------------------------------------------------------

_TAILOR_SYSTEM = (
    "You tailor resume bullets to a job description. You get bullets as JSON {id: text} and a list of "
    "skills the job names that no bullet mentions yet. Rewrite only bullets where one or more of those "
    "skills honestly fits the work described, working the skill in using the job's exact wording. Rules: "
    "keep every number, percentage, name and fact exactly as written; add no new numbers, employers, "
    "products or claims; keep the bullet's meaning; start with a strong past-tense verb; at most 220 "
    "characters per bullet. Reply with a JSON object {id: new text} containing only the bullets you "
    "changed. No prose, no markdown. " + untrusted.RULE
)


def bullet_ok(old: str, new: str, protected: list) -> Optional[str]:
    """None when `new` may replace `old`, else why not."""
    if not new or not new.strip():
        return "empty"
    if sorted(model.numbers(old)) != sorted(model.numbers(new)):
        return "numbers changed"
    for name in protected:
        if name and name.lower() in old.lower() and name.lower() not in new.lower():
            return f"dropped {name}"
    if render.bullet_lines(new) > 2:
        return "over 2 lines"
    return None


def llm_rewrite(resume: dict, missing: list, jd_text: str, protected: list) -> tuple:
    """({id: new text} accepted, {id: why rejected}). One strong-tier call, task resume_tailor."""
    from aria.llm import llm_client
    current = model.bullets(resume)
    if not missing or not current:
        return {}, {}
    block, _flags = untrusted.guard(jd_text[:6000], "Job description")
    user = (f"Skills to work in (job's wording): {json.dumps([s.text for s in missing])}\n\n"
            f"Bullets:\n{json.dumps(current, indent=1, ensure_ascii=False)}\n\n{block}")

    def validate(raw: str):
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            return None
        got = json.loads(m.group(0))
        return got if isinstance(got, dict) else None

    try:
        _resp, got = llm_client.complete_checked("resume_tailor", [{"role": "user", "content": user}],
                                                 validate=validate, system=_TAILOR_SYSTEM, max_tokens=2500)
    except Exception as exc:
        logger.warning("tailor: rewrite failed: %s", exc)
        return {}, {}
    ok, bad = {}, {}
    for bid, text in (got or {}).items():
        if bid not in current:
            continue
        new = " ".join(str(text).split())
        why = bullet_ok(current[bid], new, protected)
        if why:
            bad[bid] = why
        elif new != current[bid]:
            ok[bid] = new
    return ok, bad


def _apply(resume: dict, new: dict) -> None:
    for key in ("experience", "projects"):
        for e in resume[key]:
            e["bullets"] = [new.get(bid, b) for bid, b in zip(e["_ids"], e["bullets"])]


# ---------------------------------------------------------------------------
# 6. code weave
# ---------------------------------------------------------------------------

def _join(names: list) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _woven(original: str, names: list, practice: bool) -> str:
    base = original.rstrip(" .;")
    if practice:
        tail = "" if names[-1].lower().endswith(("practices", "principles")) else " practices"
        return f"{base}, applying {_join(names)}{tail}"
    word = "with" if re.search(r"\busing\b", base) else "using"
    return f"{base} {word} {_join(names)}"


def weave(resume: dict, missing: list, original: dict) -> dict:
    """Appends each skill still in no bullet to the bullet it fits best, keeping ≤ 2 lines.
    original: {id: text before weaving}. Returns {id: [names woven]}."""
    plan: dict = {}                              # id → (names, practice-names)
    texts = {bid: t for bid, t in model.bullets(resume).items()}
    order = list(texts)                          # experience first, most recent first, then projects

    def candidate(bid: str, s: Skill) -> tuple:
        t = texts[bid]
        same = sum(1 for it in skills.extract(t) if it.category == s.category) if s.category else 0
        recency = -order.index(bid) / max(1, len(order))
        return (same, bid.startswith("e"), recency)

    for s in missing:
        practice = s.category == "Practices"
        for bid in sorted(texts, key=lambda b: candidate(b, s), reverse=True):
            names, prac = plan.get(bid, ([], []))
            if len(names) + len(prac) >= MAX_WOVEN_PER_BULLET:
                continue
            names2, prac2 = (names, prac + [s.text]) if practice else (names + [s.text], prac)
            text = original[bid]
            if names2:
                text = _woven(text, names2, False)
            if prac2:
                text = _woven(text, prac2, True)
            if render.bullet_lines(text) <= 2:
                plan[bid] = (names2, prac2)
                texts[bid] = text
                break
    _apply(resume, {bid: texts[bid] for bid in plan})
    return {bid: n + p for bid, (n, p) in plan.items()}


# ---------------------------------------------------------------------------
# 7. one page
# ---------------------------------------------------------------------------

def fit_one_page(resume: dict, wanted: list, dropped: list) -> bool:
    """Drops least-relevant bullets (each entry keeps one), then the least-relevant project
    (one stays), until the resume fits one page. Returns whether it fits."""
    bullets_txt = lambda: model.bullet_text(resume)   # noqa: E731
    while not render.fits(resume):
        best = None
        for key in ("projects", "experience"):
            for e in resume[key]:
                if len(e["bullets"]) <= 1:
                    continue
                for j, b in enumerate(e["bullets"]):
                    # a bullet that carries a JD skill no other bullet has is the last to go
                    others = bullets_txt().replace(b, "")
                    sole = sum(1 for s in wanted if skills.present(s, b) and not skills.present(s, others))
                    rank = (sole, relevance(b, wanted), key == "experience", -j)
                    if best is None or rank < best[0]:
                        best = (rank, e, j, key)
        if best is not None:
            _rank, e, j, key = best
            dropped.append(f"bullet {e['_ids'][j]}: {e['bullets'][j]}")
            del e["bullets"][j]
            del e["_ids"][j]
            continue
        if len(resume["projects"]) > 1:
            worst = min(range(len(resume["projects"])),
                        key=lambda i: relevance(" ".join(resume["projects"][i]["bullets"]), wanted))
            dropped.append(f"project {resume['projects'][worst]['title']}")
            del resume["projects"][worst]
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------

def _protected(base: dict) -> list:
    out = [e["company"] for e in base["experience"]] + [p["title"] for p in base["projects"]]
    out += [e["school"] for e in base["education"]]
    return [x for x in out if x]


def tailor(base: dict, jd_text: str, jd_title: str = "", use_llm: bool = True) -> Tailored:
    resume = model.with_ids(copy.deepcopy(base))
    out = Tailored(resume=resume, headline_before=resume["headline"])
    try:
        wanted = jd_skills(jd_text, use_llm=use_llm)
        out.jd_skills = wanted
        out.added_skills = merge_skills(resume, wanted)
        resume["headline"] = headline(resume, jd_title, wanted)
        reorder(resume, wanted)
        original = model.bullets(resume)
        missing = [s for s in wanted if not skills.present(s, model.bullet_text(resume))]
        if use_llm and missing:
            new, rejected = llm_rewrite(resume, missing, jd_text, _protected(base))
            if new:
                out.llm_used = True
                _apply(resume, new)
                out.changes += [{"id": bid, "old": original[bid], "new": t, "how": "llm"} for bid, t in new.items()]
            out.notes += [f"rewrite of {bid} rejected: {why}" for bid, why in rejected.items()]
        before_weave = model.bullets(resume)
        missing = [s for s in wanted if not skills.present(s, model.bullet_text(resume))]
        woven = weave(resume, missing, before_weave)
        reorder(resume, wanted)
        after = model.bullets(resume)
        for bid in woven:
            prior = next((c for c in out.changes if c["id"] == bid), None)
            if prior:
                prior["new"], prior["how"] = after[bid], "llm+code"
            else:
                out.changes.append({"id": bid, "old": original[bid], "new": after[bid], "how": "code"})
        if not fit_one_page(resume, wanted, out.dropped):
            out.notes.append("still over one page after dropping every optional bullet and project")
    except Exception as exc:                                # never raises: whatever was done stays
        logger.error("tailor failed: %s", exc)
        out.notes.append(f"tailoring stopped early: {exc}")
    return out


def verify(base: dict, tailored: dict) -> list:
    """Problems where tailoring changed a fact (employers, roles, dates, schools, degrees,
    certifications, name, contact) or a bullet's numbers. [] = clean."""
    base, new = model.normalize(base), model.normalize(tailored)
    problems = []
    fb, fn = model.facts(base), model.facts(new)
    for k in ("experience", "education", "certifications", "name", "contact"):
        if fb[k] != fn[k]:
            problems.append(f"{k} changed")
    base_projects = set(fb["projects"])
    problems += [f"project changed: {p[0]}" for p in fn["projects"] if p not in base_projects]
    base_bullets = model.bullets(base)
    for bid, text in model.bullets(tailored).items():
        old = base_bullets.get(bid)
        if old is None:
            problems.append(f"unknown bullet {bid}")
        elif sorted(model.numbers(old)) != sorted(model.numbers(text)):
            problems.append(f"numbers changed in {bid}")
    return problems
