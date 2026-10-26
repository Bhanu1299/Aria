"""features/ats/answer_check.py — an LLM-written answer checked against THAT job's resume.

After the LLM writes an open-text answer, code checks it against the job's tailored
resume.json (the base resume when the job has none):
  number    every number / percentage must appear on the resume
  employer  "at <Name>", "worked for <Name>" … must be an employer on the resume (or the
            company being applied to)
  school    "<Name> University / College / Institute" must be a school on the resume
  degree    B.S. / M.S. / PhD / MBA / Bachelor's / Master's must match a resume degree
  project   "built <Name>", "my project <Name>" … must be a project or employer on the resume
  skill     a lexicon skill that's on neither the resume nor the job description
Then: violations → one strong-tier rewrite (task answer_fix) with the violations listed →
still failing → the answer without the offending sentences. Never asks the user, never
leaves the field empty (when every sentence offends, the one with the fewest problems stays).

    check(answer, resume, jd_text="", company="") → [Violation]
    strip_violations(text, resume, jd_text="", company="") → str
    enforce(answer, resume, jd_text, company, label, rewrite_fn=None) → (text, log)
        log = {"violations": [...], "action": "clean|rewritten|stripped|kept_least_bad",
               "after": [...]}

Pure code apart from the one rewrite. Never raises.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from aria.features.ats import untrusted
from aria.features.resume import model, skills

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    kind: str          # number | employer | school | degree | project | skill
    text: str
    sentence: str


_BREAK = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"(])|\n+")
_ABBREV = re.compile(r"(?:[A-Za-z]\.){1,3}|(?:Dr|Mr|Ms|Mrs|Jr|Sr|St|Inc|Co|Corp|Ltd|vs|etc|e\.g|i\.e|approx|No)\.",
                     re.I)
_NAME = r"([A-Z][\w&.'-]*(?:\s+(?:of\s+|&\s+)?[A-Z][\w&.'-]*){0,3})"
_EMPLOYER_CUE = re.compile(r"\b(?P<cue>(?i:at|for|joined|with the team at|interned at|worked at|employed by|company))\s+"
                           + _NAME)
_EMPLOYER_VERB = re.compile(r"\b(?:worked|interned|served|employed|engineer|developer|intern)\b", re.I)
_PROJECT_CUE = re.compile(r"\b(?:built|created|launched|shipped|developed|project(?:\s+called|\s+named)?|called|named)\s+"
                          + _NAME)
_SCHOOL = re.compile(r"((?:[A-Z][\w&.'-]*\s+){0,4}(?:University|College|Institute|School of [A-Z]\w+)"
                     r"(?:\s+of\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2})?)")
_DEGREE = re.compile(r"\b(B\.?\s?S\.?|M\.?\s?S\.?|B\.?\s?A\.?|M\.?\s?A\.?|Ph\.?\s?D\.?|MBA|B\.?\s?Tech|M\.?\s?Tech|"
                     r"[Bb]achelor(?:'s)?|[Mm]aster(?:'s)?|[Dd]octorate)(?![A-Za-z])")
# Capitalised words that are never names of employers or projects.
_COMMON = {"i", "my", "the", "a", "an", "this", "that", "our", "your", "their", "it", "we", "in", "on", "at",
           "monday", "tuesday", "wednesday", "thursday", "friday", "january", "february", "march", "april",
           "may", "june", "july", "august", "september", "october", "november", "december", "english",
           "american", "us", "u.s", "usa", "api", "apis", "ai", "ml", "llm", "llms", "ci", "data", "scale",
           "speed", "work", "home", "first", "one", "two", "three", "every", "each", "when", "while",
           "after", "before", "then", "there", "here", "what", "why", "how", "both", "these", "those",
           "production", "team", "teams", "engineering", "software", "backend", "frontend", "full", "stack",
           "people", "customers", "users", "school", "university", "college", "graduate", "undergrad"}


def _norm_num(n: str) -> str:
    n = n.lower().replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
    n = re.sub(r"\s*percent$", "%", n)
    return n.rstrip(".")


def _numbers(text: str) -> list:
    text = re.sub(r"(\d)\s+percent\b", r"\1%", text or "", flags=re.I)
    return [_norm_num(n) for n in model.numbers(text)]


def _degree_key(s: str) -> str:
    s = re.sub(r"[^a-z]", "", s.lower())
    return {"bs": "b", "ba": "b", "btech": "b", "bachelor": "b", "bachelors": "b", "ms": "m", "ma": "m",
            "mtech": "m", "master": "m", "masters": "m", "mba": "mba", "phd": "d", "doctorate": "d"}.get(s, s)


def _known(resume: dict, jd_text: str, company: str) -> dict:
    r = model.normalize(resume)
    text = model.plain_text(r)
    names = [e["company"] for e in r["experience"]] + [p["title"] for p in r["projects"]] + \
        [e["school"] for e in r["education"]] + ([company] if company else [])
    return {"text": text, "low": text.lower(), "jd_low": (jd_text or "").lower(),
            "numbers": set(_numbers(text)),
            "names": [n.lower() for n in names if n],
            "schools": [e["school"].lower() for e in r["education"] if e["school"]],
            "degrees": {_degree_key(m.group(1)) for e in r["education"] for m in _DEGREE.finditer(e["degree"])}}


def _name_known(name: str, k: dict) -> bool:
    n = name.lower().strip(" .,'")
    if not n or n.split()[0] in _COMMON or skills.canonical(name):
        return True
    if any(n in x or x in n for x in k["names"]):
        return True
    return n in k["low"] or n in k["jd_low"]


def check(answer: str, resume: dict, jd_text: str = "", company: str = "") -> list:
    try:
        k = _known(resume, jd_text, company)
        out: list = []
        for s in _sentences(answer):
            s = s.strip()
            if not s:
                continue
            for n in _numbers(s):
                if n not in k["numbers"]:
                    out.append(Violation("number", n, s))
            for sm in _SCHOOL.finditer(s):
                school = sm.group(1).strip()
                if not any(school.lower() in x or x in school.lower() for x in k["schools"]):
                    out.append(Violation("school", school, s))
            for dm in _DEGREE.finditer(s):
                if _degree_key(dm.group(1)) not in k["degrees"]:
                    out.append(Violation("degree", dm.group(1), s))
            school_spans = [sm.span() for sm in _SCHOOL.finditer(s)]
            for cue, kind in ((_EMPLOYER_CUE, "employer"), (_PROJECT_CUE, "project")):
                for nm in cue.finditer(s):
                    if any(a <= nm.start(nm.lastindex) < b for a, b in school_spans):
                        continue
                    opens = kind == "employer" and nm.start() == 0 and nm.group("cue").lower() == "at"
                    if kind == "employer" and not opens and not _EMPLOYER_VERB.search(s):
                        continue
                    name = nm.group(nm.lastindex).rstrip(".,'’")
                    if not _name_known(name, k):
                        out.append(Violation(kind, name, s))
            for sk in skills.extract(s):
                if not skills.present(sk, k["text"]) and not skills.present(sk, jd_text or ""):
                    out.append(Violation("skill", sk.text, s))
        unique, seen = [], set()
        for v in out:
            if (v.kind, v.text, v.sentence) not in seen:
                seen.add((v.kind, v.text, v.sentence))
                unique.append(v)
        return unique
    except Exception as exc:
        logger.warning("answer_check.check failed: %s", exc)
        return []


def _sentences(text: str) -> list:
    """Sentences with their trailing whitespace (so "".join() rebuilds the text); an
    abbreviation ("M.S.", "Dr.", "e.g.") never ends one."""
    text = text or ""
    out, start = [], 0
    for m in _BREAK.finditer(text):
        before = text[start:m.start()].rstrip("\"')]")
        last = before.split()[-1] if before.split() else ""
        if "\n" not in m.group(0) and _ABBREV.fullmatch(last):
            continue
        out.append(text[start:m.end()])
        start = m.end()
    if text[start:].strip():
        out.append(text[start:])
    return [s for s in out if s.strip()]


def strip_violations(text: str, resume: dict, jd_text: str = "", company: str = "") -> str:
    bad = {v.sentence for v in check(text, resume, jd_text, company)}
    kept = [s for s in _sentences(text) if s.strip() not in bad]
    return re.sub(r"[ \t]+", " ", "".join(kept)).strip()


_FIX_SYSTEM = (
    "You fix a job-application answer so every fact in it is on the candidate's resume. You get the "
    "answer, the problems found, and the resume. Rewrite the answer: remove or replace each problem "
    "with something the resume does say; keep the rest, the voice and the length. Plain text only, "
    "no preamble. " + untrusted.RULE
)


def _llm_fix(answer: str, violations: list, resume: dict, label: str, max_chars: int = 0) -> str:
    from aria.llm import llm_client
    probs = "\n".join(f"- {v.kind} “{v.text}” is not on the resume" for v in violations)
    question, _flags = untrusted.guard(label, "Question from the form")
    user = (f"{question}\n\nAnswer:\n{answer}\n\nProblems:\n{probs}\n\n"
            f"Resume:\n{model.plain_text(resume)}" + (f"\n\nHard limit: {max_chars} characters." if max_chars else ""))
    try:
        resp = llm_client.complete(messages=[{"role": "user", "content": user}], task="answer_fix",
                                   system=_FIX_SYSTEM, max_tokens=600)
    except Exception as exc:
        logger.warning("answer_check: rewrite failed: %s", exc)
        return ""
    if getattr(resp, "stop_reason", "") == "error":
        return ""
    return (resp.text or "").strip().strip('"')


def enforce(answer: str, resume: Optional[dict], jd_text: str = "", company: str = "", label: str = "",
            rewrite_fn: Optional[Callable] = None, max_chars: int = 0) -> tuple:
    """(final answer, log). rewrite_fn(answer, violations, resume, label, max_chars) → str
    defaults to one strong-tier LLM rewrite."""
    log: dict = {"violations": [], "action": "clean", "after": []}
    if not answer or not resume:
        return answer, log
    found = check(answer, resume, jd_text, company)
    log["violations"] = [asdict(v) for v in found]
    if not found:
        return answer, log
    fix = rewrite_fn or _llm_fix
    try:
        fixed = fix(answer, found, resume, label, max_chars) or ""
    except Exception as exc:
        logger.warning("answer_check: rewrite_fn raised: %s", exc)
        fixed = ""
    if max_chars and len(fixed) > max_chars:
        from aria.features.ats.answers import fit_limit
        fixed = fit_limit(fixed, max_chars)
    if fixed:
        again = check(fixed, resume, jd_text, company)
        if not again:
            log["action"] = "rewritten"
            return fixed, log
        log["after"] = [asdict(v) for v in again]
    source = fixed if fixed and len(check(fixed, resume, jd_text, company)) <= len(found) else answer
    stripped = strip_violations(source, resume, jd_text, company)
    if stripped:
        log["action"] = "stripped"
        return stripped, log
    sents = _sentences(source)
    least = min(sents, key=lambda s: len(check(s, resume, jd_text, company))) if sents else source
    log["action"] = "kept_least_bad"
    return least.strip(), log


def describe(log: dict) -> str:
    """One review note for the field: "" when the answer was clean."""
    if not log or not log.get("violations"):
        return ""
    what = ", ".join(sorted({f"{v['kind']} {v['text']}" for v in log["violations"]}))[:160]
    return {"rewritten": f"answer check: rewritten (not on your resume: {what})",
            "stripped": f"answer check: removed sentences (not on your resume: {what})",
            "kept_least_bad": f"answer check: couldn't fully fix ({what}) — please read it"}.get(log["action"], "")
