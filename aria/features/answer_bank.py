"""features/answer_bank.py — the user's answers to application-form questions.

Seed (hand-edited): ~/.aria/answer_bank.json (template: answer_bank.example.json)

    {
      "answers": {"first_name": "...", "sponsorship": null, "eeo": {"gender": "..."}, ...},
      "aliases": {"sponsorship": ["do you need visa sponsorship", ...]},   # extends defaults
      "salary_rule": {"floor": 120000}
    }

Learned answers live in aria.db (table learned_answers), not in the seed file — the
seed's own "learned" block, if any, is read as a lower-priority starting set.
null / "" / "<placeholder>" = unset: the field is left for the user (needs_me).

lookup(label, options=None, job=None) → Answer | None. Order:
    1. exact learned        a question the user answered before, word for word
                            (its key's bank value still wins if it has one)
    2. exact alias          normalized label == an alias or answer key name
    3. pattern              built-in regex per key (e.g. /sponsor/ → sponsorship) —
                            real forms phrase these a hundred ways
    4. fuzzy                rapidfuzz token_sort_ratio >= 90 over aliases + learned

It NEVER invents: the value always comes from the file (or identity.json for basic
contact fields). A matched key with no value, a placeholder value ("<…>"), or a
value that fits none of the field's options returns None — the caller leaves the
field for the user.

learn(question, answer, source) records an answer the user gave (by voice or in
review) in aria.db so the same question is answered from the bank next time.

Two more rules, both set from the dashboard and stored in the seed file:
  always_ask   ["salary_rule", "why do you want to work here", ...] — keys or questions
               Aria must never fill on its own: lookup → None, and callers mark the
               field needs_me without an LLM guess (always_ask(label)).
  job_answers  {"acme | software engineer": {"<question>": "<answer>"}} — answers for
               one job only, checked before everything else for that job.

Edits (set_answer / set_learned / set_job_answer / set_always_ask / set_salary_floor)
back the seed file up once before the first write, and log every change in aria.db
(answer_changes) so undo(change_id) can put the old value back.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from rapidfuzz import fuzz, process

import aria.state.db as db
from aria.core import paths as _aria_paths

logger = logging.getLogger(__name__)

BANK_PATH = Path.home() / ".aria" / "answer_bank.json"
FUZZY_THRESHOLD = 90
DEFAULT_SALARY_FLOOR = 120000

# Canonical key → label variants (normalized form). The file's "aliases" extend these.
# Every field the bank knows has its synonyms here; the patterns below catch longer wordings.
DEFAULT_ALIASES: dict[str, list[str]] = {
    "first_name": ["first name", "given name", "legal first name", "forename", "first"],
    "last_name": ["last name", "surname", "family name", "legal last name", "last"],
    "full_name": ["full name", "name", "your name", "applicant name", "candidate name"],
    "legal_name": ["legal name", "full legal name", "legal full name", "name as it appears on id"],
    "email": ["email", "email address", "e-mail", "e-mail address", "contact email", "your email"],
    "phone": ["phone", "phone number", "mobile", "mobile phone", "mobile phone number",
              "cell phone", "cell", "telephone", "contact number", "mobile number"],
    "linkedin": ["linkedin", "linkedin profile", "linkedin url", "linkedin profile url", "linkedin profile link"],
    "github": ["github", "github url", "github profile", "github link", "github username"],
    "website": ["website", "portfolio", "personal website", "portfolio url", "website url", "personal site",
                "portfolio link", "other website"],
    "preferred_name": ["preferred name", "preferred first name", "nickname", "name you go by"],
    "location": ["location", "current location", "location city", "where are you located", "where are you based",
                 "current city"],
    "address": ["address", "current address", "home address", "mailing address", "full address"],
    "address_line1": ["address line 1", "street address", "street address line 1", "address 1", "street"],
    "city": ["city", "town", "city town"],
    "state": ["state", "state province", "province", "region", "state region"],
    "zip": ["zip", "zip code", "postal code", "zip postal code", "postcode"],
    "country": ["country", "country of residence", "country region", "location of residence"],
    "school": ["school", "university", "college", "institution", "school name", "university name",
               "college name", "school university", "college university", "university college",
               "name of school", "school attended", "educational institution"],
    "degree": ["degree", "degree type", "highest degree", "highest degree earned", "degree earned",
               "degree obtained", "highest level of education", "level of education", "education level",
               "degree level"],
    "major": ["major", "field of study", "discipline", "area of study", "concentration", "major field of study",
              "program of study", "course of study", "specialization"],
    "gpa": ["gpa", "grade point average", "cumulative gpa", "overall gpa", "gpa out of 4.0"],
    "grad_date": ["graduation date", "expected graduation date", "expected graduation", "anticipated graduation date",
                  "graduation month and year", "when do you graduate", "degree completion date"],
    "grad_month": ["graduation month", "end date month", "end month"],
    "grad_year": ["graduation year", "end date year", "end year", "year of graduation", "class year"],
    "edu_start_month": ["education start month", "start date month", "start month"],
    "edu_start_year": ["education start year", "start date year", "start year"],
    "current_company": ["current company", "current employer", "most recent employer", "employer"],
    "relocation": ["willing to relocate", "are you willing to relocate", "open to relocation", "relocation"],
    "sponsorship": ["sponsorship", "visa sponsorship", "do you require sponsorship", "need sponsorship"],
    "work_authorization": ["work authorization", "are you legally authorized to work in the united states",
                           "authorized to work", "legally authorized to work", "work eligibility"],
    "years_experience": ["years of experience", "how many years of experience do you have", "total years of experience",
                         "years of relevant experience", "years of professional experience"],
    "salary_rule": ["salary expectations", "desired salary", "expected salary", "salary requirements",
                    "compensation expectations", "desired compensation"],
    "start_date": ["start date", "earliest start date", "when can you start", "availability", "available start date"],
    "referral_source": ["how did you hear about us", "how did you hear about this opportunity",
                        "how did you hear about this position", "how did you hear about this job",
                        "how did you find out about this position", "source", "referral source",
                        "where did you hear about us", "where did you find this job"],
    "eeo.gender": ["gender", "gender identity", "sex"],
    "eeo.race": ["race", "race ethnicity", "ethnicity"],
    "eeo.hispanic_latino": ["are you hispanic latino", "hispanic latino", "hispanic or latino"],
    "eeo.veteran": ["veteran status", "protected veteran status", "veteran"],
    "eeo.disability": ["disability status", "disability"],
    "eeo.pronouns": ["pronouns", "preferred pronouns", "your pronouns"],
}

# Step 3 (synonym). Checked in order — more specific keys first (hispanic before race,
# referral source before school: "where did you learn about this position").
DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("referral_source", r"\bhow did you (?:hear|find out|learn|come across)\b|\bwhere did you (?:hear|find|learn|see)\b|"
                        r"\bsource where you (?:learned|heard|found)|\breferral source\b"),
    ("sponsorship", r"\bsponsor"),
    ("work_authorization", r"\bauthori[sz]ed to work|\bwork authori[sz]ation|\beligible to work|\bright to work"),
    ("relocation", r"\brelocat"),
    ("grad_date", r"\bgraduat(?:e|ion)\b.*\b(?:date|when|year)|\bwhen (?:did|do|will) you graduate|"
                  r"\b(?:anticipated|expected) graduation\b"),
    ("school", r"\b(?:school|university|college|institution)\b.*\b(?:attend|graduat|name|studied)|"
               r"\bcurrently attending\b|\bgraduated from\b|\bwhere did you (?:study|go to school)|"
               r"\bname of (?:your )?(?:school|college|university|institution)"),
    ("major", r"\bfield of study\b|\bmajor\b|\barea of study\b|\bprogram of study\b|^discipline$"),
    ("degree", r"\blevel of education\b|\bhighest (?:degree|education)\b|\bdegree\b"),
    ("gpa", r"\bgpa\b|\bgrade point\b"),
    ("start_date", r"\b(?:earliest|available|availability) .*\bstart\b|\bstart date\b|\bwhen can you start"),
    ("years_experience", r"\byears? of (?:professional |relevant |work |industry )?experience"),
    ("salary_rule", r"\b(?:salary|compensation|pay) (?:expectation|requirement|range you)|\bdesired (?:salary|pay|compensation)"),
    ("eeo.hispanic_latino", r"\bhispanic\b|\blatin[oax]\b"),
    ("eeo.race", r"^race\b|\brace\b.*\bethnicity|\bethnicity\b"),
    ("eeo.gender", r"^gender\b|\bgender identity\b|^sex$"),
    ("eeo.veteran", r"\bveteran"),
    ("eeo.disability", r"\bdisabilit"),
    ("eeo.transgender", r"\btransgender"),
    ("eeo.lgbtq", r"\blgbtq|\bsexual orientation"),
    ("eeo.pronouns", r"\bpronoun"),
]

# Keys whose answers are yes/no: their patterns only apply to yes/no questions, so
# "What is the address ... if you relocate ..." doesn't get "Yes".
YES_NO_KEYS = {"relocation", "sponsorship", "work_authorization", "conflict_of_interest",
               "outside_business", "hybrid_onsite_ok", "eeo.hispanic_latino", "eeo.transgender",
               "eeo.lgbtq"}
_OPEN_QUESTION = re.compile(r"^(?:what|where|which|when|how|who|why|please (?:describe|explain|list|provide)|describe|explain|list)\b")
# ...and the reverse: a value key (school, degree) never answers a yes/no question
# ("Do you have a bachelor's degree in CS?" is not asking for your degree).
_YES_NO_QUESTION = re.compile(r"^(?:do|does|are|is|have|has|will|would|can|could|did|were|was)\b")
# Answered from the job (the sheet's source) first; the bank's value is the default.
JOB_KEYS = {"referral_source"}

_PLACEHOLDER = re.compile(r"^<.*>$")
_YES = {"yes", "y", "true"}
_NO = {"no", "n", "false"}


@dataclass
class Answer:
    key: str                 # canonical key, or "learned"
    value: str
    source: str              # "bank" | "learned"
    matched: str             # the alias / question / pattern that matched
    score: float = 100.0


def normalize(label: str) -> str:
    s = (label or "").lower()
    s = s.replace("’", "'").replace("*", " ")
    s = re.sub(r"\((?:required|optional)\)|\brequired\b$", " ", s)
    s = re.sub(r"[^\w\s'@.+]", " ", s)        # punctuation (incl. hyphens) → space: self-identify = self identify
    s = re.sub(r"\s+", " ", s).strip(" .:?")
    return s


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------

_lock = threading.Lock()


def _empty() -> dict:
    return {"answers": {}, "aliases": {}, "learned": {}, "salary_rule": {"floor": DEFAULT_SALARY_FLOOR},
            "job_answers": {}, "always_ask": []}


def load(path: Optional[Path] = None) -> dict:
    """Seed file + learned answers from aria.db (db wins on the same question)."""
    path = path or BANK_PATH
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:
        logger.error("answer_bank: can't read %s: %s", path, exc)
        data = {}
    bank = _empty()
    for k in bank:
        if isinstance(data.get(k), type(bank[k])):
            bank[k] = data[k]
    bank["learned"] = {**bank["learned"], **_db_learned()}
    return bank


def _db_learned() -> dict:
    conn = None
    try:
        conn = db.get_connection()
        return {r["question"]: r["answer"] for r in conn.execute("SELECT question, answer FROM learned_answers")}
    except Exception as exc:
        logger.error("answer_bank: can't read learned answers: %s", exc)
        return {}
    finally:
        if conn is not None:
            conn.close()


def _identity_answers() -> dict:
    """Basic contact fields from identity.json — used only where the bank has no value."""
    try:
        ident = json.loads(_aria_paths.IDENTITY_JSON.read_text())
    except (OSError, ValueError):
        return {}
    name = str(ident.get("name") or "").strip()
    parts = name.split()
    out = {
        "full_name": name,
        "first_name": parts[0] if parts else "",
        "last_name": parts[-1] if len(parts) > 1 else "",
        "email": ident.get("email", ""),
        "phone": ident.get("phone", ""),
        "linkedin": ident.get("linkedin", ""),
        "github": ident.get("github", ""),
        "location": ident.get("location", ""),
    }
    return {k: str(v).strip() for k, v in out.items() if str(v or "").strip()}


# A key with no value of its own may use another key's value (same fact, other name).
_SAME_AS = {"full_name": "legal_name", "legal_name": "full_name", "preferred_name": "first_name"}


def _value(bank: dict, key: str) -> Optional[str]:
    """The stored value for a canonical key, or None if unset / placeholder. A few keys
    are derived from another stored fact when unset (never invented):
    grad_month / grad_year ← grad_date; major ← the degree's "…, <major>" / "… in <major>"."""
    val = _own_value(bank, key)
    if val is None and key in _SAME_AS:
        val = _own_value(bank, _SAME_AS[key])
    if val is None and key in ("grad_month", "grad_year"):
        ym = parse_month_year(_own_value(bank, "grad_date") or "")
        if ym:
            val = MONTH_NAMES[ym[1] - 1] if key == "grad_month" else str(ym[0])
    if val is None and key == "major":
        m = re.search(r"(?:,| in )\s*([^,]+)$", _own_value(bank, "degree") or "")
        val = m.group(1).strip() if m else None
    return val


# ---------------------------------------------------------------------------
# Dates and education options
# ---------------------------------------------------------------------------

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
_MONTHS = {m.lower()[:3]: i + 1 for i, m in enumerate(MONTH_NAMES)}
_MONTH_YEAR = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{4})\b", re.I)


def parse_month_year(text: str) -> Optional[tuple[int, int]]:
    """(year, month) from "05/18/2026", "2026-05-18", "05/2026", "May 2026", "May 18, 2026"."""
    t = (text or "").strip()
    m = re.fullmatch(r"(\d{1,2})/(?:\d{1,2}/)?(\d{4})", t)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", t)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:\d{1,2},?\s+)?(\d{4})\b", t, re.I)
    if m:
        return int(m.group(2)), _MONTHS[m.group(1).lower()[:3]]
    return None


def choose_date_option(value: str, options: Sequence[str], today: Optional[date] = None) -> Optional[str]:
    """The option a graduation date falls in: "already graduated" once it has passed,
    else a "<Month YYYY> - <Month YYYY>" range or "after / or later" option containing it."""
    ym = parse_month_year(value)
    if not ym:
        return None
    today = today or date.today()
    if ym < (today.year, today.month):
        done = [o for o in options if re.search(r"\balready graduated\b|\bhave graduated\b|^graduated\b", o, re.I)]
        return done[0] if len(done) == 1 else None
    for o in options:
        ends = [(int(y), _MONTHS[mo.lower()[:3]]) for mo, y in _MONTH_YEAR.findall(o)]
        if len(ends) == 2 and ends[0] <= ym <= ends[1]:
            return o
        if len(ends) == 1 and re.search(r"\bafter\b", o, re.I) and ym > ends[0]:
            return o
        if len(ends) == 1 and re.search(r"\bor later\b|\bonwards?\b", o, re.I) and ym >= ends[0]:
            return o
        if len(ends) == 1 and re.search(r"\bbefore\b", o, re.I) and ym < ends[0]:
            return o
    return None


_SEASONS = {"spring": (1, 5), "summer": (6, 7), "fall": (8, 12), "autumn": (8, 12), "winter": (12, 12)}


def graduation_window(question: str, grad_date: str) -> Optional[str]:
    """"Yes"/"No" for "my graduation date will be Fall 2026 or Spring 2027"-style checks,
    computed from the bank's graduation date. None when the question names no terms."""
    ym = parse_month_year(grad_date)
    terms = re.findall(r"\b(spring|summer|fall|autumn|winter)\s+(\d{4})\b", question or "", re.I)
    if not ym or not terms:
        return None
    for season, year in terms:
        lo, hi = _SEASONS[season.lower()]
        if int(year) == ym[0] and lo <= ym[1] <= hi:
            return "Yes"
    return "No"


_DEGREE_LEVELS = [   # (level, words in a value, words in an option) — most specific first
    ("mba", r"\bm\.?b\.?a\b|business administration", r"business administration|\bm\.?b\.?a\b"),
    ("phd", r"\bph\.?\s?d\b|doctor of philosophy|doctorate", r"ph\.?\s?d|doctor of philosophy|doctorate"),
    ("master", r"\bmaster|\bm\.?s\b|\bm\.?sc\b|\bm\.?eng\b|\bm\.?tech\b|\bm\.?a\b", r"master(?!.*business)"),
    ("bachelor", r"\bbachelor|\bb\.?s\b|\bb\.?sc\b|\bb\.?e\b|\bb\.?tech\b|\bb\.?a\b|undergrad", r"bachelor"),
    ("associate", r"\bassociate", r"associate"),
]


def choose_degree(value: str, options: Sequence[str]) -> Optional[str]:
    """The option at the same level as the degree in the bank ("Master of Science, CS" →
    "Master's Degree"). MBA and PhD never fall back to a generic master's/doctor option."""
    v = (value or "").lower()
    for level, in_value, in_option in _DEGREE_LEVELS:
        if re.search(in_value, v):
            hits = [o for o in options if re.search(in_option, o.lower())]
            return hits[0] if len(hits) == 1 else None
    return None


def _own_value(bank: dict, key: str) -> Optional[str]:
    node = bank["answers"]
    for part in key.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    if node is None and "." not in key:
        node = _identity_answers().get(key)
    if node is None or isinstance(node, dict):
        return None
    val = str(node).strip()
    if not val or _PLACEHOLDER.match(val):
        return None
    return val


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

_MONEY = r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?([kK])?"
_RANGE = re.compile(_MONEY + r"\s*(?:-|–|—|to)\s*" + _MONEY)


def _money(num: str, k: Optional[str]) -> int:
    value = float(num.replace(",", ""))
    return int(round(value * 1000)) if k else int(round(value))


def parse_salary_range(text: str) -> Optional[tuple[int, int]]:
    """First annual "$X – $Y" range in a job posting. Hourly-looking ranges → None."""
    for m in _RANGE.finditer(text or ""):
        low, high = _money(m.group(1), m.group(2)), _money(m.group(3), m.group(4))
        if low >= 1000 and high >= low:
            return low, high
    return None


def salary_answer(salary_range: Optional[Sequence[int]], floor: int = DEFAULT_SALARY_FLOOR) -> Optional[str]:
    """User's rule: the posting's low end, never below the floor; no range → None (review)."""
    if not salary_range:
        return None
    return str(max(int(salary_range[0]), floor))


def _salary_for(bank: dict, job: Optional[dict]) -> Optional[str]:
    if not job:
        return None
    rng = job.get("salary_range") or parse_salary_range(job.get("description", ""))
    floor = int(bank.get("salary_rule", {}).get("floor") or DEFAULT_SALARY_FLOOR)
    return salary_answer(rng, floor)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def _core(option: str) -> str:
    """Option text without a trailing dial code or parenthetical."""
    return re.sub(r"\s*(?:\+\d[\d ]*|\([^)]*\))$", "", normalize(option)).strip()


def choose_option(value: str, options: Sequence[str]) -> Optional[str]:
    """The option that means `value`, or None. Never picks an unrelated option."""
    opts = [o for o in options if str(o).strip()]
    if not opts:
        return None
    nv = normalize(value)
    for o in opts:
        if normalize(o) == nv:
            return o
    core = [o for o in opts if _core(o) == nv]           # "United States +1", "Master's (MS)"
    if len(core) == 1:
        return core[0]
    if nv in _YES | _NO:
        want = _YES if nv in _YES else _NO
        hits = [o for o in opts if normalize(o).split(" ")[0] in want]
        if len(hits) == 1:
            return hits[0]
    best = process.extractOne(nv, {o: normalize(o) for o in opts}, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= FUZZY_THRESHOLD:
        return best[2]
    starts = [o for o in opts if normalize(o).startswith(nv) or nv.startswith(normalize(o))]
    return starts[0] if len(starts) == 1 and len(nv) >= 4 else None


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _key_label(key: str) -> str:
    return normalize(key.split(".")[-1].replace("_", " "))


def _answer_keys(node: dict, prefix: str = "") -> list[str]:
    keys = []
    for k, v in node.items():
        full = f"{prefix}{k}"
        keys.extend(_answer_keys(v, full + ".") if isinstance(v, dict) else [full])
    return keys


def _aliases(bank: dict) -> dict[str, str]:
    """normalized alias → canonical key. Later layers win:
    default key names < default variants < file aliases < the file's own answer keys
    (so a label that IS a key name, e.g. "City", goes to that key, not another key's alias)."""
    out: dict[str, str] = {}
    for key in DEFAULT_ALIASES:
        out[_key_label(key)] = key
    for key, variants in DEFAULT_ALIASES.items():
        for v in variants:
            out[normalize(v)] = key
    for key, variants in bank.get("aliases", {}).items():
        for v in variants or []:
            out[normalize(v)] = key
    for key in _answer_keys(bank.get("answers", {})):
        out[_key_label(key)] = key
    return out


def _resolve(bank: dict, key: str, matched: str, score: float, options, job,
             question: str = "") -> Optional[Answer]:
    if key == "salary_rule":
        value = _salary_for(bank, job)
    elif key in JOB_KEYS:            # the sheet's source for this job, else your default answer
        value = str((job or {}).get("source") or "").strip() or _value(bank, key)
    else:
        value = _value(bank, key)
    if value is None:
        return None
    if options:
        if key == "degree":
            chosen = choose_degree(value, options)
        elif key == "grad_date" and {normalize(o) for o in options} <= _YES | _NO:
            yn = graduation_window(question, value)
            chosen = choose_option(yn, options) if yn else None
        elif key == "grad_date":
            chosen = choose_date_option(value, options) or choose_option(value, options)
        else:
            chosen = choose_option(value, options)
        if chosen is None:
            return None
        value = chosen
    return Answer(key=key, value=value, source="bank", matched=matched, score=score)


def _learned(bank: dict, question: str, matched: str, score: float, options) -> Optional[Answer]:
    value = str(bank["learned"].get(question, "")).strip()
    if not value:
        return None
    if options:
        value = choose_option(value, options)
        if value is None:
            return None
    return Answer(key="learned", value=value, source="learned", matched=matched, score=score)


def lookup(
    label: str,
    options: Optional[Sequence[str]] = None,
    job: Optional[dict] = None,
    path: Optional[Path] = None,
    llm: bool = False,
) -> Optional[Answer]:
    """Answer for a form field label from the user's bank, or None. Never raises.
    llm=True allows the last match step (a small LLM picks a bank KEY; never a value)."""
    try:
        return _lookup(load(path), label, options, job, llm)
    except Exception as exc:
        logger.error("answer_bank.lookup(%r) failed: %s", label, exc)
        return None


def lookup_key(key: str, options: Optional[Sequence[str]] = None, job: Optional[dict] = None,
               path: Optional[Path] = None, label: str = "") -> Optional[Answer]:
    """The bank's answer for a known key (e.g. a form field identified by its name, not its
    label). Never raises."""
    try:
        return _resolve(load(path), key, key, 100.0, options, job, normalize(label))
    except Exception as exc:
        logger.error("answer_bank.lookup_key(%r) failed: %s", key, exc)
        return None


@dataclass
class Match:
    key: Optional[str] = None          # canonical key
    question: Optional[str] = None     # a learned question, word for word
    score: float = 0.0
    how: str = ""                      # learned | exact | alias | synonym | llm | ""


def _exact_keys(bank: dict) -> dict[str, str]:
    keys = list(DEFAULT_ALIASES) + _answer_keys(bank.get("answers", {}))
    return {_key_label(k): k for k in keys}


def match_field(bank: dict, q: str, llm: bool = False) -> Match:
    """Which bank field a normalized label means, in order:
    learned (word for word) → exact field name → alias → synonym (pattern, then fuzzy)
    → small-LLM key pick (llm=True only). No values are read here."""
    aliases = _aliases(bank)
    learned = {normalize(k): k for k in bank["learned"]}
    if q in learned:
        return Match(aliases.get(q), learned[q], 100.0, "learned")
    exact = _exact_keys(bank)
    if q in exact:
        return Match(exact[q], None, 100.0, "exact")
    if q in aliases:
        return Match(aliases[q], None, 100.0, "alias")
    yes_no_q = bool(_YES_NO_QUESTION.match(q))
    for key, pattern in DEFAULT_PATTERNS:
        if not re.search(pattern, q):
            continue
        if key in YES_NO_KEYS and _OPEN_QUESTION.match(q):
            continue
        if key not in YES_NO_KEYS and yes_no_q:
            continue
        return Match(key, None, 100.0, "synonym")
    choices = {f"a:{a}": a for a in aliases}
    choices.update({f"l:{n}": n for n in learned})
    best = process.extractOne(q, choices, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= FUZZY_THRESHOLD:
        tag = best[2]
        if tag.startswith("a:"):
            return Match(aliases[tag[2:]], None, best[1], "synonym")
        return Match(None, learned[tag[2:]], best[1], "synonym")
    if llm:
        key = _llm_key(bank, q)
        if key:
            return Match(key, None, 0.0, "llm")
    return Match()


def _match(bank: dict, q: str) -> tuple[Optional[str], Optional[str], float, str]:
    """(canonical key, learned question, score, how) — kept for callers of the old shape."""
    m = match_field(bank, q)
    return m.key, m.question, m.score, m.how


# ---------------------------------------------------------------------------
# Step 4: a small LLM picks which bank KEY a question means (never a value)
# ---------------------------------------------------------------------------

_llm_cache: dict[str, Optional[str]] = {}
_KEY_SYSTEM = (
    "You map a job-application question to ONE field of the applicant's answer bank. "
    "Reply with JSON only: {\"key\": \"<one of the listed keys>\"} or {\"key\": \"NONE\"} when no field "
    "answers the question. Never invent a key. A yes/no question only maps to a yes/no field."
)


def _complete_checked(task: str, messages: list, validate, **kw):
    from aria.llm import llm_client
    return llm_client.complete_checked(task, messages, validate=validate, **kw)


def _llm_key(bank: dict, q: str) -> Optional[str]:
    if q in _llm_cache:
        return _llm_cache[q]
    keys = sorted(set(DEFAULT_ALIASES) | set(_answer_keys(bank.get("answers", {}))))
    listing = "\n".join(f"- {k}: {_key_label(k)}" + (" (yes/no)" if k in YES_NO_KEYS else "") for k in keys)

    def validate(text: str):
        try:
            got = json.loads(re.search(r"\{.*\}", text or "", re.S).group(0)).get("key")
        except Exception:
            return None
        if got == "NONE":
            return {"key": None}
        return {"key": got} if got in keys else None

    try:
        _resp, out = _complete_checked(
            "form_field_match", [{"role": "user", "content": f"Fields:\n{listing}\n\nQuestion: {q}"}],
            validate=validate, system=_KEY_SYSTEM, max_tokens=60)
    except Exception as exc:
        logger.warning("answer_bank: LLM field match failed for %r: %s", q, exc)
        return None
    key = (out or {}).get("key")
    _llm_cache[q] = key
    return key


def job_key(job: Optional[dict]) -> str:
    """The key a job's own answers are stored under: "company | role", normalized."""
    job = job or {}
    company, role = normalize(job.get("company", "")), normalize(job.get("role", ""))
    return f"{company} | {role}" if company or role else ""


def _always_ask(bank: dict, q: str) -> bool:
    items = {normalize(i) if " " in str(i) else str(i) for i in bank.get("always_ask") or []}
    if not items or not q:
        return False
    return q in items or (_match(bank, q)[0] or "") in items


def always_ask(label: str, path: Optional[Path] = None) -> bool:
    """True when the user asked to answer this field themselves, every time. Never raises."""
    try:
        return _always_ask(load(path), normalize(label))
    except Exception as exc:
        logger.error("answer_bank.always_ask(%r) failed: %s", label, exc)
        return False


def fillable_answers(bank: dict, job: Optional[dict] = None) -> dict:
    """Answers safe to hand a browser agent: set, not marked always-ask, plus this job's
    own answers (keyed by question) when a job is given."""
    skip = set(bank.get("always_ask") or [])
    out = {k: v for k, v in bank["answers"].items()
           if v not in (None, "") and not isinstance(v, dict) and k not in skip and k != "salary_rule"}
    if job:
        out.update({q: v for q, v in (bank.get("job_answers") or {}).get(job_key(job), {}).items()
                    if str(v).strip() and q not in skip})
    return out


def _lookup(bank: dict, label: str, options, job, llm: bool = False) -> Optional[Answer]:
    q = normalize(label)
    if not q or _always_ask(bank, q):
        return None
    own = (bank.get("job_answers") or {}).get(job_key(job), {}) if job else {}
    if q in own and str(own[q]).strip():
        value = str(own[q]).strip()
        if options:
            value = choose_option(value, options)
        if value:
            logger.info("answer_bank: %r → this job's answer via job", q)
            return Answer(key="job", value=value, source="job", matched=q)
    m = match_field(bank, q, llm=llm)
    if m.how:
        logger.info("answer_bank: %r → %s via %s", q, m.key or "learned", m.how)
    if m.key:
        found = _resolve(bank, m.key, q, m.score, options, job, q)
        if found:
            return found
    if m.question:
        return _learned(bank, m.question, m.question, m.score, options)
    return None


def matched_key(label: str, path: Optional[Path] = None, llm: bool = False) -> Optional[str]:
    """The canonical key a label maps to, even when the user hasn't answered it yet.

    Lets callers tell "a question we know but have no answer for" (leave it for
    the user) apart from "a question we've never seen". Never raises.
    """
    try:
        q = normalize(label)
        return match_field(load(path), q, llm=llm).key if q else None
    except Exception as exc:
        logger.error("answer_bank.matched_key(%r) failed: %s", label, exc)
        return None


def learn(question: str, answer: str, source: str = "user") -> bool:
    """Remember the user's answer to a question in aria.db. False if nothing was saved.

    source: "voice" | "review" | "user" — where the answer came from.
    """
    q, a = normalize(question), (answer or "").strip()
    if not q or not a:
        return False
    conn = None
    try:
        with _lock:
            conn = db.get_connection()
            conn.execute(
                "INSERT INTO learned_answers (question, answer, source, updated_at) "
                "VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(question) DO UPDATE SET answer=excluded.answer, "
                "source=excluded.source, updated_at=excluded.updated_at",
                (q, a, source),
            )
            conn.commit()
        logger.info("answer_bank: learned %r → %r (%s)", q, a, source)
        return True
    except Exception as exc:
        logger.error("answer_bank.learn(%r) failed: %s", question, exc)
        return False
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Edits (dashboard): seed file + aria.db, every change logged for undo
# ---------------------------------------------------------------------------

_MISSING = object()


def _read_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}


def backup_path(path: Optional[Path] = None) -> Optional[Path]:
    """The backup made before the first dashboard write, if there is one."""
    path = path or BANK_PATH
    found = sorted(path.parent.glob(path.stem + ".backup-*.json"))
    return found[0] if found else None


def _write_file(data: dict, path: Path) -> None:
    """Atomic write; the first write ever backs the original up next to it."""
    if path.exists() and backup_path(path) is None:
        shutil.copy2(path, path.with_name(f"{path.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _log_change(target: str, key: str, old, new, job: str = "", source: str = "dashboard") -> int:
    conn = db.get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO answer_changes (target, key, job, old, new, source) VALUES (?,?,?,?,?,?)",
            (target, key, job, None if old is _MISSING else json.dumps(old),
             None if new is _MISSING else json.dumps(new), source))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _get_nested(node: dict, key: str):
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _set_nested(node: dict, key: str, value) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    if value is _MISSING:
        node.pop(parts[-1], None)
    else:
        node[parts[-1]] = value


def _apply(target: str, key: str, value, job: str = "", path: Optional[Path] = None):
    """Set (or, with _MISSING, remove) one stored value. Returns the previous value."""
    path = path or BANK_PATH
    with _lock:
        if target == "learned":
            old = _db_learned().get(key, _MISSING)
            conn = db.get_connection()
            try:
                if value is _MISSING:
                    conn.execute("DELETE FROM learned_answers WHERE question=?", (key,))
                else:
                    conn.execute(
                        "INSERT INTO learned_answers (question, answer, source, updated_at) VALUES (?,?,?,datetime('now')) "
                        "ON CONFLICT(question) DO UPDATE SET answer=excluded.answer, source=excluded.source, "
                        "updated_at=excluded.updated_at", (key, str(value), "dashboard"))
                conn.commit()
            finally:
                conn.close()
            return old
        data = _read_file(path)
        if target == "answer":
            node = data.setdefault("answers", {})
            old = _get_nested(node, key)
            _set_nested(node, key, value)
        elif target == "job":
            jobs = data.setdefault("job_answers", {})
            old = jobs.get(job, {}).get(key, _MISSING)
            if value is _MISSING:
                jobs.get(job, {}).pop(key, None)
                if job in jobs and not jobs[job]:
                    jobs.pop(job)
            else:
                jobs.setdefault(job, {})[key] = value
        elif target == "always_ask":
            items = list(data.get("always_ask") or [])
            old = key in items
            want = value is not _MISSING and bool(value)
            data["always_ask"] = ([i for i in items if i != key] + ([key] if want else []))
        elif target == "alias":                       # key = field, job = the question wording
            aliases = data.setdefault("aliases", {})
            items = list(aliases.get(key) or [])
            old = True if job in items else _MISSING
            items = [i for i in items if i != job] + ([job] if value is not _MISSING and value else [])
            if items:
                aliases[key] = items
            else:
                aliases.pop(key, None)
        elif target == "salary_floor":
            rule = data.setdefault("salary_rule", {})
            old = rule.get("floor", _MISSING)
            if value is _MISSING:
                rule.pop("floor", None)
            else:
                rule["floor"] = int(value)
        else:
            raise ValueError(f"unknown target {target!r}")
        _write_file(data, path)
        return old


def _edit(target: str, key: str, value, job: str = "", source: str = "dashboard",
          path: Optional[Path] = None) -> int:
    old = _apply(target, key, value, job, path)
    new = value
    if target == "always_ask":
        new = value is not _MISSING and bool(value)
    logger.info("answer_bank: %s %r changed (%s)", target, key, source)   # never the value
    return _log_change(target, key, old, new, job, source)


def set_answer(key: str, value: Optional[str], source: str = "dashboard", path: Optional[Path] = None) -> int:
    """Set a canonical key's value in the seed file ("" / None → unset, left for you)."""
    if not key or not re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)?", key):
        raise ValueError(f"bad answer key {key!r}")
    v = (value or "").strip()
    return _edit("answer", key, v if v else None, source=source, path=path)


def set_learned(question: str, value: Optional[str], source: str = "dashboard") -> int:
    """Set or (empty value) remove a learned answer in aria.db."""
    q = normalize(question)
    if not q:
        raise ValueError("empty question")
    v = (value or "").strip()
    return _edit("learned", q, v if v else _MISSING, source=source)


def set_job_answer(job: str, question: str, value: Optional[str], source: str = "dashboard",
                   path: Optional[Path] = None) -> int:
    """An answer for one job only (job = job_key(...)); empty value removes it."""
    q = normalize(question)
    if not q or not job:
        raise ValueError("question and job are required")
    v = (value or "").strip()
    return _edit("job", q, v if v else _MISSING, job=job, source=source, path=path)


def add_alias(key: str, question: str, source: str = "dashboard", path: Optional[Path] = None) -> Optional[int]:
    """Remember a question's wording for a field, so it matches by alias next time.
    None when that wording already matches the field exactly or by alias."""
    q = normalize(question)
    if not q or not re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)?", key or ""):
        return None
    m = match_field(load(path), q)
    if m.key == key and m.how in ("exact", "alias"):
        return None
    return _edit("alias", key, True, job=q, source=source, path=path)


def save_answer_for_question(question: str, value: str, key: Optional[str] = None, source: str = "dashboard",
                             path: Optional[Path] = None) -> list:
    """The user answered a question from a review. A question that maps to a bank field
    (`key`, else matched) sets that field AND saves the wording as an alias; one that
    maps to nothing is learned word for word. Returns the change ids."""
    v = (value or "").strip()
    if not v:
        raise ValueError("empty answer")
    key = key or match_field(load(path), normalize(question)).key
    if key and key != "salary_rule":
        ids = [set_answer(key, v, source=source, path=path)]
        alias = add_alias(key, question, source=source, path=path)
        return ids + ([alias] if alias else [])
    return [set_learned(question, v, source=source)]


def set_always_ask(item: str, on: bool, source: str = "dashboard", path: Optional[Path] = None) -> int:
    """item: a canonical key (e.g. "salary_rule") or a question (normalized here)."""
    key = item if re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)?", item or "") else normalize(item)
    if not key:
        raise ValueError("empty item")
    return _edit("always_ask", key, bool(on), source=source, path=path)


def set_salary_floor(floor: int, source: str = "dashboard", path: Optional[Path] = None) -> int:
    floor = int(floor)
    if not 10_000 <= floor <= 10_000_000:
        raise ValueError("salary floor must be an annual amount")
    return _edit("salary_floor", "floor", floor, source=source, path=path)


def history(limit: int = 50) -> list:
    conn = db.get_connection()
    try:
        rows = conn.execute("SELECT * FROM answer_changes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def undo(change_id: int, path: Optional[Path] = None) -> int:
    """Put back the value from before change `change_id`. Returns the new change's id."""
    conn = db.get_connection()
    try:
        row = conn.execute("SELECT * FROM answer_changes WHERE id=?", (int(change_id),)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"no change #{change_id}")
    if row["undone_by"]:
        raise ValueError(f"change #{change_id} was already undone")
    old = _MISSING if row["old"] is None else json.loads(row["old"])
    if row["target"] == "always_ask":
        old = bool(old) if old is not _MISSING else False
    new_id = _edit(row["target"], row["key"], old, job=row["job"], source=f"undo #{change_id}", path=path)
    conn = db.get_connection()
    try:
        conn.execute("UPDATE answer_changes SET undone_by=? WHERE id=?", (new_id, int(change_id)))
        conn.commit()
    finally:
        conn.close()
    return new_id
