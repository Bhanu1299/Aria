"""features/answer_bank.py — the user's answers to application-form questions.

File: ~/.aria/answer_bank.json (outside the repo; template: answer_bank.example.json)

    {
      "answers": {"first_name": "...", "sponsorship": "...", "eeo": {"gender": "..."}, ...},
      "aliases": {"sponsorship": ["do you need visa sponsorship", ...]},   # extends defaults
      "learned": {"<normalized question>": "<answer>"},
      "salary_rule": {"floor": 120000}
    }

lookup(label, options=None, job=None) → Answer | None. Order:
    1. exact alias          normalized label == an alias of a canonical key
    2. exact learned        a question the user answered before, word for word
    3. pattern              built-in regex per key (e.g. /sponsor/ → sponsorship) —
                            real forms phrase these a hundred ways
    4. fuzzy                rapidfuzz token_sort_ratio >= 90 over aliases + learned

It NEVER invents: the value always comes from the file (or identity.json for basic
contact fields). A matched key with no value, a placeholder value ("<…>"), or a
value that fits none of the field's options returns None — the caller leaves the
field for the user.

learn(question, answer) records an answer the user gave (by voice or in review)
so the same question is answered from the bank next time.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from rapidfuzz import fuzz, process

from aria.core import paths as _aria_paths

logger = logging.getLogger(__name__)

BANK_PATH = Path.home() / ".aria" / "answer_bank.json"
FUZZY_THRESHOLD = 90
DEFAULT_SALARY_FLOOR = 120000

# Canonical key → label variants (normalized form). The file's "aliases" extend these.
DEFAULT_ALIASES: dict[str, list[str]] = {
    "first_name": ["first name", "given name", "legal first name"],
    "last_name": ["last name", "surname", "family name", "legal last name"],
    "full_name": ["full name", "name", "your name"],
    "legal_name": ["legal name", "full legal name", "legal full name"],
    "email": ["email", "email address", "e-mail", "e-mail address"],
    "phone": ["phone", "phone number", "mobile", "mobile phone", "mobile phone number",
              "cell phone", "telephone"],
    "linkedin": ["linkedin", "linkedin profile", "linkedin url", "linkedin profile url"],
    "github": ["github", "github url", "github profile"],
    "website": ["website", "portfolio", "personal website", "portfolio url", "website url"],
    "location": ["location", "city", "current location", "location city"],
    "address": ["address", "street address", "current address", "home address", "mailing address"],
    "grad_date": ["graduation date", "expected graduation date", "graduation year"],
    "relocation": ["willing to relocate", "are you willing to relocate"],
    "sponsorship": ["sponsorship", "visa sponsorship", "do you require sponsorship"],
    "work_authorization": ["work authorization", "are you legally authorized to work in the united states"],
    "years_experience": ["years of experience", "how many years of experience do you have"],
    "salary_rule": ["salary expectations", "desired salary", "expected salary", "salary requirements",
                    "compensation expectations", "desired compensation"],
    "start_date": ["start date", "earliest start date", "when can you start"],
    "eeo.gender": ["gender"],
    "eeo.race": ["race", "race ethnicity", "ethnicity"],
    "eeo.hispanic_latino": ["are you hispanic latino", "hispanic latino"],
    "eeo.veteran": ["veteran status", "protected veteran status"],
    "eeo.disability": ["disability status"],
}

# Step 3. Checked in order — more specific keys first (hispanic before race).
DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("sponsorship", r"\bsponsor"),
    ("work_authorization", r"\bauthori[sz]ed to work|\bwork authori[sz]ation|\beligible to work|\bright to work"),
    ("relocation", r"\brelocat"),
    ("grad_date", r"\bgraduat(?:e|ion)\b.*\b(?:date|when|year)|\bwhen did you graduate"),
    ("start_date", r"\b(?:earliest|available|availability) .*\bstart\b|\bstart date\b|\bwhen can you start"),
    ("years_experience", r"\byears? of (?:professional |relevant |work |industry )?experience"),
    ("salary_rule", r"\b(?:salary|compensation|pay) (?:expectation|requirement|range you)|\bdesired (?:salary|pay|compensation)"),
    ("eeo.hispanic_latino", r"\bhispanic\b|\blatin[oax]\b"),
    ("eeo.race", r"^race\b|\brace\b.*\bethnicity|\bethnicity\b"),
    ("eeo.gender", r"^gender\b|\bgender identity\b|^sex$"),
    ("eeo.veteran", r"\bveteran"),
    ("eeo.disability", r"\bdisabilit"),
]

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
    return {"answers": {}, "aliases": {}, "learned": {}, "salary_rule": {"floor": DEFAULT_SALARY_FLOOR}}


def load(path: Path = BANK_PATH) -> dict:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:
        logger.error("answer_bank: can't read %s: %s", path, exc)
        data = {}
    bank = _empty()
    for k in bank:
        if isinstance(data.get(k), dict):
            bank[k] = data[k]
    return bank


def _save(bank: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(bank, indent=2, ensure_ascii=False))
    tmp.replace(path)


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


def _value(bank: dict, key: str) -> Optional[str]:
    """The stored value for a canonical key, or None if unset / placeholder."""
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
    """User's rule: low end >= floor → floor; otherwise the posting's range; no range → None (review)."""
    if not salary_range:
        return None
    low, high = int(salary_range[0]), int(salary_range[1])
    if low >= floor:
        return str(floor)
    return f"{low}-{high}"


def _salary_for(bank: dict, job: Optional[dict]) -> Optional[str]:
    if not job:
        return None
    rng = job.get("salary_range") or parse_salary_range(job.get("description", ""))
    floor = int(bank.get("salary_rule", {}).get("floor") or DEFAULT_SALARY_FLOOR)
    return salary_answer(rng, floor)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def choose_option(value: str, options: Sequence[str]) -> Optional[str]:
    """The option that means `value`, or None. Never picks an unrelated option."""
    opts = [o for o in options if str(o).strip()]
    if not opts:
        return None
    nv = normalize(value)
    for o in opts:
        if normalize(o) == nv:
            return o
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

def _aliases(bank: dict) -> dict[str, str]:
    """normalized alias → canonical key (file aliases override defaults)."""
    out: dict[str, str] = {}
    for key in DEFAULT_ALIASES:
        out[normalize(key.split(".")[-1].replace("_", " "))] = key
    for key, variants in DEFAULT_ALIASES.items():
        for v in variants:
            out[normalize(v)] = key
    for key, variants in bank.get("aliases", {}).items():
        for v in variants or []:
            out[normalize(v)] = key
    return out


def _resolve(bank: dict, key: str, matched: str, score: float, options, job) -> Optional[Answer]:
    value = _salary_for(bank, job) if key == "salary_rule" else _value(bank, key)
    if value is None:
        return None
    if options:
        value = choose_option(value, options)
        if value is None:
            return None
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
    path: Path = BANK_PATH,
) -> Optional[Answer]:
    """Answer for a form field label from the user's bank, or None. Never raises."""
    try:
        return _lookup(load(path), label, options, job)
    except Exception as exc:
        logger.error("answer_bank.lookup(%r) failed: %s", label, exc)
        return None


def _lookup(bank: dict, label: str, options, job) -> Optional[Answer]:
    q = normalize(label)
    if not q:
        return None
    aliases = _aliases(bank)
    learned = {normalize(k): k for k in bank["learned"]}

    # 1. exact alias
    if q in aliases:
        return _resolve(bank, aliases[q], q, 100.0, options, job)
    # 2. exact learned
    if q in learned:
        return _learned(bank, learned[q], q, 100.0, options)
    # 3. built-in pattern
    for key, pattern in DEFAULT_PATTERNS:
        if re.search(pattern, q):
            return _resolve(bank, key, pattern, 100.0, options, job)
    # 4. fuzzy over aliases + learned questions
    choices = {f"a:{a}": a for a in aliases}
    choices.update({f"l:{n}": n for n in learned})
    best = process.extractOne(q, choices, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= FUZZY_THRESHOLD:
        tag = best[2]
        if tag.startswith("a:"):
            return _resolve(bank, aliases[tag[2:]], tag[2:], best[1], options, job)
        return _learned(bank, learned[tag[2:]], tag[2:], best[1], options)
    return None


def learn(question: str, answer: str, path: Path = BANK_PATH) -> bool:
    """Remember the user's answer to a question. Returns False if nothing was saved."""
    q, a = normalize(question), (answer or "").strip()
    if not q or not a:
        return False
    try:
        with _lock:
            bank = load(path)
            bank["learned"][q] = a
            _save(bank, path)
        logger.info("answer_bank: learned %r → %r", q, a)
        return True
    except Exception as exc:
        logger.error("answer_bank.learn(%r) failed: %s", question, exc)
        return False
