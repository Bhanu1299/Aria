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
    "preferred_name": ["preferred name", "preferred first name"],
    "location": ["location", "city", "current location", "location city"],
    "address": ["address", "current address", "home address", "mailing address"],
    "address_line1": ["address line 1", "street address", "street address line 1", "address 1", "street"],
    "city": ["city", "town", "city town"],
    "state": ["state", "state province", "province", "region"],
    "zip": ["zip", "zip code", "postal code", "zip postal code", "postcode"],
    "country": ["country", "country of residence"],
    "school": ["school", "university", "college", "school name", "university name"],
    "degree": ["degree", "degree type", "highest degree", "highest level of education"],
    "current_company": ["current company", "current employer", "most recent employer"],
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
    "eeo.pronouns": ["pronouns", "preferred pronouns", "your pronouns"],
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
    """The stored value for a canonical key, or None if unset / placeholder."""
    val = _own_value(bank, key)
    if val is None and key in _SAME_AS:
        val = _own_value(bank, _SAME_AS[key])
    return val


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
    path: Optional[Path] = None,
) -> Optional[Answer]:
    """Answer for a form field label from the user's bank, or None. Never raises."""
    try:
        return _lookup(load(path), label, options, job)
    except Exception as exc:
        logger.error("answer_bank.lookup(%r) failed: %s", label, exc)
        return None


def _match(bank: dict, q: str) -> tuple[Optional[str], Optional[str], float, str]:
    """(canonical key, learned question, score, how) for a normalized label — no values read."""
    aliases = _aliases(bank)
    learned = {normalize(k): k for k in bank["learned"]}
    if q in learned:
        return aliases.get(q), learned[q], 100.0, "learned"
    if q in aliases:
        return aliases[q], None, 100.0, "alias"
    for key, pattern in DEFAULT_PATTERNS:
        if re.search(pattern, q) and not (key in YES_NO_KEYS and _OPEN_QUESTION.match(q)):
            return key, None, 100.0, "pattern"
    choices = {f"a:{a}": a for a in aliases}
    choices.update({f"l:{n}": n for n in learned})
    best = process.extractOne(q, choices, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= FUZZY_THRESHOLD:
        tag = best[2]
        if tag.startswith("a:"):
            return aliases[tag[2:]], None, best[1], "fuzzy"
        return None, learned[tag[2:]], best[1], "fuzzy"
    return None, None, 0.0, ""


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


def _lookup(bank: dict, label: str, options, job) -> Optional[Answer]:
    q = normalize(label)
    if not q or _always_ask(bank, q):
        return None
    own = (bank.get("job_answers") or {}).get(job_key(job), {}) if job else {}
    if q in own and str(own[q]).strip():
        value = str(own[q]).strip()
        if options:
            value = choose_option(value, options)
        if value:
            return Answer(key="job", value=value, source="job", matched=q)
    key, question, score, _how = _match(bank, q)
    if key:
        found = _resolve(bank, key, q, score, options, job)
        if found:
            return found
    if question:
        return _learned(bank, question, question, score, options)
    return None


def matched_key(label: str, path: Optional[Path] = None) -> Optional[str]:
    """The canonical key a label maps to, even when the user hasn't answered it yet.

    Lets callers tell "a question we know but have no answer for" (leave it for
    the user) apart from "a question we've never seen". Never raises.
    """
    try:
        q = normalize(label)
        return _match(load(path), q)[0] if q else None
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
