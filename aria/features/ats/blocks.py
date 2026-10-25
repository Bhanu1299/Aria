"""features/ats/blocks.py — a form's repeating blocks (Workday: Work Experience 1, 2 …; Education
1, 2 …) filled from your lists in the answer bank (answer_bank.get_list), block i = entry i.

2026-10-25 (Meijer): Workday's own resume parse made jobs out of the PROJECTS section and gave
both schools "MA"; the bank had one school and no jobs at all. Now (user's call: "replace with
my list"):
  sync(page, section, n)    deletes / adds Workday panels until there are exactly n
                            (the trash button on the last panel; the section's "Add" button)
  assign(plans, raw_by_fid, lists)
                            every field inside a block gets entry i's value — overwriting what
                            Workday parsed — keyed by the field's data-fkit-id
                            ("workExperience-12--jobTitle": block workExperience-12, sub jobTitle)
Projects never go into work experience. A section with no entries is left as the page has it.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# the form's block name (data-fkit-id prefix) → answer-bank list
SECTION_LIST = {"workExperience": "experience", "education": "education"}
# Workday's section group: <div role="group" aria-labelledby="Work-Experience-section">
SECTION_GROUPS = {"experience": ("Work-Experience", "Employment-History", "Work-History", "Employment-Detail"),
                  "education": ("Education",)}
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_PRESENT = re.compile(r"(?i)^\s*(?:present|current|now|ongoing)\s*$")

COUNT_JS = r"""
(prefixes) => {
  const sec = Array.from(document.querySelectorAll('[role=group][aria-labelledby$="-section"]'))
    .find(g => prefixes.some(p => g.getAttribute('aria-labelledby') === p + '-section'));
  if (!sec) return -1;
  return sec.querySelectorAll('[role=group][aria-labelledby$="-panel"]').length;
}
"""

DELETE_LAST_JS = r"""
(prefixes) => {
  const sec = Array.from(document.querySelectorAll('[role=group][aria-labelledby$="-section"]'))
    .find(g => prefixes.some(p => g.getAttribute('aria-labelledby') === p + '-section'));
  if (!sec) return false;
  const panels = sec.querySelectorAll('[role=group][aria-labelledby$="-panel"]');
  if (!panels.length) return false;
  const last = panels[panels.length - 1];
  const btn = Array.from(last.querySelectorAll('button')).find(b =>
      b.querySelector('.wd-icon-trash') || b.getAttribute('data-automation-id') === 'panel-set-delete-button' ||
      /^(delete|remove)\b/i.test((b.getAttribute('aria-label') || b.innerText || '').trim()));
  if (!btn) return false;
  btn.click();
  return true;
}
"""

ADD_JS = r"""
(prefixes) => {
  const sec = Array.from(document.querySelectorAll('[role=group][aria-labelledby$="-section"]'))
    .find(g => prefixes.some(p => g.getAttribute('aria-labelledby') === p + '-section'));
  if (!sec) return false;
  const btn = Array.from(sec.querySelectorAll('button')).find(b =>
      !b.closest('[role=group][aria-labelledby$="-panel"]') &&
      (b.getAttribute('data-automation-id') === 'add-button' || /^add\b/i.test((b.innerText || b.getAttribute('aria-label') || '').trim())));
  if (!btn) return false;
  btn.click();
  return true;
}
"""


def _wait_count(page, prefixes: tuple, want_change_from: int, secs: float = 4.0) -> int:
    deadline = time.time() + secs
    n = want_change_from
    while time.time() < deadline:
        time.sleep(0.3)
        n = page.evaluate(COUNT_JS, list(prefixes))
        if n != want_change_from:
            break
    return n


def sync(page, section: str, want: int) -> Optional[int]:
    """Make the page show exactly `want` blocks for `section` ("experience" | "education").
    Returns the count it ended with, or None when the page has no such section. Never raises."""
    prefixes = SECTION_GROUPS.get(section)
    if not prefixes or want < 0:
        return None
    try:
        n = page.evaluate(COUNT_JS, list(prefixes))
        if n < 0:
            return None
        for _ in range(12):
            if n > want:
                if not page.evaluate(DELETE_LAST_JS, list(prefixes)):
                    break
            elif n < want:
                if not page.evaluate(ADD_JS, list(prefixes)):
                    break
            else:
                break
            got = _wait_count(page, prefixes, n)
            if got == n:                       # the click did nothing: don't loop on it
                break
            n = got
        return n
    except Exception as exc:
        logger.warning("blocks.sync(%s) failed: %s", section, exc)
        return None


# ---------------------------------------------------------------------------
# Values: which entry field a block's field gets
# ---------------------------------------------------------------------------

def is_current(entry: dict) -> bool:
    return bool(_PRESENT.match(entry.get("end") or ""))


def _date_part(text: str, label: str) -> str:
    """'Jan 2025' / '01/2025' / '2025' → the month ('01') or year ('2025') a Workday date box wants."""
    t = (text or "").strip()
    if not t or _PRESENT.match(t):
        return ""
    year = re.search(r"\b(19\d{2}|20\d{2})\b", t)
    month = None
    m = re.match(r"(?i)([a-z]{3})[a-z]*\.?\s+\d{4}", t)
    if m and m.group(1).lower() in _MONTHS:
        month = _MONTHS[m.group(1).lower()]
    m = re.match(r"^(\d{1,2})/(\d{4})$", t)
    if m:
        month = int(m.group(1))
    lab = (label or "").lower()
    if lab.startswith("month") or lab == "mm":
        return f"{month:02d}" if month else ""
    if lab.startswith("year") or lab == "yyyy":
        return year.group(1) if year else ""
    if month and year:
        return f"{month:02d}/{year.group(1)}"
    return year.group(1) if year else t


_EXPERIENCE = {"jobTitle": "title", "companyName": "company", "company": "company", "location": "location",
               "roleDescription": "description", "description": "description"}
_EDUCATION = {"schoolName": "school", "school": "school", "degree": "degree", "fieldOfStudy": "field",
              "gradeAverage": "gpa", "gpa": "gpa"}


def value_for(list_name: str, sub: str, label: str, entry: dict) -> Optional[str]:
    """The value block field `sub` gets from `entry`, "" for "leave it empty", None = not ours."""
    if list_name == "experience":
        if sub in _EXPERIENCE:
            return entry.get(_EXPERIENCE[sub]) or ""
        if sub == "currentlyWorkHere":
            return "Yes" if is_current(entry) else "No"
        if sub == "startDate":
            return _date_part(entry.get("start", ""), label)
        if sub == "endDate":
            return "" if is_current(entry) else _date_part(entry.get("end", ""), label)
    if list_name == "education":
        if sub in _EDUCATION:
            return entry.get(_EDUCATION[sub]) or ""
        if sub in ("startDate", "firstYearAttended"):
            return _date_part(entry.get("start", ""), label if sub == "startDate" else "year")
        if sub in ("endDate", "lastYearAttended"):
            return _date_part(entry.get("end", ""), label if sub == "endDate" else "year")
    return None


def assign(plans: list, raw_by_fid: dict, lists: dict) -> int:
    """Every planned field inside a block with an entry gets the entry's value (source "bank",
    overwriting the page). Block order = the order blocks appear on the page. Returns how many."""
    order: dict = {}
    for r in raw_by_fid.values():
        fkit = r.get("fkit") or ""
        block, _, _sub = fkit.partition("--")
        section = block.split("-")[0]
        if section in SECTION_LIST and block:
            seen = order.setdefault(section, [])
            if block not in seen:
                seen.append(block)
    done = 0
    for p in plans:
        r = raw_by_fid.get(p.field.name) or {}
        block, _, sub = (r.get("fkit") or "").partition("--")
        section = block.split("-")[0]
        list_name = SECTION_LIST.get(section)
        entries = lists.get(list_name) or []
        if not list_name or not entries or block not in order.get(section, []):
            continue
        i = order[section].index(block)
        if i >= len(entries):
            continue
        value = value_for(list_name, sub, p.field.label, entries[i])
        if value is None:
            continue
        if value:
            p.value, p.source = value, "bank"
            p.note = f"from your {list_name} list, entry {i + 1}"
        elif p.field.required:
            p.value, p.source = "", "needs_me"
            p.note = f"empty in your {list_name} list, entry {i + 1} — add it in My answers"
        else:
            p.value, p.source, p.note = "", "skipped", ""
        done += 1
    return done
