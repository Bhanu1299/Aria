"""features/ats/greenhouse.py — deterministic Greenhouse application filler. NEVER submits.

Pipeline (fill()):
  1. open      navigate to the form (embeds are opened at their iframe URL)
  2. parse     parse_form(html): the page's window.__remixContext jobPost is the source of
               truth (labels, field types, required, every select option, pay ranges,
               description); a DOM pass adds inputs it doesn't list (country,
               candidate-location, hispanic_ethnicity, consent checkboxes)
  3. options   selects with no known options (DOM-only) are opened once to read them
  4. plan      plan(): answer bank first; a question the bank KNOWS but the user hasn't
               answered (e.g. sponsorship: null) → needs_me, never an LLM guess;
               required unknown short fields → SMALL-tier guess (llm_guess);
               "why us"-style textareas → FRONTIER draft (llm_draft);
               EEO / demographic questions are never guessed
  5. fill      type / pick / upload via agent_browser — no button is ever clicked
  6. read back every field's actual DOM value → review rows

Review row: {name, label, kind, value, source, required, note}
  source: bank | llm_guess | llm_draft | needs_me | skipped
"""
from __future__ import annotations

import html as _html
import json
import logging
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional

import aria.features.answer_bank as answer_bank
import aria.web.agent_browser as agent_browser
from aria.core import paths as _aria_paths
from aria.features.ats import answers as llm_answers
from aria.features.ats.detect import greenhouse_form_url

logger = logging.getLogger(__name__)

SOURCES = ("bank", "llm_guess", "llm_draft", "needs_me", "skipped")

_REMIX = re.compile(r"window\.__remixContext\s*=\s*(\{.*?\});\s*</script>", re.S)
_TYPE_KIND = {
    "input_text": "text",
    "textarea": "textarea",
    "input_file": "file",
    "multi_value_single_select": "select",
    "multi_value_multi_select": "multiselect",
}
# Alternate inputs Greenhouse renders next to a file upload ("Enter manually").
_ALT_TEXT_FIELDS = {"resume_text", "cover_letter_text"}
_DRAFT_LABEL = re.compile(
    r"\bwhy\b|\bdescribe\b|\btell us\b|\bwhat (?:excites|interests|draws|motivates)\b|\bcover letter\b", re.I
)
# Agreements, attestations and consents are the user's to give — never filled for them.
_ATTESTATION = re.compile(
    r"\bagree|\backnowledg|\barbitrat|\bconsent|\bcertif|\battest|\bpolicy\b|\bterms\b|"
    r"\bprivacy\b|\bplease (?:read|review)\b|\bsignature\b|\bsign\b",
    re.I,
)
_FORM_READY = '#application-form, #application_form, [id="first_name"]'


@dataclass
class FormField:
    name: str                 # DOM id
    label: str
    kind: str                 # text | textarea | file | select | multiselect | location | checkbox
    required: bool = False
    options: list = field(default_factory=list)
    section: str = "questions"  # questions | eeoc | demographic | dom


@dataclass
class Planned:
    field: FormField
    value: str = ""
    source: str = "skipped"
    note: str = ""


@dataclass
class ParsedForm:
    fields: list
    title: str = ""
    company: str = ""
    description: str = ""
    salary_range: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Parsing (pure — tested against saved pages)
# ---------------------------------------------------------------------------

class _DomScan(HTMLParser):
    """Inputs / textareas / selects with an id, in page order, plus label[for] text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[dict] = []
        self.labels: dict[str, str] = {}
        self._label_for: Optional[str] = None
        self._label_text: list[str] = []
        self._select: Optional[dict] = None
        self._option: Optional[list] = None

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "label":
            self._label_for, self._label_text = a.get("for") or None, []
        elif tag in ("input", "textarea", "select") and a.get("id"):
            el = {"tag": tag, **a, "options": []}
            self.elements.append(el)
            if tag == "select":
                self._select = el
        elif tag == "option" and self._select is not None:
            self._option = []

    def handle_endtag(self, tag):
        if tag == "label" and self._label_for:
            text = re.sub(r"\s+", " ", "".join(self._label_text)).strip().rstrip("*").strip()
            self.labels.setdefault(self._label_for, text)
            self._label_for = None
        elif tag == "option" and self._option is not None and self._select is not None:
            text = "".join(self._option).strip()
            if text and text.lower() not in ("select...", "please select", "--"):
                self._select["options"].append(text)
            self._option = None
        elif tag == "select":
            self._select = None

    def handle_data(self, data):
        if self._label_for is not None:
            self._label_text.append(data)
        if self._option is not None:
            self._option.append(data)


def _dom_kind(el: dict) -> Optional[str]:
    if el["tag"] == "textarea":
        return "textarea"
    if el["tag"] == "select":
        return "select"
    itype = (el.get("type") or "text").lower()
    if itype in ("hidden", "submit", "button", "image", "search"):
        return None
    if el.get("role") == "combobox":
        return "location" if "location" in el["id"] else "select"
    if itype == "file":
        return "file"
    if itype in ("checkbox", "radio"):
        return "checkbox"
    return "text"


def _remix_job_post(html: str) -> Optional[dict]:
    m = _REMIX.search(html or "")
    if not m:
        return None
    try:
        ctx = json.loads(m.group(1))
    except ValueError:
        return None
    for route in (ctx.get("state", {}).get("loaderData") or {}).values():
        if isinstance(route, dict) and isinstance(route.get("jobPost"), dict):
            return route["jobPost"]
    return None


def html_to_text(fragment: str) -> str:
    text = re.sub(r"<(?:br|/p|/li|/h\d|/div)[^>]*>", "\n", fragment or "", flags=re.I)
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.I)
    text = _html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _money(s) -> Optional[int]:
    digits = re.sub(r"[^\d.]", "", str(s or ""))
    try:
        return int(float(digits)) if digits else None
    except ValueError:
        return None


def _salary_range(pay_ranges) -> Optional[tuple]:
    for pr in pay_ranges or []:
        lo, hi = _money(pr.get("min")), _money(pr.get("max"))
        if lo and hi and lo >= 1000:
            return (lo, hi)
    return None


def _option_labels(values) -> list:
    out = []
    for v in values or []:
        label = (v.get("label") or v.get("name")) if isinstance(v, dict) else v
        if label is not None and str(label).strip():
            out.append(str(label).strip())
    return out


def parse_form(html: str, schema_html: str = "") -> ParsedForm:
    """html: the form's DOM. schema_html: the same page as served over plain HTTP —
    the live browser strips the __remixContext script after hydration."""
    scan = _DomScan()
    scan.feed(html or "")
    fields: list[FormField] = []
    seen: set = set()

    def add(f: FormField) -> None:
        if f.name and f.name not in seen and f.name not in _ALT_TEXT_FIELDS:
            seen.add(f.name)
            fields.append(f)

    post = _remix_job_post(schema_html) or _remix_job_post(html)
    title = company = description = ""
    salary = None
    if post:
        title, company = post.get("title", ""), post.get("company_name", "")
        description = html_to_text(post.get("content", ""))
        salary = _salary_range(post.get("pay_ranges"))
        for q in post.get("questions") or []:
            for fl in q.get("fields") or []:
                kind = _TYPE_KIND.get(fl.get("type"))
                if kind:
                    add(FormField(fl["name"], q.get("label", "").strip(), kind, bool(q.get("required")),
                                  _option_labels(fl.get("values")), "questions"))
        for sec in post.get("eeoc_sections") or []:
            for q in sec.get("questions") or []:
                for fl in q.get("fields") or []:
                    add(FormField(fl["name"], q.get("label", "").strip(), _TYPE_KIND.get(fl.get("type"), "select"),
                                  bool(q.get("required")), _option_labels(fl.get("values")), "eeoc"))
        demo = post.get("demographic_questions") or {}
        for q in demo.get("questions") or []:
            key = ((q.get("answer_type") or {}).get("key") or "").upper()
            add(FormField(str(q.get("id")), str(q.get("name", "")).strip(),
                          "multiselect" if key == "MULTI_SELECT" else "select",
                          bool(q.get("required")), _option_labels(q.get("answer_options")), "demographic"))

    # Schema fields the page renders under a different id (candidate_location → the
    # candidate-location autocomplete): let the DOM element stand in for them.
    dom_ids = {el["id"] for el in scan.elements}
    for f in list(fields):
        alt = f.name.replace("_", "-")
        if f.name not in dom_ids and alt != f.name and alt in dom_ids:
            fields.remove(f)
            seen.discard(f.name)
            seen.discard(alt)

    # DOM-only inputs (and the whole form on legacy boards without a remix context)
    for el in scan.elements:
        kind = _dom_kind(el)
        if not kind or el["id"] in seen:
            continue
        if "requiredinput" in (el.get("class") or "").lower():
            continue
        label = scan.labels.get(el["id"]) or el.get("aria-label") or el.get("placeholder") or ""
        if not label:
            continue
        add(FormField(el["id"], label, kind, el.get("aria-required") == "true" or "required" in el,
                      el.get("options", []), "dom"))

    # Fill EEO "Are you Hispanic/Latino?" before "Race" (race renders only after it).
    order = {"hispanic_ethnicity": -1}
    fields.sort(key=lambda f: order.get(f.name, 0))
    return ParsedForm(fields, title, company, description, salary)


# ---------------------------------------------------------------------------
# Planning (pure apart from the injected LLM functions)
# ---------------------------------------------------------------------------

def _identity() -> dict:
    try:
        return json.loads(_aria_paths.IDENTITY_JSON.read_text())
    except (OSError, ValueError):
        return {}


def resume_path() -> Optional[str]:
    """resume_path from the answer bank, else identity.json — only if the file exists."""
    bank = answer_bank.load()
    candidates = [bank["answers"].get("resume_path"), _identity().get("resume_path")]
    for c in candidates:
        if c and Path(str(c)).expanduser().is_file():
            return str(Path(str(c)).expanduser())
    return None


def build_profile(job: dict) -> str:
    """Facts the LLM may use: the user's own bank answers, identity, and the sheet row."""
    bank = answer_bank.load()
    lines = []
    for k, v in bank["answers"].items():
        if k in ("eeo", "salary_rule") or v in (None, "") or isinstance(v, dict):
            continue
        if not str(v).startswith("<"):
            lines.append(f"{k}: {v}")
    ident = _identity()
    for k in ("summary", "skills", "experience", "education", "target_roles"):
        if ident.get(k):
            lines.append(f"{k}: {json.dumps(ident[k]) if not isinstance(ident[k], str) else ident[k]}")
    fit = next((v for k, v in (job.get("fields") or {}).items() if k.lower().startswith("why it's a fit")), "")
    if fit:
        lines.append(f"notes on this role and the candidate's relevant work:\n{fit}")
    return "\n".join(lines)


def plan(
    form: ParsedForm,
    job: dict,
    profile: str,
    guess_fn: Callable = llm_answers.guess_short,
    draft_fn: Callable = llm_answers.draft_long,
    resume: Optional[str] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> list:
    job_ctx = {**job, "salary_range": form.salary_range or job.get("salary_range"),
               "description": form.description or job.get("description", "")}
    plans: list[Planned] = []
    to_guess: list[Planned] = []
    to_draft: list[Planned] = []

    for f in form.fields:
        p = Planned(f)
        plans.append(p)
        if f.kind == "file":
            if f.name == "resume" or "resume" in f.label.lower() or "cv" in f.label.lower().split():
                if resume:
                    p.value, p.source = resume, "bank"
                else:
                    p.source, p.note = ("needs_me" if f.required else "skipped"), "no resume_path set"
            elif f.required:
                p.source, p.note = "needs_me", "file upload"
            continue
        if f.kind == "checkbox":
            p.source = "needs_me" if f.required else "skipped"
            p.note = "consent/checkbox — your call" if f.required else ""
            continue

        if _ATTESTATION.search(f.label) or any(_ATTESTATION.search(o) for o in f.options):
            p.source = "needs_me" if f.required else "skipped"
            p.note = "agreement — yours to accept"
            continue

        found = answer_bank.lookup(f.label, options=f.options or None, job=job_ctx)
        if found:
            p.value, p.source = found.value, "bank"
            continue
        if answer_bank.matched_key(f.label):
            p.source = "needs_me" if f.required else "skipped"
            p.note = "no answer in your bank yet"
            continue
        if f.section in ("eeoc", "demographic"):
            p.source = "needs_me" if f.required else "skipped"
            continue
        if f.kind == "textarea":
            if f.required or _DRAFT_LABEL.search(f.label):
                to_draft.append(p)
            continue
        if f.required:
            to_guess.append(p)

    if to_guess:
        guesses = guess_fn([{"name": p.field.name, "label": p.field.label, "options": p.field.options}
                            for p in to_guess], profile)
        for p in to_guess:
            if p.field.name in guesses:
                p.value, p.source = guesses[p.field.name], "llm_guess"
            else:
                p.source, p.note = "needs_me", "not answerable from your profile"

    job_text = f"{form.title or job.get('role', '')} at {form.company or job.get('company', '')}\n\n{job_ctx['description']}"
    for p in to_draft:
        if progress_fn:
            progress_fn(f"Drafting an answer for: {p.field.label[:60]}")
        text = draft_fn(p.field.label, job_text, profile)
        if text:
            p.value, p.source = text, "llm_draft"
        else:
            p.source = "needs_me" if p.field.required else "skipped"
            p.note = "draft failed"
    return plans


def build_review(plans: list, dom_values: dict) -> list:
    """Review rows from what was planned vs what the DOM actually holds."""
    rows = []
    for p in plans:
        f = p.field
        actual = dom_values.get(f.name)
        source, note = p.source, p.note
        value = actual if actual is not None else ""
        if f.kind == "file" and p.value and actual:
            value = Path(p.value).name
        if p.value and actual and re.fullmatch(r"\+\d{1,4}", actual):   # country picker shows the dial code
            value = p.value if p.value.endswith(actual) else f"{p.value} ({actual})"
        if p.value and not actual:
            source = "needs_me" if f.required else "skipped"
            note = "not on the form" if actual is None else "didn't stick — fill it in"
        rows.append({"name": f.name, "label": f.label, "kind": f.kind, "value": value,
                     "source": source, "required": f.required, "note": note})
    return rows


# ---------------------------------------------------------------------------
# Browser (every call runs on the agent_browser worker thread)
# ---------------------------------------------------------------------------

def _sel(name: str) -> str:
    return f'[id="{name}"]'          # Greenhouse ids can be all digits — not valid as #id


def _options_locator(page, name: str):
    return page.locator(f'[id^="react-select-{name}-option"]')


def _read_options(page, name: str) -> list:
    loc = page.locator(_sel(name))
    if loc.count() == 0:
        return []
    try:
        loc.first.click(timeout=4000)
        _options_locator(page, name).first.wait_for(timeout=2500)
        return [t.strip() for t in _options_locator(page, name).all_inner_texts() if t.strip()]
    except Exception:
        return []
    finally:
        page.keyboard.press("Escape")


_US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "dc": "district of columbia",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois",
    "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
}


def match_location(value: str, options: list) -> Optional[str]:
    """The autocomplete option for "City, ST" / "City, State": same city as the first
    part AND the state present. Anything less → None (left for the user)."""
    parts = [x.strip().lower() for x in (value or "").split(",") if x.strip()]
    if not parts:
        return None
    city = parts[0]
    region = parts[1] if len(parts) > 1 else ""
    region_names = {region, _US_STATES.get(region, region)} - {""}
    for opt in options:
        op = [x.strip().lower() for x in opt.split(",")]
        if op and op[0] == city and (not region_names or region_names & set(op[1:])):
            return opt
    return None


def _pick(page, name: str, value: str, type_text: str = "") -> bool:
    loc = page.locator(_sel(name))
    if loc.count() == 0:
        return False
    loc.first.click(timeout=4000)
    if type_text:
        loc.first.press_sequentially(type_text, delay=40)
    opts = _options_locator(page, name)
    try:
        opts.first.wait_for(timeout=5000 if type_text else 2500)
    except Exception:
        page.keyboard.press("Escape")
        return False
    texts = [t.strip() for t in opts.all_inner_texts()]
    if type_text:                                   # autocomplete (location)
        target = match_location(value, texts)
    else:
        target = value if value in texts else answer_bank.choose_option(value, texts)
    if target is None:
        page.keyboard.press("Escape")
        return False
    opts.nth(texts.index(target)).click(timeout=3000)
    page.keyboard.press("Escape")
    return True


def _fill_one(page, p: Planned) -> bool:
    """Fill one planned field. False if the element isn't on the page (yet)."""
    f, sel = p.field, _sel(p.field.name)
    if page.locator(sel).count() == 0:
        return False
    if f.kind == "file":
        page.set_input_files(sel, p.value, timeout=10000)
    elif f.kind in ("text", "textarea"):
        page.fill(sel, p.value, timeout=5000)
    elif f.kind == "location":
        _pick(page, f.name, p.value, type_text=p.value.split(",")[0].strip())
    elif f.kind in ("select", "multiselect"):
        el = page.locator(sel).first
        if el.evaluate("e => e.tagName") == "SELECT":
            page.select_option(sel, label=p.value, timeout=5000)
        else:
            _pick(page, f.name, p.value)
    return True


_READBACK_JS = """
(names) => {
  const out = {};
  for (const n of names) {
    const el = document.getElementById(n);
    if (!el) { out[n] = null; continue; }
    if (el.type === 'file') {
      const box = el.closest('[class*="file-upload"], .field-wrapper, div') || document.body;
      const name = el.files && el.files.length ? el.files[0].name : '';
      out[n] = name || ((box.innerText || '').match(/[\\w.() -]+\\.(pdf|docx?|txt|rtf)/i) || [''])[0];
      continue;
    }
    if (el.getAttribute('role') === 'combobox') {
      const ctl = el.closest('.select__control') || el.closest('[class*="control"]');
      if (!ctl) { out[n] = ''; continue; }
      const multi = Array.from(ctl.querySelectorAll('.select__multi-value__label')).map(e => e.innerText.trim());
      const single = ctl.querySelector('.select__single-value');
      out[n] = multi.length ? multi.join('; ') : (single ? single.innerText.trim() : '');
      continue;
    }
    if (el.tagName === 'SELECT') { const o = el.options[el.selectedIndex]; out[n] = o && el.value ? o.text.trim() : ''; continue; }
    if (el.type === 'checkbox' || el.type === 'radio') { out[n] = el.checked ? 'checked' : ''; continue; }
    out[n] = el.value || '';
  }
  return out;
}
"""


_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _fetch_schema_html(url: str) -> str:
    """The form page over plain HTTP (carries window.__remixContext). "" on failure."""
    try:
        import httpx
        r = httpx.get(url, headers={"User-Agent": _UA}, follow_redirects=True, timeout=15.0)
        return r.text if r.status_code == 200 else ""
    except Exception as exc:
        logger.warning("greenhouse: schema fetch failed for %s: %s", url, exc)
        return ""


def _open(url: str, progress_fn) -> str:
    """Navigate to the form; return its live DOM HTML."""
    agent_browser.navigate(url, settle_secs=2.5)
    html = agent_browser.run(lambda page: page.content())
    form_url = greenhouse_form_url(url, html)
    if form_url != url:
        if progress_fn:
            progress_fn("It's an embedded Greenhouse form — opening it directly.")
        agent_browser.navigate(form_url, settle_secs=2.5)

    def _ready(page):
        page.wait_for_selector(_FORM_READY, timeout=15000)
        return page.content()
    return agent_browser.run(_ready)


@dataclass
class GreenhouseResult:
    form_url: str = ""
    title: str = ""
    company: str = ""
    description: str = ""
    review: list = field(default_factory=list)
    error: str = ""


def fill(job: dict, progress_fn: Optional[Callable[[str], None]] = None) -> GreenhouseResult:
    """Open, parse, plan, fill and read back. Never submits, never raises."""
    say = progress_fn or (lambda _msg: None)
    result = GreenhouseResult()
    try:
        say("Opening the Greenhouse application.")
        html = _open(job["url"], progress_fn)
        result.form_url = agent_browser.run(lambda page: page.url)
        form = parse_form(html, _fetch_schema_html(result.form_url))
        result.title, result.company, result.description = form.title, form.company, form.description
        if not form.fields:
            result.error = "I couldn't find any form fields on that page."
            return result

        unknown = [f for f in form.fields if f.kind in ("select", "multiselect") and not f.options]
        if unknown:
            found = agent_browser.run(lambda page: {f.name: _read_options(page, f.name) for f in unknown})
            for f in unknown:
                f.options = found.get(f.name, [])

        say(f"Found {len(form.fields)} fields. Working out answers.")
        plans = plan(form, job, build_profile(job), resume=resume_path(), progress_fn=progress_fn)
        to_fill = [p for p in plans if p.value and p.source in ("bank", "llm_guess", "llm_draft")]
        say(f"Filling {len(to_fill)} fields.")

        def _fill_all(page):
            missing = []
            for p in to_fill:
                try:
                    if not _fill_one(page, p):
                        missing.append(p)
                except Exception as exc:
                    logger.warning("greenhouse: fill %s (%s) failed: %s", p.field.name, p.field.label, exc)
            if missing:                    # conditional fields (e.g. Race) appear after others are set
                time.sleep(1.0)
                for p in missing:
                    try:
                        _fill_one(page, p)
                    except Exception as exc:
                        logger.warning("greenhouse: retry %s failed: %s", p.field.name, exc)
            return page.evaluate(_READBACK_JS, [p.field.name for p in plans])

        dom_values = agent_browser.run(_fill_all)
        result.review = build_review(plans, dom_values)
    except Exception as exc:
        logger.error("greenhouse.fill failed: %s", exc)
        result.error = f"The Greenhouse form fill failed: {exc}"
    return result
