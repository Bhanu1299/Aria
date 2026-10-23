"""features/ats/form_engine.py — deterministic filler for any application form (Ashby, Lever,
Jobvite, SmartRecruiters, company forms …). NEVER submits. Code controls the flow; the
LLM only writes open-text answers (planner.py).

Per page:
  1. scan      SCAN_JS reads every visible field — label, kind, required, options, char
               limit — and tags each with data-aria-fid. Hidden file inputs count (styled
               upload buttons hide them); radios / checkboxes sharing a name are one field.
  2. plan      planner.plan(): bank facts / stored sensitive answers / LLM open text /
               UNKNOWN (needs_me) / consent
  3. fill      one field per browser call: type, pick a native option, click a radio,
               open a custom dropdown and click the matching option, attach a file
  4. re-scan   fields revealed by an answer are planned and filled, up to MAX_ROUNDS
  5. verify    read every value back + validation errors; refill a wrong one up to
               MAX_RETRIES times, then failed
  6. next      only when verify passed and nothing required is left: click a visible
               Next / Continue (never Submit / Apply / Send) and repeat on the new page
  7. stop      no Next button (the final page) → stop; the user approves the submit
  Walls (CAPTCHA / login / email code / a privacy notice to accept) are checked before
  every scan: wait_fn(kind) — the session says which tab needs you and waits — then go on.

Every step is a `page` function (tested with a real headless page); fill() runs them on
the agent_browser worker thread, one call per step, so a pause holds between steps.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import aria.features.answer_bank as answer_bank
from aria.features.ats import planner
from aria.features.ats.planner import FILLED_SOURCES, FormField, ParsedForm, Planned

logger = logging.getLogger(__name__)

MAX_ROUNDS = 4        # re-scans per page for fields revealed by answers
MAX_RETRIES = 2       # refills per field before it's failed
MAX_PAGES = 8

SCAN_JS = r"""
() => {
  const clean = t => (t || '').replace(/\s+/g, ' ').trim();
  const GENERIC = /^\s*(?:type here|start typing|select|choose|please select|enter|search|e\.?g\.?|ex:|example|mm|dd|yyyy)\b|^\s*-+\s*(?:no answer|select|choose|none)?\s*-*\s*$/i;
  const NOISE = /^\d+\s*\/\s*\d+$|characters? (left|remaining)|^(required|optional)$/i;
  const vis = el => { if (!el) return false; const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  // ids carry a per-page-load prefix: after a reload (e.g. a privacy notice accepted) new
  // fields never reuse an old field's id
  if (!window.__ariaPfx) window.__ariaPfx = Math.random().toString(36).slice(2, 6);
  let next = window.__ariaFid || 0;
  const fid = el => { if (!el.dataset.ariaFid) { next += 1; el.dataset.ariaFid = window.__ariaPfx + '-' + next; } window.__ariaFid = next; return el.dataset.ariaFid; };
  const CONTROL = 'input, select, textarea, [role=combobox], [role=listbox], [role=radiogroup]';
  const LABELISH = 'legend, label, h1, h2, h3, h4, h5, h6, [class*="label" i], [class*="question" i], [class*="title" i]';

  function before(c, el) { return !!(c.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING); }
  function usable(c, el) {
    if (c.contains(el) || c.querySelector(CONTROL)) return false;
    if (c.tagName === 'LABEL' && c.htmlFor && c.htmlFor !== el.id && document.getElementById(c.htmlFor)) return false;
    const t = clean(c.innerText || c.textContent);
    return t.length > 1 && t.length < 400 && !NOISE.test(t) && vis(c);
  }
  // The nearest label-like text before `el` inside its closest ancestors (up to maxDepth).
  function nearLabel(el, maxDepth) {
    let node = el;
    for (let depth = 0; depth < (maxDepth || 7) && node.parentElement; depth++) {
      node = node.parentElement;
      if (node === document.body) break;
      for (const sel of [LABELISH, 'p, span, div']) {
        const cands = Array.from(node.querySelectorAll(sel)).filter(c => before(c, el) && usable(c, el) &&
          (sel === LABELISH || c.children.length === 0));
        if (cands.length) return clean(cands[cands.length - 1].innerText || cands[cands.length - 1].textContent);
      }
    }
    return '';
  }
  function ownLabel(el) {
    if (el.labels && el.labels.length) { const t = clean(el.labels[0].innerText || el.labels[0].textContent); if (t) return t; }
    const lb = el.getAttribute('aria-labelledby');
    if (lb) { const t = clean(lb.split(/\s+/).map(id => { const n = document.getElementById(id); return n ? (n.innerText || n.textContent) : ''; }).join(' ')); if (t) return t; }
    const al = el.getAttribute('aria-label'); if (al && !GENERIC.test(al)) return clean(al);
    return '';
  }
  function optionText(el) {
    const own = ownLabel(el); if (own) return own;
    const p = el.closest('label'); if (p) return clean(p.innerText || p.textContent);
    const sib = el.nextElementSibling || el.parentElement;
    return clean(sib ? (sib.innerText || sib.textContent) : el.value);
  }
  function req(el, raw) {
    return !!(el.required || el.getAttribute('aria-required') === 'true' || /[*✱]/.test(raw || '') || /\(required\)/i.test(raw || ''));
  }
  const strip = t => clean((t || '').replace(/[*✱]+/g, ' ').replace(/\((required|optional)\)/ig, ' '));
  // Workday-style stable ids on the control or its wrapper — adapters map them to bank keys
  const aidOf = el => { const n = el.closest('[data-automation-id]'); return n ? n.getAttribute('data-automation-id') : ''; };
  const out = [];
  const groups = {};

  for (const el of document.querySelectorAll('input, textarea, select, [role=combobox]')) {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || (tag === 'input' ? 'text' : tag)).toLowerCase();
    if (['hidden', 'submit', 'button', 'image', 'reset', 'password'].includes(type)) continue;
    if (el.disabled || el.readOnly && type !== 'file' && el.getAttribute('role') !== 'combobox') continue;
    if (el.closest('[aria-hidden="true"]') && type !== 'file') continue;
    if (el.closest('[class*="autofill" i]')) continue;      // "autofill from resume" drop zones aren't questions
    if (type === 'radio' || type === 'checkbox') {
      // checkboxes named q-1, q-2 … (or q[1], q[2]) are one question
      const stem = type === 'checkbox' ? (el.name || '').replace(/(?:[-_]\d+|\[\d*\])$/, '') : el.name;
      const key = type + ':' + (stem || fid(el));
      (groups[key] = groups[key] || []).push(el);
      continue;
    }
    if (type !== 'file' && !vis(el)) continue;
    if (el.getAttribute('role') === 'combobox' && tag !== 'input' && el.querySelector('input')) continue;
    // own label → text right next to it → a meaningful placeholder ("City") → text further up
    let raw = ownLabel(el);
    const ph = el.placeholder && !GENERIC.test(el.placeholder) ? clean(el.placeholder) : '';
    const FILEWORD = /^\s*(?:file|files|upload|upload a file|attach|attach a file|choose file|browse|drop files? here)\s*$/i;
    if (!raw || GENERIC.test(raw) || raw.length < 2 || (type === 'file' && FILEWORD.test(raw)))
      raw = nearLabel(el, 3) || ph || nearLabel(el, 7) || raw;
    if (type === 'file' && (!raw || FILEWORD.test(raw))) {   // hidden upload widgets: name the slot from its text
      let n = el;
      for (let i = 0; i < 4 && n.parentElement; i++) {
        n = n.parentElement;
        const m = (n.textContent || '').match(/\b(resume|cv|cover letter|portfolio|transcript)\b/i);
        if (m) { raw = m[1].charAt(0).toUpperCase() + m[1].slice(1); break; }
      }
    }
    if (!raw) raw = el.name || '';
    let kind = tag === 'textarea' ? 'textarea' : tag === 'select' ? 'select' : type === 'file' ? 'file'
      : type === 'date' ? 'date' : 'text';
    const combo = el.getAttribute('role') === 'combobox' || el.getAttribute('aria-autocomplete') === 'list' ||
      el.getAttribute('aria-haspopup') === 'listbox';
    if (combo && tag !== 'select') kind = 'combobox';
    let options = [];
    if (tag === 'select') options = Array.from(el.options).filter(o => o.value !== '' && clean(o.text) && !GENERIC.test(clean(o.text)))
      .map(o => clean(o.text)).filter(Boolean);
    out.push({fid: fid(el), kind: kind, label: strip(raw), required: req(el, raw), options: options,
              max_chars: el.maxLength > 0 && el.maxLength < 100000 ? el.maxLength : 0,
              accept: el.getAttribute('accept') || '', name: el.name || el.id || '', aid: aidOf(el)});
  }
  for (const key of Object.keys(groups)) {
    const els = groups[key];
    const shown = els.filter(e => vis(e) || (e.labels && e.labels[0] && vis(e.labels[0])) || vis(e.parentElement));
    if (!shown.length) continue;
    const first = shown[0];
    const type = key.split(':')[0];
    const fs = first.closest('fieldset');
    const legend = fs && fs.querySelector('legend');
    let raw = legend ? clean(legend.innerText) : '';
    const group = first.closest('[role=radiogroup], [role=group]');
    if (!raw && group) raw = ownLabel(group);
    if (!raw) raw = nearLabel(first);
    if (type === 'checkbox' && shown.length === 1) {
      const own = optionText(first);
      raw = own && own.length > 12 ? own : clean((raw ? raw + ' ' : '') + own);
      out.push({fid: fid(first), kind: 'checkbox', label: strip(raw), required: req(first, raw) || req(first, own),
                options: [], max_chars: 0, accept: '', name: first.name || '', aid: aidOf(first)});
      continue;
    }
    shown.forEach(e => fid(e));
    out.push({fid: fid(first), kind: type === 'radio' ? 'radio' : 'checkgroup', label: strip(raw),
              required: shown.some(e => req(e, raw)), options: shown.map(optionText), max_chars: 0, accept: '',
              name: first.name || '', aid: aidOf(first), members: shown.map(e => e.dataset.ariaFid)});
  }
  return out;
}
"""

READ_JS = r"""
(fields) => {
  const clean = t => (t || '').replace(/\s+/g, ' ').trim();
  const out = {};
  const by = f => document.querySelector('[data-aria-fid="' + f + '"]');
  const errorOf = el => {
    const ids = (el.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
    for (const id of ids) { const n = document.getElementById(id); const t = n && clean(n.innerText);
      if (t && /required|invalid|error|must|please|enter a|select a/i.test(t)) return t; }
    const box = el.closest('fieldset, li, [class*="field" i], [class*="question" i], .form-group') || el.parentElement;
    const err = box && box.querySelector('[class*="error" i]:not(:empty), [role=alert]');
    return err ? clean(err.innerText).slice(0, 160) : '';
  };
  for (const f of fields) {
    const el = by(f.fid);
    if (!el) { out[f.fid] = {value: null, error: ''}; continue; }
    let value = '';
    if (f.kind === 'radio' || f.kind === 'checkgroup') {
      const picked = (f.members || [f.fid]).map(by).filter(e => e && e.checked);
      value = picked.map(e => { const l = (e.labels && e.labels[0]) || e.closest('label');
        return clean(l ? l.innerText : e.value); }).join('; ');
    } else if (f.kind === 'checkbox') {
      value = el.checked ? 'checked' : '';
    } else if (f.kind === 'file') {
      const name = el.files && el.files.length ? el.files[0].name : '';
      const box = el.closest('[class*="upload" i], [class*="file" i], fieldset, li, div') || document.body;
      value = name || ((box.innerText || '').match(/[\w.() -]+\.(pdf|docx?|txt|rtf)/i) || [''])[0];
    } else if (el.tagName === 'SELECT') {
      const o = el.options[el.selectedIndex]; value = o && el.value ? clean(o.text) : '';
    } else if (f.kind === 'combobox') {
      const box = el.closest('[class*="control" i], [class*="select" i], [class*="combobox" i], [class*="dropdown" i]') || el.parentElement;
      const shown = box ? clean(box.innerText) : '';
      value = clean(el.value) || shown;
    } else {
      value = el.value || '';
    }
    out[f.fid] = {value: value, error: el.getAttribute('aria-invalid') === 'true' ? (errorOf(el) || 'invalid') : errorOf(el)};
  }
  return out;
}
"""

_NEXT_TEXT = re.compile(r"^\s*(?:next|continue|save (?:and|&) continue|next (?:step|page)|proceed)\b", re.I)
_SUBMIT_TEXT = re.compile(r"\b(?:submit|send|apply|finish|complete)\b", re.I)
_OPTION_SEL = ('[role="option"], [role="listbox"] li, [class*="option" i]:not(select):not(option), '
               '[class*="dropdown-results" i] > *, [class*="suggestion" i], [class*="autocomplete" i] li, .pac-item')
_MARK_PRE_JS = """(sel) => document.querySelectorAll(sel).forEach(e => {
  const r = e.getBoundingClientRect(); if (r.width > 0 && r.height > 0) e.setAttribute('data-aria-pre', '1'); })"""
# One-line inputs that are typeaheads on many forms: type, then click the suggestion.
_TYPEAHEAD_LABEL = re.compile(r"\blocation\b|\bcity\b|where are you (?:based|located)", re.I)


@dataclass
class EngineResult:
    form_url: str = ""
    pages: int = 0
    review: list = field(default_factory=list)
    filled: dict = field(default_factory=dict)
    stopped_at: str = ""          # "final page" | "needs you" | "failed fields" | "no fields" | "page limit"
    error: str = ""


# ---------------------------------------------------------------------------
# Page steps (pure Playwright on a `page`)
# ---------------------------------------------------------------------------

def scan(page) -> list:
    """Visible fields on the page as FormFields (name = data-aria-fid), plus the raw scan."""
    return page.evaluate(SCAN_JS)


def to_fields(raw: list, field_key: Optional[Callable[[dict], str]] = None) -> list:
    """Raw scan rows → FormFields. field_key(raw) names the bank key for a field the ATS
    identifies by its own name / data-automation-id (the adapter's map), else ""."""
    out = []
    for r in raw:
        label = r.get("label") or ""
        if not label:
            continue
        f = FormField(r["fid"], label, r["kind"], bool(r.get("required")), list(r.get("options") or []), "dom",
                      int(r.get("max_chars") or 0), key=(field_key(r) if field_key else "") or "")
        out.append(f)
    return out


def _loc(page, fid: str):
    return page.locator(f'[data-aria-fid="{fid}"]').first


def _iso_date(value: str) -> str:
    m = re.match(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$", value or "")
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return value


def _choose(value: str, options: list) -> Optional[str]:
    if value in options:
        return value
    return answer_bank.choose_option(value, options)


def _pick_combobox(page, fid: str, value: str, wait_ms: int = 4000) -> Optional[str]:
    """Custom dropdown / typeahead: click, type, wait for options, click the match.
    Returns the option clicked, or None."""
    el = _loc(page, fid)
    # Only this dropdown's options: the listbox it controls, else options that appeared on
    # opening it (anything already visible belongs to another question).
    page.evaluate(_MARK_PRE_JS, _OPTION_SEL)
    el.click(timeout=4000)
    typed = re.split(r"[,(]", value)[0].strip()[:40]
    try:
        if el.evaluate("e => e.tagName === 'INPUT'"):
            el.fill("")
            el.press_sequentially(typed, delay=30)
    except Exception:
        pass
    owned = el.get_attribute("aria-controls") or el.get_attribute("aria-owns") or ""
    owned = owned.split()[0] if owned.split() else ""
    if owned and page.locator(f'[id="{owned}"]').count():
        opts = page.locator(f'[id="{owned}"]').locator('[role="option"], li')
    else:
        opts = page.locator(_OPTION_SEL)
    try:
        opts.first.wait_for(state="visible", timeout=wait_ms)
    except Exception:
        try:
            deadline = time.time() + wait_ms / 1000.0
            while time.time() < deadline and not any(
                    opts.nth(i).is_visible() and opts.nth(i).get_attribute("data-aria-pre") is None
                    for i in range(opts.count())):
                time.sleep(0.2)
        except Exception:
            pass
    time.sleep(0.3)
    visible = [i for i in range(opts.count())
               if opts.nth(i).is_visible() and opts.nth(i).get_attribute("data-aria-pre") is None]
    page.evaluate("() => document.querySelectorAll('[data-aria-pre]').forEach(e => e.removeAttribute('data-aria-pre'))")
    if not visible:
        page.keyboard.press("Escape")
        return None
    texts = [opts.nth(i).inner_text().strip() for i in visible]
    target = _choose(value, texts)
    if target is None:
        from aria.features.ats.greenhouse import match_location
        target = match_location(value, texts)
    if target is None:
        page.keyboard.press("Escape")
        return None
    opts.nth(visible[texts.index(target)]).click(timeout=3000)
    return target


def fill_one(page, p: Planned, members: Optional[list] = None) -> bool:
    """Fill one planned field on `page`. False = couldn't."""
    f, value = p.field, p.value
    el = _loc(page, f.name)
    if el.count() == 0:
        return False
    if f.kind == "file":
        el.set_input_files(planner.upload_path(p), timeout=10000)
        return True
    if f.kind == "text" and _TYPEAHEAD_LABEL.search(f.label) and p.category != "open_text":
        picked = _pick_combobox(page, f.name, value, wait_ms=2500)
        if picked:
            p.value = picked
            return True
    if f.kind in ("text", "textarea"):
        el.fill(value, timeout=5000)
        el.evaluate("e => e.dispatchEvent(new Event('blur', {bubbles: true}))")
        return True
    if f.kind == "date":
        el.fill(_iso_date(value), timeout=5000)
        return True
    if f.kind == "select":
        target = _choose(value, f.options)
        if target is None:
            return False
        el.select_option(label=target, timeout=5000)
        return True
    if f.kind == "checkbox":
        want = str(value).strip().lower() in ("yes", "true", "checked", "y", "1")
        if want != el.is_checked():
            el.check(force=True, timeout=3000) if want else el.uncheck(force=True, timeout=3000)
        return True
    if f.kind in ("radio", "checkgroup"):
        wanted = [v.strip() for v in str(value).split(";") if v.strip()] if f.kind == "checkgroup" else [value]
        ok = False
        for v in wanted:
            target = _choose(v, f.options)
            if target is None or not members:
                continue
            box = _loc(page, members[f.options.index(target)])
            box.check(force=True, timeout=3000)
            ok = True
        return ok
    if f.kind == "combobox":
        picked = _pick_combobox(page, f.name, value)
        if picked:
            p.value = picked           # verify against the option actually chosen
        return bool(picked)
    return False


def read(page, raw: list) -> dict:
    """{fid: {value, error}} for the scanned fields."""
    return page.evaluate(READ_JS, [{"fid": r["fid"], "kind": r["kind"], "members": r.get("members")} for r in raw])


def matches(p: Planned, got: dict) -> bool:
    """Does the read-back value show what was planned?"""
    actual = str((got or {}).get("value") or "").strip()
    if (got or {}).get("error"):
        return False
    if not actual:
        return False
    f, want = p.field, str(p.value).strip()
    if f.kind == "checkbox":
        yes = want.lower() in ("yes", "true", "checked", "y", "1")
        return (actual == "checked") == yes
    if f.kind == "file":
        return Path(planner.upload_path(p)).name in actual or actual.lower().endswith((".pdf", ".doc", ".docx", ".txt"))
    if f.kind in ("text", "textarea", "date"):
        norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
        return norm(actual) == norm(_iso_date(want) if f.kind == "date" else want) or norm(actual).startswith(norm(want)[:40])
    norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    wanted = [v.strip() for v in want.split(";") if v.strip()] if f.kind == "checkgroup" else [want]
    targets = [norm((_choose(v, f.options) if f.options else None) or v) for v in wanted]
    if f.kind in ("select", "radio", "checkgroup", "multiselect"):
        # native selects / radios read back the exact option text: the whole option must match
        got = {norm(x) for x in actual.split(";")} if f.kind in ("checkgroup", "multiselect") else {norm(actual)}
        return all(t in got for t in targets)
    a = norm(actual)                 # custom dropdowns read back their box text: the option as a whole phrase
    return all(re.search(r"(?<!\w)" + re.escape(t) + r"(?![\w,])", a) for t in targets)


def find_next(page):
    """A visible Next / Continue control that isn't a submit, or None."""
    for loc in page.locator("button, a[role=button], input[type=submit], input[type=button], [role=button]").all():
        try:
            if not loc.is_visible():
                continue
            text = (loc.inner_text() or loc.get_attribute("value") or "").strip()
        except Exception:
            continue
        if _NEXT_TEXT.match(text) and not _SUBMIT_TEXT.search(text):
            return loc
    return None


# ---------------------------------------------------------------------------
# Orchestration (agent_browser worker thread)
# ---------------------------------------------------------------------------

def _status(source: str) -> str:
    return {"needs_me": "unknown", "consent": "consent", "failed": "failed", "skipped": "skipped"}.get(source, "filled")


def wall(page) -> Optional[str]:
    from aria.features.applying import walls
    got = page.evaluate(walls.WALL_JS)
    return walls.detect(got.get("text", ""), got.get("url", ""), got.get("frames") or [], bool(got.get("agreement")))


def fill(job: dict, description: str = "", progress_fn: Optional[Callable[[str], None]] = None,
         llm_available: bool = True, run: Optional[Callable] = None,
         guess_fn: Optional[Callable] = None, draft_fn: Optional[Callable] = None,
         wait_fn: Optional[Callable[[str], bool]] = None,
         field_key: Optional[Callable[[dict], str]] = None) -> EngineResult:
    """Fill the form open in the current agent_browser tab, page by page. Never clicks a
    submit. `run(fn)` executes fn(page) on the browser thread (agent_browser.run by default).
    Never raises — a pause/stop gate's Stopped is a BaseException and passes through."""
    if run is None:
        import aria.web.agent_browser as agent_browser
        run = agent_browser.run
    say = progress_fn or (lambda _m: None)
    result = EngineResult()
    resume = planner.resume_path()
    profile = planner.build_profile(job, resume)
    company, role = job.get("company", ""), job.get("role", "")
    kw = {k: v for k, v in (("guess_fn", guess_fn), ("draft_fn", draft_fn)) if v is not None}
    all_plans: list = []
    try:
        for page_no in range(1, MAX_PAGES + 1):
            result.pages = page_no
            raw_by_fid: dict = {}
            plans_here: list = []
            blocked = ""
            for _round in range(MAX_ROUNDS + 2):
                kind = run(wall)
                if kind:
                    if wait_fn is not None and wait_fn(kind):
                        time.sleep(1.0)
                        continue
                    blocked = kind
                    break
                raw = run(scan)
                new = [r for r in raw if r["fid"] not in raw_by_fid]
                for r in raw:
                    raw_by_fid[r["fid"]] = r
                fields = to_fields(new, field_key)
                if not fields:
                    break
                say(f"Page {page_no}: {len(fields)} new field{'s' if len(fields) != 1 else ''}.")
                form = ParsedForm(fields, title=role, company=company, description=description)
                plans = planner.plan(form, job, profile, resume=resume, progress_fn=progress_fn,
                                     llm_available=llm_available, **kw)
                plans_here.extend(plans)
                for p in plans:
                    if p.value and p.source in FILLED_SOURCES:
                        members = raw_by_fid[p.field.name].get("members")
                        try:
                            ok = run(lambda page, p=p, m=members: fill_one(page, p, m))
                        except Exception as exc:
                            logger.warning("form_engine: fill %s failed: %s", p.field.label[:60], exc)
                            ok = False
                        if not ok:
                            p.note = "couldn't set it"
                time.sleep(0.6)       # let conditional fields render
            if blocked:
                from aria.features.applying import walls
                result.stopped_at = f"blocked by {walls.NAMES.get(blocked, blocked)}"
            if not plans_here and page_no == 1 and not blocked:
                result.stopped_at = "no fields"
                return result

            # verify, with retries
            todo = [p for p in plans_here if p.value and p.source in FILLED_SOURCES]
            raws = [raw_by_fid[p.field.name] for p in todo]
            got = run(lambda page: read(page, raws)) if raws else {}
            for _retry in range(MAX_RETRIES):
                wrong = [p for p in todo if (got.get(p.field.name) or {}).get("value") is not None
                         and not matches(p, got.get(p.field.name))]
                if not wrong:
                    break
                for p in wrong:
                    members = raw_by_fid[p.field.name].get("members")
                    try:
                        run(lambda page, p=p, m=members: fill_one(page, p, m))
                    except Exception as exc:
                        logger.warning("form_engine: retry %s failed: %s", p.field.label[:60], exc)
                time.sleep(0.5)
                got = run(lambda page: read(page, raws))
            for p in todo:
                g = got.get(p.field.name) or {}
                if g.get("value") is None:          # the page re-rendered without it (a gate, a reload)
                    p.source, p.note = "skipped", "no longer on the page"
                elif not matches(p, g):
                    p.note = g.get("error") or p.note or "didn't stick after 2 retries"
                    p.source = "failed" if p.field.required else "skipped"
            all_plans.extend((page_no, p, (got.get(p.field.name) or {}).get("value")) for p in plans_here)

            failed = [p for p in plans_here if p.source == "failed"]
            unknown = [p for p in plans_here if p.source == "needs_me"]
            if blocked:
                break
            if failed or unknown:
                result.stopped_at = "failed fields" if failed else "needs you"
                break
            nxt = run(lambda page: find_next(page) is not None)
            if not nxt:
                result.stopped_at = "final page"
                break
            say(f"Page {page_no} verified. Going to the next page.")
            before = run(lambda page: page.url)
            run(lambda page: find_next(page).click(timeout=5000))
            time.sleep(2.0)
            after_raw = run(scan)
            if run(lambda page: page.url) == before and all(r["fid"] in raw_by_fid for r in after_raw):
                result.stopped_at = "next did nothing (a validation message is likely showing)"
                break
        else:
            result.stopped_at = "page limit"
        result.form_url = run(lambda page: page.url)
    except Exception as exc:
        logger.error("form_engine.fill failed: %s", exc)
        result.error = f"The form fill failed: {exc}"

    for page_no, p, actual in all_plans:
        f = p.field
        value = actual if actual else (p.value if p.source in FILLED_SOURCES else "")
        if f.kind == "file" and p.value and p.source in FILLED_SOURCES:
            value = Path(planner.upload_path(p)).name
        result.review.append({"name": f.name, "label": f.label, "kind": f.kind, "category": p.category,
                              "value": value, "source": p.source, "required": f.required, "note": p.note,
                              "page": page_no, "status": _status(p.source)})
        result.filled[f.name] = {"label": f.label, "kind": f.kind, "value": value, "source": p.source}
    return result
