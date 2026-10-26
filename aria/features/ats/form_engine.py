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
  6. next      click a visible Next / Continue (never Submit / Apply / Send) and repeat on the
               new page — also with a field left open (it stays on the review list); only a
               page that refuses to move stops the run there
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
from aria.features.ats import blocks, planner
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
  // The nearest label-like element before `el` inside its closest ancestors (up to maxDepth).
  function nearLabelEl(el, maxDepth) {
    let node = el;
    for (let depth = 0; depth < (maxDepth || 7) && node.parentElement; depth++) {
      node = node.parentElement;
      if (node === document.body) break;
      for (const sel of [LABELISH, 'p, span, div']) {
        const cands = Array.from(node.querySelectorAll(sel)).filter(c => before(c, el) && usable(c, el) &&
          (sel === LABELISH || c.children.length === 0));
        if (cands.length) return cands[cands.length - 1];
      }
    }
    return null;
  }
  function nearLabel(el, maxDepth) {
    const n = nearLabelEl(el, maxDepth);
    return n ? clean(n.innerText || n.textContent) : '';
  }
  // Required shown by styling, not text: a "required" class on the label (Ashby's _required_x)
  // or an asterisk drawn by CSS (::after / ::before) — innerText never includes either.
  function marked(n) {
    if (!n || n.nodeType !== 1) return false;
    const c = typeof n.className === 'string' ? n.className : '';
    if (/(?:^|[\s_-])required/i.test(c) && !/optional|not-required/i.test(c)) return true;
    try {
      const drawn = (getComputedStyle(n, '::after').content || '') + (getComputedStyle(n, '::before').content || '');
      if (/[*✱]/.test(drawn)) return true;
    } catch (e) {}
    return false;
  }
  function labelFor(el) {
    const key = el.id || el.name;
    if (!key) return null;
    try { return document.querySelector('label[for="' + CSS.escape(key) + '"]'); } catch (e) { return null; }
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
    return !!(el.required || el.getAttribute('aria-required') === 'true' || /[*✱]/.test(raw || '') || /\(required\)/i.test(raw || '')
      || marked(el.labels && el.labels[0]) || marked(labelFor(el)) || marked(nearLabelEl(el, 4)));
  }
  const strip = t => clean((t || '').replace(/[*✱]+/g, ' ').replace(/\((required|optional)\)/ig, ' '));
  // Workday-style stable ids on the control or its wrapper — adapters map them to bank keys
  const aidOf = el => { const n = el.closest('[data-automation-id]'); return n ? n.getAttribute('data-automation-id') : ''; };
  // Workday's repeating blocks (workExperience-22--jobTitle, education-40--degree, webAddress-19--url)
  const fkitOf = el => { const n = el.closest('[data-fkit-id]'); return n ? n.getAttribute('data-fkit-id') : ''; };
  const sectionOf = el => fkitOf(el).split('-')[0];
  // a picker's own option list (Workday prompts draw every option as a radio) is not a question
  const PICKER = '[data-automation-id="activeListContainer"], [data-automation-id="promptOption"], [data-automation-id="promptLeafNode"], [data-automation-id="selectedItemList"]';
  const out = [];
  const groups = {};

  for (const el of document.querySelectorAll('input, textarea, select, [role=combobox], button[aria-haspopup="listbox"]')) {
    const tag = el.tagName.toLowerCase();
    const listButton = tag === 'button';                        // Workday's single-select ("Degree: Select One")
    const type = listButton ? 'text' : (el.getAttribute('type') || (tag === 'input' ? 'text' : tag)).toLowerCase();
    if (['hidden', 'submit', 'button', 'image', 'reset', 'password'].includes(type)) continue;
    if (el.closest(PICKER)) continue;
    if (el.disabled || el.readOnly && type !== 'file' && el.getAttribute('role') !== 'combobox') continue;
    if (el.closest('[aria-hidden="true"]') && type !== 'file') continue;
    if (el.closest('[class*="autofill" i]')) continue;      // "autofill from resume" drop zones aren't questions
    if (type === 'radio' || type === 'checkbox') {
      // Ashby-style Yes / No buttons keep a hidden checkbox beside them: the buttons are the field
      if (el.parentElement && el.parentElement.querySelectorAll('button[aria-pressed], button[data-option]').length >= 2) continue;
      // checkboxes named q-1, q-2 … (or q[1], q[2]) are one question
      const stem = type === 'checkbox' ? (el.name || '').replace(/(?:[-_]\d+|\[\d*\])$/, '') : el.name;
      // …but never across Workday's repeating blocks: every job's "I currently work here" box is
      // named currentlyWorkHere (2026-10-25: Meijer's 4 jobs became one 4-box question)
      const block = fkitOf(el).split('--')[0];
      const key = type + ':' + (stem || fid(el)) + (block ? '@' + block : '');
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
      el.getAttribute('aria-haspopup') === 'listbox' || el.getAttribute('data-automation-id') === 'searchBox' ||
      !!el.closest('[data-automation-id="multiselectInputContainer"], [data-uxi-widget-type="multiselect"], [data-uxi-widget-type="selectinput"]');
    if ((combo || listButton) && tag !== 'select') kind = 'combobox';
    let options = [];
    if (tag === 'select') options = Array.from(el.options).filter(o => o.value !== '' && clean(o.text) && !GENERIC.test(clean(o.text)))
      .map(o => clean(o.text)).filter(Boolean);
    out.push({fid: fid(el), kind: kind, label: strip(raw), required: req(el, raw), options: options,
              max_chars: el.maxLength > 0 && el.maxLength < 100000 ? el.maxLength : 0,
              accept: el.getAttribute('accept') || '', name: el.name || el.id || '', aid: aidOf(el), section: sectionOf(el), fkit: fkitOf(el)});
  }
  // Choices drawn as buttons (Ashby's Yes / No): one field, each button an option.
  const bgroups = new Map();
  for (const b of document.querySelectorAll('button[aria-pressed], button[data-option]')) {
    if (!vis(b) || b.closest('nav, header, [role=toolbar]')) continue;
    const g = bgroups.get(b.parentElement) || [];
    g.push(b);
    bgroups.set(b.parentElement, g);
  }
  for (const [box, btns] of bgroups) {
    if (btns.length < 2) continue;
    const hidden = box.querySelector('input[type=checkbox], input[type=radio]');
    const lab = (hidden && labelFor(hidden)) || nearLabelEl(btns[0], 4);
    const raw = lab ? clean(lab.innerText || lab.textContent) : '';
    if (!raw) continue;
    btns.forEach(b => fid(b));
    out.push({fid: fid(btns[0]), kind: 'buttons', label: strip(raw), required: marked(lab) || /[*✱]/.test(raw),
              options: btns.map(b => clean(b.innerText || b.getAttribute('aria-label') || '')), max_chars: 0, accept: '',
              name: (hidden && hidden.name) || '', aid: aidOf(btns[0]), members: btns.map(b => b.dataset.ariaFid)});
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
    if (!raw && fs) raw = ownLabel(fs);                       // <fieldset aria-label="…"> (RippleMatch)
    if (!raw) raw = nearLabel(first);
    if (type === 'checkbox' && shown.length === 1) {
      const own = optionText(first);
      raw = own && own.length > 12 ? own : clean((raw ? raw + ' ' : '') + own);
      out.push({fid: fid(first), kind: 'checkbox', label: strip(raw), required: req(first, raw) || req(first, own),
                options: [], max_chars: 0, accept: '', name: first.name || '', aid: aidOf(first), section: sectionOf(first), fkit: fkitOf(first)});
      continue;
    }
    shown.forEach(e => fid(e));
    out.push({fid: fid(first), kind: type === 'radio' ? 'radio' : 'checkgroup', label: strip(raw),
              required: shown.some(e => req(e, raw)), options: shown.map(optionText), max_chars: 0, accept: '',
              name: first.name || '', aid: aidOf(first), section: sectionOf(first), fkit: fkitOf(first), members: shown.map(e => e.dataset.ariaFid)});
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
    if (f.kind === 'buttons') {
      const on = (f.members || [f.fid]).map(by).filter(e => e && (e.getAttribute('aria-pressed') === 'true' ||
        /(?:^|[\s_-])(?:selected|active|checked|pressed)(?:$|[\s_-])/i.test(e.className || '')));
      value = on.map(e => clean(e.innerText)).join('; ');
    } else if (f.kind === 'radio' || f.kind === 'checkgroup') {
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
    } else if (f.kind === 'combobox' && el.closest('[data-automation-id="multiSelectContainer"]')) {
      value = Array.from(el.closest('[data-automation-id="multiSelectContainer"]').querySelectorAll('[data-automation-id="selectedItem"]'))
        .map(e => clean((e.getAttribute('aria-label') || e.innerText || '').replace(/, press delete to clear value\.?$/i, ''))).join('; ');
    } else if (f.kind === 'combobox' && el.tagName === 'BUTTON') {
      const t = clean(el.innerText);
      value = /^select one$|^select$|^choose/i.test(t) ? '' : t;
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
# Buttons drawn as custom elements (UKG / UltiPro's <ukg-button>, data-tag-name="button"):
# marked so a plain locator can find them.
MARK_CUSTOM_BUTTONS_JS = """() => { let n = 0; document.querySelectorAll('*').forEach(e => {
  if (/-BUTTON$/.test(e.tagName) || e.getAttribute('data-tag-name') === 'button') { e.setAttribute('data-aria-btn', '1'); n++; } });
  return n; }"""
_PROMPT_JS = """e => !e.form && (e.getAttribute('data-automation-id') === 'searchBox' ||
  !!e.closest('[data-automation-id="multiselectInputContainer"], [data-uxi-widget-type="multiselect"], [data-uxi-widget-type="selectinput"]'))"""
_STEP_JS = """() => { const s = document.querySelector('[data-automation-id="progressBarActiveStep"], [aria-current="step"]');
  return s ? (s.innerText || '').trim().slice(0, 120) : ''; }"""
_PLAIN_OPTION_SEL = 'li, [role="menuitem"], [role="menuitemradio"]'
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


def _open(page, el) -> None:
    """Open a custom dropdown. React-select draws its placeholder over the input (it takes the
    click) and keeps a 1px "dummy input" for unsearchable ones (outside the viewport to
    Playwright): then focus it and open it from the keyboard."""
    try:
        el.click(timeout=2000)
        return
    except Exception:
        pass
    try:
        box = el.locator("xpath=ancestor::*[contains(@class, 'control')][1]")
        if box.count():
            box.first.click(timeout=2000)
            return
    except Exception:
        pass
    el.focus(timeout=2000)
    page.keyboard.press("ArrowDown")


def _shown_options(page, opts) -> list:
    """Indexes of the options that are visible now and weren't before the dropdown opened."""
    try:
        return [i for i in range(opts.count())
                if opts.nth(i).is_visible() and opts.nth(i).get_attribute("data-aria-pre") is None]
    except Exception:
        return []


def _wait_options(page, opts, wait_ms: int) -> list:
    deadline = time.time() + wait_ms / 1000.0
    while True:
        shown = _shown_options(page, opts)
        if shown or time.time() >= deadline:
            break
        time.sleep(0.2)
    time.sleep(0.3)
    return _shown_options(page, opts)


def _pick_combobox(page, fid: str, value: str, wait_ms: int = 4000,
                   chooser: Optional[Callable[[str, list], Optional[str]]] = None) -> Optional[str]:
    """Custom dropdown / typeahead: open, type, wait for options, click the match. A search
    box that lists nothing until Enter (Workday's prompts) gets Enter; nothing matching the
    typed text → the whole list, unfiltered; a category that opens a second list (Workday:
    "Recruiter" → "Recruiter contacted me") → the match in that list too.
    Returns the option clicked, or None."""
    choose = chooser or _choose
    el = _loc(page, fid)
    # Only this dropdown's options: the listbox it controls, else options that appeared on
    # opening it (anything already visible belongs to another question).
    page.evaluate(_MARK_PRE_JS, _OPTION_SEL + ", " + _PLAIN_OPTION_SEL)
    _open(page, el)
    typed = re.split(r"[,(]", value)[0].strip()[:40]
    searchable = prompt = False
    try:
        searchable = bool(el.evaluate("e => e.tagName === 'INPUT' && !e.readOnly && e.getAttribute('inputmode') !== 'none'"))
        # Enter only ever goes to a search box outside any <form> (Workday's prompts): in a form
        # field, Enter SUBMITS the form
        prompt = searchable and bool(el.evaluate(_PROMPT_JS))
        if searchable:
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
    visible = _wait_options(page, opts, wait_ms if not prompt else min(wait_ms, 1500))
    if not visible and prompt:
        el.press("Enter")
        visible = _wait_options(page, opts, wait_ms)
    if not visible and not owned:          # options drawn as plain list items next to the box
        opts = page.locator(_PLAIN_OPTION_SEL)
        visible = _wait_options(page, opts, 1000)
    texts = [opts.nth(i).inner_text().strip() for i in visible]
    target = choose(value, texts) if texts else None
    if target is None and texts:
        from aria.features.ats.greenhouse import match_location
        target = match_location(value, texts)
    if target is None and prompt and typed:
        # the typed words matched nothing useful: the full list, unfiltered
        try:
            el.fill("")
            el.press("Enter")
        except Exception:
            pass
        visible = _wait_options(page, opts, 1500)
        texts = [opts.nth(i).inner_text().strip() for i in visible]
        target = choose(value, texts) if texts else None
    if target is None:
        page.evaluate("() => document.querySelectorAll('[data-aria-pre]').forEach(e => e.removeAttribute('data-aria-pre'))")
        page.keyboard.press("Escape")
        return None
    opts.nth(visible[texts.index(target)]).click(timeout=3000)
    # a category that opened its own list: options now shown that weren't there before the click
    time.sleep(0.6)
    before = set(texts)
    again = _shown_options(page, opts)
    sub = [opts.nth(i).inner_text().strip() for i in again]
    if sub and not (set(sub) & before):
        inner = choose(value, sub) or (sub[0] if len(sub) == 1 else None)
        if inner is not None:
            opts.nth(again[sub.index(inner)]).click(timeout=3000)
            target = inner
    page.evaluate("() => document.querySelectorAll('[data-aria-pre]').forEach(e => e.removeAttribute('data-aria-pre'))")
    return target


_DEGREE_LABEL = re.compile(r"\bdegree\b|\blevel of education\b|\bhighest education\b", re.I)


def _chooser(f: FormField) -> Optional[Callable[[str, list], Optional[str]]]:
    """Options seen only once the dropdown is open get the bank's own matching rules."""
    if answer_bank.is_referral_label(f.label):
        return answer_bank.choose_referral
    if _DEGREE_LABEL.search(f.label or ""):           # "Master of Science, CS" → "Master's Degree"
        # never a fuzzy pick: "Master of Science" fuzzy-matched Meijer's "MA" (2026-10-25)
        return lambda value, options: answer_bank.choose_degree(value, options) or (value if value in options else None)
    return None


def _tick(page, box, on: bool = True) -> None:
    """Check / uncheck a radio or checkbox. Visually-hidden inputs (RippleMatch's 1px radios
    "outside of the viewport") get their label clicked, else a JS click — the page's own handlers run."""
    try:
        box.check(force=True, timeout=3000) if on else box.uncheck(force=True, timeout=3000)
        return
    except Exception:
        pass
    if box.is_checked() == on:
        return
    label = box.locator("xpath=ancestor::label[1]")
    if not label.count():
        bid = box.get_attribute("id") or ""
        label = page.locator(f'label[for="{bid}"]') if bid else label
    try:
        if label.count():
            label.first.click(timeout=3000)
            if box.is_checked() == on:
                return
    except Exception:
        pass
    box.evaluate("e => e.click()")


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
    if f.kind == "text" and el.get_attribute("role") == "spinbutton":
        # Workday's date boxes (Month / Year): React ignores .fill(); typed digits are what it reads
        el.click(timeout=3000)
        el.press("Backspace")
        el.press("Backspace")
        el.press_sequentially(value, delay=40)
        el.evaluate("e => e.dispatchEvent(new Event('blur', {bubbles: true}))")
        return True
    if f.kind in ("text", "textarea"):
        el.fill(value, timeout=5000)
        el.evaluate("e => e.dispatchEvent(new Event('blur', {bubbles: true}))")
        return True
    if f.kind == "date":
        el.fill(_iso_date(value), timeout=5000)
        return True
    if f.kind == "select":
        target = (_chooser(f) or _choose)(value, f.options)
        if target is None:
            return False
        el.select_option(label=target, timeout=5000)
        p.value = target                   # verify against the option actually chosen
        return True
    if f.kind == "checkbox":
        want = str(value).strip().lower() in ("yes", "true", "checked", "y", "1")
        if want != el.is_checked():
            _tick(page, el, want)
        return True
    if f.kind == "buttons":
        target = _choose(value, f.options)
        if target is None or not members:
            return False
        _loc(page, members[f.options.index(target)]).click(timeout=3000)
        return True
    if f.kind in ("radio", "checkgroup"):
        wanted = [v.strip() for v in str(value).split(";") if v.strip()] if f.kind == "checkgroup" else [value]
        ok = False
        for v in wanted:
            target = _choose(v, f.options)
            if target is None or not members:
                continue
            _tick(page, _loc(page, members[f.options.index(target)]))
            ok = True
        return ok
    if f.kind == "combobox":
        picked = _pick_combobox(page, f.name, value, chooser=_chooser(f))
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
    f, want = p.field, str(p.value).strip()
    if f.kind == "checkbox":                 # unticked is an answer too ("I currently work here": No)
        yes = want.lower() in ("yes", "true", "checked", "y", "1")
        return (actual == "checked") == yes
    if not actual:
        return False
    if re.fullmatch(r"\d{1,4}", actual) and re.fullmatch(r"\d{1,4}", want):
        return int(actual) == int(want)       # Workday's month box reads back "1" for "01"
    if f.kind == "file":
        return Path(planner.upload_path(p)).name in actual or actual.lower().endswith((".pdf", ".doc", ".docx", ".txt"))
    if f.kind in ("text", "textarea", "date"):
        norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
        return norm(actual) == norm(_iso_date(want) if f.kind == "date" else want) or norm(actual).startswith(norm(want)[:40])
    norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    wanted = [v.strip() for v in want.split(";") if v.strip()] if f.kind == "checkgroup" else [want]
    targets = [norm((_choose(v, f.options) if f.options else None) or v) for v in wanted]
    if f.kind in ("select", "radio", "checkgroup", "multiselect", "buttons"):
        # native selects / radios read back the exact option text: the whole option must match
        got = {norm(x) for x in actual.split(";")} if f.kind in ("checkgroup", "multiselect") else {norm(actual)}
        return all(t in got for t in targets)
    a = norm(actual)                 # custom dropdowns read back their box text: the option as a whole phrase
    if all(re.search(r"(?<!\w)" + re.escape(t) + r"(?![\w,])", a) for t in targets):
        return True
    # an answer with punctuation in it reads back with other punctuation ("University at Buffalo,
    # SUNY" → "University at Buffalo - SUNY"); a bare "No" never matches "No, I do not …"
    loose = lambda t: re.sub(r"\s+", " ", answer_bank.normalize(t)).strip()
    return all(loose(t) != t and re.search(r"(?<!\w)" + re.escape(loose(t)) + r"(?!\w)", loose(actual)) for t in targets)


def find_next(page):
    """A visible Next / Continue control that isn't a submit, or None."""
    try:
        page.evaluate(MARK_CUSTOM_BUTTONS_JS)
    except Exception:
        pass
    for loc in page.locator("button, a[role=button], input[type=submit], input[type=button], [role=button], "
                            "[data-aria-btn]").all():
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

# Repeating blocks the answer bank knows nothing about: one per job / website on your resume.
# What the page holds there (Workday's autofill from your resume) stays; the bank never
# overwrites it (2026-10-24: Meijer's work dates got your graduation year).
PAGE_OWNED = {"workExperience", "webAddress", "language", "certification", "award"}
# ...and inside education blocks, the dates are the block's own (the bank's grad year isn't "From").
_EDU_DATE = re.compile(r"^(?:month|year|from|to|start|end)\b", re.I)
_PLACEHOLDER = re.compile(r"^(?:n/?a|none|null|-+|select(?: one)?|choose(?: one)?|please select|expanded|collapsed)$", re.I)


def keep_page_values(plans: list, raw_by_fid: dict, current: dict) -> None:
    """A field that already holds an answer — from an ATS's resume autofill or your saved profile
    there — keeps it when the bank has none (never asked again, never re-typed), and keeps it
    always inside the repeating blocks above. Consent boxes and files are never kept this way."""
    for p in plans:
        raw = raw_by_fid.get(p.field.name) or {}
        if p.category in ("consent", "upload") or p.field.kind in ("file", "checkbox"):
            continue
        value = str((current.get(p.field.name) or {}).get("value") or "").strip()
        has = bool(value) and not _PLACEHOLDER.match(value)
        section = raw.get("section") or ""
        owned = section in PAGE_OWNED or (section == "education" and _EDU_DATE.match(p.field.label or ""))
        if owned and p.source in FILLED_SOURCES:
            if has:
                p.value, p.source, p.note = value, "page", "from the page (your resume's autofill) — the bank doesn't overwrite it"
            elif p.field.required:
                p.value, p.source, p.note = "", "needs_me", "empty in this block — yours to fill"
            else:
                p.value, p.source, p.note = "", "skipped", ""
        elif has and p.source in ("needs_me", "skipped"):
            p.value, p.source, p.note = value, "page", "already on the page"


def _status(source: str) -> str:
    return {"needs_me": "unknown", "consent": "consent", "failed": "failed", "skipped": "skipped"}.get(source, "filled")


def signature(page) -> tuple:
    """What page of the form this is: URL, the wizard's current step, the field labels. A Next
    that leaves all three the same didn't move (Workday re-draws the same page — new elements,
    same questions — when it refuses to go on)."""
    try:
        step = page.evaluate(_STEP_JS)
    except Exception:
        step = ""
    return page.url, step, tuple(sorted(r.get("label") or "" for r in scan(page)))


def _settled_signature(run: Callable, max_secs: float = 10.0) -> tuple:
    """The page's signature once it has fields and stopped changing (a wizard's spinner between steps)."""
    deadline = time.time() + max_secs
    last = run(signature)
    while time.time() < deadline:
        time.sleep(0.8)
        now = run(signature)
        if now == last and now[2]:
            return now
        last = now
    return last


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
    resume = planner.resume_for(job, description, job.get("role", ""), llm=llm_available)
    profile = planner.build_profile(job, resume)
    company, role = job.get("company", ""), job.get("role", "")
    kw = {k: v for k, v in (("guess_fn", guess_fn), ("draft_fn", draft_fn)) if v is not None}
    lists = {name: answer_bank.get_list(name) for name in set(blocks.SECTION_LIST.values())}
    all_plans: list = []
    try:
        for page_no in range(1, MAX_PAGES + 1):
            result.pages = page_no
            raw_by_fid: dict = {}
            before: dict = {}             # what each field held when first seen
            plans_here: list = []
            blocked = ""
            # repeating blocks (Workday's jobs / schools): exactly as many as your lists have
            synced = [run(lambda page, n=name, k=len(entries): blocks.sync(page, n, k))
                      for name, entries in lists.items() if entries]
            if any(c is not None for c in synced):
                time.sleep(0.8)
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
                current = run(lambda page, new=new: read(page, new))
                before.update(current)
                keep_page_values(plans, raw_by_fid, current)
                blocks.assign(plans, raw_by_fid, lists)
                plans_here.extend(plans)
                for p in plans:
                    if p.value and p.source in FILLED_SOURCES:
                        if matches(p, current.get(p.field.name)):
                            continue           # already says it (an autofill, a saved profile): never re-typed
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
                shown = str(g.get("value") or "").strip()
                if g.get("value") is None:          # the page re-rendered without it (a gate, a reload)
                    p.source, p.note = "skipped", "no longer on the page"
                elif not matches(p, g) and shown and not g.get("error") and not _PLACEHOLDER.match(shown) \
                        and p.field.kind in ("combobox", "select") and (before.get(p.field.name) or {}).get("value"):
                    # couldn't pick the bank's option, but the page already had an answer there: keep it
                    p.value, p.source, p.note = shown, "page", f"kept what the page had (couldn't pick “{p.value}”)"
                elif not matches(p, g):
                    p.note = g.get("error") or p.note or "didn't stick after 2 retries"
                    p.source = "failed" if p.field.required else "skipped"
            all_plans.extend((page_no, p, (got.get(p.field.name) or {}).get("value")) for p in plans_here)

            failed = [p for p in plans_here if p.source == "failed"]
            unknown = [p for p in plans_here if p.source == "needs_me" and p.field.required]
            if blocked:
                break
            nxt = run(lambda page: find_next(page) is not None)
            if not nxt:
                result.stopped_at = "final page"
                break
            # A field left open never stops the run by itself (user 2026-10-25: go all the way to
            # the last Submit): Next is tried anyway — only a page that refuses to move stops it.
            say(f"Page {page_no} done. Going to the next page.")
            before = run(signature)
            run(lambda page: find_next(page).click(timeout=5000))
            time.sleep(1.5)
            if _settled_signature(run) == before:
                result.stopped_at = ("failed fields" if failed else "needs you") if failed or unknown \
                    else "next did nothing (a validation message is likely showing)"
                break
        else:
            result.stopped_at = "page limit"
        result.form_url = run(lambda page: page.url)
    except Exception as exc:
        logger.error("form_engine.fill failed: %s", exc)
        result.error = f"The form fill failed: {exc}"

    for page_no, p, actual in all_plans:
        f = p.field
        value = actual if actual else (p.value if p.source in FILLED_SOURCES or p.source == "page" else "")
        if f.kind == "file" and p.value and p.source in FILLED_SOURCES:
            value = Path(planner.upload_path(p)).name
        result.review.append({"name": f.name, "label": f.label, "kind": f.kind, "category": p.category,
                              "value": value, "source": p.source, "required": f.required, "note": p.note,
                              "page": page_no, "status": _status(p.source)})
        result.filled[f.name] = {"label": f.label, "kind": f.kind, "value": value, "source": p.source}
    return result
