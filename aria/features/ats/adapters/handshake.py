"""adapters/handshake.py — Handshake is near one-click: Apply opens a dialog with your resume
already chosen from the documents on your account.

fill: sign-in wall → wait for you; click Apply (never "Submit application"); then
  · the dialog → read which resume is selected and compare it with resume_path:
      ours → filled (no re-upload: the account's copy is preferred)
      another one → needs you ("Handshake has X selected")
      none, with an upload box → resume_path uploaded via set_input_files
    other documents the dialog requires (transcript, cover letter) → needs you
  · "Apply externally" (a new tab / the employer's site) → resolve that page and hand it to
    its own adapter
Stops there: the dialog's Submit is yours.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from aria.features.ats import apply_target, corrections, planner
from aria.features.ats.adapters import Adapter, ApplyContext, FillReport
from aria.features.ats.apply_target import Tab

logger = logging.getLogger(__name__)

_EXTERNAL = re.compile(r"externally|company|employer", re.I)

DIALOG_JS = r"""() => {
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const d = Array.from(document.querySelectorAll('[role=dialog], dialog[open], [aria-modal=true]')).filter(vis).pop();
  if (!d) return null;
  const clean = t => (t || '').replace(/\s+/g, ' ').trim();
  const picked = [];
  d.querySelectorAll('[aria-selected=true], [aria-checked=true], [class*="singleValue" i], [class*="single-value" i], [class*="selected" i]')
    .forEach(e => picked.push(clean(e.innerText || e.textContent)));
  d.querySelectorAll('input[type=radio]:checked, input[type=checkbox]:checked').forEach(e => {
    const l = (e.labels && e.labels[0]) || e.closest('label') || e.parentElement;
    picked.push(clean(l ? (l.innerText || l.textContent) : ''));
  });
  d.querySelectorAll('select').forEach(s => { if (s.selectedIndex >= 0 && s.value) picked.push(clean(s.options[s.selectedIndex].text)); });
  const file = d.querySelector('input[type=file]');
  if (file && !file.dataset.ariaUpload) file.dataset.ariaUpload = '1';
  return {text: clean(d.innerText).slice(0, 4000), picked: picked.filter(Boolean), upload: !!file};
}"""

_REQUIRED_DOC = re.compile(r"\b(transcript|cover letter|work sample|writing sample)\b[^.]{0,40}\brequired\b|"
                           r"\brequired\b[^.]{0,40}\b(transcript|cover letter|work sample|writing sample)\b", re.I)


def _norm(name: str) -> str:
    """"Alex_Resume.pdf · uploaded Oct 2" → "alexresume"."""
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"\.(?:pdf|docx?)\b.*$", "", (name or "").lower()))


def resume_choice(picked: list, resume: Optional[str]) -> tuple:
    """(status, shown): ok (our resume is selected) · wrong (another one is) · none."""
    want = _norm(Path(resume).name) if resume else ""
    for name in picked:
        got = _norm(name)
        if want and got and (want in got or got in want):
            return "ok", name
    docs = [n for n in picked if re.search(r"\.(?:pdf|docx?)\b|resume|cv\b", n, re.I)]
    return ("wrong", docs[0]) if docs else ("none", "")


def required_documents(text: str) -> list:
    """Documents the dialog says are required besides the resume."""
    found = []
    for m in _REQUIRED_DOC.finditer(text or ""):
        doc = (m.group(1) or m.group(2)).lower()
        if doc not in found:
            found.append(doc)
    return found


def _row(label: str, value: str, source: str, note: str = "", kind: str = "file") -> dict:
    status = {"needs_me": "unknown", "failed": "failed"}.get(source, "filled")
    return {"name": label.lower().replace(" ", "_"), "label": label, "kind": kind, "category": "upload",
            "value": value, "source": source, "required": True, "note": note, "page": 1, "status": status}


class HandshakeAdapter(Adapter):
    name = "handshake"

    def _walls_cleared(self, tab: Tab, ctx: ApplyContext) -> bool:
        for _ in range(3):
            kind = apply_target._wall(tab)
            if not kind:
                return True
            if ctx.wait_fn is None or not ctx.wait_fn(kind):
                corrections.record(ctx.job, "login_failed" if kind == "login" else kind, "login",
                                   f"Handshake: {kind} not cleared", ats=self.name, run=tab.run)
                return False
            apply_target.settle(tab)
        return apply_target._wall(tab) is None

    def login(self, tab: Tab, ctx: ApplyContext) -> bool:
        return self._walls_cleared(tab, ctx)

    def _external(self, tab: Tab, ctx: ApplyContext) -> FillReport:
        from aria.features.ats import adapters
        target = apply_target.resolve_apply_target(tab, ctx.job, ctx.progress_fn, ctx.wait_fn)
        nxt = adapters.pick(target.ats)
        if nxt.name == self.name:
            return FillReport(error="Handshake's external apply link didn't lead to a form.", form_url=target.url)
        ctx.progress_fn(f"Handshake sends this one to the employer's site ({target.identified or 'its own form'}).")
        ctx.target = target
        if not nxt.login(tab, ctx):
            return FillReport(error="I couldn't get past the employer site's sign-in page.", form_url=target.url)
        return nxt.fill(tab, ctx)

    def fill(self, tab: Tab, ctx: ApplyContext) -> FillReport:
        here = tab.run(lambda page: page.url)
        info = tab.run(lambda page: page.evaluate(DIALOG_JS))
        if info is None:
            pick = apply_target.pick_affordance(tab.run(lambda page: page.evaluate(apply_target.AFFORDANCES_JS)), here)
            if pick is None:
                corrections.record(ctx.job, "apply_not_found", "fill", "no Apply button on the Handshake posting",
                                   ats=self.name, run=tab.run)
                ctx.report = FillReport(error="I couldn't find Handshake's Apply button.", form_url=here)
                return ctx.report
            if pick["action"] == "follow" or _EXTERNAL.search(pick["text"]):
                if pick["action"] == "follow":
                    tab.navigate(pick["url"], settle_secs=1.0)
                else:
                    apply_target.click_outcome(tab, pick["i"])
                ctx.report = self._external(tab, ctx)
                return ctx.report
            ctx.progress_fn("Clicking Handshake's Apply.")
            outcome = apply_target.click_outcome(tab, pick["i"])
            if outcome in ("new_tab", "same_tab") and "joinhandshake.com" not in apply_target.settle(tab):
                ctx.report = self._external(tab, ctx)
                return ctx.report
            info = tab.run(lambda page: page.evaluate(DIALOG_JS))
            if info is None:
                corrections.record(ctx.job, "apply_click_nothing", "fill", f"Handshake: “{pick['text']}” opened no dialog",
                                   ats=self.name, run=tab.run)
                ctx.report = FillReport(error="Handshake's Apply didn't open its application dialog.", form_url=here)
                return ctx.report

        resume = planner.resume_path()
        status, shown = resume_choice(info["picked"], resume)
        rows = []
        if status == "ok":
            rows.append(_row("Resume", shown, "bank", "already on your Handshake account"))
        elif status == "wrong":
            rows.append(_row("Resume", shown, "needs_me",
                             f"Handshake has “{shown}” selected — pick {Path(resume).name if resume else 'your resume'}"))
        elif info["upload"] and resume:
            try:
                tab.run(lambda page: page.locator('[data-aria-upload="1"]').first.set_input_files(resume))
                rows.append(_row("Resume", Path(resume).name, "bank", "uploaded — none was selected"))
            except Exception as exc:
                logger.warning("handshake: resume upload failed: %s", exc)
                rows.append(_row("Resume", "", "failed", f"upload failed: {exc}"))
        else:
            rows.append(_row("Resume", "", "needs_me", "no resume selected and no resume_path to upload"))
        for doc in required_documents(info["text"]):
            rows.append(_row(doc.title(), "", "needs_me", "Handshake requires it — choose one in the dialog"))
        ctx.report = FillReport(rows=rows, filled={r["name"]: {"label": r["label"], "kind": "file", "value": r["value"],
                                                               "source": r["source"]} for r in rows if r["source"] == "bank"},
                                stopped_at="Handshake dialog — Submit is yours", form_url=tab.run(lambda page: page.url))
        return ctx.report
