"""features/ats/adapters — one small class per applicant system, all with the same four steps.

    detect(tab, url, html) -> bool          is this page served by this ATS
    login(tab, ctx)        -> bool          True = past sign-in (or none needed); False = handed to you
    fill(tab, ctx)         -> FillReport    prepare() (the ATS's own steps), then the form engine
    verify(tab, ctx)       -> ReadbackReport  what's on the page now, field by field

Adapter itself IS the generic adapter: the shared field-label dictionary (answer bank →
planner) through form_engine, with the vision loop only when no fields are found. An ATS
adapter overrides prepare() / login() and keeps everything else. pick(name) → its adapter,
the generic one for anything unknown. Adapters take a Tab (apply_target.Tab), never a raw
Playwright page: every page call is one short job on the browser worker, so a wall can be
waited out without freezing other tabs. Nothing here clicks Submit.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from aria.features.ats.apply_target import ApplyTarget, Tab
from aria.features.ats.detect import identify

logger = logging.getLogger(__name__)


@dataclass
class ApplyContext:
    job: dict                                       # the sheet row (url, company, role, row, notes)
    target: ApplyTarget
    progress_fn: Callable[[str], None] = lambda _m: None
    wait_fn: Optional[Callable[[str], bool]] = None  # a wall → True once the user cleared it
    report: Optional["FillReport"] = None           # set by fill(), read by verify()


@dataclass
class FillReport:
    rows: list = field(default_factory=list)        # review rows: name, label, kind, value, source, status, page
    filled: dict = field(default_factory=dict)      # name → {label, kind, value, source}
    pages: int = 1
    stopped_at: str = ""
    form_url: str = ""
    error: str = ""
    title: str = ""
    company: str = ""
    description: str = ""

    @property
    def unknown(self) -> list:
        return [r.get("label") or r.get("name", "") for r in self.rows if r.get("source") == "needs_me"]


@dataclass
class ReadbackReport:
    ok: bool
    rows: list                                      # {label, value, status} — what the page holds
    mismatches: list                                # labels that didn't stick


class Adapter:
    name = "generic"
    # (pattern on the field's name / data-automation-id, bank key): fields the ATS names the
    # same way on every form are filled by key, whatever their label says
    FIELD_KEYS: tuple = ()
    # No fields found → the see-think-act vision loop tries. Off where the page structure is
    # known (Workday's Review page has no fields by design).
    VISION_FALLBACK = True

    def field_key(self, raw: dict) -> str:
        for ident in (raw.get("aid") or "", raw.get("name") or ""):
            for pattern, key in self.FIELD_KEYS:
                if ident and re.fullmatch(pattern, ident, re.I):
                    return key
        return ""

    def detect(self, tab: Tab, url: str, html: str) -> bool:
        return identify(url, html) == self.name

    def login(self, tab: Tab, ctx: ApplyContext) -> bool:
        return True

    def prepare(self, tab: Tab, ctx: ApplyContext) -> None:
        """ATS-specific steps between the resolved page and the form (none here)."""

    def fill(self, tab: Tab, ctx: ApplyContext) -> FillReport:
        import aria.web.agent_browser as agent_browser
        from aria.features import ats
        from aria.features.ats import form_engine
        from aria.features.ats.greenhouse import _gateway_up

        self.prepare(tab, ctx)
        llm_up = _gateway_up()
        eng = form_engine.fill(ctx.job, ctx.target.posting_text, progress_fn=ctx.progress_fn,
                               llm_available=llm_up, run=tab.run, wait_fn=ctx.wait_fn, field_key=self.field_key)
        report = FillReport(rows=eng.review, filled=eng.filled, pages=eng.pages, stopped_at=eng.stopped_at,
                            form_url=eng.form_url or tab.run(lambda page: page.url), error=eng.error)
        if eng.stopped_at == "no fields" and not eng.error and self.VISION_FALLBACK:
            with agent_browser.use_tab(tab.id, gate=agent_browser.current_gate()):
                status, report.rows = ats._vision_fallback(ctx.job, ctx.progress_fn, agent_browser)
            report.filled, report.stopped_at = {}, "vision fallback"
            if status not in ("confirm", "needs_input"):
                report.error = f"No form fields found, and the browser agent stopped early ({status}). Check the form."
        elif not llm_up:
            report.error = report.error or "The LLM gateway was down, so open-text answers weren't written."
        ctx.report = report
        return report

    def verify(self, tab: Tab, ctx: ApplyContext) -> ReadbackReport:
        """The fill already read every value back from the page (2 refills each); this
        reports it. Values are the page's, not the plan's."""
        rows = [{"label": r.get("label") or r.get("name", ""), "value": r.get("value", ""),
                 "status": r.get("status") or ("unknown" if r.get("source") == "needs_me" else "filled")}
                for r in (ctx.report.rows if ctx.report else [])]
        bad = [r["label"] for r in rows if r["status"] == "failed"]
        return ReadbackReport(ok=bool(rows) and not bad, rows=rows, mismatches=bad)


def _registry() -> dict:
    from aria.features.ats.adapters.ashby import AshbyAdapter
    from aria.features.ats.adapters.greenhouse import GreenhouseAdapter
    from aria.features.ats.adapters.handshake import HandshakeAdapter
    from aria.features.ats.adapters.icims import IcimsAdapter
    from aria.features.ats.adapters.lever import LeverAdapter
    from aria.features.ats.adapters.workday import WorkdayAdapter
    return {a.name: a for a in (GreenhouseAdapter(), LeverAdapter(), AshbyAdapter(), IcimsAdapter(),
                                HandshakeAdapter(), WorkdayAdapter(), Adapter())}


def pick(name: str) -> Adapter:
    reg = _registry()
    return reg.get(name) or reg["generic"]
