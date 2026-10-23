"""adapters/greenhouse.py — Greenhouse forms through the existing schema-driven filler
(ats/greenhouse.py: the form's own JSON schema, then planner → fill → read back)."""
from __future__ import annotations

from aria.features.ats.adapters import Adapter, ApplyContext, FillReport
from aria.features.ats.apply_target import Tab


class GreenhouseAdapter(Adapter):
    name = "greenhouse"

    def fill(self, tab: Tab, ctx: ApplyContext) -> FillReport:
        from aria.features.ats import greenhouse
        job = dict(ctx.job, url=ctx.target.url or ctx.job.get("url", ""))
        gh = greenhouse.fill(job, ctx.progress_fn, tab=tab.id)
        ctx.report = FillReport(rows=gh.review, filled=gh.filled, form_url=gh.form_url, error=gh.error,
                                title=gh.title, company=gh.company, description=gh.description)
        return ctx.report
