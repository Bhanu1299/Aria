"""adapters/ashby.py — Ashby: the posting jobs.ashbyhq.com/<company>/<id> opens on its Overview
tab; the form is <id>/application. System fields carry stable names (_systemfield_name …)."""
from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from aria.features.ats import apply_target
from aria.features.ats.adapters import Adapter, ApplyContext
from aria.features.ats.apply_target import Tab


def form_url(url: str) -> str:
    """jobs.ashbyhq.com/acme/<id> → jobs.ashbyhq.com/acme/<id>/application."""
    u = urlparse(url)
    parts = [p for p in u.path.split("/") if p]
    if len(parts) == 2:
        return urlunparse(u._replace(path="/" + "/".join(parts + ["application"]), fragment=""))
    return url


class AshbyAdapter(Adapter):
    name = "ashby"
    FIELD_KEYS = (
        (r"_systemfield_name", "full_name"), (r"_systemfield_email", "email"),
        (r"_systemfield_phone(?:_number)?", "phone"), (r"_systemfield_location", "location"),
        (r"_systemfield_linkedin(?:_url)?", "linkedin"), (r"_systemfield_github(?:_url)?", "github"),
    )

    def prepare(self, tab: Tab, ctx: ApplyContext) -> None:
        want = form_url(ctx.target.url)
        if want != ctx.target.url and tab.run(lambda page: page.evaluate(apply_target.FIELDS_JS)) < apply_target.INLINE_FIELDS:
            ctx.progress_fn("Opening the Ashby application tab.")
            tab.navigate(want, settle_secs=1.0)
            ctx.target.url = apply_target.settle(tab)
