"""adapters/lever.py — Lever: the posting is jobs.lever.co/<company>/<id>, the form is the same
URL + /apply. Field names are Lever's own on every form (name, email, phone, org, location,
urls[LinkedIn] …), so they're filled by key; custom cards[…] questions go through labels."""
from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from aria.features.ats import apply_target
from aria.features.ats.adapters import Adapter, ApplyContext
from aria.features.ats.apply_target import Tab


def form_url(url: str) -> str:
    """jobs.lever.co/acme/123 → jobs.lever.co/acme/123/apply (unchanged if already there)."""
    u = urlparse(url)
    parts = [p for p in u.path.split("/") if p]
    if len(parts) == 2:
        return urlunparse(u._replace(path="/" + "/".join(parts + ["apply"]), query="", fragment=""))
    return url


class LeverAdapter(Adapter):
    name = "lever"
    FIELD_KEYS = (
        (r"name", "full_name"), (r"email", "email"), (r"phone", "phone"), (r"org", "current_company"),
        (r"location", "location"), (r"urls\[linkedin\]", "linkedin"), (r"urls\[github\]", "github"),
    )

    def prepare(self, tab: Tab, ctx: ApplyContext) -> None:
        want = form_url(ctx.target.url)
        if want != ctx.target.url and tab.run(lambda page: page.evaluate(apply_target.FIELDS_JS)) < apply_target.INLINE_FIELDS:
            ctx.progress_fn("Opening the Lever application form.")
            tab.navigate(want, settle_secs=1.0)
            ctx.target.url = apply_target.settle(tab)
