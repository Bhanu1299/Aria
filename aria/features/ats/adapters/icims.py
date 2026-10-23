"""adapters/icims.py — iCIMS: the portal renders inside #icims_content_iframe on the company's
own page; the form is opened at the frame's URL (in_iframe=1) so filling never crosses a frame.
Its sign-in page is a login wall: the form engine waits for you there (credentials: phase 4)."""
from __future__ import annotations

from aria.features.ats import apply_target
from aria.features.ats.adapters import Adapter, ApplyContext
from aria.features.ats.apply_target import Tab

FRAME_JS = """() => { const f = document.querySelector('#icims_content_iframe, iframe[src*="icims.com"]');
  return f ? (f.src || '') : ''; }"""


class IcimsAdapter(Adapter):
    name = "icims"

    def prepare(self, tab: Tab, ctx: ApplyContext) -> None:
        src = tab.run(lambda page: page.evaluate(FRAME_JS))
        if src:
            ctx.progress_fn("Opening the iCIMS form outside its frame.")
            tab.navigate(src, settle_secs=1.0)
            ctx.target.url = apply_target.settle(tab)
