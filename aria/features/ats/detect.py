"""features/ats/detect.py — which applicant-tracking system serves a job URL.

identify(url, html)  → "greenhouse" | "lever" | "ashby" | "workday" | None   (recognition only)
detect(url, html)    → the ATS name only when Aria has a filler for it (ADAPTERS), else None
greenhouse_form_url(url, html) → the URL that renders the Greenhouse form itself

Greenhouse shows up three ways:
  job-boards.greenhouse.io/<board>/jobs/<id>      (current boards; form on the page)
  boards.greenhouse.io/<board>/jobs/<id>          (legacy; usually redirects to job-boards)
  <company site>?gh_jid=<id>                      (embedded: an iframe#grnhse_iframe whose src
                                                   is /embed/job_app?for=<board>&token=<id>)
For embeds the form is opened at the iframe's URL directly, so filling never has
to cross a frame boundary.
"""
from __future__ import annotations

import html as _html
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

ADAPTERS = {"greenhouse"}

_GREENHOUSE_EMBED = "https://job-boards.greenhouse.io/embed/job_app"
_IFRAME_SRC = re.compile(
    r"""<iframe[^>]*\bsrc=["']([^"']*greenhouse\.io/embed/job_app[^"']*)["']""", re.I
)
_EMBED_BOARD = re.compile(r"greenhouse\.io/embed/job_(?:board|app)[^\"']*?[?&]for=([\w-]+)", re.I)


def _host(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower()


def _gh_jid(url: str) -> Optional[str]:
    jid = parse_qs(urlparse(url or "").query).get("gh_jid", [""])[0]
    return jid if jid.isdigit() else None


def identify(url: str, html: str = "") -> Optional[str]:
    host = _host(url)
    if host == "greenhouse.io" or host.endswith(".greenhouse.io") or _gh_jid(url):
        return "greenhouse"
    if host == "jobs.lever.co" or host.endswith(".lever.co"):
        return "lever"
    if host == "jobs.ashbyhq.com" or host.endswith(".ashbyhq.com"):
        return "ashby"
    if host.endswith("myworkdayjobs.com") or host.endswith(".workday.com"):
        return "workday"
    if html and (_IFRAME_SRC.search(html) or "grnhse_iframe" in html or "grnhse_app" in html):
        return "greenhouse"
    return None


def detect(url: str, html: str = "") -> Optional[str]:
    name = identify(url, html)
    return name if name in ADAPTERS else None


def greenhouse_form_url(url: str, html: str = "") -> str:
    """Where the Greenhouse application form renders for this job URL."""
    if _host(url).endswith("greenhouse.io"):
        return url
    if html:
        m = _IFRAME_SRC.search(html)
        if m:
            src = _html.unescape(m.group(1))
            return "https:" + src if src.startswith("//") else src
    jid = _gh_jid(url)
    if jid:
        board = _EMBED_BOARD.search(html or "")
        suffix = f"&for={board.group(1)}" if board else ""
        return f"{_GREENHOUSE_EMBED}?token={jid}{suffix}"
    return url
