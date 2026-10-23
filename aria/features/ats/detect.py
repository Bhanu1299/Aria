"""features/ats/detect.py — which applicant-tracking system serves a job URL.

identify(url, html)  → "greenhouse" | "lever" | "ashby" | "workday" | "icims" | "handshake" | … | None
                       (by host, else by marks the ATS leaves in a company page; recognition only)
detect(url, html)    → the ATS name only when it fills straight from the job URL (ADAPTERS), else None
domain(url)          → the domain an unknown ATS is logged under
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


# host suffix → ATS. Named ones without an adapter still go to the generic form engine,
# but aren't logged as unknown.
_HOSTS = (
    ("greenhouse.io", "greenhouse"), ("lever.co", "lever"), ("ashbyhq.com", "ashby"),
    ("myworkdayjobs.com", "workday"), ("myworkdaysite.com", "workday"), ("workday.com", "workday"),
    ("icims.com", "icims"), ("joinhandshake.com", "handshake"), ("jobvite.com", "jobvite"),
    ("smartrecruiters.com", "smartrecruiters"), ("workable.com", "workable"), ("bamboohr.com", "bamboohr"),
    ("applytojob.com", "jazzhr"), ("recruitee.com", "recruitee"), ("breezy.hr", "breezy"),
    ("taleo.net", "taleo"), ("successfactors.com", "successfactors"), ("oraclecloud.com", "oracle"),
)
# What an ATS leaves in a page it serves under a company's own domain.
_DOM_MARKS = (
    ("workday", re.compile(r"data-automation-id=[\"'](?:jobPostingHeader|adventureButton|applyManually|"
                           r"legalNameSection_firstName|bottom-navigation-next-button)", re.I)),
    ("icims", re.compile(r"icims_content_iframe|iCIMS_MainWrapper", re.I)),
    ("ashby", re.compile(r"ashby_embed|jobs\.ashbyhq\.com/[\w-]+/embed", re.I)),
)


def _host_ats(host: str) -> Optional[str]:
    for suffix, name in _HOSTS:
        if host == suffix or host.endswith("." + suffix):
            return name
    return None


def identify(url: str, html: str = "") -> Optional[str]:
    """The ATS behind a URL (by host), else by what it embeds in the page; None if unknown."""
    if _gh_jid(url):
        return "greenhouse"
    name = _host_ats(_host(url))
    if name:
        return name
    if html and (_IFRAME_SRC.search(html) or "grnhse_iframe" in html or "grnhse_app" in html):
        return "greenhouse"
    for name, mark in _DOM_MARKS:
        if html and mark.search(html):
            return name
    return None


def domain(url: str) -> str:
    """Registrable-ish domain for the unknown-ATS log: careers.acme.co.uk → acme.co.uk."""
    parts = _host(url).split(".")
    keep = 3 if len(parts) > 2 and len(parts[-1]) == 2 and parts[-2] in {"co", "com", "ac", "org", "gov"} else 2
    return ".".join(parts[-keep:])


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
