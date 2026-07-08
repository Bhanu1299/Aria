"""
websearch.py — fast web search via DuckDuckGo's HTML endpoint.

search(query, max_results=5) -> [{"title", "url", "snippet"}]
snippets_text(query)          -> one text block ready for the summarizer

No browser, no API key, no JavaScript — a single POST to html.duckduckgo.com.
This is the primary search path; the Playwright/Google scrape in the core
plugin is the fallback when this returns nothing.

Never raises — error paths return [] / "".
"""

from __future__ import annotations

import logging
import re
import urllib.parse

import requests

logger = logging.getLogger(__name__)

_ENDPOINT = "https://html.duckduckgo.com/html/"
_TIMEOUT = 6
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _TAG_RE.sub("", html).strip()


def _decode_url(href: str) -> str:
    """DDG wraps results in //duckduckgo.com/l/?uddg=<encoded> redirects."""
    if "uddg=" in href:
        try:
            qs = urllib.parse.urlparse(href).query
            target = urllib.parse.parse_qs(qs).get("uddg", [""])[0]
            if target:
                return target
        except Exception:
            pass
    return href


def search(query: str, max_results: int = 5) -> list:
    """Search DuckDuckGo. Returns [{"title","url","snippet"}]. Never raises."""
    try:
        resp = requests.post(
            _ENDPOINT, data={"q": query}, headers=_HEADERS, timeout=_TIMEOUT
        )
        if resp.status_code != 200:
            logger.warning("DDG search HTTP %s for %r", resp.status_code, query)
            return []
        links = _RESULT_LINK_RE.findall(resp.text)
        snippets = [_strip_tags(s) for s in _SNIPPET_RE.findall(resp.text)]
        results = []
        for i, (href, title_html) in enumerate(links[:max_results]):
            results.append({
                "title": _strip_tags(title_html),
                "url": _decode_url(href),
                "snippet": snippets[i] if i < len(snippets) else "",
            })
        return results
    except Exception as exc:
        logger.error("DDG search failed for %r: %s", query, exc)
        return []


def snippets_text(query: str, max_results: int = 5) -> str:
    """Results as one text block for the summarizer. "" when nothing found."""
    results = search(query, max_results=max_results)
    if not results:
        return ""
    parts = []
    for r in results:
        parts.append(f"{r['title']}\n{r['snippet']}\n({r['url']})")
    return "\n\n".join(parts)
