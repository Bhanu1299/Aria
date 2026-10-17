"""web_search tool: Google-SERP fallback when DuckDuckGo throttles (HTTP 202)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from aria.plugins.core import CorePlugin

_SERP = (
    "Web results\nRegal Elmwood Center Movie Tickets and Showtimes\n"
    "Jailer 2 (Tamil) 5:45pm 8:35pm Fri Oct 16"
)


def _tool(browser):
    return CorePlugin(browser=browser)._web_search_tool()


def _run(browser, params):
    with patch("aria.web.websearch.snippets_text", return_value="") as ddg, \
         patch("aria.features.summarizer.summarize",
               side_effect=lambda page_text, query, instructions: f"SUMMARY<{page_text}>") as summ:
        out = _tool(browser).execute(params)
    return out, ddg, summ


def test_ddg_empty_uses_google_serp_text_even_with_no_links():
    browser = MagicMock()
    browser.fetch.return_value = _SERP
    browser.extract_links.return_value = []
    out, _, _ = _run(browser, {"query": "Jailer 2 showtimes Buffalo NY"})
    assert "5:45pm" in out
    browser.fetch.assert_called_once()
    assert "google.com/search" in browser.fetch.call_args[0][0]


def test_serp_unusable_falls_back_to_result_links():
    browser = MagicMock()
    browser.fetch.side_effect = [None, "page body about weather"]
    browser.extract_links.return_value = ["https://weather.example/buffalo"]
    out, _, _ = _run(browser, {"query": "weather"})
    assert "page body about weather" in out


def test_nothing_found_returns_graceful_message():
    browser = MagicMock()
    browser.fetch.return_value = None
    browser.extract_links.return_value = []
    out, _, _ = _run(browser, {"query": "weather"})
    assert out == "I couldn't find a good result for that search."


def test_location_not_appended_when_city_already_in_query():
    browser = MagicMock()
    browser.fetch.return_value = _SERP
    with patch("aria.core.config.CURRENT_LOCATION", "Buffalo, New York"):
        _, ddg, _ = _run(browser, {"query": "weather Buffalo NY", "location_sensitive": "true"})
    assert ddg.call_args[0][0] == "weather Buffalo NY"


def test_location_appended_when_missing():
    browser = MagicMock()
    browser.fetch.return_value = _SERP
    with patch("aria.core.config.CURRENT_LOCATION", "Buffalo, New York"):
        _, ddg, _ = _run(browser, {"query": "weather today", "location_sensitive": "true"})
    assert ddg.call_args[0][0] == "weather today Buffalo, New York"
