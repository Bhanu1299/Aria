"""Tests for websearch.py — DuckDuckGo HTML search, no browser required."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aria.web.websearch as websearch

_FIXTURE = """
<html><body>
<div class="result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=abc">Example Title One</a>
  <a class="result__snippet" href="#">First snippet about the query.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://direct.example.org/x">Second Result</a>
  <a class="result__snippet" href="#">Second snippet, with <b>bold</b> text.</a>
</div>
</body></html>
"""


def _fake_response(text=_FIXTURE, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


def test_search_parses_titles_snippets_and_decoded_urls():
    with patch.object(websearch.requests, "post", return_value=_fake_response()):
        results = websearch.search("test query")
    assert len(results) == 2
    assert results[0]["title"] == "Example Title One"
    assert results[0]["url"] == "https://example.com/page"      # uddg redirect decoded
    assert results[0]["snippet"] == "First snippet about the query."
    assert results[1]["url"] == "https://direct.example.org/x"  # direct link kept
    assert results[1]["snippet"] == "Second snippet, with bold text."  # tags stripped


def test_search_returns_empty_on_http_error():
    with patch.object(websearch.requests, "post", return_value=_fake_response(status=403)):
        assert websearch.search("q") == []


def test_search_never_raises_on_network_failure():
    with patch.object(websearch.requests, "post", side_effect=OSError("offline")):
        assert websearch.search("q") == []


def test_search_respects_max_results():
    many = _FIXTURE * 5  # 10 results
    with patch.object(websearch.requests, "post", return_value=_fake_response(text=many)):
        assert len(websearch.search("q", max_results=3)) == 3


def test_snippets_text_block_for_summarizer():
    with patch.object(websearch.requests, "post", return_value=_fake_response()):
        block = websearch.snippets_text("test query")
    assert "Example Title One" in block
    assert "First snippet about the query." in block


def test_snippets_text_empty_when_no_results():
    with patch.object(websearch.requests, "post", return_value=_fake_response(text="<html></html>")):
        assert websearch.snippets_text("q") == ""
