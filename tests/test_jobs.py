"""Tests for jobs.py — pure functions and mockable Groq paths."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock
import json


# ── format_spoken_results ────────────────────────────────────────────────────

def test_format_spoken_results_empty():
    import jobs
    result = jobs.format_spoken_results([])
    assert result == "I couldn't find any job listings. Try rephrasing your search."


def test_format_spoken_results_single():
    import jobs
    r = jobs.format_spoken_results([
        {"index": 1, "title": "Software Engineer", "company": "Stripe",
         "location": "New York, NY", "posted": "2 days ago", "platform": "LinkedIn"}
    ])
    assert r.startswith("Found 1 job.")
    assert "First" in r
    assert "Stripe" in r
    assert "2 days ago" in r


def test_format_spoken_results_five():
    import jobs
    results = [
        {"index": i + 1, "title": f"Job {i+1}", "company": f"Co {i+1}",
         "location": "Remote", "posted": "today", "platform": "LinkedIn"}
        for i in range(5)
    ]
    spoken = jobs.format_spoken_results(results)
    assert spoken.startswith("Found 5 jobs.")
    for ordinal in ["First", "Second", "Third", "Fourth", "Fifth"]:
        assert ordinal in spoken


def test_format_spoken_results_missing_location_and_posted():
    import jobs
    r = jobs.format_spoken_results([
        {"index": 1, "title": "SWE", "company": "Acme",
         "location": "", "posted": "", "platform": "LinkedIn"}
    ])
    assert "SWE at Acme" in r
    # No trailing comma/empty fields
    assert ", ." not in r


# ── _parse_query ─────────────────────────────────────────────────────────────

def test_parse_query_success():
    import jobs
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"role": "software engineer", "location": "New York"}'

    with patch.object(jobs, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        role, location = jobs._parse_query("find me software engineer jobs in New York")

    assert role == "software engineer"
    assert location == "New York"


def test_parse_query_groq_failure_falls_back_to_raw_query():
    import jobs
    with patch.object(jobs, "_get_client", side_effect=RuntimeError("API down")):
        role, location = jobs._parse_query("find backend jobs remote")

    assert role == "find backend jobs remote"
    assert location == ""


def test_parse_query_bad_json_falls_back():
    import jobs
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "not json"

    with patch.object(jobs, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        role, location = jobs._parse_query("some query")

    assert role == "some query"
    assert location == ""


# ── search_jobs deduplication ─────────────────────────────────────────────────

def test_search_jobs_deduplicates():
    import jobs
    duplicated = [
        {"title": "SWE", "company": "Stripe", "location": "NYC",
         "posted": "today", "url": "https://li.com/1", "platform": "LinkedIn"},
        {"title": "SWE", "company": "Stripe", "location": "NYC",
         "posted": "today", "url": "https://ind.com/1", "platform": "Indeed"},
        {"title": "Backend", "company": "Plaid", "location": "Remote",
         "posted": "1 day ago", "url": "https://li.com/2", "platform": "LinkedIn"},
    ]
    with patch.object(jobs, "_parse_query", return_value=("SWE", "NYC")), \
         patch.object(jobs, "_get_search_url", return_value="https://example.com"), \
         patch.object(jobs, "_extract_jobs_from_page", side_effect=[
             duplicated[:2], duplicated[2:]
         ]):
        results = jobs.search_jobs("find me SWE jobs in NYC")

    titles_companies = [(r["title"], r["company"]) for r in results]
    assert ("SWE", "Stripe") in titles_companies
    # Duplicate should be removed
    assert sum(1 for t, c in titles_companies if t == "SWE" and c == "Stripe") == 1


def test_search_jobs_returns_empty_when_all_sources_fail():
    import jobs
    with patch.object(jobs, "_parse_query", return_value=("SWE", "NYC")), \
         patch.object(jobs, "_get_search_url", return_value=None):
        results = jobs.search_jobs("find me SWE jobs")

    assert results == []


# ── _vision_ask error handling ───────────────────────────────────────────────

def test_vision_ask_returns_empty_string_on_groq_failure():
    import jobs
    with patch.object(jobs, "_get_client", side_effect=RuntimeError("API down")):
        result = jobs._vision_ask("fake_b64", "extract something")

    assert result == ""
