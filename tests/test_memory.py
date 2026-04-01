"""Tests for memory.py session state."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import memory


def setup_function():
    """Reset session before each test."""
    memory.session.clear()


def test_store_and_get_by_index():
    jobs = [
        {"index": 1, "title": "SWE", "company": "Stripe"},
        {"index": 2, "title": "Backend", "company": "Plaid"},
    ]
    memory.store_jobs(jobs)
    assert memory.get_job_by_index(1) == jobs[0]
    assert memory.get_job_by_index(2) == jobs[1]


def test_get_job_out_of_bounds_returns_none():
    memory.store_jobs([{"index": 1, "title": "SWE", "company": "Stripe"}])
    assert memory.get_job_by_index(0) is None
    assert memory.get_job_by_index(2) is None
    assert memory.get_job_by_index(-1) is None


def test_get_job_empty_session_returns_none():
    assert memory.get_job_by_index(1) is None


def test_store_jobs_overwrites_previous():
    memory.store_jobs([{"index": 1, "title": "Old"}])
    memory.store_jobs([{"index": 1, "title": "New"}])
    assert memory.get_job_by_index(1)["title"] == "New"
