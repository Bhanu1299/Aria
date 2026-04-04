"""
Tests for memory.py persistence and thread safety.
Uses a temp DB so tests never touch ~/.aria/aria.db.
"""
import json
import threading
import importlib
import pytest


@pytest.fixture(autouse=True)
def isolated_memory(monkeypatch, tmp_path):
    """Patch db.DB_PATH and reload memory module fresh for each test."""
    import db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test_aria.db"))
    # Force memory to reload so _load_from_db() runs against the temp DB
    import memory
    importlib.reload(memory)
    yield
    importlib.reload(memory)  # clean up after test


def test_store_and_retrieve_jobs():
    import memory
    jobs = [{"title": "SWE", "company": "Acme", "url": "https://example.com"}]
    memory.store_jobs(jobs)
    assert memory.get_job_by_index(1)["title"] == "SWE"
    assert memory.get_job_by_index(2) is None


def test_store_last_search_and_get():
    import memory
    memory.store_last_search("SWE jobs in New York")
    assert memory.get_last_search() == "SWE jobs in New York"


def test_persistent_memory_survives_reload(tmp_path, monkeypatch):
    """Simulate a restart: store, reload module, confirm value is still there."""
    import db
    db_file = str(tmp_path / "persist_test.db")
    monkeypatch.setattr(db, "DB_PATH", db_file)

    import memory
    importlib.reload(memory)
    memory.store_last_search("backend engineer remote")

    # Simulate restart by reloading the module
    importlib.reload(memory)
    assert memory.get_last_search() == "backend engineer remote"


def test_jobs_expire_after_24h(tmp_path, monkeypatch):
    """Jobs written with past expiry should not load on restart."""
    import db
    db_file = str(tmp_path / "expire_test.db")
    monkeypatch.setattr(db, "DB_PATH", db_file)

    import memory
    importlib.reload(memory)

    # Manually write an expired job row
    conn = db.get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO memory (key, value, updated_at, expires_at) VALUES (?,?,datetime('now'),?)",
        ("last_jobs", json.dumps([{"title": "Old Job"}]), "2020-01-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    # Reload — expired row should not load
    importlib.reload(memory)
    assert memory.get_job_by_index(1) is None


def test_thread_safety():
    """Multiple threads writing simultaneously must not corrupt the session dict."""
    import memory
    errors = []

    def writer(i):
        try:
            memory.store_last_search(f"query {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    # Final state should be a valid query string (some thread won the race)
    result = memory.get_last_search()
    assert result.startswith("query ")
