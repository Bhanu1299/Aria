import os
import sqlite3
import tempfile
import pytest

# Patch DB_PATH before importing db so tests don't touch ~/.aria/aria.db
@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    db_file = str(tmp_path / "test_aria.db")
    import db
    monkeypatch.setattr(db, "DB_PATH", db_file)
    yield db_file

def test_get_connection_creates_file():
    import db
    conn = db.get_connection()
    conn.close()
    assert os.path.exists(db.DB_PATH)

def test_applications_table_exists():
    import db
    conn = db.get_connection()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "applications" in tables

def test_memory_table_exists():
    import db
    conn = db.get_connection()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "memory" in tables

def test_get_connection_idempotent():
    import db
    conn1 = db.get_connection()
    conn1.close()
    conn2 = db.get_connection()
    conn2.close()
    # Should not raise on repeated calls
