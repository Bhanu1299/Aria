"""
Tests for plan_context.py and memory plan persistence.
"""
import json
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


def test_plan_context_defaults():
    from plan_context import PlanContext

    ctx = PlanContext(
        goal="find flights and book",
        steps=[
            {
                "id": 1,
                "description": "find",
                "intent_type": "browser_task",
                "params": {},
                "result_key": "r1",
                "depends_on": [],
            }
        ],
    )
    assert ctx.results == {}
    assert ctx.current_step == 0
    assert ctx.retry_count == 0


def test_plan_context_to_dict():
    from plan_context import PlanContext

    ctx = PlanContext(
        goal="test",
        steps=[],
        results={"r1": "some result"},
        current_step=2,
        retry_count=1,
    )
    d = ctx.to_dict()
    assert d["goal"] == "test"
    assert d["results"] == {"r1": "some result"}
    assert d["current_step"] == 2
    assert d["retry_count"] == 1


def test_plan_context_from_dict():
    from plan_context import PlanContext

    orig_dict = {
        "goal": "test goal",
        "steps": [{"id": 1, "description": "step 1"}],
        "results": {"r1": "result 1"},
        "current_step": 1,
        "retry_count": 2,
    }
    ctx = PlanContext.from_dict(orig_dict)
    assert ctx.goal == "test goal"
    assert ctx.steps == [{"id": 1, "description": "step 1"}]
    assert ctx.results == {"r1": "result 1"}
    assert ctx.current_step == 1
    assert ctx.retry_count == 2


def test_memory_store_and_get_last_plan(tmp_path, monkeypatch):
    """Test that PlanContext can be stored to memory and retrieved."""
    import db
    import memory
    from plan_context import PlanContext

    # Point db at temp file
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    importlib.reload(memory)

    ctx = PlanContext(
        goal="g", steps=[], results={"k": "v"}, current_step=1, retry_count=0
    )
    memory.store_last_plan(ctx.to_dict())
    loaded = memory.get_last_plan()
    assert loaded is not None
    assert loaded["goal"] == "g"
    assert loaded["results"] == {"k": "v"}
    assert loaded["current_step"] == 1

    # Simulate restart: reload memory module to clear in-memory session and re-read from DB
    importlib.reload(memory)
    loaded_after_restart = memory.get_last_plan()
    assert loaded_after_restart is not None
    assert loaded_after_restart["goal"] == "g"
    assert loaded_after_restart["results"] == {"k": "v"}
