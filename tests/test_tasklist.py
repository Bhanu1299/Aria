"""tests/test_tasklist.py — TaskList + update_tasks tool unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.core.tasklist import TaskList, make_task_tool


def test_set_tasks_and_render():
    tl = TaskList()
    tl.set_tasks([
        {"content": "open browser", "status": "completed"},
        {"content": "search flights", "status": "in_progress"},
        {"content": "report price", "status": "pending"},
    ])
    rendered = tl.render()
    assert "1. [completed] open browser" in rendered
    assert "2. [in_progress] search flights" in rendered
    assert "3. [pending] report price" in rendered


def test_unfinished_returns_non_completed():
    tl = TaskList()
    tl.set_tasks([
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ])
    unfinished = tl.unfinished()
    assert [t["content"] for t in unfinished] == ["b", "c"]
    assert not tl.is_complete()


def test_is_complete_when_all_done_or_empty():
    tl = TaskList()
    assert tl.is_complete()  # empty list counts as complete
    tl.set_tasks([{"content": "a", "status": "completed"}])
    assert tl.is_complete()


def test_invalid_status_rejected_list_unchanged():
    tl = TaskList()
    tl.set_tasks([{"content": "a", "status": "pending"}])
    tool = make_task_tool(tl)
    result = tool.execute({"tasks": [{"content": "b", "status": "doing"}]})
    assert "invalid" in result.lower()
    assert [t["content"] for t in tl.tasks] == ["a"]


def test_tool_execute_replaces_list_and_returns_rendered():
    tl = TaskList()
    tool = make_task_tool(tl)
    assert tool.name == "update_tasks"
    result = tool.execute({"tasks": [
        {"content": "find the file", "status": "in_progress"},
        {"content": "summarize it", "status": "pending"},
    ]})
    assert "find the file" in result
    assert "summarize it" in result
    assert len(tl.tasks) == 2


def test_clear_empties_list():
    tl = TaskList()
    tl.set_tasks([{"content": "a", "status": "pending"}])
    tl.clear()
    assert tl.tasks == []
    assert tl.is_complete()
