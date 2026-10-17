"""Regression: core tools must resolve their lazy (function-level) imports.

The 2026-07-07 flat-root → aria/ package refactor left `import computer_use,
notifier` inside browser_task and `import memory, jobs` inside the jobs tool.
Module-level smoke tests can't see those, so the tools failed on every call.
These tests execute the tools with heavy dependencies stubbed.
"""
from __future__ import annotations

import ast
import pathlib
from importlib.machinery import PathFinder
from unittest.mock import patch

from aria.core.tool import ToolRegistry
from aria.plugins.core import CorePlugin

_REPO = pathlib.Path(__file__).resolve().parent.parent


def _tool(name: str):
    registry = ToolRegistry()
    CorePlugin().register(registry)
    return registry.get(name)


def test_browser_task_runs_research_loop():
    with patch("aria.web.computer_use.research_loop", return_value="found 3 hotels") as loop, \
         patch("aria.ui.notifier.notify_if_slow") as notify:
        result = _tool("browser_task").execute({"goal": "cheapest hotel in Arlington"})
    assert result == "found 3 hotels"
    loop.assert_called_once()
    assert loop.call_args.kwargs["goal"] == "cheapest hotel in Arlington"
    notify.assert_called_once()


def test_jobs_tool_searches_and_caches():
    listings = [{"title": "ML Engineer", "company": "Acme"}]
    with patch("aria.state.memory.get_cached_jobs", return_value=None), \
         patch("aria.state.memory.store_jobs") as store, \
         patch("aria.state.memory.store_last_search"), \
         patch("aria.state.memory.store_cached_jobs"), \
         patch("aria.features.jobs.search_jobs", return_value=listings) as search, \
         patch("aria.features.jobs.format_spoken_results", return_value="1 job found"):
        result = _tool("jobs").execute({"query": "ml engineer"})
    assert result == "1 job found"
    search.assert_called_once_with("ml engineer")
    store.assert_called_once_with(listings)


# Optional third-party packages whose name collides with an aria module stem.
_THIRD_PARTY_SAME_NAME = {"whatsapp"}


def test_no_stale_flat_root_imports():
    """Every import of a module that used to live at repo root must use aria.*"""
    flat_names = {p.stem for p in _REPO.joinpath("aria").rglob("*.py")}
    flat_names -= {"__init__"} | _THIRD_PARTY_SAME_NAME
    offenders = []
    for path in _REPO.joinpath("aria").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            for mod in mods:
                top = mod.split(".")[0]
                # PathFinder searches sys.path only — immune to fake modules
                # other tests leave in sys.modules
                if top in flat_names and PathFinder.find_spec(top) is None:
                    offenders.append(f"{path.relative_to(_REPO)}:{node.lineno} import {mod}")
    assert not offenders, "stale pre-refactor imports:\n" + "\n".join(offenders)
