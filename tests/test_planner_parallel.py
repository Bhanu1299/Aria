"""
tests/test_planner_parallel.py — TDD tests for parallel planner execution.

Tests for:
  - _classify_dependencies(): partitions steps into batches
  - _execute_batch(): runs steps concurrently via ThreadPoolExecutor
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from planner import _classify_dependencies, _execute_batch
from plan_context import PlanContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(
    id: int,
    intent_type: str,
    depends_on: list[int] | None = None,
    result_key: str | None = None,
) -> dict:
    """Build a minimal valid step dict for testing."""
    return {
        "id": id,
        "description": f"Step {id} description",
        "intent_type": intent_type,
        "params": {"query": f"query {id}"},
        "result_key": result_key or f"result_{id}",
        "depends_on": depends_on if depends_on is not None else [],
    }


# ---------------------------------------------------------------------------
# Test 1: independent steps detected as one parallel batch
# ---------------------------------------------------------------------------

def test_independent_steps_detected():
    """
    Two steps both with intent_type='web_search' and depends_on=[]
    should end up in the same parallel batch (one batch of length 2).
    """
    step1 = _make_step(1, "web_search", depends_on=[])
    step2 = _make_step(2, "web_search", depends_on=[])

    batches = _classify_dependencies([step1, step2])

    assert len(batches) == 1, f"Expected 1 batch, got {len(batches)}"
    assert len(batches[0]) == 2, f"Expected both steps in same batch, got {len(batches[0])}"
    assert step1 in batches[0]
    assert step2 in batches[0]


# ---------------------------------------------------------------------------
# Test 2: dependent / serial steps run in separate batches
# ---------------------------------------------------------------------------

def test_dependent_steps_run_serially():
    """
    Step 1 = web_search (no deps) → parallel-eligible.
    Step 2 = browser_task with depends_on=[1] → serial (not in _PARALLEL_INTENTS).
    They must end up in separate batches.
    """
    step1 = _make_step(1, "web_search", depends_on=[])
    step2 = _make_step(2, "browser_task", depends_on=[1])

    batches = _classify_dependencies([step1, step2])

    assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"
    assert batches[0] == [step1]
    assert batches[1] == [step2]


# ---------------------------------------------------------------------------
# Test 3: parallel batch collects results from all steps
# ---------------------------------------------------------------------------

def test_parallel_batch_results_collected():
    """
    Mock handle_intent_fn returns a string immediately.
    _execute_batch() with 2 parallel steps should return a dict
    with both result_keys populated.
    """
    step1 = _make_step(1, "web_search", result_key="r1")
    step2 = _make_step(2, "knowledge", result_key="r2")

    def fake_handle(intent, description):
        return f"answer for {intent['type']}"

    ctx = PlanContext(goal="test", steps=[step1, step2])

    results = _execute_batch([step1, step2], ctx, fake_handle)

    assert "r1" in results, "result_key r1 missing from batch results"
    assert "r2" in results, "result_key r2 missing from batch results"
    assert results["r1"] != "", "r1 should have a non-empty result"
    assert results["r2"] != "", "r2 should have a non-empty result"


# ---------------------------------------------------------------------------
# Test 4: single-step plan produces [[step]] (no parallel overhead)
# ---------------------------------------------------------------------------

def test_single_step_plan_not_parallelized():
    """
    A single step should produce exactly [[step]] — one batch, one step.
    No ThreadPoolExecutor overhead (batch of 1 takes the fast path).
    """
    step = _make_step(1, "web_search", depends_on=[])

    batches = _classify_dependencies([step])

    assert batches == [[step]], f"Expected [[step]], got {batches}"
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert batches[0][0] is step
