"""tests/test_llm_tasks.py — TASK_TIERS is the only place tiers are chosen; small → frontier escalation."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aria.llm import meter
from aria.llm.base import LLMResponse
from aria.llm.client import LLMClient
from aria.llm.tasks import FRONTIER, SMALL, TASK_TIERS, VISION, tier_for

ARIA = Path(__file__).parent.parent / "aria"


def _sources():
    for path in ARIA.rglob("*.py"):
        if "llm" in path.relative_to(ARIA).parts[:1]:
            continue
        yield path, path.read_text()


def test_every_task_maps_to_a_tier_or_explicit_model():
    for task, tier in TASK_TIERS.items():
        assert tier in (SMALL, FRONTIER, VISION) or re.match(r"^[a-z0-9][\w./-]+$", tier), task


def test_every_task_named_in_code_exists():
    used = set()
    for _path, src in _sources():
        used |= set(re.findall(r'task="([\w-]+)"', src))
        used |= set(re.findall(r'tier_for\("([\w-]+)"\)', src))
        used |= set(re.findall(r'complete_checked\(\s*"([\w-]+)"', src))
    missing = used - set(TASK_TIERS)
    assert not missing, f"tasks used but not in TASK_TIERS: {missing}"
    assert {"agent", "job_query_parse"} <= used


def test_no_call_site_hard_codes_a_tier():
    offenders = []
    for path, src in _sources():
        for m in re.finditer(r'\btier\s*=\s*"(\w+)"|\bmodel\s*=\s*"(small|frontier|vision|smart|cheap|fast)"', src):
            offenders.append(f"{path.relative_to(ARIA)}: {m.group(0)}")
    assert not offenders, "use task=... / tier_for(...) from aria/llm/tasks.py:\n" + "\n".join(offenders)


def test_unknown_task_is_an_error():
    with pytest.raises(KeyError):
        tier_for("not_a_task")


def _resp(text, stop="end_turn"):
    return LLMResponse(text=text, tool_calls=[], stop_reason=stop, provider_used="t", model_used="t")


def _client(small_text, frontier_text):
    small, frontier = MagicMock(), MagicMock()
    small.complete.return_value = _resp(small_text)
    frontier.complete.return_value = _resp(frontier_text)
    return LLMClient({"small": small, "frontier": frontier, "smart": frontier}), small, frontier


def _is_yes_no(text):
    return text if text in ("Yes", "No") else None


def test_task_routes_to_its_tier():
    client, small, frontier = _client("ok", "ok")
    client.complete([{"role": "user", "content": "x"}], task="job_query_parse")
    assert small.complete.called and not frontier.complete.called


def test_valid_small_output_is_not_escalated():
    client, small, frontier = _client("Yes", "No")
    before = meter.snapshot()
    _resp_, value = client.complete_checked("form_short_answer", [{"role": "user", "content": "q"}], _is_yes_no)
    assert value == "Yes"
    assert not frontier.complete.called
    assert meter.since(before)["escalations"] == 0


def test_invalid_small_output_retries_once_on_frontier():
    client, small, frontier = _client("Maybe?", "No")
    before = meter.snapshot()
    _resp_, value = client.complete_checked("form_short_answer", [{"role": "user", "content": "q"}], _is_yes_no)
    assert value == "No"
    assert small.complete.call_count == 1 and frontier.complete.call_count == 1
    assert meter.since(before)["escalations"] == 1


def test_frontier_task_is_never_escalated_further():
    client, small, frontier = _client("x", "still invalid")
    _resp_, value = client.complete_checked("form_free_text", [{"role": "user", "content": "q"}], _is_yes_no)
    assert value is None and frontier.complete.call_count == 1 and not small.complete.called


def test_gateway_error_counts_as_invalid():
    client, small, frontier = _client("Yes", "No")
    small.complete.return_value = _resp("I can't reach the LLM gateway.", stop="error")
    _resp_, value = client.complete_checked("form_short_answer", [{"role": "user", "content": "q"}], _is_yes_no)
    assert value == "No"


def test_validator_exception_is_invalid():
    client, small, frontier = _client("Yes", "No")

    def boom(_text):
        raise ValueError("bad")
    _resp_, value = client.complete_checked("form_short_answer", [{"role": "user", "content": "q"}], boom)
    assert value is None and frontier.complete.call_count == 1


def test_escalations_reach_the_flight_log(tmp_path, monkeypatch):
    import json
    import aria.observability.flight_recorder as fr
    log = tmp_path / "flight.jsonl"
    monkeypatch.setattr(fr, "_LOG_PATH", str(log))
    fr.record("q", "a", 1.0, usage={"cost_usd": 0.001, "tokens_in": 1, "tokens_out": 1, "escalations": 2})
    assert json.loads(log.read_text())["usage"]["escalations"] == 2
