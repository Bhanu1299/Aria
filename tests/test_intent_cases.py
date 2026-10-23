"""tests/test_intent_cases.py — the real-utterance set (tests/data/intent_cases.json), offline half:
exact-phrase cases are claimed by their pre-handler; agent cases are never stolen by a fast path.
The agent's own choices are measured live by scripts/manual_intent_eval.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aria.dispatch.parse import normalize
from aria.dispatch.rules import RuleMatch, RuleSet
from aria.features.applying import commands as c

CASES = json.loads((Path(__file__).parent / "data" / "intent_cases.json").read_text())["cases"]

_PRE = {
    "dashboard": lambda q: c._DASHBOARD.fullmatch(q),
    "by_hand": lambda q: c._BY_HAND_IT.fullmatch(q) or c._BY_HAND_CO.fullmatch(q),
    "submit": lambda q: c._SUBMIT_IT.fullmatch(q) or c._SUBMIT_CO.fullmatch(q),
    "mark": lambda q: c._MARK.search(q),
}
_ANY_PRE = (c._DASHBOARD, c._START_LEAD, c._DONE, c._CLOSE_ALL, c._SUBMIT_IT, c._BY_HAND_IT,
            c._SKIP_IT, c._CLOSE_IT)


@pytest.mark.parametrize("case", [x for x in CASES if x["layer"] == "pre"], ids=lambda x: x["said"])
def test_pre_handler_claims_it(case):
    assert _PRE[case["handler"]](normalize(case["said"]))


@pytest.mark.parametrize("case", [x for x in CASES if x["layer"] == "agent"], ids=lambda x: x["said"])
def test_no_fast_path_steals_an_agent_case(case):
    q = normalize(case["said"])
    assert not any(rx.fullmatch(q) for rx in _ANY_PRE)
    assert not isinstance(RuleSet.load().match(case["said"]), RuleMatch)


def test_cases_are_well_formed():
    for x in CASES:
        assert x["layer"] in ("pre", "agent") and x["said"].strip()
        if x["layer"] == "pre":
            assert x["handler"] in _PRE
        else:
            assert x["tools"] and isinstance(x.get("args", {}), dict)
