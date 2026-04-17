"""
Tests for plan_context.py and memory plan persistence.
"""
import importlib
import json
import pytest
from unittest.mock import patch, MagicMock
from plan_context import PlanContext
import planner


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


# ---------------------------------------------------------------------------
# Detection layer tests
# ---------------------------------------------------------------------------


def test_is_multi_step_conjunction_two_verbs():
    assert planner.is_multi_step("find the cheapest flight and then book it") is True


def test_is_multi_step_and_add():
    assert planner.is_multi_step("search for flights and add the best one to my calendar") is True


def test_is_multi_step_single_action():
    assert planner.is_multi_step("what is the weather today") is False


def test_is_multi_step_no_conjunction():
    assert planner.is_multi_step("search for jobs in New York") is False


def test_is_multi_step_borderline_calls_groq():
    # One conjunction, one verb — should call _groq_is_multi
    with patch("planner._groq_is_multi", return_value=True) as mock:
        result = planner.is_multi_step("find flights and then depart tomorrow")
        mock.assert_called_once()
        assert result is True


def test_validate_steps_valid():
    steps = [
        {"id": 1, "description": "d1", "intent_type": "browser_task",
         "params": {"browser_goal": "g"}, "result_key": "r1", "depends_on": []},
        {"id": 2, "description": "d2", "intent_type": "knowledge",
         "params": {"query": "q"}, "result_key": "r2", "depends_on": [1]},
    ]
    assert planner._validate_steps(steps) == steps


def test_validate_steps_too_few():
    steps = [{"id": 1, "description": "d", "intent_type": "browser_task",
              "params": {}, "result_key": "r1", "depends_on": []}]
    assert planner._validate_steps(steps) is None


def test_validate_steps_too_many():
    steps = [{"id": i, "description": "d", "intent_type": "browser_task",
              "params": {}, "result_key": f"r{i}", "depends_on": []}
             for i in range(1, 7)]  # 6 steps
    assert planner._validate_steps(steps) is None


def test_validate_steps_unknown_intent():
    steps = [
        {"id": 1, "description": "d1", "intent_type": "browser_task",
         "params": {}, "result_key": "r1", "depends_on": []},
        {"id": 2, "description": "d2", "intent_type": "INVALID_TYPE",
         "params": {}, "result_key": "r2", "depends_on": []},
    ]
    assert planner._validate_steps(steps) is None


def test_validate_steps_missing_field():
    steps = [
        {"id": 1, "description": "d1", "intent_type": "browser_task",
         "params": {}, "depends_on": []},  # missing result_key
        {"id": 2, "description": "d2", "intent_type": "knowledge",
         "params": {}, "result_key": "r2", "depends_on": []},
    ]
    assert planner._validate_steps(steps) is None


def test_validate_steps_missing_depends_on():
    steps = [
        {"id": 1, "description": "d1", "intent_type": "browser_task",
         "params": {}, "result_key": "r1"},  # missing depends_on
        {"id": 2, "description": "d2", "intent_type": "knowledge",
         "params": {}, "result_key": "r2", "depends_on": []},
    ]
    assert planner._validate_steps(steps) is None


def _make_groq_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_VALID_STEPS_JSON = json.dumps([
    {"id": 1, "description": "Search Kayak for flights",
     "intent_type": "browser_task",
     "params": {"browser_goal": "find cheap flights NYC"},
     "result_key": "flight_result", "depends_on": []},
    {"id": 2, "description": "Book the flight",
     "intent_type": "browser_task",
     "params": {"browser_goal": "book flight: {{flight_result}}"},
     "result_key": "booking_result", "depends_on": [1]},
])


def test_generate_plan_valid():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_groq_response(_VALID_STEPS_JSON)
    with patch("planner._get_client", return_value=mock_client):
        result = planner.generate_plan("find flights and book cheapest")
    assert result is not None
    assert len(result) == 2
    assert result[0]["intent_type"] == "browser_task"


def test_generate_plan_returns_none_on_invalid_json():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_groq_response("not json at all")
    with patch("planner._get_client", return_value=mock_client):
        result = planner.generate_plan("whatever")
    assert result is None


def test_generate_plan_returns_none_on_bad_intent():
    bad = json.dumps([
        {"id": 1, "description": "d", "intent_type": "BOGUS",
         "params": {}, "result_key": "r1", "depends_on": []},
        {"id": 2, "description": "d", "intent_type": "browser_task",
         "params": {}, "result_key": "r2", "depends_on": []},
    ])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_groq_response(bad)
    with patch("planner._get_client", return_value=mock_client):
        result = planner.generate_plan("whatever")
    assert result is None


def test_revise_plan_valid():
    original = json.loads(_VALID_STEPS_JSON)
    revised_json = json.dumps([
        {"id": 1, "description": "Search Google Flights",
         "intent_type": "browser_task",
         "params": {"browser_goal": "find cheap flights NYC on Google Flights"},
         "result_key": "flight_result", "depends_on": []},
        {"id": 2, "description": "Book the flight",
         "intent_type": "browser_task",
         "params": {"browser_goal": "book flight: {{flight_result}}"},
         "result_key": "booking_result", "depends_on": [1]},
    ])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_groq_response(revised_json)
    with patch("planner._get_client", return_value=mock_client):
        result = planner.revise_plan(original, "use Google Flights instead of Kayak")
    assert result is not None
    assert "Google Flights" in result[0]["description"]


def test_substitute_placeholders_replaces_key():
    params = {"browser_goal": "book flight: {{flight_result}}"}
    results = {"flight_result": "Delta $320 8am Apr 25"}
    out = planner._substitute_placeholders(params, results)
    assert out["browser_goal"] == "book flight: Delta $320 8am Apr 25"


def test_substitute_placeholders_no_match():
    params = {"browser_goal": "find hotels in NYC"}
    results = {"flight_result": "Delta $320"}
    out = planner._substitute_placeholders(params, results)
    assert out == params


def test_substitute_placeholders_empty_results():
    params = {"query": "weather NYC"}
    out = planner._substitute_placeholders(params, {})
    assert out == params


def test_step_to_intent_browser_task():
    step = {
        "id": 1, "description": "search Kayak",
        "intent_type": "browser_task",
        "params": {"browser_goal": "find flights NYC"},
        "result_key": "r1", "depends_on": [],
    }
    intent = planner._step_to_intent(step)
    assert intent["type"] == "browser_task"
    assert intent["browser_goal"] == "find flights NYC"
    assert intent["url"] == ""
    assert intent["app_name"] == ""


def test_step_to_intent_knowledge():
    step = {
        "id": 1, "description": "what is the capital of France",
        "intent_type": "knowledge",
        "params": {"query": "capital of France"},
        "result_key": "r1", "depends_on": [],
    }
    intent = planner._step_to_intent(step)
    assert intent["type"] == "knowledge"
    assert intent["query"] == "capital of France"


def test_is_failure_empty():
    assert planner._is_failure(None) is True
    assert planner._is_failure("") is True


def test_is_failure_stuck_phrase():
    assert planner._is_failure("I got stuck and couldn't complete that.") is True
    assert planner._is_failure("I ran into an error while researching.") is True


def test_is_failure_success():
    assert planner._is_failure("The cheapest flight is Delta at $320.") is False
    assert planner._is_failure("Done. Added to your calendar.") is False
