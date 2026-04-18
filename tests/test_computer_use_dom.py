from unittest.mock import patch, MagicMock
import computer_use

SAMPLE_SNAPSHOT = """URL: https://www.amazon.com/dp/B0D1XD1ZV3
TITLE: Apple AirPods Pro - Amazon

INTERACTIVE[3]:
[0] BUTTON   #add-to-cart-button              "Add to Cart"
[1] BUTTON   #buy-now-button                  "Buy Now"
[2] INPUT    [name="field-keywords"]          ""

PAGE TEXT:
Apple AirPods Pro $249.00 In Stock"""


def _mock_groq_response(content: str):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = content
    return mock_client


def test_dom_decide_returns_click_with_selector():
    mock_client = _mock_groq_response(
        '{"action": "click", "selector": "#add-to-cart-button", "reason": "add to cart"}'
    )
    with patch("computer_use._get_client", return_value=mock_client):
        result = computer_use._dom_decide(
            snapshot=SAMPLE_SNAPSHOT,
            goal="add AirPods Pro to cart",
            context_data={},
            step=1,
            max_steps=30,
        )
    assert result["action"] == "click"
    assert result["selector"] == "#add-to-cart-button"


def test_dom_decide_returns_stuck_on_llm_error():
    with patch("computer_use._get_client", side_effect=RuntimeError("no key")):
        result = computer_use._dom_decide(
            snapshot=SAMPLE_SNAPSHOT,
            goal="some goal",
            context_data={},
            step=1,
            max_steps=10,
        )
    assert result["action"] == "stuck"


def test_dom_decide_returns_stuck_on_invalid_action():
    mock_client = _mock_groq_response('{"action": "fly", "reason": "invalid"}')
    with patch("computer_use._get_client", return_value=mock_client):
        result = computer_use._dom_decide(
            snapshot=SAMPLE_SNAPSHOT,
            goal="goal",
            context_data={},
            step=1,
            max_steps=10,
        )
    assert result["action"] == "stuck"


def test_dom_decide_passes_history_to_prompt():
    mock_client = _mock_groq_response(
        '{"action": "scroll", "direction": "down", "amount": 400, "reason": "scroll"}'
    )
    history = [{"step": 1, "action": "click", "selector": "#btn", "reason": "test"}]
    with patch("computer_use._get_client", return_value=mock_client):
        result = computer_use._dom_decide(
            snapshot=SAMPLE_SNAPSHOT,
            goal="goal",
            context_data={},
            step=2,
            max_steps=30,
            history=history,
        )
    call_args = mock_client.chat.completions.create.call_args
    user_content = call_args[1]["messages"][1]["content"]
    assert "Step 1: click" in user_content
    assert result["action"] == "scroll"
