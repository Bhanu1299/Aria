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


def test_dom_research_decide_returns_navigate():
    mock_client = _mock_groq_response(
        '{"action": "navigate", "url": "https://amazon.com", "reason": "go to amazon"}'
    )
    with patch("computer_use._get_client", return_value=mock_client):
        result = computer_use._dom_research_decide(
            snapshot=SAMPLE_SNAPSHOT,
            goal="search amazon for airpods",
            step=1,
            max_steps=80,
            history=[],
            collected_data=[],
        )
    assert result["action"] == "navigate"
    assert result["url"] == "https://amazon.com"


def test_dom_research_decide_returns_done():
    mock_client = _mock_groq_response(
        '{"action": "done", "summary": "AirPods Pro costs $249.", "reason": "found price"}'
    )
    with patch("computer_use._get_client", return_value=mock_client):
        result = computer_use._dom_research_decide(
            snapshot=SAMPLE_SNAPSHOT,
            goal="find airpods price",
            step=3,
            max_steps=80,
            history=[],
            collected_data=[{"label": "price", "value": "$249"}],
        )
    assert result["action"] == "done"
    assert "$249" in result["summary"]


def test_dom_research_decide_returns_stuck_on_error():
    with patch("computer_use._get_client", side_effect=RuntimeError("api down")):
        result = computer_use._dom_research_decide(
            snapshot=SAMPLE_SNAPSHOT,
            goal="goal",
            step=1,
            max_steps=10,
            history=[],
            collected_data=[],
        )
    assert result["action"] == "stuck"


def test_dom_research_decide_click_text_is_valid():
    mock_client = _mock_groq_response(
        '{"action": "click_text", "text": "Add to Cart", "reason": "add product"}'
    )
    with patch("computer_use._get_client", return_value=mock_client):
        result = computer_use._dom_research_decide(
            snapshot=SAMPLE_SNAPSHOT,
            goal="add to cart",
            step=2,
            max_steps=80,
            history=[],
            collected_data=[],
        )
    assert result["action"] == "click_text"
    assert result["text"] == "Add to Cart"


def _fake_browser_run(page):
    def _run(fn):
        fn(page)
    return _run


def test_execute_click_with_selector():
    mock_page = MagicMock()
    with patch("agent_browser.run", side_effect=_fake_browser_run(mock_page)):
        computer_use.execute({"action": "click", "selector": "#add-to-cart-button"})
    mock_page.locator.assert_called_with("#add-to-cart-button")
    mock_page.locator.return_value.first.click.assert_called_once_with(timeout=3000)


def test_execute_click_text():
    mock_page = MagicMock()
    with patch("agent_browser.run", side_effect=_fake_browser_run(mock_page)):
        computer_use.execute({"action": "click_text", "text": "Add to Cart"})
    mock_page.locator.assert_called_with('text="Add to Cart"')
    mock_page.locator.return_value.first.click.assert_called_once_with(timeout=3000)


def test_execute_type_with_selector():
    mock_page = MagicMock()
    with patch("agent_browser.run", side_effect=_fake_browser_run(mock_page)):
        computer_use.execute({"action": "type", "selector": "#search", "text": "AirPods Pro"})
    mock_page.locator.assert_called_with("#search")
    mock_page.locator.return_value.first.fill.assert_called_once_with("AirPods Pro", timeout=3000)


def test_execute_click_coordinates_unchanged():
    mock_page = MagicMock()
    with patch("agent_browser.run", side_effect=_fake_browser_run(mock_page)):
        computer_use.execute({"action": "click", "x": 640, "y": 450})
    mock_page.mouse.click.assert_called_once_with(640, 450)


def test_execute_type_without_selector_uses_keyboard():
    mock_page = MagicMock()
    with patch("agent_browser.run", side_effect=_fake_browser_run(mock_page)):
        with patch("time.sleep"):
            computer_use.execute({"action": "type", "text": "hi"})
    assert mock_page.keyboard.type.call_count == 2  # one call per character


def test_run_loop_uses_dom_path_when_elements_present():
    rich_snapshot = "URL: https://example.com\nTITLE: Test\n\nINTERACTIVE[10]:\n" + \
        "\n".join(f"[{i}] BUTTON #btn{i} \"Button {i}\"" for i in range(10)) + \
        "\n\nPAGE TEXT:\nsome text"

    with patch("dom_browser.get_dom_snapshot", return_value=(rich_snapshot, 10)), \
         patch("computer_use._dom_decide", return_value={"action": "confirm"}) as mock_dom, \
         patch("computer_use.decide") as mock_vision, \
         patch("computer_use.take_screenshot", return_value="fakeb64"), \
         patch("computer_use.execute"), \
         patch("computer_use._human_sleep"):
        status, _ = computer_use.run_loop("fill a form", {}, max_steps=5)

    assert status == "confirm"
    mock_dom.assert_called()
    mock_vision.assert_not_called()


def test_run_loop_falls_back_to_vision_on_thin_dom():
    thin_snapshot = "URL: https://example.com\nTITLE: CAPTCHA\n\nINTERACTIVE[2]:\n[0] BUTTON #v \"Verify\"\n\nPAGE TEXT:\nverify"

    with patch("dom_browser.get_dom_snapshot", return_value=(thin_snapshot, 2)), \
         patch("computer_use.decide", return_value={"action": "confirm"}) as mock_vision, \
         patch("computer_use._dom_decide") as mock_dom, \
         patch("computer_use.take_screenshot", return_value="fakeb64"), \
         patch("dom_browser.save_debug_screenshot"), \
         patch("computer_use.execute"), \
         patch("computer_use._human_sleep"):
        status, _ = computer_use.run_loop("fill a form", {}, max_steps=5)

    assert status == "confirm"
    mock_vision.assert_called()
    mock_dom.assert_not_called()


def test_research_loop_uses_dom_path_when_elements_present():
    rich_snapshot = "URL: https://amazon.com\nTITLE: Amazon\n\nINTERACTIVE[10]:\n" + \
        "\n".join(f"[{i}] BUTTON #btn{i} \"Btn {i}\"" for i in range(10)) + \
        "\n\nPAGE TEXT:\nAmazon homepage"

    with patch("dom_browser.get_dom_snapshot", return_value=(rich_snapshot, 10)), \
         patch("computer_use._dom_research_decide",
               return_value={"action": "done", "summary": "Task complete."}) as mock_dom, \
         patch("computer_use._research_decide") as mock_vision, \
         patch("computer_use.take_screenshot", return_value="fakeb64"), \
         patch("computer_use._human_sleep"):
        result = computer_use.research_loop("find airpods price")

    assert result == "Task complete."
    mock_dom.assert_called()
    mock_vision.assert_not_called()


def test_research_loop_falls_back_to_vision_on_thin_dom():
    thin_snapshot = "URL: https://example.com\nTITLE: CAPTCHA\n\nINTERACTIVE[1]:\n[0] BUTTON #v \"Verify\"\n\nPAGE TEXT:\nverify"

    with patch("dom_browser.get_dom_snapshot", return_value=(thin_snapshot, 1)), \
         patch("computer_use._research_decide",
               return_value={"action": "done", "summary": "Verified."}) as mock_vision, \
         patch("computer_use._dom_research_decide") as mock_dom, \
         patch("computer_use.take_screenshot", return_value="fakeb64"), \
         patch("dom_browser.save_debug_screenshot"), \
         patch("computer_use._human_sleep"):
        result = computer_use.research_loop("handle captcha")

    assert result == "Verified."
    mock_vision.assert_called()
    mock_dom.assert_not_called()
