from unittest.mock import patch, MagicMock
import dom_browser


def _fake_run(page):
    """Returns a fake agent_browser.run that calls fn(page)."""
    def _run(fn):
        return fn(page)
    return _run


def test_get_dom_snapshot_returns_tuple():
    mock_page = MagicMock()
    mock_page.url = "https://example.com"
    mock_page.title.return_value = "Example Domain"
    mock_page.evaluate.side_effect = [
        [
            {"tag": "BUTTON", "selector": "#btn", "text": "Click me", "href": ""},
            {"tag": "INPUT",  "selector": "#q",   "text": "search",   "href": ""},
        ],
        "Some page content here",
    ]
    with patch("agent_browser.run", side_effect=_fake_run(mock_page)):
        snapshot, count = dom_browser.get_dom_snapshot()
    assert count == 2
    assert "https://example.com" in snapshot
    assert "BUTTON" in snapshot
    assert "#btn" in snapshot
    assert "Click me" in snapshot
    assert "PAGE TEXT:" in snapshot
    assert "Some page content" in snapshot


def test_get_dom_snapshot_includes_interactive_count_in_header():
    mock_page = MagicMock()
    mock_page.url = "https://example.com"
    mock_page.title.return_value = "Test"
    mock_page.evaluate.side_effect = [
        [{"tag": "BUTTON", "selector": "#x", "text": "Go", "href": ""}],
        "body text",
    ]
    with patch("agent_browser.run", side_effect=_fake_run(mock_page)):
        snapshot, count = dom_browser.get_dom_snapshot()
    assert "INTERACTIVE[1]" in snapshot
    assert count == 1


def test_get_dom_snapshot_returns_empty_on_error():
    with patch("agent_browser.run", side_effect=RuntimeError("browser dead")):
        snapshot, count = dom_browser.get_dom_snapshot()
    assert snapshot == ""
    assert count == 0


def test_get_dom_snapshot_thin_dom():
    mock_page = MagicMock()
    mock_page.url = "https://example.com/captcha"
    mock_page.title.return_value = "CAPTCHA"
    mock_page.evaluate.side_effect = [
        [{"tag": "BUTTON", "selector": "#verify", "text": "Verify", "href": ""}],
        "Please verify you are human",
    ]
    with patch("agent_browser.run", side_effect=_fake_run(mock_page)):
        snapshot, count = dom_browser.get_dom_snapshot()
    assert count == 1
    assert count < 5


def test_get_dom_snapshot_zero_elements():
    mock_page = MagicMock()
    mock_page.url = "https://example.com/loading"
    mock_page.title.return_value = "Loading..."
    mock_page.evaluate.side_effect = [[], ""]
    with patch("agent_browser.run", side_effect=_fake_run(mock_page)):
        snapshot, count = dom_browser.get_dom_snapshot()
    assert count == 0
    assert "INTERACTIVE[0]" in snapshot
