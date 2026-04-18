# DOM-First Browser Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace screenshot→vision loops in `computer_use.py` with DOM snapshot→text model as the primary execution path, using vision only as fallback for thin-DOM pages (<5 interactive elements).

**Architecture:** Add `get_dom_snapshot()` to `dom_browser.py`, add `_dom_decide()` and `_dom_research_decide()` to `computer_use.py` backed by Groq text model, update `execute()` to handle selector-based actions, then wire both loops to call DOM path first.

**Tech Stack:** Python 3.11+, Playwright (via `agent_browser.run()`), Groq `llama-3.3-70b-versatile` (text), existing Groq vision + Claude vision fallbacks

---

## File Map

| File | Change |
|------|--------|
| `dom_browser.py` | Add `get_dom_snapshot() -> tuple[str, int]` |
| `computer_use.py` | Add `_CU_DOM_SYSTEM`, `_CU_DOM_GENERAL_SYSTEM`, `_DOM_TEXT_MODEL`, `_VALID_DOM_FORM_ACTIONS`, `_dom_decide()`, `_dom_research_decide()`; update `execute()`; update `_VALID_GENERAL_ACTIONS`; wire `run_loop()` and `research_loop()` |
| `tests/test_dom_snapshot.py` | New — unit tests for `get_dom_snapshot()` |
| `tests/test_computer_use_dom.py` | New — unit tests for `_dom_decide`, `_dom_research_decide`, `execute()` selector paths |

---

## Task 1: `get_dom_snapshot()` in `dom_browser.py`

**Files:**
- Modify: `dom_browser.py`
- Test: `tests/test_dom_snapshot.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dom_snapshot.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_dom_snapshot.py -v
```

Expected: `AttributeError: module 'dom_browser' has no attribute 'get_dom_snapshot'`

- [ ] **Step 3: Implement `get_dom_snapshot()` in `dom_browser.py`**

Add this after the existing `save_debug_screenshot` function (end of file):

```python
_DOM_EXTRACT_JS = """
(() => {
    const results = [];
    const els = document.querySelectorAll('button, a[href], input, select, textarea');
    for (const el of els) {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) continue;
        const style = getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') continue;

        const tag = el.tagName.toUpperCase();

        let selector = '';
        if (el.id) {
            selector = '#' + CSS.escape(el.id);
        } else if (el.name) {
            selector = el.tagName.toLowerCase() + '[name="' + el.name + '"]';
        } else if (el.getAttribute('aria-label')) {
            selector = '[aria-label="' + el.getAttribute('aria-label').replace(/"/g, '\\"') + '"]';
        } else {
            selector = el.tagName.toLowerCase();
        }

        let text = '';
        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') {
            text = el.value || el.getAttribute('placeholder') || el.getAttribute('aria-label') || '';
        } else {
            text = (el.innerText || el.textContent || '').trim().slice(0, 60);
        }

        results.push({
            tag: tag,
            selector: selector,
            text: text,
            href: el.href || '',
        });
    }
    return results;
})()
"""


def get_dom_snapshot() -> tuple[str, int]:
    """Extract page DOM as compact text digest.

    Returns (snapshot_text, interactive_element_count). Never raises.
    All Playwright calls go through agent_browser.run().
    """
    def _do(page):
        try:
            url = page.url
            title = page.title()
            elements = page.evaluate(_DOM_EXTRACT_JS)
            body_text = page.evaluate("(document.body.innerText || '').slice(0, 800)")
        except Exception as exc:
            logger.warning("get_dom_snapshot JS failed: %s", exc)
            return ("", 0)

        lines = [f"URL: {url}", f"TITLE: {title}", "", f"INTERACTIVE[{len(elements)}]:"]
        for i, el in enumerate(elements):
            href_part = f' href="{el["href"][:50]}"' if el.get("href") else ""
            text_part = f' "{el["text"]}"' if el.get("text") else ""
            lines.append(f"[{i}] {el['tag']:<8} {el['selector']:<40}{href_part}{text_part}")

        lines.extend(["", "PAGE TEXT:", body_text])
        return ("\n".join(lines), len(elements))

    try:
        result = agent_browser.run(_do)
        return result if result else ("", 0)
    except Exception as exc:
        logger.warning("get_dom_snapshot failed: %s", exc)
        return ("", 0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_dom_snapshot.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add dom_browser.py tests/test_dom_snapshot.py
git commit -m "feat: add get_dom_snapshot() to dom_browser"
```

---

## Task 2: `_dom_decide()` in `computer_use.py`

**Files:**
- Modify: `computer_use.py`
- Test: `tests/test_computer_use_dom.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_computer_use_dom.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_dom_decide_returns_click_with_selector tests/test_computer_use_dom.py::test_dom_decide_returns_stuck_on_llm_error tests/test_computer_use_dom.py::test_dom_decide_returns_stuck_on_invalid_action tests/test_computer_use_dom.py::test_dom_decide_passes_history_to_prompt -v
```

Expected: `AttributeError: module 'computer_use' has no attribute '_dom_decide'`

- [ ] **Step 3: Add `_DOM_TEXT_MODEL`, `_CU_DOM_SYSTEM`, `_VALID_DOM_FORM_ACTIONS`, and `_dom_decide()` to `computer_use.py`**

Add after the existing `_VALID_GENERAL_ACTIONS` line (around line 43):

```python
_VALID_DOM_FORM_ACTIONS = {"click", "click_text", "type", "scroll", "key", "confirm", "stuck", "needs_input"}
_DOM_TEXT_MODEL = "llama-3.3-70b-versatile"
```

Add after the existing `_CU_SYSTEM` block (after line ~110):

```python
_CU_DOM_SYSTEM = """You control a Chrome browser. You receive a DOM snapshot of the current page.
Complete form-filling tasks: fill required fields, advance through form pages, stop before submitting.

Return ONLY a JSON object — no markdown, no explanation:

{"action": "click",       "selector": "#submit-btn",                  "reason": "clicking Next"}
{"action": "click_text",  "text": "Continue",                         "reason": "clicking Continue button"}
{"action": "type",        "selector": "#first-name", "text": "Bhanu", "reason": "filling first name"}
{"action": "scroll",      "direction": "down", "amount": 400,         "reason": "revealing more fields"}
{"action": "key",         "key": "Tab",                               "reason": "advancing to next field"}
{"action": "confirm",     "reason": "all required fields filled, form ready to submit"}
{"action": "needs_input", "field": "Years of experience",             "reason": "required, not in context"}
{"action": "stuck",       "reason": "CAPTCHA detected — cannot proceed"}

Rules:
- Use selectors from the INTERACTIVE list. Prefer #id selectors.
- If a field already has a value, skip it and move to the next empty field.
- If all visible fields are filled, click Next/Continue to advance.
- Never click Submit or Apply — return "confirm" when the form is fully complete.
- Use "needs_input" when a required field is empty and not in Context data.
- Use "stuck" only as last resort (CAPTCHA, hard login wall, repeated failures).
"""
```

Add after `_CU_DOM_SYSTEM` and before `_human_sleep`:

```python
def _dom_decide(
    snapshot: str,
    goal: str,
    context_data: dict,
    step: int,
    max_steps: int,
    history: list[dict] | None = None,
) -> dict:
    """
    Groq text model decision for form-fill loop (DOM-first path).
    Never raises — returns {"action": "stuck"} on any error.
    """
    user_text = (
        f"Step {step} of {max_steps}.\n\n"
        f"Goal: {goal}\n\n"
        f"Context (use this data to fill fields):\n{json.dumps(context_data, indent=2)}\n\n"
        f"Current page DOM snapshot:\n{snapshot}"
    )
    if history:
        recent = history[-5:]
        lines = []
        for h in recent:
            desc = f"  Step {h['step']}: {h['action']}"
            if h.get("selector"): desc += f" selector={h['selector']!r}"
            if h.get("text"):     desc += f" text={h['text']!r}"
            if h.get("key"):      desc += f" key={h['key']}"
            if h.get("reason"):   desc += f" — {h['reason']}"
            lines.append(desc)
        user_text += (
            "\n\nPrevious actions already executed (do NOT repeat):\n"
            + "\n".join(lines)
        )
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_DOM_TEXT_MODEL,
            messages=[
                {"role": "system", "content": _CU_DOM_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = _extract_first_json(raw)
        parsed = json.loads(raw)
        if parsed.get("action") not in _VALID_DOM_FORM_ACTIONS:
            raise ValueError(f"Unknown action: {parsed.get('action')!r}")
        logger.info(
            "DOM step %d/%d  action=%r  selector=%s  text=%r  reason=%r",
            step, max_steps,
            parsed.get("action"),
            parsed.get("selector", "-"),
            parsed.get("text", ""),
            parsed.get("reason", ""),
        )
        return parsed
    except Exception as exc:
        logger.error("_dom_decide failed (step %d): %s", step, exc)
        return {"action": "stuck", "reason": f"LLM error: {exc}"}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_dom_decide_returns_click_with_selector tests/test_computer_use_dom.py::test_dom_decide_returns_stuck_on_llm_error tests/test_computer_use_dom.py::test_dom_decide_returns_stuck_on_invalid_action tests/test_computer_use_dom.py::test_dom_decide_passes_history_to_prompt -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add computer_use.py tests/test_computer_use_dom.py
git commit -m "feat: add _dom_decide() text-model path for form-fill loop"
```

---

## Task 3: `_dom_research_decide()` in `computer_use.py`

**Files:**
- Modify: `computer_use.py`
- Test: `tests/test_computer_use_dom.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_computer_use_dom.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_dom_research_decide_returns_navigate tests/test_computer_use_dom.py::test_dom_research_decide_returns_done tests/test_computer_use_dom.py::test_dom_research_decide_returns_stuck_on_error tests/test_computer_use_dom.py::test_dom_research_decide_click_text_is_valid -v
```

Expected: `AttributeError: module 'computer_use' has no attribute '_dom_research_decide'`

- [ ] **Step 3: Add `_CU_DOM_GENERAL_SYSTEM` and `_dom_research_decide()` to `computer_use.py`**

Add `_CU_DOM_GENERAL_SYSTEM` right after `_CU_DOM_SYSTEM`:

```python
_CU_DOM_GENERAL_SYSTEM = """You control a Chrome browser. You receive a DOM snapshot of the current page.
Complete ANY browser task: research, shopping, messaging, form submission, booking, or anything else.

Return ONLY a JSON object — no markdown, no explanation:

{"action": "navigate",    "url": "https://amazon.com",                    "reason": "go to Amazon"}
{"action": "click",       "selector": "#add-to-cart-button",              "reason": "clicking Add to Cart"}
{"action": "click_text",  "text": "Add to Cart",                          "reason": "clicking by visible text"}
{"action": "type",        "selector": "#search", "text": "AirPods Pro",   "reason": "typing search query"}
{"action": "scroll",      "direction": "down", "amount": 400,             "reason": "revealing more content"}
{"action": "key",         "key": "Enter",                                 "reason": "submitting search"}
{"action": "extract",     "label": "price", "value": "$249",              "reason": "recording price"}
{"action": "confirm",     "summary": "About to add AirPods to cart...",   "reason": "irreversible action"}
{"action": "needs_input", "field": "delivery address",                    "reason": "required, not in context"}
{"action": "done",        "summary": "AirPods Pro added to cart.",        "reason": "task complete"}
{"action": "stuck",       "reason": "Login required, no session"}

Rules:
- Use selectors from the INTERACTIVE list. Prefer #id selectors.
- Use "navigate" to jump directly to URLs — faster than clicking menus.
- Use "extract" to record any data worth reporting to the user.
- Use "confirm" BEFORE any irreversible action (checkout, purchase, send, submit, delete).
- Use "done" when task is fully complete with a clear spoken summary.
- Use "needs_input" when a required field is missing from context.
- Use "stuck" only as last resort. Always try alternatives first.
- Do NOT re-extract data you already have. Do NOT re-navigate to a page you just left.
- If approaching step limit with partial results, use "done" with what you have.
"""
```

Add `_dom_research_decide()` after `_dom_decide()`:

```python
def _dom_research_decide(
    snapshot: str,
    goal: str,
    step: int,
    max_steps: int,
    history: list[dict],
    collected_data: list[dict],
) -> dict:
    """
    Groq text model decision for research loop (DOM-first path).
    Never raises — returns {"action": "stuck"} on any error.
    """
    _VALID_DOM_RESEARCH_ACTIONS = _VALID_GENERAL_ACTIONS | {"click_text"}
    user_text = (
        f"Step {step} of {max_steps}.\n\n"
        f"Goal: {goal}\n\n"
        f"Data collected so far ({len(collected_data)} items):\n"
        f"{json.dumps(collected_data, indent=2)}\n\n"
        f"Current page DOM snapshot:\n{snapshot}"
    )
    if history:
        recent = history[-6:]
        lines = []
        for h in recent:
            desc = f"  Step {h['step']}: {h['action']}"
            if h.get("url"):      desc += f" url={h['url']!r}"
            if h.get("selector"): desc += f" selector={h['selector']!r}"
            if h.get("label"):    desc += f" label={h['label']!r} value={h.get('value', '')!r}"
            if h.get("text"):     desc += f" text={h['text']!r}"
            if h.get("key"):      desc += f" key={h['key']}"
            if h.get("reason"):   desc += f" — {h['reason']}"
            lines.append(desc)
        user_text += (
            "\n\nPrevious actions (do NOT repeat):\n" + "\n".join(lines)
            + "\n\nDo NOT re-extract data you already have. "
              "If you have all needed data, return 'done' with a complete summary."
        )
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_DOM_TEXT_MODEL,
            messages=[
                {"role": "system", "content": _CU_DOM_GENERAL_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = _extract_first_json(raw)
        parsed = json.loads(raw)
        if parsed.get("action") not in _VALID_DOM_RESEARCH_ACTIONS:
            raise ValueError(f"Unknown action: {parsed.get('action')!r}")
        logger.info(
            "DOM research step %d/%d  action=%r  selector=%s  reason=%r",
            step, max_steps,
            parsed.get("action"),
            parsed.get("selector", "-"),
            parsed.get("reason", ""),
        )
        return parsed
    except Exception as exc:
        logger.error("_dom_research_decide failed (step %d): %s", step, exc)
        return {"action": "stuck", "reason": f"LLM error: {exc}"}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_dom_research_decide_returns_navigate tests/test_computer_use_dom.py::test_dom_research_decide_returns_done tests/test_computer_use_dom.py::test_dom_research_decide_returns_stuck_on_error tests/test_computer_use_dom.py::test_dom_research_decide_click_text_is_valid -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add computer_use.py tests/test_computer_use_dom.py
git commit -m "feat: add _dom_research_decide() text-model path for research loop"
```

---

## Task 4: Update `execute()` for selector-based actions

**Files:**
- Modify: `computer_use.py`
- Test: `tests/test_computer_use_dom.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_computer_use_dom.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_execute_click_with_selector tests/test_computer_use_dom.py::test_execute_click_text tests/test_computer_use_dom.py::test_execute_type_with_selector tests/test_computer_use_dom.py::test_execute_click_coordinates_unchanged tests/test_computer_use_dom.py::test_execute_type_without_selector_uses_keyboard -v
```

Expected: failures because `click_text` is unknown and `click` ignores `selector`.

- [ ] **Step 3: Update `execute()` in `computer_use.py`**

Replace the `_do` inner function inside `execute()` (currently at lines ~406–425):

```python
    def _do(page):
        if act == "click":
            if "selector" in action:
                page.locator(action["selector"]).first.click(timeout=3000)
            else:
                page.mouse.click(int(action.get("x", 0)), int(action.get("y", 0)))
        elif act == "click_text":
            page.locator(f'text="{action.get("text", "")}"').first.click(timeout=3000)
        elif act == "type":
            if "selector" in action:
                page.locator(action["selector"]).first.fill(
                    action.get("text", ""), timeout=3000
                )
            else:
                for char in action.get("text", ""):
                    page.keyboard.type(char)
                    time.sleep(random.uniform(0.03, 0.12))
        elif act == "scroll":
            direction = action.get("direction", "down")
            amount = int(action.get("amount", 400))
            page.mouse.wheel(0, amount if direction == "down" else -amount)
        elif act == "key":
            page.keyboard.press(action.get("key", "Tab"))
        elif act in ("confirm", "stuck", "navigate", "extract", "done", "needs_input"):
            pass  # caller handles these
        else:
            raise ValueError(f"Unknown action type: {act!r}")
```

Also update the `_VALID_GENERAL_ACTIONS` set at the top of the file to include `click_text`:

```python
_VALID_GENERAL_ACTIONS = {"click", "click_text", "type", "scroll", "key", "navigate", "extract", "confirm", "needs_input", "done", "stuck"}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_execute_click_with_selector tests/test_computer_use_dom.py::test_execute_click_text tests/test_computer_use_dom.py::test_execute_type_with_selector tests/test_computer_use_dom.py::test_execute_click_coordinates_unchanged tests/test_computer_use_dom.py::test_execute_type_without_selector_uses_keyboard -v
```

Expected: `5 passed`

- [ ] **Step 5: Run the full test file to confirm no regressions**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py tests/test_dom_snapshot.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add computer_use.py tests/test_computer_use_dom.py
git commit -m "feat: update execute() with selector-based click, click_text, type"
```

---

## Task 5: Wire DOM-first into `run_loop()`

**Files:**
- Modify: `computer_use.py`
- Test: `tests/test_computer_use_dom.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_computer_use_dom.py`:

```python
def test_run_loop_uses_dom_path_when_elements_present():
    """run_loop should call _dom_decide, not decide, when DOM has >=5 elements."""
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
    """run_loop should call decide (vision) when DOM has <5 elements."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_run_loop_uses_dom_path_when_elements_present tests/test_computer_use_dom.py::test_run_loop_falls_back_to_vision_on_thin_dom -v
```

Expected: `FAILED` — current `run_loop` always uses `take_screenshot` + `decide`.

- [ ] **Step 3: Rewrite the top of the loop body in `run_loop()`**

In `run_loop()`, replace the three lines at the start of the `for` loop body:

```python
        b64 = take_screenshot()
        if b64 is None:
            logger.error("Screenshot failed at step %d — aborting loop", step)
            return ("stuck", None)

        dom_browser.save_debug_screenshot(f"step_{step:02d}")

        action = decide(b64, goal, context_data, step, max_steps, history=history)
```

With:

```python
        snapshot, interactive_count = dom_browser.get_dom_snapshot()
        _vision_mode = interactive_count < 5

        if _vision_mode:
            b64 = take_screenshot()
            if b64 is None:
                logger.error("Screenshot failed at step %d — aborting loop", step)
                return ("stuck", None)
            dom_browser.save_debug_screenshot(f"step_{step:02d}")
            action = decide(b64, goal, context_data, step, max_steps, history=history)
        else:
            b64 = None
            action = _dom_decide(snapshot, goal, context_data, step, max_steps, history=history)
```

Then replace the post-action no_change detection block:

```python
        post_b64 = take_screenshot()
        if post_b64 is not None:
            pre_hash = hashlib.md5(b64.encode()).hexdigest()
            post_hash = hashlib.md5(post_b64.encode()).hexdigest()
            if pre_hash == post_hash:
                history.append({
                    "step": step,
                    "action": "no_change",
                    "reason": "page did not visibly change after last action — try a different approach",
                })
```

With:

```python
        if _vision_mode:
            post_b64 = take_screenshot()
            if post_b64 is not None:
                pre_hash = hashlib.md5(b64.encode()).hexdigest()
                post_hash = hashlib.md5(post_b64.encode()).hexdigest()
                if pre_hash == post_hash:
                    history.append({
                        "step": step,
                        "action": "no_change",
                        "reason": "page did not visibly change after last action — try a different approach",
                    })
        else:
            post_snapshot, _ = dom_browser.get_dom_snapshot()
            pre_interactive = snapshot.split("PAGE TEXT:")[0]
            post_interactive = post_snapshot.split("PAGE TEXT:")[0]
            if pre_interactive == post_interactive:
                history.append({
                    "step": step,
                    "action": "no_change",
                    "reason": "page did not change after last action — try a different approach",
                })
```

Also update the history append after execute to use `selector` instead of `x/y` in DOM mode. Replace:

```python
        history.append({
            "step": step,
            "action": action["action"],
            "x": action.get("x"),
            "y": action.get("y"),
            "text": action.get("text"),
            "key": action.get("key"),
            "reason": action.get("reason", ""),
        })
```

With:

```python
        history.append({
            "step": step,
            "action": action["action"],
            "selector": action.get("selector"),
            "x": action.get("x"),
            "y": action.get("y"),
            "text": action.get("text"),
            "key": action.get("key"),
            "reason": action.get("reason", ""),
        })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_run_loop_uses_dom_path_when_elements_present tests/test_computer_use_dom.py::test_run_loop_falls_back_to_vision_on_thin_dom -v
```

Expected: `2 passed`

- [ ] **Step 5: Run the full test suite**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py tests/test_dom_snapshot.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add computer_use.py
git commit -m "feat: wire DOM-first path into run_loop()"
```

---

## Task 6: Wire DOM-first into `research_loop()`

**Files:**
- Modify: `computer_use.py`
- Test: `tests/test_computer_use_dom.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_computer_use_dom.py`:

```python
def test_research_loop_uses_dom_path_when_elements_present():
    """research_loop should call _dom_research_decide, not _research_decide, for rich DOM."""
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
    """research_loop should call _research_decide (vision) for thin DOM."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_research_loop_uses_dom_path_when_elements_present tests/test_computer_use_dom.py::test_research_loop_falls_back_to_vision_on_thin_dom -v
```

Expected: `FAILED` — current `research_loop` always uses `take_screenshot` + `_research_decide`.

- [ ] **Step 3: Rewrite the top of the loop body in `research_loop()`**

In `research_loop()`, replace the three lines at the start of the `for` loop body:

```python
        b64 = take_screenshot()
        if b64 is None:
            logger.error("research_loop screenshot failed at step %d", step)
            return "I ran into a browser error while researching that."

        dom_browser.save_debug_screenshot(f"research_step_{step:02d}")

        if _use_claude:
            if _claude_steps_used >= _CLAUDE_MAX_STEPS:
                ...
            action = _claude_research_decide(b64, goal, step, max_steps, history, collected_data)
            _claude_steps_used += 1
        else:
            action = _research_decide(b64, goal, step, max_steps, history, collected_data)
```

With:

```python
        snapshot, interactive_count = dom_browser.get_dom_snapshot()
        _vision_mode = interactive_count < 5

        if _vision_mode or _use_claude:
            b64 = take_screenshot()
            if b64 is None:
                logger.error("research_loop screenshot failed at step %d", step)
                return "I ran into a browser error while researching that."
            dom_browser.save_debug_screenshot(f"research_step_{step:02d}")

        if _use_claude:
            if _claude_steps_used >= _CLAUDE_MAX_STEPS:
                logger.warning("research_loop hit Claude step cap (%d)", _CLAUDE_MAX_STEPS)
                if collected_data:
                    partial = "; ".join(f"{d['label']}: {d['value']}" for d in collected_data)
                    return f"I used my full research budget. Here's what I found: {partial}"
                return "I couldn't complete the research within the step budget."
            action = _claude_research_decide(b64, goal, step, max_steps, history, collected_data)
            _claude_steps_used += 1
        elif _vision_mode:
            action = _research_decide(b64, goal, step, max_steps, history, collected_data)
        else:
            action = _dom_research_decide(snapshot, goal, step, max_steps, history, collected_data)
```

Then replace the post-action no_change detection block in `research_loop()`:

```python
            post_b64 = take_screenshot()
            if post_b64 is not None:
                pre_hash = hashlib.md5(b64.encode()).hexdigest()
                post_hash = hashlib.md5(post_b64.encode()).hexdigest()
                if pre_hash == post_hash:
                    history.append({
                        "step": step, "action": "no_change",
                        "reason": "page did not visibly change — try a different approach",
                    })
```

With:

```python
            if _vision_mode or _use_claude:
                post_b64 = take_screenshot()
                if post_b64 is not None:
                    pre_hash = hashlib.md5(b64.encode()).hexdigest()
                    post_hash = hashlib.md5(post_b64.encode()).hexdigest()
                    if pre_hash == post_hash:
                        history.append({
                            "step": step, "action": "no_change",
                            "reason": "page did not visibly change — try a different approach",
                        })
            else:
                post_snapshot, _ = dom_browser.get_dom_snapshot()
                pre_interactive = snapshot.split("PAGE TEXT:")[0]
                post_interactive = post_snapshot.split("PAGE TEXT:")[0]
                if pre_interactive == post_interactive:
                    history.append({
                        "step": step, "action": "no_change",
                        "reason": "page did not change after last action — try a different approach",
                    })
```

Also add `selector` to the history entry in `research_loop()`. Replace the block:

```python
        h_entry = {"step": step, "action": action["action"], "reason": action.get("reason", "")}
        if action["action"] == "navigate":  h_entry["url"] = action.get("url", "")
        elif action["action"] == "extract": h_entry["label"] = action.get("label", ""); h_entry["value"] = action.get("value", "")
        elif action["action"] == "click":   h_entry["x"] = action.get("x"); h_entry["y"] = action.get("y")
        elif action["action"] == "type":    h_entry["text"] = action.get("text", "")
        elif action["action"] == "key":     h_entry["key"] = action.get("key", "")
        history.append(h_entry)
```

With:

```python
        h_entry = {"step": step, "action": action["action"], "reason": action.get("reason", "")}
        if action["action"] == "navigate":   h_entry["url"] = action.get("url", "")
        elif action["action"] == "extract":  h_entry["label"] = action.get("label", ""); h_entry["value"] = action.get("value", "")
        elif action["action"] == "click":    h_entry["selector"] = action.get("selector"); h_entry["x"] = action.get("x"); h_entry["y"] = action.get("y")
        elif action["action"] == "click_text": h_entry["text"] = action.get("text", "")
        elif action["action"] == "type":     h_entry["selector"] = action.get("selector"); h_entry["text"] = action.get("text", "")
        elif action["action"] == "key":      h_entry["key"] = action.get("key", "")
        history.append(h_entry)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py::test_research_loop_uses_dom_path_when_elements_present tests/test_computer_use_dom.py::test_research_loop_falls_back_to_vision_on_thin_dom -v
```

Expected: `2 passed`

- [ ] **Step 5: Run the full test suite**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/test_computer_use_dom.py tests/test_dom_snapshot.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add computer_use.py tests/test_computer_use_dom.py
git commit -m "feat: wire DOM-first path into research_loop()"
```

---

## Final Verification

- [ ] **Run the entire test suite**

```bash
cd /Users/bhanuteja/Documents/trae_projects/Aria && source venv/bin/activate && python -m pytest tests/ -v
```

Expected: all existing tests pass plus the new ones.

- [ ] **Smoke test (manual)**

Start Aria, press the hotkey, say: *"Search for AirPods Pro on Amazon and add it to my cart."*

Watch the logs — you should see `DOM step N/M` lines instead of vision calls. The loop should click `#add-to-cart-button` directly without getting stuck.
