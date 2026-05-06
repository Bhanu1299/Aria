# DOM-First Browser Redesign — Design Spec
**Date:** 2026-04-18
**Status:** Approved

## Problem

`run_loop()` and `research_loop()` in `computer_use.py` both take a screenshot every step, encode it as base64, and pass it to a vision model that guesses pixel coordinates. This produces:

- **Stuck loops** — model repeatedly clicks wrong coordinates, recovery scroll fires, loop aborts
- **Slow execution** — vision calls are slower than text calls
- **Wasted cost** — vision tokens are expensive, especially on the Claude fallback

Observed failure: "add AirPods Pro to cart on Amazon" failed 3 times across planner retries because the vision model couldn't reliably click the Add to Cart button.

## Goal

Replace the screenshot → vision path with a DOM snapshot → text model path as the primary execution strategy. Keep vision as a fallback for pages with thin DOMs (<5 interactive elements: CAPTCHAs, canvas-heavy pages, unrendered SPAs).

Works on any website — Amazon, flights, movie bookings, Google search, forms, anything.

---

## Design

### 1. `get_dom_snapshot()` — new function in `dom_browser.py`

Runs inline JS to extract a compact text digest of the current page. Returns a plain string.

**Output format:**
```
URL: https://www.amazon.com/dp/B0D1XD1ZV3
TITLE: Apple AirPods Pro (2nd Gen) - Amazon

INTERACTIVE[5]:
[0] BUTTON  id="add-to-cart-button"        "Add to Cart"
[1] BUTTON  id="buy-now-button"            "Buy Now"
[2] SELECT  id="quantity"                  "Quantity: 1"
[3] INPUT   placeholder="Search Amazon"   name="field-keywords"
[4] LINK    href="/dp/..."                 "See all buying options"

PAGE TEXT:
Apple AirPods Pro (2nd generation) with MagSafe Case. $249.00. In Stock...
```

**JS extraction rules:**
- Walk `button`, `a[href]`, `input`, `select`, `textarea` elements
- Skip elements where `display:none` or `visibility:hidden` or bounding rect is 0×0
- Per element: tag, best selector (prefer `#id`, fallback to `[name=...]`, fallback to `[aria-label=...]`), visible text (trimmed, max 60 chars)
- Truncate page body text at 800 chars
- Return element count alongside snapshot string so caller can check thin-DOM threshold

**Signature:**
```python
def get_dom_snapshot() -> tuple[str, int]:
    """Returns (snapshot_text, interactive_element_count). Never raises."""
```

All Playwright calls go through `agent_browser.run()` per the worker thread rule.

---

### 2. New action schema

Text model returns the same JSON envelope as before. `click` and `type` use selectors instead of coordinates:

| Action | New fields | Notes |
|--------|-----------|-------|
| `click` | `selector` (CSS) OR `x,y` (vision fallback) | Detects mode by key presence |
| `click_text` | `text` | Clicks first visible element whose text matches |
| `type` | `selector` + `text` OR just `text` (vision fallback) | |
| `navigate` | `url` | Unchanged |
| `scroll` | `direction`, `amount` | Unchanged |
| `key` | `key` | Unchanged |
| `extract` | `label`, `value` | Unchanged |
| `confirm` | `summary` | Unchanged |
| `done` | `summary` | Unchanged |
| `needs_input` | `field` | Unchanged |
| `stuck` | `reason` | Unchanged |

---

### 3. New decision functions in `computer_use.py`

**`_dom_decide(snapshot, goal, context_data, step, max_steps, history) -> dict`**
- Replaces `decide()` as the primary path in `run_loop()`
- Model: Groq `llama-3.3-70b-versatile` (text, no vision, fast, free tier)
- System prompt: rewrite of `_CU_SYSTEM` — instructs model it receives a DOM snapshot, must return selector-based actions

**`_dom_research_decide(snapshot, goal, step, max_steps, history, collected_data) -> dict`**
- Replaces `_research_decide()` as the primary path in `research_loop()`
- Same model as above
- System prompt: rewrite of `_CU_GENERAL_SYSTEM` for DOM mode

Both functions: never raise, return `{"action": "stuck"}` on any error.

---

### 4. Fallback chain

```
get_dom_snapshot() → Groq text model        ← PRIMARY (fast, free, selector-based)
       ↓ interactive_count < 5
take_screenshot() → Groq vision model       ← existing _research_decide / decide
       ↓ 2 consecutive stucks, no data
take_screenshot() → Claude vision           ← existing _claude_research_decide
```

Thin DOM threshold: `< 5` interactive elements. Catches CAPTCHAs, canvas-heavy pages, unrendered SPAs. Normal pages have 10–100+ elements so this fires rarely.

---

### 5. `execute()` changes

Add selector-based branches before existing coordinate path. Detection: presence of `"selector"` key.

```python
if act == "click":
    if "selector" in action:
        page.locator(action["selector"]).first.click(timeout=3000)
    elif "x" in action:
        page.mouse.click(int(action["x"]), int(action["y"]))

elif act == "click_text":
    page.locator(f'text="{action["text"]}"').first.click(timeout=3000)

elif act == "type":
    if "selector" in action:
        page.locator(action["selector"]).first.fill(action.get("text", ""), timeout=3000)
    else:
        for char in action.get("text", ""):
            page.keyboard.type(char)
            time.sleep(random.uniform(0.03, 0.12))
```

All other action types unchanged.

---

### 6a. `no_change` detection in DOM mode

Both loops currently take a second screenshot after each action and MD5-compare it to the pre-action screenshot. If identical, they append a `no_change` history entry to warn the model.

In DOM mode: take a second `get_dom_snapshot()` after the action and compare the interactive element list (not the full text). If the element list is byte-identical, append the same `no_change` history entry. This is cheaper than a screenshot and still catches cases where a click had no effect.

---

### 6. What does NOT change

- `run_loop()` and `research_loop()` loop structure and control flow
- Stuck/repetition detection (3-identical-actions → recovery → abort)
- Groq → Claude fallback trigger logic
- `agent_browser.run()` threading rule — all Playwright calls still go through it
- All existing `dom_browser.py` helpers (`fill_if_empty`, `click_by_text`, `find_empty_required_fields`, etc.)
- `linkedin_applicator.py` — untouched
- `planner.py` — untouched

---

## Files changed

| File | Change |
|------|--------|
| `dom_browser.py` | Add `get_dom_snapshot() -> tuple[str, int]` |
| `computer_use.py` | Add `_dom_decide()`, `_dom_research_decide()`, new system prompts; update `execute()`; swap screenshot call at top of each loop iteration |

---

## Success criteria

1. "Add X to cart on Amazon" completes without stuck loop
2. General research tasks (flights, movies, products) complete faster than with vision
3. Vision fallback still works when `interactive_count < 5`
4. No regressions in `run_loop` (LinkedIn Easy Apply still works)
