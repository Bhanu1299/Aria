"""
dom_browser.py — Thin Playwright DOM helper module for Aria.

Provides safe, non-raising helper functions for interacting with form fields,
clicking elements, inspecting page text, and capturing debug screenshots.
All Playwright calls go through agent_browser.run() for thread safety.
"""

from __future__ import annotations

import logging
import os
import time

import agent_browser

logger = logging.getLogger(__name__)


def fill_if_empty(selector: str, value: str) -> bool:
    """Fill input/textarea only if current value is blank.
    Returns True if filled, False if already had content or not found.
    """
    def _do(page):
        try:
            current = page.input_value(selector, timeout=2000)
        except Exception:
            return False
        if current.strip():
            return False
        try:
            page.fill(selector, value, timeout=2000)
            return True
        except Exception:
            return False

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.warning("fill_if_empty(%r) failed: %s", selector, exc)
        return False


def click_by_text(text: str, tag: str = "button", timeout_ms: int = 8000) -> bool:
    """Click first VISIBLE element matching tag whose innerText contains `text`.

    Uses wait_for_selector(state='visible') so hidden duplicates (e.g. mobile
    layout buttons) are skipped and slow SPA rendering is tolerated.
    Returns True if found and clicked, False otherwise.
    """
    def _do(page):
        try:
            el = page.wait_for_selector(
                f'{tag}:has-text("{text}")',
                state="visible",
                timeout=timeout_ms,
            )
            if el:
                el.click(timeout=2000)
                return True
        except Exception:
            pass
        return False

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.warning("click_by_text(%r, tag=%r) failed: %s", text, tag, exc)
        return False


def get_field_value(selector: str) -> str:
    """Return current value of input/textarea. Empty string if not found."""
    def _do(page):
        try:
            return page.input_value(selector, timeout=2000)
        except Exception:
            return ""

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.warning("get_field_value(%r) failed: %s", selector, exc)
        return ""


def find_empty_required_fields() -> list[dict]:
    """Return list of dicts: {label, selector, field_type} for all visible empty required fields."""

    JS_CODE = """
    (() => {
        const results = [];
        const els = document.querySelectorAll('input, select, textarea');
        for (const el of els) {
            // Must be visible
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) continue;
            const style = getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') continue;

            // Must be required
            const isRequired = el.hasAttribute('required') ||
                               el.getAttribute('aria-required') === 'true';
            if (!isRequired) continue;

            // Must be empty
            if (el.value && el.value.trim() !== '') continue;

            // Determine label
            let label = el.getAttribute('aria-label') ||
                        el.getAttribute('placeholder') || '';
            if (!label) {
                // Check for associated <label>
                if (el.id) {
                    const labelEl = document.querySelector(`label[for="${el.id}"]`);
                    if (labelEl) label = labelEl.innerText.trim();
                }
                if (!label) {
                    const parent = el.closest('label');
                    if (parent) label = parent.innerText.trim();
                }
            }
            if (!label) label = el.name || el.type || 'unknown';

            // Build selector
            let selector = '';
            if (el.id) {
                selector = '#' + CSS.escape(el.id);
            } else if (el.name) {
                selector = el.tagName.toLowerCase() + '[name="' + el.name + '"]';
            } else {
                // nth-of-type fallback
                const tag = el.tagName.toLowerCase();
                const siblings = el.parentElement ?
                    Array.from(el.parentElement.querySelectorAll(':scope > ' + tag)) : [];
                const idx = siblings.indexOf(el) + 1;
                selector = tag + ':nth-of-type(' + idx + ')';
            }

            // Field type
            const tag = el.tagName.toLowerCase();
            let field_type = 'text';
            if (tag === 'select') field_type = 'select';
            else if (tag === 'textarea') field_type = 'textarea';
            else {
                const t = (el.type || 'text').toLowerCase();
                if (['file', 'radio', 'checkbox'].includes(t)) field_type = t;
                else field_type = 'text';
            }

            results.push({ label, selector, field_type });
        }
        return results;
    })()
    """

    def _do(page):
        try:
            return page.evaluate(JS_CODE)
        except Exception:
            return []

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.warning("find_empty_required_fields failed: %s", exc)
        return []


def page_has_text(text: str) -> bool:
    """Return True if visible page body text contains `text` (case-insensitive)."""
    def _do(page):
        try:
            body_text = page.evaluate("document.body.innerText")
            return text.lower() in body_text.lower()
        except Exception:
            return False

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.warning("page_has_text(%r) failed: %s", text, exc)
        return False


def save_debug_screenshot(label: str) -> None:
    """Save screenshot to /tmp/aria_debug/<timestamp>_<label>.jpg.
    Creates /tmp/aria_debug/ if not exists. Never raises.
    """
    def _do(page):
        debug_dir = "/tmp/aria_debug"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        path = os.path.join(debug_dir, f"{timestamp}_{safe_label}.jpg")
        try:
            screenshot_bytes = page.screenshot(type="jpeg", quality=80)
            with open(path, "wb") as f:
                f.write(screenshot_bytes)
            logger.debug("Debug screenshot saved: %s", path)
        except Exception as exc:
            logger.warning("Failed to save screenshot %s: %s", path, exc)

    try:
        agent_browser.run(_do)
    except Exception as exc:
        logger.warning("save_debug_screenshot(%r) failed: %s", label, exc)


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
            const tag = el.tagName.toLowerCase();
            const siblings = el.parentElement
                ? Array.from(el.parentElement.querySelectorAll(':scope > ' + tag))
                : [];
            const idx = siblings.indexOf(el) + 1;
            selector = tag + ':nth-of-type(' + idx + ')';
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
        except Exception as exc:
            logger.warning("get_dom_snapshot page access failed: %s", exc)
            return ("", 0)
        try:
            elements = page.evaluate(_DOM_EXTRACT_JS)
            body_text = page.evaluate("(document.body.innerText || '').slice(0, 800)")
        except Exception as exc:
            logger.warning("get_dom_snapshot JS failed on %s: %s", url, exc)
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
        return result if result is not None else ("", 0)
    except Exception as exc:
        logger.warning("get_dom_snapshot failed: %s", exc)
        return ("", 0)
