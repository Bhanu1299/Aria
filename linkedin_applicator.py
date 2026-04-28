"""
linkedin_applicator.py — LinkedIn Easy Apply automation for Aria.

Drives the multi-page LinkedIn Easy Apply modal by filling fields from
identity.json context_data, uploading resume, and advancing through pages
until the review/submit screen is reached.
"""

from __future__ import annotations

import logging
import os
import time

import agent_browser
import dom_browser

logger = logging.getLogger(__name__)

_MAX_PAGES = 20
_MAX_VOICE_ASKS = 5


def run_linkedin_application(
    job: dict,
    context_data: dict,
    voice_ask_fn,
) -> tuple[str, dict | None]:
    """Drive LinkedIn Easy Apply flow.

    Returns ("confirm"|"stuck"|"needs_input", data_or_none).
    """
    logger.info(
        "Applying to %s at %s — %s",
        job.get("title", "unknown"), job.get("company", "unknown"), job.get("url", ""),
    )

    try:
        return _run_linkedin_flow(job, context_data, voice_ask_fn)
    except Exception as exc:
        logger.exception("Unexpected error in LinkedIn application flow: %s", exc)
        return ("stuck", {"reason": "unexpected_error"})


def _run_linkedin_flow(
    job: dict,
    context_data: dict,
    voice_ask_fn,
) -> tuple[str, dict | None]:
    """Internal flow — may raise; caller wraps in try/except."""
    # ------------------------------------------------------------------
    # 1. Session check — detect sign-in *form*, not just nav-bar text
    # ------------------------------------------------------------------
    try:
        sign_in_form = agent_browser.run(
            lambda page: page.locator('form input[type="password"]').count() > 0
        )
    except Exception:
        sign_in_form = False

    if sign_in_form:
        logger.warning("LinkedIn sign-in form detected — not logged in")
        return ("stuck", {"reason": "not_logged_in"})

    # Debug: screenshot before button search
    dom_browser.save_debug_screenshot("apply_before_button_search")

    # Log current URL to confirm we're on the right page
    try:
        current_url = agent_browser.run(lambda page: page.url)
        logger.info("Page URL before button search: %s", current_url)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Click Easy Apply — text-based, then CSS-class fallback
    # ------------------------------------------------------------------
    clicked = _click_easy_apply_button()

    if not clicked:
        dom_browser.save_debug_screenshot("apply_button_NOT_FOUND")
        try:
            url_now = agent_browser.run(lambda page: page.url)
            logger.warning("Current page URL when button not found: %s", url_now)
        except Exception:
            pass
        logger.warning("Neither 'Easy Apply' nor 'Apply' button found")
        return ("stuck", None)

    logger.info("Easy Apply button clicked — waiting for modal")
    time.sleep(2)
    dom_browser.save_debug_screenshot("apply_after_button_click")

    # ------------------------------------------------------------------
    # 3. Wait for the Easy Apply modal to appear
    # ------------------------------------------------------------------
    modal_ready = _wait_for_modal(timeout_secs=8)
    if not modal_ready:
        logger.warning("Easy Apply modal did not appear after button click")
        dom_browser.save_debug_screenshot("apply_modal_NOT_FOUND")
        return ("stuck", {"reason": "modal_not_found"})

    logger.info("Easy Apply modal detected")

    # ------------------------------------------------------------------
    # 4. Multi-page form loop
    # ------------------------------------------------------------------
    voice_ask_count = 0

    for page_num in range(1, _MAX_PAGES + 1):
        dom_browser.save_debug_screenshot(f"apply_page_{page_num}")

        # a. Find empty fields in the modal (not just required ones)
        try:
            fields = _find_modal_fields()
        except Exception as exc:
            logger.warning("find_modal_fields error: %s", exc)
            fields = []

        logger.info("Page %d: found %d empty fields", page_num, len(fields))

        # b. Fill each field
        for field in fields:
            label = field.get("label", "")
            selector = field.get("selector", "")
            field_type = field.get("field_type", "text")

            # Resume / CV file upload
            if field_type == "file" and _label_is_resume(label):
                resume_path = context_data.get("resume_path", "")
                if resume_path:
                    _upload_resume(selector, resume_path)
                continue

            # Select / dropdown
            if field_type == "select":
                matched = _match_field_label(label, context_data)
                if matched:
                    _select_option(selector, matched)
                elif voice_ask_count < _MAX_VOICE_ASKS:
                    answer = _safe_voice_ask(voice_ask_fn, label)
                    voice_ask_count += 1
                    if answer:
                        _select_option(selector, answer)
                continue

            # Radio buttons
            if field_type == "radio":
                matched = _match_field_label(label, context_data)
                if matched:
                    _click_radio_option(selector, matched)
                continue

            # Text / textarea
            matched = _match_field_label(label, context_data)
            if matched:
                dom_browser.fill_if_empty(selector, matched)
            elif voice_ask_count < _MAX_VOICE_ASKS:
                answer = _safe_voice_ask(voice_ask_fn, label)
                voice_ask_count += 1
                if answer:
                    dom_browser.fill_if_empty(selector, answer)

        # c. Check for review / submit page FIRST
        if _check_review_or_submit():
            logger.info("Review/submit page reached on page %d", page_num)
            return ("confirm", None)

        # d. Try "Next" / "Continue" / "Review" buttons inside modal
        if _click_modal_advance_button():
            logger.info("Advanced to next page from page %d", page_num)
            time.sleep(2)
            continue

        # e. Scroll modal and retry
        _scroll_modal_down()
        time.sleep(1)

        if _check_review_or_submit():
            return ("confirm", None)

        if _click_modal_advance_button():
            logger.info("Advanced to next page from page %d (after scroll)", page_num)
            time.sleep(2)
            continue

        logger.warning("Stuck on page %d — no advance button found", page_num)
        dom_browser.save_debug_screenshot(f"apply_stuck_page_{page_num}")
        return ("stuck", None)

    logger.warning("Exceeded %d page iterations", _MAX_PAGES)
    return ("stuck", None)


# -----------------------------------------------------------------------
# Button clicking
# -----------------------------------------------------------------------

def _click_easy_apply_button() -> bool:
    """Try multiple strategies to find and click the Easy Apply button."""
    # Strategy 1: Text-based matching (longer timeout for SPA)
    if dom_browser.click_by_text("Easy Apply", tag="button", timeout_ms=12000):
        return True
    if dom_browser.click_by_text("Easy Apply", tag="[role='button']"):
        return True

    # Strategy 2: LinkedIn-specific CSS class selectors
    def _try_css(page):
        selectors = [
            'button.jobs-apply-button',
            '.jobs-apply-button--top-card button',
            '.jobs-s-apply button',
            'button[aria-label*="Easy Apply"]',
            'button[aria-label*="Apply to"]',
            '.jobs-apply-button button',
            # Sometimes it's a top-level button with specific data attributes
            'button[data-control-name="jobdetails_topcard_inapply"]',
        ]
        for sel in selectors:
            try:
                el = page.wait_for_selector(sel, state="visible", timeout=3000)
                if el:
                    logger.info("Found apply button via CSS: %s", sel)
                    el.click(timeout=2000)
                    return True
            except Exception:
                continue
        return False

    try:
        if agent_browser.run(_try_css):
            return True
    except Exception:
        pass

    # Strategy 3: Generic "Apply" text (last resort — may be external apply)
    if dom_browser.click_by_text("Apply", tag="button"):
        return True
    if dom_browser.click_by_text("Apply", tag="[role='button']"):
        return True

    return False


def _wait_for_modal(timeout_secs: int = 8) -> bool:
    """Wait for the LinkedIn Easy Apply modal to appear."""
    def _do(page):
        modal_selectors = [
            '.jobs-easy-apply-modal',
            '.jobs-easy-apply-content',
            'div[data-test-modal]',
            '[aria-labelledby*="easy-apply"]',
            # Generic modal dialog
            'div[role="dialog"]',
        ]
        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            for sel in modal_selectors:
                try:
                    el = page.query_selector(sel)
                    if el and el.is_visible():
                        return True
                except Exception:
                    continue
            time.sleep(0.5)
        return False

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.warning("_wait_for_modal error: %s", exc)
        return False


def _click_modal_advance_button() -> bool:
    """Click Next/Continue/Review inside the modal."""
    # Try each advance button text
    for text in ["Next", "Continue", "Review"]:
        if dom_browser.click_by_text(text, tag="button", timeout_ms=3000):
            return True
    # CSS fallback for LinkedIn's specific footer buttons
    def _do(page):
        selectors = [
            '.jobs-easy-apply-modal footer button[aria-label="Continue to next step"]',
            '.jobs-easy-apply-modal footer button[aria-label="Next"]',
            '.jobs-easy-apply-modal footer button[aria-label="Review your application"]',
            'button[aria-label="Continue to next step"]',
            'button[aria-label="Review your application"]',
        ]
        for sel in selectors:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click(timeout=2000)
                    return True
            except Exception:
                continue
        return False

    try:
        return agent_browser.run(_do)
    except Exception:
        return False


def _check_review_or_submit() -> bool:
    """Check if we're on the review/submit page."""
    for text in [
        "Review your application",
        "Submit application",
        "Review and submit",
        "Your application will be sent",
    ]:
        if dom_browser.page_has_text(text):
            return True
    return False


def _scroll_modal_down() -> None:
    """Scroll the Easy Apply modal content down."""
    def _do(page):
        # Try scrolling the modal content area
        for sel in ['.jobs-easy-apply-modal', '.jobs-easy-apply-content', 'div[role="dialog"]']:
            try:
                el = page.query_selector(sel)
                if el:
                    el.evaluate('el => el.scrollTop += 400')
                    return
            except Exception:
                continue
        # Fallback: mouse wheel
        page.mouse.wheel(0, 400)

    try:
        agent_browser.run(_do)
    except Exception:
        pass


# -----------------------------------------------------------------------
# Modal-scoped field detection
# -----------------------------------------------------------------------

def _find_modal_fields() -> list[dict]:
    """Find empty fields inside the Easy Apply modal (not just required ones)."""

    JS_CODE = """
    (() => {
        // Find the modal container
        const modal = document.querySelector('.jobs-easy-apply-modal')
                   || document.querySelector('.jobs-easy-apply-content')
                   || document.querySelector('div[role="dialog"]')
                   || document.body;

        const results = [];
        const els = modal.querySelectorAll('input, select, textarea');
        for (const el of els) {
            // Must be visible
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) continue;
            const style = getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') continue;

            // Skip hidden inputs, submit buttons, search fields
            const inputType = (el.type || '').toLowerCase();
            if (['hidden', 'submit', 'button', 'image'].includes(inputType)) continue;

            // Check if empty (or file input with no files)
            if (inputType === 'file') {
                // Always include file inputs — let Python decide
            } else if (el.tagName === 'SELECT') {
                // Include selects where first/blank option is selected
                if (el.selectedIndex > 0 && el.value) continue;
            } else {
                if (el.value && el.value.trim() !== '') continue;
            }

            // Determine label
            let label = el.getAttribute('aria-label') ||
                        el.getAttribute('placeholder') || '';
            if (!label) {
                if (el.id) {
                    const labelEl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                    if (labelEl) label = labelEl.innerText.trim();
                }
                if (!label) {
                    const parent = el.closest('label');
                    if (parent) label = parent.innerText.trim();
                }
                if (!label) {
                    // Walk up to find nearby text
                    const wrapper = el.closest('.fb-dash-form-element, .artdeco-text-input, [data-test-form-element]');
                    if (wrapper) {
                        const lbl = wrapper.querySelector('label, .artdeco-text-input--label, span.t-14');
                        if (lbl) label = lbl.innerText.trim();
                    }
                }
            }
            if (!label) label = el.name || inputType || 'unknown';

            // Build selector
            let selector = '';
            if (el.id) {
                selector = '#' + CSS.escape(el.id);
            } else if (el.name) {
                selector = el.tagName.toLowerCase() + '[name="' + el.name + '"]';
            } else {
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
                if (['file', 'radio', 'checkbox'].includes(inputType)) field_type = inputType;
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
        logger.warning("_find_modal_fields failed: %s", exc)
        return []


def _click_radio_option(selector: str, value: str) -> bool:
    """Click a radio button option matching value text."""
    def _do(page):
        # Find the radio group container and click the matching label
        try:
            # Look for a label near the radio that contains the value text
            radio = page.query_selector(selector)
            if not radio:
                return False
            parent = radio.evaluate_handle('el => el.closest("fieldset") || el.parentElement.parentElement')
            labels = parent.query_selector_all('label')
            low_val = value.lower()
            for lbl in labels:
                txt = lbl.inner_text().strip().lower()
                if low_val in txt or txt in low_val:
                    lbl.click()
                    return True
        except Exception:
            pass
        return False

    try:
        return agent_browser.run(_do)
    except Exception:
        return False


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _match_field_label(label: str, context_data: dict) -> str | None:
    """Fuzzy match field label to context_data value. Returns value or None."""
    low = label.lower()

    # Direct matches
    if "phone" in low or "mobile" in low:
        return context_data.get("phone", "")
    if "email" in low:
        return context_data.get("email", "")
    if "first name" in low:
        name = context_data.get("name", "")
        return name.split()[0] if name else ""
    if "last name" in low or "surname" in low:
        name = context_data.get("name", "")
        return name.split()[-1] if name else ""
    if "full name" in low or "your name" in low:
        return context_data.get("name", "")
    if "linkedin" in low or "profile url" in low:
        return context_data.get("linkedin", "")
    if "location" in low or "city" in low or "where" in low:
        return context_data.get("location", "")
    if "website" in low or "github" in low or "portfolio" in low:
        return context_data.get("github", "")
    if "summary" in low or "headline" in low or "about" in low:
        return context_data.get("summary", "")

    # Common questions with safe defaults
    if "authorized" in low or "eligible" in low or "legally" in low:
        return "Yes"
    if "sponsorship" in low or "visa" in low:
        return "No"
    if "currently" in low and ("work" in low or "employ" in low):
        return "No"
    # Years of experience
    if "year" in low and "experience" in low:
        return "3"
    # Salary / compensation
    if "salary" in low or "compensation" in low or "pay" in low:
        return ""  # skip — don't auto-fill salary

    return None


def _label_is_resume(label: str) -> bool:
    """Check if a field label refers to a resume or CV upload."""
    low = label.lower()
    return "resume" in low or "cv" in low


def _upload_resume(selector: str, resume_path: str) -> bool:
    """Upload resume file via Playwright set_input_files."""
    expanded = os.path.expanduser(resume_path)
    if not os.path.isfile(expanded):
        logger.warning("Resume file not found: %s", expanded)
        return False

    def _do(page):
        page.set_input_files(selector, expanded)
        return True

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.warning("Resume upload failed: %s", exc)
        return False


def _select_option(selector: str, value: str) -> bool:
    """Try to select a dropdown option by label text."""
    def _do(page):
        try:
            page.select_option(selector, label=value, timeout=2000)
            return True
        except Exception:
            pass
        # Fallback: try by value
        try:
            page.select_option(selector, value=value, timeout=2000)
            return True
        except Exception:
            pass
        # Fallback: try partial label match
        try:
            options = page.query_selector_all(f'{selector} option')
            low_val = value.lower()
            for opt in options:
                opt_text = opt.inner_text().strip().lower()
                if low_val in opt_text or opt_text in low_val:
                    opt_value = opt.get_attribute("value")
                    if opt_value:
                        page.select_option(selector, value=opt_value, timeout=2000)
                        return True
        except Exception:
            pass
        return False

    try:
        return agent_browser.run(_do)
    except Exception as exc:
        logger.warning("select_option(%r, %r) failed: %s", selector, value, exc)
        return False


def _safe_voice_ask(voice_ask_fn, label: str) -> str:
    """Call voice_ask_fn safely, return empty string on any error."""
    try:
        result = voice_ask_fn(label)
        return result.strip() if result else ""
    except Exception as exc:
        logger.warning("voice_ask_fn(%r) failed: %s", label, exc)
        return ""
