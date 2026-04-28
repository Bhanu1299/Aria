"""
tests/test_browser.py — Real browser integration test for Aria's BrowserExecutor.

Runs a live Chromium session against claude.ai.
No mocks. Requires an active claude.ai login stored in the browser profile.

Expected output:
    [TEST] browser.py
    [BROWSER] Starting Chromium...
    [BROWSER] Navigated to claude.ai
    Browser opened. Check: does it steal focus? (3 second pause to observe)
    [BROWSER] Typing question...
    [BROWSER] Waiting for response...
    [BROWSER] Response received (143 chars)
    PASS — got response containing "4"
"""

import sys
import time
import os

# Ensure the project root is on the path so imports work regardless of how
# the script is invoked (e.g. `python tests/test_browser.py` from repo root).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from browser import BrowserExecutor  # noqa: E402


def run_test() -> None:
    print("[TEST] browser.py")

    browser = BrowserExecutor()

    # -----------------------------------------------------------------------
    # 1. Start the browser
    # -----------------------------------------------------------------------
    try:
        browser.start()
    except Exception as exc:
        print(f"FAIL — browser.start() raised an exception: {exc}")
        return

    # -----------------------------------------------------------------------
    # 2. Focus-steal observation window
    # -----------------------------------------------------------------------
    print("Browser opened. Check: does it steal focus? (3 second pause to observe)")
    time.sleep(3)

    # -----------------------------------------------------------------------
    # 3. Ask a simple arithmetic question
    # -----------------------------------------------------------------------
    QUESTION = "What is 2 + 2?"
    try:
        response = browser.execute(QUESTION)
    except Exception as exc:
        print(f"FAIL — browser.ask() raised an exception: {exc}")
        browser.stop()
        return

    # -----------------------------------------------------------------------
    # 4. Stop the browser
    # -----------------------------------------------------------------------
    browser.stop()

    # -----------------------------------------------------------------------
    # 5. Verify result
    # -----------------------------------------------------------------------

    # Special sentinel returned when claude.ai shows a login wall.
    if response == "__LOGIN_REQUIRED__":
        print(
            "FAIL — Claude.ai login required. "
            "Log in to claude.ai in the browser first, then re-run."
        )
        return

    # Timed out — browser.ask() already spoke the error.
    if response == "":
        print("FAIL — browser.ask() returned an empty string (timeout or DOM error).")
        return

    if not isinstance(response, str):
        print(f"FAIL — expected a str response, got {type(response).__name__}.")
        return

    if "4" not in response:
        print(
            f'FAIL — response does not contain "4".\n'
            f"Response was:\n{response[:500]}"
        )
        return

    print(f'PASS — got response containing "4"')


if __name__ == "__main__":
    run_test()
