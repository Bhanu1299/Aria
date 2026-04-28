"""
tests/test_hotkey.py — Real integration test for HotkeyListener.

Run from the project root:
    python tests/test_hotkey.py

The test starts HotkeyListener, waits 5 seconds for the user to press
the configured hotkey (default: ⌥ Space), then reports PASS or FAIL.

Requires macOS Accessibility permission for pynput.
"""

import os
import sys
import time

# Ensure the project root is on the path so `hotkey` and `config` import cleanly.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from hotkey import HotkeyListener  # noqa: E402


def main() -> None:
    print("[TEST] hotkey.py")

    press_detected: list[bool] = [False]   # mutable container so inner fn can write
    release_detected: list[bool] = [False]

    def on_press() -> None:
        press_detected[0] = True
        print("  → on_press_cb fired")

    def on_release() -> None:
        release_detected[0] = True
        print("  → on_release_cb fired")

    listener = HotkeyListener(on_press_cb=on_press, on_release_cb=on_release)

    try:
        listener.start()
    except SystemExit:
        # start() already printed the Accessibility error; propagate the failure.
        print("FAIL — could not start listener (check Accessibility permission)")
        sys.exit(1)

    print("Hold ⌥ Space now... (waiting 5 seconds)")

    # Poll for early success so the test exits the moment both events fire.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if press_detected[0] and release_detected[0]:
            break
        time.sleep(0.1)

    listener.stop()

    if press_detected[0]:
        print("PASS — hotkey detected")
        if not release_detected[0]:
            print("  (note: release callback was not observed — key may still be held)")
    else:
        print("FAIL — hotkey not detected (check Accessibility permission)")
        sys.exit(1)


if __name__ == "__main__":
    main()
