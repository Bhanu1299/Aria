"""
menubar.py — Aria menu bar icon controller

Shows a single icon in the macOS menu bar.
Cycles through: IDLE → LISTENING → THINKING → DONE → IDLE
Called from main.py; set_state() is safe to call from background threads.
"""

from __future__ import annotations

import os
import subprocess

import rumps

LOG_PATH = os.path.expanduser("~/.aria/logs/aria.log")  # written by Aria.app's launcher

ICONS = {
    "IDLE":      "◉",
    "LISTENING": "🎙",
    "THINKING":  "⏳",
    "DONE":      "✓",
    "CODING":    "⚙",
}


class AriaMenuBar(rumps.App):
    def __init__(self):
        super().__init__(
            name="Aria",
            title=ICONS["IDLE"],
            menu=[
                rumps.MenuItem("Show Log", callback=self._show_log),
                rumps.MenuItem("Quit", callback=self._quit),
            ],
            quit_button=None,  # disable default Quit so ours is the only one
        )

    def _show_log(self, _sender):
        if os.path.exists(LOG_PATH):
            subprocess.Popen(["open", "-a", "Console", LOG_PATH])
        else:
            rumps.alert("Aria", "No log yet — the log is written when Aria runs from Aria.app.")

    def _quit(self, _sender):
        rumps.quit_application()

    def set_state(self, state: str):
        """Update the menu bar icon to reflect the current state.

        Safe to call from any thread — rumps allows title mutation from
        background threads on macOS.

        Args:
            state: One of "IDLE", "LISTENING", "THINKING", "DONE".
        """
        icon = ICONS.get(state)
        if icon is None:
            print(f"[MENUBAR] Unknown state: {state!r} — ignoring")
            return
        self.title = icon
        print(f"[MENUBAR] State → {state}")

    def set_state_label(self, label: str):
        """Set an arbitrary label on the menu bar icon (used by coder for live feedback)."""
        self.title = label

    def stop(self):
        """Quit the rumps application. Safe to call from any thread."""
        rumps.quit_application()

    def run(self):
        """Start the rumps main loop (blocks the calling thread)."""
        # Menu-bar-only: no Python rocket in the Dock, never steals focus.
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            NSApplication.sharedApplication().setActivationPolicy_(
                NSApplicationActivationPolicyAccessory)
        except Exception:
            pass
        super().run()


if __name__ == "__main__":
    import time
    import threading

    app = AriaMenuBar()

    def cycle():
        for state in ["LISTENING", "THINKING", "DONE", "IDLE"]:
            time.sleep(1)
            app.set_state(state)

    t = threading.Thread(target=cycle, daemon=True)
    t.start()
    app.run()
