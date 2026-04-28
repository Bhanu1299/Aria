"""
hotkey.py — Global hotkey listener for Aria.

Listens for a configurable key+modifier combo (default: ⌥ Space) using
pynput.keyboard.GlobalHotKeys.  Calls on_press_cb() on first activation
and on_release_cb() when the hotkey is released.

macOS requirement: Terminal (or the app running Aria) must be granted
Accessibility access in System Settings → Privacy & Security → Accessibility.
"""

import sys
import threading
from typing import Callable

from config import HOTKEY_KEY, HOTKEY_MODS


def _build_combo(key: str, mods: list[str]) -> str:
    """
    Build a pynput GlobalHotKeys combo string from key and modifier names.

    Examples
    --------
    _build_combo("space", ["alt"])        → "<alt>+<space>"
    _build_combo("space", ["alt", "cmd"]) → "<alt>+<cmd>+<space>"
    """
    parts = [f"<{m}>" for m in mods] + [f"<{key}>"]
    return "+".join(parts)


class HotkeyListener:
    """
    Wraps pynput GlobalHotKeys to provide press / release semantics.

    Parameters
    ----------
    on_press_cb : Callable[[], None]
        Called once each time the hotkey is pressed down.
    on_release_cb : Callable[[], None]
        Called once each time the hotkey is released.
    """

    def __init__(
        self,
        on_press_cb: Callable[[], None],
        on_release_cb: Callable[[], None],
    ) -> None:
        self._on_press_cb = on_press_cb
        self._on_release_cb = on_release_cb

        self._combo: str = _build_combo(HOTKEY_KEY, HOTKEY_MODS)
        self._pressed: bool = False
        self._lock = threading.Lock()

        # The pynput listener object; created in start().
        self._hotkey_listener = None

    # ------------------------------------------------------------------
    # Internal callbacks passed to GlobalHotKeys
    # ------------------------------------------------------------------

    def _on_activate(self) -> None:
        """Fired by pynput when the full combo is pressed."""
        with self._lock:
            if self._pressed:
                return          # already held — ignore repeat events
            self._pressed = True
        try:
            self._on_press_cb()
        except Exception as exc:
            print(f"[HotkeyListener] on_press_cb raised an exception: {exc}")

    def _on_deactivate(self) -> None:
        """Fired by pynput when the full combo is released."""
        with self._lock:
            if not self._pressed:
                return          # already released — ignore spurious events
            self._pressed = False
        try:
            self._on_release_cb()
        except Exception as exc:
            print(f"[HotkeyListener] on_release_cb raised an exception: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start listening for the hotkey in a background thread (non-blocking).

        Raises SystemExit if pynput cannot register (Accessibility missing).
        """
        from pynput import keyboard as _kb  # imported here so the error is localised

        hotkeys = {
            self._combo: self._on_activate,
        }

        try:
            self._hotkey_listener = _kb.GlobalHotKeys(hotkeys)
            self._hotkey_listener.start()
        except Exception as exc:
            err = str(exc).lower()
            if (
                "accessibility" in err
                or "axobserver" in err
                or "permission" in err
                or "ax" in err
            ):
                print(
                    "\n[Aria] Cannot register global hotkey — Accessibility permission missing.\n"
                    "  Fix: System Settings → Privacy & Security → Accessibility\n"
                    "       → click '+' and add Terminal (or your terminal app).\n"
                    f"  (error detail: {exc})\n"
                )
            else:
                print(f"\n[Aria] Failed to start hotkey listener: {exc}\n")
            sys.exit(1)

        # GlobalHotKeys does not expose a release callback directly.
        # We use a separate Listener that monitors key-up events and fires
        # _on_deactivate when all configured modifier keys are lifted.
        self._release_listener = _kb.Listener(
            on_release=self._handle_key_release,
        )
        try:
            self._release_listener.start()
        except Exception as exc:
            print(f"\n[Aria] Failed to start release listener: {exc}\n")
            self._hotkey_listener.stop()
            sys.exit(1)

        print(f"[Aria] Hotkey listener active — combo: {self._combo}")

    def _handle_key_release(self, key) -> None:
        """
        Called for every key-release event by the secondary Listener.

        If any of the configured modifier keys is released while the combo
        is considered "held", we treat that as a release of the full hotkey.
        """
        from pynput import keyboard as _kb

        # Build the set of pynput Key objects for our modifiers.
        mod_map = {
            "alt":   [_kb.Key.alt, _kb.Key.alt_l, _kb.Key.alt_r],
            "cmd":   [_kb.Key.cmd, _kb.Key.cmd_l, _kb.Key.cmd_r],
            "ctrl":  [_kb.Key.ctrl, _kb.Key.ctrl_l, _kb.Key.ctrl_r],
            "shift": [_kb.Key.shift, _kb.Key.shift_l, _kb.Key.shift_r],
        }

        released_key_name: str | None = None
        if hasattr(key, "char"):
            released_key_name = key.char
        else:
            released_key_name = key.name if hasattr(key, "name") else None

        # Check if the released key is the primary key or a modifier.
        primary_key_name = HOTKEY_KEY  # e.g. "space"
        is_primary = released_key_name == primary_key_name

        is_modifier = False
        for mod_str in HOTKEY_MODS:
            candidates = mod_map.get(mod_str, [])
            if key in candidates:
                is_modifier = True
                break

        if is_primary or is_modifier:
            self._on_deactivate()

    def stop(self) -> None:
        """Stop the hotkey listener and the release listener."""
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception as exc:
                print(f"[HotkeyListener] Error stopping hotkey listener: {exc}")
            self._hotkey_listener = None

        if hasattr(self, "_release_listener") and self._release_listener is not None:
            try:
                self._release_listener.stop()
            except Exception as exc:
                print(f"[HotkeyListener] Error stopping release listener: {exc}")
            self._release_listener = None

        print("[Aria] Hotkey listener stopped.")
