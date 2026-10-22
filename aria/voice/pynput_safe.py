"""voice/pynput_safe.py — stop pynput from crashing Aria on macOS.

pynput's keyboard Listener calls the Text Input Source (TIS) API on its own
thread when it starts (keycode_context(), to learn the keyboard layout).
Aria starts listeners while the main thread launches the menu-bar app, which
uses the same API — and macOS aborts the process when TIS/TSM is called from
two threads at once (SIGABRT, "Text Input Sources … called in two threads
concurrently", seen 2026-10-16).

install() reads the layout once, on the main thread, and makes pynput reuse
it. Trade-off: switching keyboard layouts mid-session doesn't update how
pynput names character keys; the hotkey (⌥ Space) doesn't depend on that.

It also fixes a pynput 1.8.1 bug: media keys (volume, play/pause) call
GlobalHotKeys._on_press(key) without the `injected` argument it requires; the
TypeError stops the listener, so ⌥ Space went dead after one media key
(seen 2026-10-22). Missing `injected` now means a real key press.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import threading

logger = logging.getLogger(__name__)

_installed = False


def _patch_media_keys() -> None:
    from pynput import keyboard
    hk = keyboard.GlobalHotKeys
    if getattr(hk, "_aria_media_safe", False):
        return
    press, release = hk._on_press, hk._on_release

    def _on_press(self, key, injected=False):
        return press(self, key, injected)

    def _on_release(self, key, injected=False):
        return release(self, key, injected)

    hk._on_press, hk._on_release, hk._aria_media_safe = _on_press, _on_release, True


def install() -> bool:
    """Call on the main thread before any pynput Listener starts. Never raises."""
    global _installed
    if _installed or sys.platform != "darwin":
        return _installed
    try:
        _patch_media_keys()
    except Exception as exc:
        logger.warning("pynput_safe: couldn't patch media-key handling: %s", exc)
    if threading.current_thread() is not threading.main_thread():
        logger.warning("pynput_safe.install() must run on the main thread — skipped")
        return False
    try:
        import pynput._util.darwin as util
        import pynput.keyboard._darwin as kbd

        with util.keycode_context() as context:  # the only TIS calls — main thread
            cached = context

        @contextlib.contextmanager
        def cached_keycode_context():
            yield cached

        # keyboard._darwin imported the name; _util.darwin uses its own global
        # (get_unicode_to_keycode_map, for Controller) — patch both.
        kbd.keycode_context = cached_keycode_context
        util.keycode_context = cached_keycode_context
        _installed = True
    except Exception as exc:
        logger.warning("pynput_safe: couldn't pre-cache keyboard layout: %s", exc)
    return _installed
