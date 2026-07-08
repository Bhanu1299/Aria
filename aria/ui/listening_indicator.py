"""
listening_indicator.py — Siri-style on-screen "I'm listening" HUD.

A small dark pill near the bottom-center of the screen with a pulsing
dot and a label. Shown whenever Aria's mic is actually open: wake word
recording, hotkey recording, and conversation-mode follow-up windows.

  show(text="Listening...") -> bool   — show (or retitle) the pill
  hide() -> bool                      — remove it

Same window rules as overlay.py: borderless, screensaver level,
click-through, joins all Spaces, shown with orderFrontRegardless so it
NEVER takes focus. All AppKit work runs on the main queue (rumps owns
the run loop). Never raises — failures return False.

Disable with ARIA_LISTENING_HUD=0.
"""

from __future__ import annotations

import logging
import math
import os
import threading

logger = logging.getLogger(__name__)

_PILL_W = 190.0
_PILL_H = 44.0
_BOTTOM_MARGIN = 70.0
_PULSE_FPS = 20.0

# Guarded by _lock: the one active window/view/timer, plus a generation
# counter so a stale hide() can't kill a newer show().
_lock = threading.Lock()
_state = {"window": None, "view": None, "timer": None, "generation": 0}


def enabled() -> bool:
    return os.getenv("ARIA_LISTENING_HUD", "1").strip().lower() not in {
        "0", "false", "off", "no",
    }


def show(text: str = "Listening...") -> bool:
    """Show the pill (or update its label if already visible). Never raises."""
    if not enabled():
        return False
    try:
        _dispatch_show(text)
        return True
    except Exception as exc:
        logger.debug("listening_indicator.show failed: %s", exc)
        return False


def hide() -> bool:
    """Remove the pill. Idempotent. Never raises."""
    try:
        with _lock:
            if _state["window"] is None:
                return False
            generation = _state["generation"]
        _dispatch_hide(generation)
        return True
    except Exception as exc:
        logger.debug("listening_indicator.hide failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# AppKit internals — everything below runs on the main queue
# ---------------------------------------------------------------------------

def _dispatch_show(text: str) -> None:
    from AppKit import NSOperationQueue

    def _on_main():
        try:
            _show_on_main(text)
        except Exception as exc:
            logger.debug("listening_indicator main-thread show failed: %s", exc)

    NSOperationQueue.mainQueue().addOperationWithBlock_(_on_main)


def _dispatch_hide(generation: int) -> None:
    from AppKit import NSOperationQueue

    def _on_main():
        try:
            _hide_on_main(generation)
        except Exception as exc:
            logger.debug("listening_indicator main-thread hide failed: %s", exc)

    NSOperationQueue.mainQueue().addOperationWithBlock_(_on_main)


def _show_on_main(text: str) -> None:
    from AppKit import (
        NSApplication, NSBackingStoreBuffered, NSBezierPath, NSColor, NSFont,
        NSFontAttributeName, NSForegroundColorAttributeName, NSMakeRect,
        NSScreen, NSTimer, NSView, NSWindow, NSWindowStyleMaskBorderless,
        NSScreenSaverWindowLevel,
    )
    import objc
    from Foundation import NSString

    with _lock:
        view = _state["view"]
        if view is not None:
            # Already visible — retitle, and claim a new generation so any
            # hide() dispatched before this show() can't remove the pill.
            _state["generation"] += 1
    if view is not None:
        view._label = str(text)
        view.setNeedsDisplay_(True)
        return

    NSApplication.sharedApplication()
    screen = NSScreen.mainScreen()
    if screen is None:
        raise RuntimeError("no main screen")
    sf = screen.frame()
    frame = NSMakeRect(
        sf.origin.x + (sf.size.width - _PILL_W) / 2.0,
        sf.origin.y + _BOTTOM_MARGIN,
        _PILL_W, _PILL_H,
    )

    class _PillView(NSView):
        def initWithFrame_label_(self, rect, label):
            self = objc.super(_PillView, self).initWithFrame_(rect)
            if self is None:
                return None
            self._label = label
            self._phase = 0.0
            return self

        def drawRect_(self, rect):
            bounds = self.bounds()
            pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, _PILL_H / 2.0, _PILL_H / 2.0
            )
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.05, 0.05, 0.10, 0.82
            ).setFill()
            pill.fill()

            # Pulsing dot — Siri-ish blue, alpha and radius breathe together.
            pulse = 0.5 + 0.5 * math.sin(self._phase)
            radius = 7.0 + 3.0 * pulse
            cx, cy = 26.0, bounds.size.height / 2.0
            dot_rect = NSMakeRect(cx - radius, cy - radius, radius * 2, radius * 2)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.35, 0.55, 1.0, 0.55 + 0.45 * pulse
            ).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(dot_rect).fill()

            attrs = {
                NSFontAttributeName: NSFont.systemFontOfSize_(14.0),
                NSForegroundColorAttributeName: NSColor.whiteColor(),
            }
            label = NSString.stringWithString_(self._label)
            size = label.sizeWithAttributes_(attrs)
            label.drawAtPoint_withAttributes_(
                (46.0, (bounds.size.height - size.height) / 2.0), attrs
            )

    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
    )
    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setLevel_(NSScreenSaverWindowLevel)
    window.setIgnoresMouseEvents_(True)          # click-through
    window.setCollectionBehavior_(1 << 0)        # join all Spaces
    view = _PillView.alloc().initWithFrame_label_(
        NSMakeRect(0, 0, _PILL_W, _PILL_H), str(text)
    )
    window.setContentView_(view)
    window.orderFrontRegardless()                # show WITHOUT taking focus

    def _tick(_timer):
        view._phase += 2.0 * math.pi / _PULSE_FPS  # full pulse ~1s
        view.setNeedsDisplay_(True)

    timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        1.0 / _PULSE_FPS, True, _tick
    )

    with _lock:
        _state["window"] = window
        _state["view"] = view
        _state["timer"] = timer
        _state["generation"] += 1


def _hide_on_main(generation: int) -> None:
    with _lock:
        if _state["generation"] != generation or _state["window"] is None:
            return  # a newer show() owns the pill now
        window = _state["window"]
        timer = _state["timer"]
        _state["window"] = None
        _state["view"] = None
        _state["timer"] = None
    try:
        if timer is not None:
            timer.invalidate()
    except Exception:
        pass
    try:
        window.orderOut_(None)
    except Exception:
        pass
