"""
overlay.py — draw annotations on top of the screen to point things out.

draw_boxes(regions, duration=6.0) -> bool
    regions: [{"x", "y", "w", "h", "label"}] in screen POINTS, top-left origin.
    Shows a transparent, click-through window with labeled boxes for `duration`
    seconds, then removes it. Returns True if the overlay was scheduled.

scale_normalized_regions(regions, screen_w, screen_h) -> list[dict]
    Converts vision-model regions (0–1000 normalized, top-left origin)
    to screen points, clamping and dropping malformed entries.

The window is borderless, sits above everything (screensaver level), ignores
all mouse events, and never takes focus — Aria's "never steal focus" rule holds.
All AppKit work is dispatched to the main thread; rumps owns the run loop.

Never raises — failure paths return False.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_DEFAULT_DURATION = 6.0
_NORM = 1000.0  # vision model returns coordinates in 0-1000

_active_windows: list = []  # keep refs so windows aren't GC'd while visible


def scale_normalized_regions(regions, screen_w: float, screen_h: float) -> list:
    """0-1000 normalized regions (top-left origin) → screen points, clamped."""
    out = []
    for r in regions or []:
        if not isinstance(r, dict):
            continue
        try:
            x = float(r["x"]) / _NORM * screen_w
            y = float(r["y"]) / _NORM * screen_h
            w = float(r["w"]) / _NORM * screen_w
            h = float(r["h"]) / _NORM * screen_h
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        x = max(0.0, min(x, screen_w))
        y = max(0.0, min(y, screen_h))
        w = min(w, screen_w - x)
        h = min(h, screen_h - y)
        if w <= 0 or h <= 0:
            continue
        out.append({"x": x, "y": y, "w": w, "h": h,
                    "label": str(r.get("label", ""))[:60]})
    return out


def _flip_y(y: float, h: float, screen_h: float) -> float:
    """Top-left origin → AppKit bottom-left origin."""
    return screen_h - y - h


def _show_on_main_thread(regions: list, duration: float) -> None:
    """Create and display the overlay window. Must be called with AppKit available."""
    from AppKit import (
        NSApplication, NSBackingStoreBuffered, NSBezierPath, NSColor, NSFont,
        NSFontAttributeName, NSForegroundColorAttributeName, NSMakeRect,
        NSOperationQueue, NSScreen, NSView, NSWindow,
        NSWindowStyleMaskBorderless, NSScreenSaverWindowLevel,
    )
    import objc

    NSApplication.sharedApplication()  # ensure the shared app exists

    screen = NSScreen.mainScreen()
    if screen is None:
        raise RuntimeError("no main screen")
    frame = screen.frame()
    screen_h = frame.size.height

    class _AnnotationView(NSView):
        def initWithFrame_regions_(self, rect, regs):
            self = objc.super(_AnnotationView, self).initWithFrame_(rect)
            if self is None:
                return None
            self._regions = regs
            return self

        def drawRect_(self, rect):
            stroke = NSColor.systemYellowColor()
            fill = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.85, 0.0, 0.12)
            font = NSFont.boldSystemFontOfSize_(15)
            for r in self._regions:
                ns_y = _flip_y(r["y"], r["h"], screen_h)
                box = NSMakeRect(r["x"], ns_y, r["w"], r["h"])
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(box, 6, 6)
                fill.setFill()
                path.fill()
                stroke.setStroke()
                path.setLineWidth_(3.0)
                path.stroke()
                label = r.get("label") or ""
                if label:
                    attrs = {
                        NSFontAttributeName: font,
                        NSForegroundColorAttributeName: NSColor.systemYellowColor(),
                    }
                    # label above the box, nudged inside the screen if needed
                    label_y = min(ns_y + r["h"] + 4, screen_h - 22)
                    label_pt = (r["x"], label_y)
                    __import__("Foundation").NSString.stringWithString_(label).drawAtPoint_withAttributes_(label_pt, attrs)

    def _build_and_show():
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setLevel_(NSScreenSaverWindowLevel)
        window.setIgnoresMouseEvents_(True)          # click-through
        window.setCollectionBehavior_(1 << 0)        # can join all spaces
        view = _AnnotationView.alloc().initWithFrame_regions_(frame, regions)
        window.setContentView_(view)
        window.orderFrontRegardless()                # show WITHOUT taking focus
        _active_windows.append(window)

        def _dismiss():
            def _close():
                try:
                    window.orderOut_(None)
                    _active_windows.remove(window)
                except Exception:
                    pass
            NSOperationQueue.mainQueue().addOperationWithBlock_(_close)

        threading.Timer(duration, _dismiss).start()

    NSOperationQueue.mainQueue().addOperationWithBlock_(_build_and_show)


def draw_boxes(regions: list, duration: float = _DEFAULT_DURATION) -> bool:
    """Show labeled boxes on screen for `duration` seconds. Never raises."""
    if not regions:
        return False
    try:
        _show_on_main_thread(regions, duration)
        return True
    except Exception as exc:
        logger.error("Overlay draw failed: %s", exc)
        return False


def screen_size() -> tuple:
    """Main screen size in points, (w, h). Falls back to (1440, 900)."""
    try:
        from AppKit import NSScreen
        f = NSScreen.mainScreen().frame()
        return float(f.size.width), float(f.size.height)
    except Exception:
        return 1440.0, 900.0
