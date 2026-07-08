"""Tests for overlay.py — screen annotation overlay coordinate math + safety."""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aria.ui.overlay as overlay


def test_scale_normalized_regions_maps_0_1000_to_screen_points():
    regions = [{"x": 500, "y": 250, "w": 100, "h": 50, "label": "error"}]
    scaled = overlay.scale_normalized_regions(regions, screen_w=1440, screen_h=900)
    assert scaled == [{"x": 720.0, "y": 225.0, "w": 144.0, "h": 45.0, "label": "error"}]


def test_scale_normalized_regions_clamps_out_of_range():
    regions = [{"x": -50, "y": 990, "w": 200, "h": 200, "label": "edge"}]
    scaled = overlay.scale_normalized_regions(regions, screen_w=1000, screen_h=1000)
    r = scaled[0]
    assert r["x"] >= 0 and r["y"] >= 0
    assert r["y"] + r["h"] <= 1000
    assert r["x"] + r["w"] <= 1000


def test_scale_normalized_regions_drops_malformed_entries():
    regions = [
        {"x": 10, "y": 10, "w": 100, "h": 100, "label": "good"},
        {"x": "junk", "label": "bad"},
        "not even a dict",
        {"x": 10, "y": 10, "w": 0, "h": 0, "label": "zero-size"},
    ]
    scaled = overlay.scale_normalized_regions(regions, screen_w=1000, screen_h=1000)
    assert len(scaled) == 1
    assert scaled[0]["label"] == "good"


def test_flip_y_converts_topleft_to_appkit_bottomleft():
    # region at top of a 900pt screen → near y=900 in AppKit coords
    assert overlay._flip_y(y=0, h=100, screen_h=900) == 800
    # region at bottom → y=0 in AppKit coords
    assert overlay._flip_y(y=800, h=100, screen_h=900) == 0


def test_draw_boxes_returns_false_for_empty_regions():
    assert overlay.draw_boxes([]) is False


def test_draw_boxes_never_raises_when_appkit_unavailable():
    with patch.object(overlay, "_show_on_main_thread", side_effect=RuntimeError("no NSApp")):
        result = overlay.draw_boxes([{"x": 10, "y": 10, "w": 50, "h": 50, "label": "x"}])
    assert result is False
