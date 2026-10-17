"""tests/test_pynput_safe.py — pynput keyboard-layout lookups stay on the main thread."""
from __future__ import annotations

import sys
import threading
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS TIS API")


def _spy(util, calls):
    """Record which thread asks for the input source, then return the real one."""
    real = util.CarbonExtra.TISCopyCurrentKeyboardInputSource

    def spy():
        calls.append(threading.current_thread().name)
        return real()
    return spy


@pytest.fixture
def fresh():
    import pynput._util.darwin as util
    import pynput.keyboard._darwin as kbd
    from aria.voice import pynput_safe
    saved = (util.keycode_context, kbd.keycode_context, pynput_safe._installed)
    pynput_safe._installed = False
    yield pynput_safe, util, kbd
    util.keycode_context, kbd.keycode_context, pynput_safe._installed = saved


def test_listener_threads_never_touch_tis_after_install(fresh):
    pynput_safe, util, kbd = fresh
    assert pynput_safe.install() is True

    calls = []
    with patch.object(util.CarbonExtra, "TISCopyCurrentKeyboardInputSource",
                      side_effect=_spy(util, calls)):
        results = []

        def listener_like():
            with kbd.keycode_context() as ctx:
                results.append(ctx)

        t = threading.Thread(target=listener_like, name="pynput-listener")
        t.start()
        t.join()
    assert calls == []                       # no TIS call from the listener thread
    assert results and results[0][1]         # but it still got real layout data


def test_unpatched_pynput_does_call_tis_from_listener_thread(fresh):
    """Control: proves the spy above would catch a TIS call if one happened."""
    pynput_safe, util, kbd = fresh
    calls = []
    with patch.object(util.CarbonExtra, "TISCopyCurrentKeyboardInputSource",
                      side_effect=_spy(util, calls)):
        def listener_like():
            with kbd.keycode_context():
                pass

        t = threading.Thread(target=listener_like, name="pynput-listener")
        t.start()
        t.join()
    assert calls == ["pynput-listener"]


def test_install_refuses_off_main_thread(fresh):
    pynput_safe, util, kbd = fresh
    before = kbd.keycode_context
    out = []
    t = threading.Thread(target=lambda: out.append(pynput_safe.install()))
    t.start()
    t.join()
    assert out == [False] and kbd.keycode_context is before


def test_install_is_idempotent(fresh):
    pynput_safe, util, kbd = fresh
    assert pynput_safe.install() is True
    patched = kbd.keycode_context
    assert pynput_safe.install() is True and kbd.keycode_context is patched
