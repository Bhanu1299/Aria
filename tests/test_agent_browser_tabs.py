from __future__ import annotations

import threading

import pytest

import aria.web.agent_browser as agent_browser
import aria.web.browser_profile as browser_profile


class FakePage:
    def __init__(self, url: str = "about:blank", visible: bool = False) -> None:
        self.url = url
        self.visible = visible
        self.closed = False
        self.fronted = 0
        self.viewport = None

    def is_closed(self) -> bool:
        return self.closed

    def evaluate(self, js: str):
        assert "visibilityState" in js
        return "visible" if self.visible else "hidden"

    def bring_to_front(self) -> None:
        self.fronted += 1

    def close(self) -> None:
        self.closed = True

    def set_viewport_size(self, size: dict) -> None:
        self.viewport = size


class FakeContext:
    def __init__(self, pages=None) -> None:
        self.pages = list(pages or [])

    def new_page(self) -> FakePage:
        p = FakePage()
        self.pages.append(p)
        return p


@pytest.fixture
def browser(monkeypatch):
    shared = FakePage("https://shared.example/")
    ctx = FakeContext([shared])
    monkeypatch.setattr(agent_browser, "_context", ctx)
    monkeypatch.setattr(agent_browser, "_page", shared)
    monkeypatch.setattr(agent_browser, "_tabs", {})
    monkeypatch.setattr(agent_browser, "_open_browser", lambda: shared)
    monkeypatch.setattr(agent_browser, "_connect", lambda: None)
    return ctx, shared


def test_use_tab_routes_default_calls(browser):
    _ctx, shared = browser
    assert agent_browser.run(lambda page: page) is shared
    with agent_browser.use_tab("apply-15"):
        inside = agent_browser.run(lambda page: page)
        explicit = agent_browser.run(lambda page: page, tab="other")
    assert inside is not shared and inside is agent_browser._tabs["apply-15"]
    assert explicit is agent_browser._tabs["other"]
    assert agent_browser.run(lambda page: page) is shared


def test_gate_runs_before_each_call_on_that_thread_only(browser):
    calls = []
    other_thread_saw = []
    with agent_browser.use_tab("apply-15", gate=lambda: calls.append(1)):
        agent_browser.run(lambda page: None)
        agent_browser.navigate("https://x.example/", settle_secs=0)
        t = threading.Thread(target=lambda: other_thread_saw.append(agent_browser.run(lambda page: page)))
        t.start()
        t.join()
    assert len(calls) == 2
    assert other_thread_saw[0] is browser[1]


def test_attach_binds_by_host_and_path(browser):
    ctx, _shared = browser
    form = FakePage("https://job-boards.greenhouse.io/scaleai/jobs/4730836005?utm=x")
    ctx.pages.append(form)
    bound = agent_browser.attach({"apply-15": "https://job-boards.greenhouse.io/scaleai/jobs/4730836005/"})
    assert bound == {"apply-15"}
    assert agent_browser.run(lambda page: page, tab="apply-15") is form


def test_attach_skips_missing_pages(browser):
    assert agent_browser.attach({"apply-16": "https://job-boards.greenhouse.io/singlestore/jobs/1"}) == set()
    assert not agent_browser.has_tab("apply-16")


def test_front_tabs_uses_the_front_window_title(browser, monkeypatch):
    """visibilityState is 'visible' in every tab under Playwright's focus emulation, so the
    front tab comes from the Chrome window title (= its active tab's title)."""
    ctx, _ = browser
    a, b = FakePage("https://a.example/f"), FakePage("https://b.example/f")
    a.title = lambda: "Job Application for SWE at Scale AI"
    b.title = lambda: "Job Application for SWE at SingleStore"
    ctx.pages += [a, b]
    agent_browser.attach({"a": a.url, "b": b.url})
    monkeypatch.setattr(agent_browser, "_front_window_title", lambda: "Job Application for SWE at SingleStore")
    assert agent_browser.front_tabs(["a", "b", "missing"]) == ["b"]
    monkeypatch.setattr(agent_browser, "_front_window_title", lambda: "Job pipeline")      # dashboard in front
    assert agent_browser.front_tabs(["a", "b"]) == []
    monkeypatch.setattr(agent_browser, "_front_window_title", lambda: None)                # can't tell
    assert agent_browser.front_tabs(["a", "b"]) == []


def test_window_title_strips_chrome_suffix():
    assert agent_browser.strip_window_title("Job Application at Scale AI - Google Chrome") == "Job Application at Scale AI"
    assert agent_browser.strip_window_title("Ookla Careers - Apply - Google Chrome - Aria") == "Ookla Careers - Apply"


def test_tabs_never_come_to_front(browser):
    """Aria's Chrome never takes focus: no tab is fronted, and there's no API to do it."""
    agent_browser.run(lambda page: None, tab="apply-15")
    name = agent_browser.open_in_tab("https://example.com/")
    assert name.startswith("link-") and agent_browser.has_tab(name)
    assert all(p.fronted == 0 for p in agent_browser._tabs.values())
    assert not hasattr(agent_browser, "bring_to_front")


def test_new_tabs_are_created_in_the_background(monkeypatch):
    """Target.createTarget(background=True): Chrome doesn't switch to the tab or raise its window."""
    sent = []
    made = FakePage()

    class Cdp:
        def send(self, method, params):
            sent.append((method, params))

        def detach(self):
            pass

    class Browser:
        def new_browser_cdp_session(self):
            return Cdp()

    class Expect:
        value = made

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Ctx:
        def expect_page(self, timeout=0):
            return Expect()

        def new_page(self):
            pytest.fail("new_page() brings the tab to front")
    monkeypatch.setattr(agent_browser, "_browser", Browser())
    monkeypatch.setattr(agent_browser, "_context", Ctx())
    assert agent_browser._new_tab() is made
    assert sent == [("Target.createTarget", {"url": "about:blank", "background": True})]


def test_chrome_launches_without_activating(monkeypatch, tmp_path):
    monkeypatch.setattr(browser_profile, "PROFILE_DIR", str(tmp_path / "profile"))
    monkeypatch.setattr(browser_profile, "_CHROME_LOG", tmp_path / "chrome.log")
    ups = iter([False, True])
    monkeypatch.setattr(browser_profile, "cdp_up", lambda port=0: next(ups, True))
    launched = []
    monkeypatch.setattr(browser_profile.subprocess, "Popen", lambda *a, **k: launched.append(a[0]))
    browser_profile.launch_chrome(wait_secs=1)
    assert launched[0][:3] == ["open", "-g", "-n"]                 # -g: don't bring Chrome forward


def test_close_tab_forgets_name(browser):
    page = agent_browser.run(lambda page: page, tab="apply-15")
    assert agent_browser.close_tab("apply-15") is True
    assert page.closed and not agent_browser.has_tab("apply-15")
    assert agent_browser.close_tab("apply-15") is False


def test_new_tabs_get_fixed_viewport(browser):
    page = agent_browser.run(lambda page: page, tab="apply-15")
    assert page.viewport == {"width": 1280, "height": 900}


def test_launch_reports_locked_profile(monkeypatch, tmp_path):
    (tmp_path / "SingletonLock").symlink_to("host-123")
    monkeypatch.setattr(browser_profile, "PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(browser_profile, "cdp_up", lambda port=0: False)
    monkeypatch.setattr(browser_profile, "_pid_alive", lambda pid: pid == 123)   # Chrome holding it is alive
    launched = []
    monkeypatch.setattr(browser_profile.subprocess, "Popen", lambda *a, **k: launched.append(a))
    with pytest.raises(RuntimeError, match="without remote debugging"):
        browser_profile.launch_chrome(wait_secs=0.2)
    assert launched == []


def test_launch_is_noop_when_cdp_up(monkeypatch):
    monkeypatch.setattr(browser_profile, "cdp_up", lambda port=0: True)
    monkeypatch.setattr(browser_profile.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    browser_profile.launch_chrome()


def test_closed_shared_page_keeps_tab_bindings(monkeypatch):
    """Closing Aria's blank tab must not drop the application-tab bindings."""
    shared = FakePage("about:blank")
    ctx = FakeContext([shared])

    class Browser:
        def is_connected(self):
            return True
    monkeypatch.setattr(agent_browser, "_browser", Browser())
    monkeypatch.setattr(agent_browser, "_context", ctx)
    monkeypatch.setattr(agent_browser, "_page", shared)
    form = FakePage("https://job-boards.greenhouse.io/scaleai/jobs/1")
    monkeypatch.setattr(agent_browser, "_tabs", {"apply-15": form})
    shared.closed = True
    page = agent_browser.run(lambda page: page)
    assert page is not shared and not page.closed
    assert agent_browser.run(lambda page: page, tab="apply-15") is form


def test_stale_singleton_lock_from_a_dead_chrome_still_launches(monkeypatch, tmp_path):
    (tmp_path / "SingletonLock").symlink_to("Some-Mac.local-999999")
    monkeypatch.setattr(browser_profile, "PROFILE_DIR", str(tmp_path))
    ups = iter([False, True])
    monkeypatch.setattr(browser_profile, "cdp_up", lambda port=0: next(ups, True))
    launched = []
    monkeypatch.setattr(browser_profile.subprocess, "Popen", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(browser_profile, "_pid_alive", lambda pid: False)
    browser_profile.launch_chrome(wait_secs=1)
    assert launched and not (tmp_path / "SingletonLock").is_symlink()
