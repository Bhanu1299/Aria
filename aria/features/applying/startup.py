"""applying/startup.py — bring up everything an application run needs, health-checked.

bring_up() → (ok, spoken message)
  1. GateLLM    started on demand (aria.llm.autostart) and /health checked
  2. Coldstart  a Python library, not a service: imported, find_emails present
  3. dashboard  its own process on 127.0.0.1:8787 (started detached if down, /health checked)
  4. Chrome     Aria's visible Chrome (launched detached, debug port on 127.0.0.1)
The first one that fails is named and nothing else runs. When all pass, the dashboard
opens in a tab of that Chrome, on the Live run tab.
"""
from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path

import httpx

import aria.core.config as config
import aria.web.agent_browser as agent_browser
from aria.core import paths

logger = logging.getLogger(__name__)

DASHBOARD_PORT = 8787
DASHBOARD_URL = f"http://127.0.0.1:{DASHBOARD_PORT}/"
DASHBOARD_TAB = "dashboard"
DASHBOARD_WAIT_SECS = 10.0
_DASHBOARD_LOG = Path.home() / ".aria" / "logs" / "dashboard.log"


def _gateway_problem():
    """None when GateLLM answers /health, else why not."""
    from aria.llm import autostart, check_health
    url = config.GATEWAY_URL
    if check_health(url) is None:
        return None
    if autostart.available(url) and autostart.ensure_running(url):
        return None
    return check_health(url) or "it didn't start"


def _import_coldstart():
    return importlib.import_module("coldstart.api")


def dashboard_up() -> bool:
    try:
        r = httpx.get(DASHBOARD_URL + "health", timeout=1.0)
        return r.status_code == 200 and r.text.strip() == "ok"
    except Exception:
        return False


def _spawn_dashboard() -> None:
    _DASHBOARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _DASHBOARD_LOG.open("ab") as log:
        subprocess.Popen([sys.executable, "-m", "aria.dashboard", "--no-browser", "--port", str(DASHBOARD_PORT)],
                         cwd=str(paths.ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         start_new_session=True)


def _fail(name: str, why) -> tuple:
    why = str(why).strip().rstrip(".") or "unknown error"
    return False, f"{name} isn't working: {why}. Stopping."


def bring_up(open_dashboard: bool = True) -> tuple:
    problem = _gateway_problem()
    if problem:
        return _fail("GateLLM", problem)

    try:
        cold = _import_coldstart()
        if not callable(getattr(cold, "find_emails", None)):
            return _fail("Coldstart", "it has no find_emails")
    except Exception as exc:
        return _fail("Coldstart", exc)

    if not dashboard_up():
        try:
            _spawn_dashboard()
        except Exception as exc:
            return _fail("The dashboard", exc)
        deadline = time.time() + DASHBOARD_WAIT_SECS
        while not dashboard_up():
            if time.time() > deadline:
                return _fail("The dashboard", f"nothing answered on port {DASHBOARD_PORT} (log: {_DASHBOARD_LOG})")
            time.sleep(0.3)

    try:
        agent_browser.run(lambda page: page.url)
    except Exception as exc:
        return _fail("Chrome", exc)

    if open_dashboard:
        try:
            agent_browser.navigate(DASHBOARD_URL + "#live", settle_secs=0.5, tab=DASHBOARD_TAB)
            agent_browser.bring_to_front(DASHBOARD_TAB)
        except Exception as exc:
            logger.warning("couldn't open the dashboard tab: %s", exc)
    return True, "GateLLM, Coldstart, the dashboard and Chrome are up."
