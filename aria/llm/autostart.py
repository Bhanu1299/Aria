"""llm/autostart.py — start GateLLM on demand, stop it when Aria stops using it.

The first LLM call that finds the gateway down launches ~/GateLLM's uvicorn,
waits for /health, and retries. A watchdog stops the gateway after
GATEWAY_IDLE_SECONDS with no calls (never mid-request), and on Aria exit.

Only a gateway Aria started is ever stopped — one you run by hand is left alone.
The pid is kept in ~/.aria/gatellm.pid so a gateway orphaned by a crashed Aria
is adopted (and idle-stopped) by the next run.

  ensure_running(base_url) → bool   start if needed; True once /health answers
  in_flight()                       context manager around each request
  available(base_url) → bool        could we start it? (local URL + GateLLM installed)
  stop()                            stop the gateway if Aria owns it
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

import aria.core.config as config

logger = logging.getLogger(__name__)

_PIDFILE = Path.home() / ".aria" / "gatellm.pid"
_LOGFILE = Path.home() / ".aria" / "logs" / "gatellm.log"
_START_TIMEOUT = 20.0
_WATCH_INTERVAL = 30.0

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_pid: int | None = None          # pid we own (our child, or adopted from the pidfile)
_last_used = time.monotonic()
_active = 0
_watchdog: threading.Thread | None = None


def _uvicorn() -> Path:
    return Path(config.GATEWAY_DIR) / ".venv" / "bin" / "uvicorn"


def _host_port(base_url: str) -> tuple[str, int] | None:
    u = urlparse(base_url)
    if u.hostname not in ("127.0.0.1", "localhost"):
        return None
    return u.hostname, u.port or 80


def available(base_url: str) -> bool:
    return config.GATEWAY_AUTOSTART and _host_port(base_url) is not None and _uvicorn().exists()


def _healthy(base_url: str) -> bool:
    try:
        return httpx.get(base_url.rstrip("/") + "/health", timeout=1.0).status_code == 200
    except Exception:
        return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _adopt_orphan() -> None:
    """Take ownership of a gateway a previous (crashed) Aria started."""
    global _pid
    try:
        pid = int(_PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return
    if _alive(pid):
        _pid = pid
        logger.info("[GateLLM] adopted running gateway pid=%d", pid)
    else:
        _PIDFILE.unlink(missing_ok=True)


def ensure_running(base_url: str) -> bool:
    """Start GateLLM if it's down and we can. True once /health answers. Never raises."""
    global _proc, _pid
    if not available(base_url):
        return False
    with _lock:
        if _healthy(base_url):
            return True
        try:
            host, port = _host_port(base_url)
            _LOGFILE.parent.mkdir(parents=True, exist_ok=True)
            log = open(_LOGFILE, "ab")
            _proc = subprocess.Popen(
                [str(_uvicorn()), "llm_gateway.api.app:app", "--host", host, "--port", str(port)],
                cwd=config.GATEWAY_DIR, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            log.close()
            _pid = _proc.pid
            _PIDFILE.write_text(str(_pid))
        except Exception as exc:
            logger.error("[GateLLM] could not start: %s", exc)
            return False
        print(f"[Aria] Starting GateLLM (pid {_pid})…")
        deadline = time.monotonic() + _START_TIMEOUT
        while time.monotonic() < deadline:
            if _proc.poll() is not None:
                logger.error("[GateLLM] exited during startup (code %s) — see %s", _proc.returncode, _LOGFILE)
                _forget()
                return False
            if _healthy(base_url):
                _start_watchdog()
                return True
            time.sleep(0.3)
        logger.error("[GateLLM] not healthy after %.0fs — see %s", _START_TIMEOUT, _LOGFILE)
        return False


@contextlib.contextmanager
def in_flight():
    """Mark a request as running so the watchdog never stops the gateway under it."""
    global _active, _last_used
    with _lock:
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1
            _last_used = time.monotonic()


def _forget() -> None:
    global _proc, _pid
    _proc = None
    _pid = None
    _PIDFILE.unlink(missing_ok=True)


def stop() -> None:
    """Stop the gateway if Aria owns it. Never raises."""
    with _lock:
        if _pid is None:
            return
        pid = _pid
        try:
            if _proc is not None:
                _proc.terminate()
                try:
                    _proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _proc.kill()
            elif _alive(pid):
                os.kill(pid, signal.SIGTERM)
            print(f"[Aria] Stopped GateLLM (pid {pid}).")
        except Exception as exc:
            logger.warning("[GateLLM] stop failed: %s", exc)
        _forget()


def _watch() -> None:
    while True:
        time.sleep(_WATCH_INTERVAL)
        with _lock:
            if _pid is None:
                continue
            idle = time.monotonic() - _last_used
            if _active or idle < config.GATEWAY_IDLE_SECONDS:
                continue
        logger.info("[GateLLM] idle %.0fs — stopping", idle)
        stop()


def _start_watchdog() -> None:
    global _watchdog
    if _watchdog is None or not _watchdog.is_alive():
        _watchdog = threading.Thread(target=_watch, name="gatellm-idle", daemon=True)
        _watchdog.start()


def init() -> None:
    """Called once at Aria startup: adopt an orphaned gateway, stop it on exit."""
    if not config.GATEWAY_AUTOSTART:
        return
    with _lock:
        _adopt_orphan()
    if _pid is not None:
        _start_watchdog()
    atexit.register(stop)
