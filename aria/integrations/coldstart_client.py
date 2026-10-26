"""integrations/coldstart_client.py — Coldstart, the email-finding service, over HTTP.

Aria and Coldstart stay decoupled: the only contract is

    POST {COLDSTART_URL}/find   {"name": "Pat Lee", "company": "Acme", "domain": "acme.com"}
    →   {"email": "pat@acme.com", "status": "verified" | "pattern_guess" | "not_found",
         "confidence": 0.93, "pattern": "{first}.{last}"}          (extra keys ignored)

    client = ColdstartClient()
    client.ensure_running() → ''                    or why Coldstart can't be reached
    client.find(name, company, domain) → Lookup     never raises; Lookup.error on failure

ensure_running() starts a local Coldstart that's down (`python -m coldstart serve` from
COLDSTART_DIR's venv, log ~/.aria/logs/coldstart.log) and stops it again when Aria exits.
One you run by hand is left alone.

Lookup.confidence is the sheet's label: verified | pattern | not found (pattern_guess
and not_found are mapped). Calls are spaced at least COLDSTART_MIN_INTERVAL seconds
apart (one client = one rate limit). A domain's pattern is cached for the client's
lifetime (seeded from the sheet with remember()), and find() answers later contacts at
that domain from the cache without calling Coldstart.
"""
from __future__ import annotations

import atexit
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

import aria.core.config as config

logger = logging.getLogger(__name__)

VERIFIED = "verified"
PATTERN = "pattern"
NOT_FOUND = "not found"
_STATUS = {"verified": VERIFIED, "valid": VERIFIED, "pattern_guess": PATTERN, "pattern": PATTERN,
           "not_found": NOT_FOUND, "not found": NOT_FOUND}

# Bare pattern spellings Coldstart or a human may write, as brace templates.
_BARE = {
    "first": "{first}", "last": "{last}", "first.last": "{first}.{last}", "firstlast": "{first}{last}",
    "first_last": "{first}_{last}", "first-last": "{first}-{last}", "flast": "{f}{last}",
    "f.last": "{f}.{last}", "firstl": "{first}{l}", "first.l": "{first}.{l}", "last.first": "{last}.{first}",
    "lastfirst": "{last}{first}", "lastf": "{last}{f}", "fl": "{f}{l}",
}
_LOGFILE = Path.home() / ".aria" / "logs" / "coldstart.log"
_START_TIMEOUT = 30.0
_start_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_TOKEN = re.compile(r"\{(first|last|f|l|first_initial|last_initial)\}")


@dataclass
class Lookup:
    email: str = ""
    confidence: str = NOT_FOUND     # verified | pattern | not found
    pattern: str = ""
    score: Optional[float] = None   # Coldstart's numeric confidence, when it sent one
    source: str = "Coldstart"
    error: str = ""                 # set when the call failed; nothing else is meaningful then

    @property
    def ok(self) -> bool:
        return not self.error


def _name_parts(name: str) -> tuple[str, str]:
    words = [re.sub(r"[^a-z]", "", w) for w in (name or "").lower().split()]
    words = [w for w in words if w]
    if len(words) < 2:
        return (words[0] if words else ""), ""
    return words[0], words[-1]


def normalize_pattern(pattern: str) -> str:
    """'{first}.{last}@acme.com' / 'first.last' / '{f}{last}' → a brace template, or '' if unusable."""
    p = (pattern or "").strip().lower().split("@")[0]
    p = p.replace("{first_initial}", "{f}").replace("{last_initial}", "{l}")
    if p in _BARE:
        return _BARE[p]
    return p if _TOKEN.search(p) and not re.search(r"[^a-z0-9{}._-]", p) else ""


def render(pattern: str, name: str, domain: str) -> str:
    """The address `pattern` gives `name` at `domain`, or '' when it can't (single-word name, bad pattern)."""
    tpl = normalize_pattern(pattern)
    first, last = _name_parts(name)
    if not tpl or not domain or not first or (("{last}" in tpl or "{l}" in tpl) and not last):
        return ""
    local = tpl.format(first=first, last=last, f=first[:1], l=last[:1])
    return f"{local}@{domain.lower()}" if local and "{" not in local else ""


def parse(data) -> Lookup:
    """Coldstart's JSON → Lookup (mapped status)."""
    if not isinstance(data, dict):
        return Lookup(error="Coldstart sent something that isn't a JSON object")
    raw = str(data.get("status") or "").strip().lower()
    label = _STATUS.get(raw)
    if label is None:
        return Lookup(error=f"Coldstart sent an unknown status {raw!r}")
    email = str(data.get("email") or "").strip()
    if label != NOT_FOUND and "@" not in email:
        return Lookup(error=f"Coldstart said {raw} but sent no address")
    try:
        score = float(data["confidence"]) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        score = None
    return Lookup(email=email if label != NOT_FOUND else "", confidence=label,
                  pattern=str(data.get("pattern") or "").strip(), score=score,
                  source=f"Coldstart ({data['source']})" if data.get("source") else "Coldstart")


class ColdstartClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        min_interval: Optional[float] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.Client] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = (base_url or config.COLDSTART_URL).rstrip("/")
        self.min_interval = config.COLDSTART_MIN_INTERVAL if min_interval is None else min_interval
        self.timeout = config.COLDSTART_TIMEOUT if timeout is None else timeout
        self._client = client
        self._clock, self._sleep = clock, sleep
        self._last: Optional[float] = None
        self._lock = threading.Lock()
        self.patterns: dict[str, str] = {}     # domain → brace template
        self.calls = 0

    # -- pattern cache -----------------------------------------------------

    def remember(self, domain: str, pattern: str) -> None:
        tpl = normalize_pattern(pattern)
        if domain and tpl:
            self.patterns.setdefault(domain.lower(), tpl)

    def from_cache(self, name: str, domain: str) -> Optional[Lookup]:
        tpl = self.patterns.get((domain or "").lower())
        email = render(tpl, name, domain) if tpl else ""
        if not email:
            return None
        return Lookup(email=email, confidence=PATTERN, pattern=tpl, source=f"Coldstart pattern cache ({domain})")

    # -- HTTP ----------------------------------------------------------------

    def _wait_turn(self) -> None:
        if self._last is not None:
            gap = self.min_interval - (self._clock() - self._last)
            if gap > 0:
                self._sleep(gap)
        self._last = self._clock()

    def _post(self, payload: dict):
        http = self._client or httpx.Client()
        try:
            r = http.post(f"{self.base_url}/find", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        finally:
            if self._client is None:
                http.close()

    def find(self, name: str, company: str, domain: str, use_cache: bool = True) -> Lookup:
        """One contact. Never raises: a failure comes back as Lookup(error=...)."""
        if use_cache:
            hit = self.from_cache(name, domain)
            if hit:
                return hit
        payload = {"name": name, "company": company, "domain": domain}
        with self._lock:
            self._wait_turn()
            self.calls += 1
            try:
                data = self._post(payload)
            except httpx.HTTPStatusError as exc:
                return Lookup(error=f"Coldstart answered HTTP {exc.response.status_code}")
            except httpx.HTTPError as exc:
                return Lookup(error=f"Coldstart unreachable at {self.base_url}: {exc}")
            except ValueError:
                return Lookup(error="Coldstart sent invalid JSON")
        out = parse(data)
        if out.ok:
            self.remember(domain, out.pattern)
        logger.info("coldstart: %s @ %s → %s %s", name, domain, out.confidence if out.ok else "error",
                    out.email or out.error)
        return out

    def ensure_running(self) -> str:
        """'' once Coldstart answers /health, starting a local one if it's down; else why not."""
        why = self.health()
        if not why:
            return ""
        port = _local_port(self.base_url)
        python = Path(config.COLDSTART_DIR) / "venv" / "bin" / "python"
        if not (config.COLDSTART_AUTOSTART and port and python.exists()):
            return f"Coldstart isn't running at {self.base_url} ({why})"
        with _start_lock:
            if _proc is None or _proc.poll() is not None:
                _start(python, port)
            deadline = self._clock() + _START_TIMEOUT
            while self._clock() < deadline:
                if not self.health():
                    logger.info("coldstart: started at %s", self.base_url)
                    return ""
                if _proc is not None and _proc.poll() is not None:
                    break
                self._sleep(0.5)
        return f"Coldstart didn't start at {self.base_url}; see {_LOGFILE}"

    def health(self) -> str:
        """'' when Coldstart answers GET /health, else why not."""
        http = self._client or httpx.Client()
        try:
            r = http.get(f"{self.base_url}/health", timeout=3.0)
            return "" if r.status_code == 200 else f"HTTP {r.status_code}"
        except httpx.HTTPError as exc:
            return str(exc) or exc.__class__.__name__
        finally:
            if self._client is None:
                http.close()


def _local_port(base_url: str) -> Optional[int]:
    u = urlparse(base_url)
    return (u.port or 80) if u.hostname in ("127.0.0.1", "localhost") else None


def _start(python: Path, port: int) -> None:
    global _proc
    _LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(_LOGFILE, "ab")
    _proc = subprocess.Popen([str(python), "-m", "coldstart", "serve", "--port", str(port)],
                             cwd=config.COLDSTART_DIR, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    logger.info("coldstart: starting (pid %s, port %s)", _proc.pid, port)


@atexit.register
def stop() -> None:
    """Stop the Coldstart Aria started, if any."""
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None
