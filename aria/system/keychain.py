"""system/keychain.py — portal passwords in the macOS Keychain, one entry per site.

get(host, username)              → Credential or None
save(host, username, password)   → True if stored
delete(host, username)           → True if removed
generate(length)                 → a strong password that satisfies the usual portal rules
                                   (upper, lower, digit, symbol; no ambiguous characters)

Keyed by the tenant host ("acme.wd5.myworkdayjobs.com") — every Workday tenant is its own
account — under service "aria-apply:<host>", account = the login email.

Rules: a password is read only where it's typed into the page. It never goes into an LLM
prompt, a log line, a review file, a correction row or the answer bank. Credential's repr
hides it, so an accidental log of the object shows nothing. Never raises: a locked or
missing Keychain → None / False.
"""
from __future__ import annotations

import logging
import secrets
import string
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SERVICE_PREFIX = "aria-apply"
_SYMBOLS = "!@#$%^&*-_=+"
_ALPHABET = "".join(c for c in string.ascii_letters + string.digits if c not in "O0Il1") + _SYMBOLS


@dataclass(frozen=True)
class Credential:
    host: str
    username: str
    password: str = field(repr=False)

    def __repr__(self) -> str:
        return f"Credential(host={self.host!r}, username={self.username!r}, password=<hidden>)"


def service(host: str) -> str:
    return f"{SERVICE_PREFIX}:{(host or '').strip().lower()}"


def _backend():
    import keyring
    return keyring


def get(host: str, username: str) -> Optional[Credential]:
    if not host or not username:
        return None
    try:
        pw = _backend().get_password(service(host), username)
    except Exception as exc:
        logger.warning("keychain: read for %s failed (%s)", host, type(exc).__name__)
        return None
    return Credential(host.lower(), username, pw) if pw else None


def save(host: str, username: str, password: str) -> bool:
    if not (host and username and password):
        return False
    try:
        _backend().set_password(service(host), username, password)
        logger.info("keychain: saved the login for %s", host)
        return True
    except Exception as exc:
        logger.warning("keychain: save for %s failed (%s)", host, type(exc).__name__)
        return False


def delete(host: str, username: str) -> bool:
    try:
        _backend().delete_password(service(host), username)
        return True
    except Exception as exc:
        logger.debug("keychain: delete for %s: %s", host, type(exc).__name__)
        return False


def generate(length: int = 20) -> str:
    length = max(12, length)
    while True:
        pw = "".join(secrets.choice(_ALPHABET) for _ in range(length))
        if (any(c.islower() for c in pw) and any(c.isupper() for c in pw)
                and any(c.isdigit() for c in pw) and any(c in _SYMBOLS for c in pw)):
            return pw
