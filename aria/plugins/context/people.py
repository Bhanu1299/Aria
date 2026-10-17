"""plugins/context/people.py — handle normalization + macOS Contacts reader.

Every way of reaching a person (email, phone, display name) becomes a
typed alias string: "email:priya@acme.com", "tel:5551234567",
"name:priya sharma". The store links aliases that co-occur (a From
header, a Contacts card, a calendar attendee) to one person.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import sqlite3
from email.utils import getaddresses

logger = logging.getLogger(__name__)

_ADDRESSBOOK_GLOBS = [
    "~/Library/Application Support/AddressBook/AddressBook-v22.abcddb",
    "~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb",
]


def email_alias(addr: str) -> str:
    addr = (addr or "").strip().strip("<>").lower()
    return f"email:{addr}" if "@" in addr else ""


def phone_alias(number: str) -> str:
    digits = re.sub(r"\D", "", number or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return f"tel:{digits}" if len(digits) >= 7 else ""


def name_alias(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip().strip("'\"")).lower()
    if not name or "@" in name or not re.search(r"[a-z]", name):
        return ""
    return f"name:{name}"


def handle_alias(handle: str) -> str:
    """Alias for a raw handle that may be an email or a phone number."""
    return email_alias(handle) if "@" in (handle or "") else phone_alias(handle)


def parse_address_header(value: str) -> list:
    """'Priya <p@x.com>, b@y.com' → [('Priya', 'p@x.com'), ('', 'b@y.com')]."""
    try:
        return [(n.strip(), a.strip()) for n, a in getaddresses([value or ""]) if a]
    except Exception:
        return []


def read_contacts() -> list | str:
    """macOS Contacts cards as [{name, handles: [raw email/phone]}].

    Returns an error string if the AddressBook databases can't be read
    (Contacts needs the same Full Disk Access as Messages).
    """
    paths = []
    for pattern in _ADDRESSBOOK_GLOBS:
        paths.extend(glob.glob(os.path.expanduser(pattern)))
    if not paths:
        return "Contacts database not found or not readable (needs Full Disk Access)."

    cards: list = []
    errors = []
    for path in paths:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            people = {}
            for pk, first, last, org in conn.execute(
                "SELECT Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION FROM ZABCDRECORD"
            ):
                name = " ".join(p for p in (first, last) if p) or (org or "")
                if name:
                    people[pk] = {"name": name, "handles": []}
            for owner, number in conn.execute(
                "SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER"
            ):
                if owner in people and number:
                    people[owner]["handles"].append(number)
            for owner, addr in conn.execute(
                "SELECT ZOWNER, ZADDRESS FROM ZABCDEMAILADDRESS"
            ):
                if owner in people and addr:
                    people[owner]["handles"].append(addr)
            conn.close()
            cards.extend(c for c in people.values() if c["handles"])
        except Exception as exc:
            errors.append(str(exc))
    if not cards and errors:
        return f"Couldn't read Contacts ({errors[0]}). Grant Full Disk Access."
    return cards
