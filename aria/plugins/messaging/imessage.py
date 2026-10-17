"""plugins/messaging/imessage.py — iMessage read/send via chat.db + osascript."""
from __future__ import annotations

import difflib
import logging
import os
import sqlite3
import subprocess
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_CHAT_DB = os.path.expanduser("~/Library/Messages/chat.db")
_PERMISSION_MSG = (
    "I need Full Disk Access to read iMessages. "
    "Go to System Settings → Privacy → Full Disk Access and add Terminal (or Aria)."
)
_NOT_RUNNING_MSG = "iMessage isn't responding. Is the Messages app running?"


class iMessageClient:
    def get_contacts(self) -> list:
        """Return recent contacts as list of {handle_id, display_name}."""
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT ROWID, id FROM handle ORDER BY ROWID DESC LIMIT 100"
            ).fetchall()
            conn.close()
            return [{"handle_id": r[1], "display_name": r[1]} for r in rows]
        except PermissionError:
            return []
        except Exception as exc:
            logger.warning("imessage.get_contacts failed: %s", exc)
            return []

    def resolve_contact(self, name: str) -> str | None:
        """
        Resolve a name to a handle ID (phone/email).
        Returns the handle_id string, or None if not found.
        """
        name = name.strip()
        if not name:
            return None
        # Direct match (phone number or email)
        if name.startswith("+") or "@" in name:
            return name
        try:
            conn = self._connect()
            rows = conn.execute("SELECT id FROM handle").fetchall()
            conn.close()
            handles = [r[0] for r in rows]
        except Exception:
            return name  # best-effort passthrough

        matches = difflib.get_close_matches(name.lower(),
                                            [h.lower() for h in handles],
                                            n=1, cutoff=0.6)
        if matches:
            idx = [h.lower() for h in handles].index(matches[0])
            return handles[idx]
        return None

    def read(self, contact: str, limit: int = 5) -> list | str:
        """
        Return up to `limit` recent messages with/from contact.
        Each item: {sender, text, timestamp, is_from_me}.
        Returns an error string on permission / not-found errors.
        """
        limit = min(max(1, limit), 20)
        handle = self.resolve_contact(contact)
        if handle is None:
            return f"I couldn't find a contact named {contact!r}. Try using their phone number."
        try:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT m.is_from_me, m.text, m.date, h.id
                FROM message m
                JOIN handle h ON m.handle_id = h.ROWID
                WHERE h.id = ?
                ORDER BY m.date DESC
                LIMIT ?
                """,
                (handle, limit),
            ).fetchall()
            conn.close()
        except PermissionError:
            return _PERMISSION_MSG
        except Exception as exc:
            logger.warning("imessage.read failed: %s", exc)
            return f"I couldn't read messages: {exc}"

        if not rows:
            return f"No messages found with {contact}."

        messages = []
        for row in reversed(rows):
            is_from_me, text, mac_ts, sender_id = row
            # macOS timestamps: seconds since 2001-01-01 (in nanoseconds for newer rows)
            try:
                if mac_ts > 1e15:
                    mac_ts = mac_ts / 1e9
                ts = datetime.fromtimestamp(mac_ts + 978307200).strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts = "unknown"
            messages.append({
                "sender": "You" if is_from_me else sender_id,
                "text": text or "(media)",
                "timestamp": ts,
                "is_from_me": bool(is_from_me),
            })
        return messages

    def recent(self, days: int = 30, limit: int = 3000) -> list | str:
        """All 1:1 messages newer than `days`, oldest first (for the context index).

        Each item: {handle, text, ts, is_from_me}; ts is epoch seconds.
        Newer macOS leaves `text` NULL and stores it in attributedBody.
        """
        cutoff_s = time.time() - 978307200 - max(1, days) * 86400
        try:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT m.is_from_me, m.text, m.attributedBody, m.date, h.id
                FROM message m
                JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.date > ?
                ORDER BY m.date ASC
                LIMIT ?
                """,
                (cutoff_s * 1e9, limit),
            ).fetchall()
            conn.close()
        except PermissionError:
            return _PERMISSION_MSG
        except sqlite3.DatabaseError as exc:
            if "authorization" in str(exc).lower():
                return _PERMISSION_MSG
            return f"I couldn't read messages: {exc}"
        except Exception as exc:
            logger.warning("imessage.recent failed: %s", exc)
            return f"I couldn't read messages: {exc}"

        out = []
        for is_from_me, text, attributed, mac_ts, handle in rows:
            body = text or decode_attributed_body(attributed)
            if not body:
                continue
            ts = mac_ts / 1e9 if mac_ts > 1e15 else mac_ts
            out.append({"handle": handle, "text": body,
                        "ts": ts + 978307200, "is_from_me": bool(is_from_me)})
        return out

    def send(self, contact: str, message: str, service: str = "auto") -> str:
        """Send a message via AppleScript. Returns 'Sent.' or error string."""
        handle = self.resolve_contact(contact)
        if handle is None:
            return f"I couldn't find a contact named {contact!r}."

        svc = "iMessage" if service == "iMessage" else "SMS" if service == "SMS" else ""
        escaped_msg = message.replace('"', '\\"').replace("\\", "\\\\")
        escaped_handle = handle.replace('"', '\\"')

        if svc:
            script = f"""
tell application "Messages"
    set targetService to 1st service whose service type = {svc}
    set targetBuddy to buddy "{escaped_handle}" of targetService
    send "{escaped_msg}" to targetBuddy
end tell
"""
        else:
            script = f"""
tell application "Messages"
    send "{escaped_msg}" to buddy "{escaped_handle}" of (1st service whose service type = iMessage)
end tell
"""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                err = result.stderr.strip()
                if "not running" in err.lower():
                    return _NOT_RUNNING_MSG
                return f"I couldn't send that message: {err or 'unknown error'}"
            return "Sent."
        except subprocess.TimeoutExpired:
            return _NOT_RUNNING_MSG
        except Exception as exc:
            return f"I couldn't send that message: {exc}"

    def _connect(self) -> sqlite3.Connection:
        if not os.path.exists(_CHAT_DB):
            raise PermissionError("chat.db not found")
        conn = sqlite3.connect(_CHAT_DB, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn


def decode_attributed_body(blob) -> str:
    """Pull the plain string out of a chat.db attributedBody (typedstream) blob.

    The NSString payload follows the class name: 5 header bytes, then a
    length byte (or 0x81 + 2-byte little-endian length), then UTF-8 text.
    Returns '' for anything that doesn't fit that shape.
    """
    if not blob:
        return ""
    try:
        data = bytes(blob).split(b"NSString", 1)[1][5:]
        if data[0] == 0x81:
            length = int.from_bytes(data[1:3], "little")
            data = data[3:3 + length]
        else:
            length = data[0]
            data = data[1:1 + length]
        return data.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
