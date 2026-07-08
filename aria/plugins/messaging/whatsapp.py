"""plugins/messaging/whatsapp.py — WhatsApp client (session-based, graceful fallback)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CREDS_DIR = Path.home() / ".aria" / "credentials" / "whatsapp"
_NOT_SETUP_MSG = (
    "WhatsApp isn't set up yet. Run: python -m aria.setup whatsapp"
)
_DISCONNECTED_MSG = "WhatsApp lost its connection. I'll try to reconnect."


class WhatsAppClient:
    """
    WhatsApp Web client backed by whatsapp-web.py when available.
    All methods return helpful strings when WhatsApp isn't configured
    or the underlying library is unavailable.
    """

    def __init__(self) -> None:
        self._client = None
        self._lock = threading.Lock()
        self._message_ids_seen: set = set()
        self._ready = False
        if self.is_configured():
            self._try_connect()

    @staticmethod
    def is_configured() -> bool:
        return _CREDS_DIR.exists() and any(_CREDS_DIR.iterdir()) if _CREDS_DIR.exists() else False

    def get_contacts(self) -> list | str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        client = self._get_client()
        if client is None:
            return _DISCONNECTED_MSG
        try:
            chats = client.get_chats()
            return [{"chat_id": c.id, "display_name": c.name or c.id} for c in chats[:50]]
        except Exception as exc:
            logger.warning("whatsapp.get_contacts failed: %s", exc)
            return f"I couldn't get WhatsApp contacts: {exc}"

    def read(self, contact: str, limit: int = 5) -> list | str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        client = self._get_client()
        if client is None:
            return _DISCONNECTED_MSG
        limit = min(max(1, limit), 20)
        try:
            chat = self._find_chat(client, contact)
            if chat is None:
                return f"I couldn't find a WhatsApp contact named {contact!r}. Try using their phone number."
            messages = chat.fetch_messages(limit=limit)
            result = []
            for m in messages:
                result.append({
                    "sender": "You" if m.from_me else (m.author or contact),
                    "text": m.body or "(media)",
                    "timestamp": _fmt_ts(m.timestamp),
                    "is_from_me": m.from_me,
                })
            return result
        except Exception as exc:
            logger.warning("whatsapp.read failed: %s", exc)
            return f"I couldn't read WhatsApp messages: {exc}"

    def send(self, contact: str, message: str) -> str:
        if not self.is_configured():
            return _NOT_SETUP_MSG
        client = self._get_client()
        if client is None:
            return _DISCONNECTED_MSG
        try:
            chat = self._find_chat(client, contact)
            if chat is None:
                return f"I couldn't find a WhatsApp contact named {contact!r}. Try using their phone number."
            msg = chat.send_message(message)
            # Debounce: record message ID to prevent echo
            if msg and hasattr(msg, "id"):
                with self._lock:
                    self._message_ids_seen.add(msg.id)
            return "Sent."
        except Exception as exc:
            logger.warning("whatsapp.send failed: %s", exc)
            return f"I couldn't send that WhatsApp message: {exc}"

    def _get_client(self):
        with self._lock:
            if self._client is not None and self._ready:
                return self._client
        return None

    def _try_connect(self) -> None:
        def _connect():
            try:
                from whatsapp import WAClient  # type: ignore
                client = WAClient(str(_CREDS_DIR))
                client.connect()
                with self._lock:
                    self._client = client
                    self._ready = True
                logger.info("whatsapp: connected")
            except ImportError:
                logger.info("whatsapp: library not available")
            except Exception as exc:
                logger.warning("whatsapp: connect failed: %s", exc)
        t = threading.Thread(target=_connect, daemon=True, name="whatsapp-connect")
        t.start()

    def _find_chat(self, client, contact: str):
        """Find a chat by contact name or phone number. Returns None if not found."""
        try:
            chats = client.get_chats()
        except Exception:
            return None
        # Direct ID match
        for chat in chats:
            if contact in (chat.id, getattr(chat, "name", "")):
                return chat
        # Fuzzy name match
        import difflib
        names = [getattr(c, "name", "") or "" for c in chats]
        matches = difflib.get_close_matches(contact.lower(),
                                            [n.lower() for n in names],
                                            n=1, cutoff=0.6)
        if matches:
            idx = [n.lower() for n in names].index(matches[0])
            return chats[idx]
        return None


def _fmt_ts(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except Exception:
        return "unknown"
