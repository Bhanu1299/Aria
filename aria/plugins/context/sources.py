"""plugins/context/sources.py — turn each app's data into context items.

Each fetcher returns (items, error). Items are store.upsert_items() dicts;
error is '' on success or a short human reason ("Google isn't connected").
Fetchers never raise — a dead source just reports its error.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from aria.plugins.context.people import email_alias, parse_address_header

logger = logging.getLogger(__name__)

_IMSG_LINES_PER_DAY = 12


def sync_contacts(store, read_contacts) -> tuple:
    """Link every Contacts card's phones + emails to its name. Returns (n, error)."""
    try:
        cards = read_contacts()
        if isinstance(cards, str):
            return 0, cards
        for card in cards:
            store.link(card["name"], card["handles"], authoritative=True)
        return len(cards), ""
    except Exception as exc:
        logger.warning("context: contacts sync failed: %s", exc)
        return 0, f"Contacts sync failed: {exc}"


def gmail_items(client, days: int, skip_ids: set) -> tuple:
    try:
        me = client.my_address()
        msgs = client.recent_messages(days=days, limit=200, skip_ids=skip_ids)
        if isinstance(msgs, str):
            return [], msgs
        items = []
        for m in msgs:
            people = []
            for field in ("from", "to", "cc"):
                for name, addr in parse_address_header(m.get(field, "")):
                    if me and email_alias(addr) == email_alias(me):
                        continue
                    people.append({"name": name, "handles": [addr]})
            sender = parse_address_header(m.get("from", ""))
            who = (sender[0][0] or sender[0][1]) if sender else ""
            items.append({
                "id": f"email:{m['id']}",
                "source": "email",
                "ts": m["ts"],
                "title": m.get("subject", "") or "(no subject)",
                "body": f"{who}: {m.get('snippet', '')}" if who else m.get("snippet", ""),
                "ref": m["id"],
                "participants": people,
            })
        return items, ""
    except Exception as exc:
        logger.warning("context: gmail fetch failed: %s", exc)
        return [], f"Gmail fetch failed: {exc}"


def calendar_items(client, past_days: int = 7, future_days: int = 14) -> tuple:
    try:
        now = datetime.now(timezone.utc)
        events = client.list_events(
            (now - timedelta(days=past_days)).isoformat(),
            (now + timedelta(days=future_days)).isoformat(),
            max_results=150,
        )
        if isinstance(events, str):
            return [], events
        items = []
        for e in events:
            ts = _parse_event_start(e.get("start", ""))
            if ts is None:
                continue
            body = " ".join(p for p in (e.get("location", ""), e.get("description", "")[:500]) if p)
            items.append({
                "id": f"calendar:{e['id']}",
                "source": "calendar",
                "ts": ts,
                "title": e.get("title", "(no title)"),
                "body": body,
                "ref": e["id"],
                "participants": [{"name": a.get("name", ""), "handles": [a.get("email", "")]}
                                 for a in e.get("attendees", []) if not a.get("self")],
            })
        return items, ""
    except Exception as exc:
        logger.warning("context: calendar fetch failed: %s", exc)
        return [], f"Calendar fetch failed: {exc}"


def imessage_items(client, days: int) -> tuple:
    """One item per (contact, local day) so a chat reads as a conversation.

    The window's first day is cut mid-day, so it's dropped — re-indexing it
    would overwrite a complete earlier item with a partial one.
    """
    try:
        msgs = client.recent(days=days)
        if isinstance(msgs, str):
            return [], msgs
        partial_day = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
        days_by_key: "OrderedDict[tuple, list]" = OrderedDict()
        for m in msgs:
            day = time.strftime("%Y-%m-%d", time.localtime(m["ts"]))
            if day <= partial_day:
                continue
            days_by_key.setdefault((m["handle"], day), []).append(m)
        items = []
        for (handle, day), convo in days_by_key.items():
            lines = [f"{'You' if m['is_from_me'] else 'Them'}: {m['text']}"
                     for m in convo[-_IMSG_LINES_PER_DAY:]]
            items.append({
                "id": f"imessage:{handle}:{day}",
                "source": "imessage",
                "ts": convo[-1]["ts"],
                "title": f"{len(convo)} message{'s' if len(convo) != 1 else ''}",
                "body": " | ".join(lines)[:800],
                "ref": handle,
                "participants": [{"name": "", "handles": [handle]}],
            })
        return items, ""
    except Exception as exc:
        logger.warning("context: imessage fetch failed: %s", exc)
        return [], f"iMessage fetch failed: {exc}"


def _parse_event_start(value: str) -> float | None:
    if not value:
        return None
    try:
        if len(value) == 10:  # all-day: YYYY-MM-DD
            return datetime.strptime(value, "%Y-%m-%d").timestamp()
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
