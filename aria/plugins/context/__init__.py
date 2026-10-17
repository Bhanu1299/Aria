"""plugins/context/__init__.py — ContextPlugin: one timeline across every app.

Gmail, Calendar and iMessage are indexed locally (~/.aria/context.db) and
linked by person via macOS Contacts + email headers, so "what's going on
with Priya?" is one tool call that sees her emails, texts and meetings
together instead of the Agent stitching three separate reads.

Background sync every 30 min (ARIA_CONTEXT_SYNC=0 disables). Nothing
leaves the machine until the Agent reads a result.
"""
from __future__ import annotations

import logging
import os
import time

import aria.core.plugin as _plugin_base
from aria.core.tool import ToolDescriptor, ToolRegistry

logger = logging.getLogger(__name__)

_MAX_ITEMS = 20
_UPCOMING_DAYS = 14
_SNIPPET = 220


class ContextPlugin(_plugin_base.PluginBase):

    def __init__(self, store=None, syncer=None, autostart: bool = True) -> None:
        self._store = store
        self._syncer = syncer
        self._autostart = autostart

    def register(self, registry: ToolRegistry) -> None:
        try:
            if self._store is None:
                from aria.plugins.context.store import ContextStore
                self._store = ContextStore()
            if self._syncer is None:
                self._syncer = _default_syncer(self._store)
        except Exception as exc:
            logger.warning("context plugin disabled: %s", exc)
            return
        registry.register(self._context_for())
        registry.register(self._context_sync())
        if self._autostart and os.getenv("ARIA_CONTEXT_SYNC", "1") != "0":
            self._syncer.start()

    def _context_for(self) -> ToolDescriptor:
        store = self._store

        def execute(params: dict) -> str:
            try:
                return lookup(store, params.get("query", ""),
                              int(params.get("days", 30) or 30),
                              params.get("kind", "auto") or "auto")
            except Exception as exc:
                logger.warning("context_for failed: %s", exc)
                return f"I couldn't search your context: {exc}"

        return ToolDescriptor(
            name="context_for",
            description=(
                "Everything about a PERSON or TOPIC across email, calendar and iMessage, "
                "merged into one timeline (upcoming meetings first, then newest activity). "
                "Use this FIRST — instead of separate gmail/imessage/calendar calls — for: "
                "'what's going on with Priya', 'catch me up on the Macro interview', "
                "'when did I last talk to mom', 'prep me for my meeting with Alex'. "
                "Results carry email_ids for gmail_read when you need a full email."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "A person's name/email/phone, or a topic"},
                    "days": {"type": "integer",
                             "description": "How far back to look (default 30)"},
                    "kind": {"type": "string", "enum": ["auto", "person", "topic"],
                             "description": "Force person or topic matching (default auto)"},
                },
                "required": ["query"],
            },
            execute=execute,
        )

    def _context_sync(self) -> ToolDescriptor:
        syncer = self._syncer

        def execute(params: dict) -> str:
            return syncer.run_once()

        return ToolDescriptor(
            name="context_sync",
            description=(
                "Refresh the cross-app context index right now (normally runs every 30 "
                "minutes). Use for: 'refresh my context', or when context_for looks stale."
            ),
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )


def _default_syncer(store):
    from aria.plugins.context.people import read_contacts
    from aria.plugins.context.sync import ContextSync
    from aria.plugins.messaging.imessage import iMessageClient
    from aria.plugins.productivity.gcal import CalendarClient
    from aria.plugins.productivity.gmail import GmailClient
    return ContextSync(store, gmail=GmailClient(), calendar=CalendarClient(),
                       imessage=iMessageClient(), read_contacts=read_contacts)


# ----------------------------------------------------------------------
# lookup + rendering (module-level so tests and other plugins can reuse)
# ----------------------------------------------------------------------

def lookup(store, query: str, days: int = 30, kind: str = "auto") -> str:
    query = (query or "").strip()
    if not query:
        return "Tell me which person or topic to look up."
    now = time.time()
    since = now - max(1, days) * 86400
    until = now + _UPCOMING_DAYS * 86400

    if kind != "topic":
        pid, name, others, how = store.resolve_person(query)
        if pid is not None:
            items = store.person_items(pid, since, until, _MAX_ITEMS)
            # A fuzzy name hit ("macro" ≈ "Marco") loses to real topic matches —
            # unless those matches are about that same person ("Pria" → Priya).
            topic = store.topic_items(query, since, until, _MAX_ITEMS) if how == "fuzzy" else []
            if topic and any(name in store.item_people_names(i["id"]) for i in topic):
                topic = []
            if topic and kind == "auto":
                return (_render_topic(store, query, topic, days, now)
                        + f"\n(Also a close name match: {name}. Ask with kind=person for them.)")
            return _render_person(store, pid, name or query, others, items, days, now)
        if kind == "person":
            return _empty(store, f"No person matching {query!r} in your contacts or messages.")

    items = store.topic_items(query, since, until, _MAX_ITEMS)
    if items:
        return _render_topic(store, query, items, days, now)
    return _empty(store, f"Nothing about {query!r} in the last {days} days "
                         "across email, calendar or iMessage.")


def _render_person(store, pid, name, others, items, days, now) -> str:
    handles = store.person_handles(pid)
    head = name + (f" ({', '.join(handles[:3])})" if handles else "")
    if not items:
        return f"{head}: known contact, but nothing in the last {days} days."
    by_source: dict = {}
    for it in items:
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1
    mix = ", ".join(f"{n} {s}" for s, n in sorted(by_source.items(), key=lambda kv: -kv[1]))
    lines = [f"{head}: {len(items)} items in the last {days} days ({mix})."]
    lines += _render_items(items, now, with_people=None)
    if others:
        lines.append(f"(Other close matches: {', '.join(others)}.)")
    return "\n".join(lines)


def _render_topic(store, query, items, days, now) -> str:
    lines = [f"{query!r}: {len(items)} items in the last {days} days."]
    lines += _render_items(items, now, with_people=store)
    return "\n".join(lines)


def _render_items(items, now, with_people) -> list:
    upcoming = sorted((i for i in items if i["ts"] > now), key=lambda i: i["ts"])
    past = [i for i in items if i["ts"] <= now]
    out = []
    if upcoming:
        out.append("Coming up:")
        out += [_line(i, with_people) for i in upcoming]
    if past:
        out.append("Recent, newest first:")
        out += [_line(i, with_people) for i in past]
    return out


def _line(item, with_people) -> str:
    when = time.strftime("%a %b %d %I:%M %p", time.localtime(item["ts"]))
    text = f"- {when} [{item['source']}] {item['title']}"
    if with_people is not None:
        names = with_people.item_people_names(item["id"])
        if names:
            text += f" — with {', '.join(names[:3])}"
    body = item["body"].replace("\n", " ").strip()
    if body:
        text += f" — {body[:_SNIPPET]}{'…' if len(body) > _SNIPPET else ''}"
    if item["source"] == "email" and item["ref"]:
        text += f" (email_id {item['ref']})"
    return text


def _empty(store, message: str) -> str:
    counts = store.counts()
    if sum(v for k, v in counts.items() if k != "people"):
        return message
    problems = [f"{src}: {st['last_error']}" for src, st in store.sync_states().items()
                if st["last_error"]]
    if problems:
        return "Nothing is indexed yet. " + "; ".join(problems)
    return "Nothing is indexed yet — the first context sync runs shortly after startup."
