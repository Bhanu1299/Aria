"""plugins/productivity/__init__.py — ProductivityPlugin: Gmail, Calendar, Cron."""
from __future__ import annotations

import logging
import re

import aria.core.plugin as _plugin_base
from aria.core.tool import ToolDescriptor, ToolRegistry

logger = logging.getLogger(__name__)

_NOT_SETUP_MSG = "Google isn't connected yet. Run: python -m aria.setup google"


def _str_prop(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _int_prop(desc: str) -> dict:
    return {"type": "integer", "description": desc}


def _bool_prop(desc: str) -> dict:
    return {"type": "boolean", "description": desc}


_SEND_YES = re.compile(r"(?:yes|yeah|yep|yup|sure|confirm(?:ed)?|go ahead|send it|send)(?:[ ,]+(?:please|send it|send|go ahead|do it))*")
_SEND_NO = re.compile(r"\b(?:no|nope|don'?t|do not|cancel|stop|wait|hold on|not yet|draft)\b")


def is_send_confirmation(reply: str) -> bool:
    """A clear spoken yes, and nothing that sounds like no."""
    q = re.sub(r"[^\w\s',]", " ", (reply or "").lower()).strip(" ,")
    q = re.sub(r"\s+", " ", q)
    return bool(q) and not _SEND_NO.search(q) and bool(_SEND_YES.fullmatch(q))


class ProductivityPlugin(_plugin_base.PluginBase):

    requires_agent = True  # cron jobs run their prompts through the live Agent

    def __init__(self, agent=None, speaker=None, voice_capture=None, transcriber=None) -> None:
        self._agent = agent
        self._speaker = speaker
        self._voice_capture = voice_capture
        self._transcriber = transcriber

    @classmethod
    def from_context(cls, ctx) -> "ProductivityPlugin":
        return cls(agent=ctx.agent, speaker=ctx.speaker,
                   voice_capture=ctx.voice_capture, transcriber=ctx.transcriber)

    def _confirm_send(self, prompt: str) -> bool:
        """Spoken yes/no before anything leaves the outbox. No voice stack → False."""
        if not (self._speaker and self._voice_capture and self._transcriber):
            return False
        try:
            self._speaker.say(prompt)
            wav = self._voice_capture.record_once(max_seconds=5)
            if wav is None:
                return False
            return is_send_confirmation(self._transcriber.transcribe(wav))
        except Exception as exc:
            logger.warning("gmail: send confirmation failed: %s", exc)
            return False

    def register(self, registry: ToolRegistry) -> None:
        from aria.plugins.productivity.gmail import GmailClient
        from aria.plugins.productivity.gcal import CalendarClient
        from aria.plugins.productivity.cron import CronScheduler
        import aria.ui.notifier as notifier

        gmail = GmailClient("personal")
        cal = CalendarClient()
        cron = CronScheduler(
            agent=self._agent,
            speaker=self._speaker,
            notifier=lambda msg: notifier.send_notification("Aria Cron", msg),
        )
        cron.start()

        registry.register(self._gmail_list(gmail))
        registry.register(self._gmail_read(gmail))
        registry.register(self._gmail_create_draft(gmail))
        registry.register(self._gmail_send(gmail))
        registry.register(self._gmail_reply(gmail))
        registry.register(self._calendar_list(cal))
        registry.register(self._calendar_create(cal))
        registry.register(self._calendar_delete(cal))
        registry.register(self._cron_create(cron))
        registry.register(self._cron_list(cron))
        registry.register(self._cron_delete(cron))
        registry.register(self._cron_toggle(cron))

    # ------------------------------------------------------------------
    # Gmail tools
    # ------------------------------------------------------------------

    def _gmail_list(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            result = client.list_messages(params.get("query", ""), int(params.get("limit", 5)))
            if isinstance(result, str):
                return result
            if not result:
                return "No emails found."
            lines = [f"{m['date']} | {m['from']} | {m['subject']}" for m in result]
            return "\n".join(lines)

        return ToolDescriptor(
            name="gmail_list",
            description=(
                "List Gmail messages matching a query. Use Gmail search syntax. "
                "Use for: 'check emails', 'emails from recruiters', 'unread emails today'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": _str_prop("Gmail search query (e.g. 'from:recruiter is:unread')"),
                    "limit": _int_prop("Number of emails (default 5, max 20)"),
                },
                "required": [],
            },
            execute=execute,
        )

    def _gmail_read(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            result = client.read_message(params["email_id"])
            if isinstance(result, str):
                return result
            return (
                f"From: {result['from']}\n"
                f"Subject: {result['subject']}\n"
                f"Date: {result['date']}\n\n"
                f"{result['body_text'][:2000]}"
            )

        return ToolDescriptor(
            name="gmail_read",
            description="Read the full body of an email by ID (from gmail_list).",
            input_schema={
                "type": "object",
                "properties": {"email_id": _str_prop("Email ID from gmail_list")},
                "required": ["email_id"],
            },
            execute=execute,
        )

    def _gmail_create_draft(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            return client.create_draft(
                params.get("to", ""), params.get("subject", ""), params["body"],
                params.get("cc", ""), params.get("reply_to_id", ""),
            )

        return ToolDescriptor(
            name="gmail_create_draft",
            description=(
                "Save an email as a DRAFT in the user's PERSONAL Gmail — never sends. The default "
                "for writing any personal email: follow-ups, replies, 'write/draft an email to ...'. "
                "Job outreach (recruiters, hiring managers, cold emails about a job) goes to "
                "job_outreach_draft instead. "
                "Set reply_to_id (from gmail_list) to draft a reply in that thread."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "to": _str_prop("Recipient email address"),
                    "subject": _str_prop("Email subject"),
                    "body": _str_prop("Email body text"),
                    "cc": _str_prop("CC address (optional)"),
                    "reply_to_id": _str_prop("Email ID to reply to (optional)"),
                },
                "required": ["body"],
            },
            execute=execute,
        )

    def _gmail_send(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            to, subject, body, cc = params["to"], params["subject"], params["body"], params.get("cc", "")
            if not self._confirm_send(f"Send the email to {to}, subject {subject}? Say yes to send."):
                return "Not sent. " + client.create_draft(to, subject, body, cc)
            return client.send_message(to, subject, body, cc)

        return ToolDescriptor(
            name="gmail_send",
            description=(
                "SEND an email — only when the user explicitly says to send it. Aria asks them to "
                "confirm out loud first; without a spoken yes it saves a draft instead. For writing "
                "or drafting an email use gmail_create_draft."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "to": _str_prop("Recipient email address"),
                    "subject": _str_prop("Email subject"),
                    "body": _str_prop("Email body text"),
                    "cc": _str_prop("CC address (optional)"),
                },
                "required": ["to", "subject", "body"],
            },
            execute=execute,
        )

    def _gmail_reply(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            email_id, body = params["email_id"], params["body"]
            if not self._confirm_send("Send that reply? Say yes to send."):
                return "Not sent. " + client.create_draft("", "", body, reply_to_id=email_id)
            return client.reply_message(email_id, body)

        return ToolDescriptor(
            name="gmail_reply",
            description=(
                "SEND a reply (preserves thread) — only when the user explicitly says to send it. "
                "Aria asks them to confirm out loud first; without a spoken yes it saves the reply "
                "as a draft. To draft a reply use gmail_create_draft with reply_to_id."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "email_id": _str_prop("Email ID to reply to"),
                    "body": _str_prop("Reply text"),
                },
                "required": ["email_id", "body"],
            },
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Calendar tools
    # ------------------------------------------------------------------

    def _calendar_list(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            result = client.list_events(
                params.get("time_min", "today"),
                params.get("time_max", ""),
                params.get("calendar_id", "primary"),
            )
            if isinstance(result, str):
                return result
            if not result:
                return "No events found."
            lines = [f"{e['start']} — {e['title']}" + (f" @ {e['location']}" if e['location'] else "")
                     for e in result]
            return "\n".join(lines)

        return ToolDescriptor(
            name="calendar_list",
            description=(
                "List Google Calendar events. Use for: 'what's on my calendar', "
                "'what do I have tomorrow', 'events this week'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "time_min": _str_prop("Start time: 'today', 'tomorrow', ISO 8601"),
                    "time_max": _str_prop("End time (optional)"),
                    "calendar_id": _str_prop("Calendar ID (default: primary)"),
                },
                "required": [],
            },
            execute=execute,
        )

    def _calendar_create(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            result = client.create_event(
                params["title"], params["start"], params["end"],
                params.get("description", ""), params.get("location", ""),
            )
            if isinstance(result, str):
                return result
            return f"Event created. ID: {result['event_id']}"

        return ToolDescriptor(
            name="calendar_create",
            description=(
                "Create a Google Calendar event. "
                "Use for: 'block 2pm to 3pm tomorrow as deep work', 'add meeting on Friday at 10am'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "title": _str_prop("Event title"),
                    "start": _str_prop("Start datetime in ISO 8601"),
                    "end": _str_prop("End datetime in ISO 8601"),
                    "description": _str_prop("Event description (optional)"),
                    "location": _str_prop("Location (optional)"),
                },
                "required": ["title", "start", "end"],
            },
            execute=execute,
        )

    def _calendar_delete(self, client) -> ToolDescriptor:
        def execute(params: dict) -> str:
            return client.delete_event(params["event_id"],
                                       params.get("calendar_id", "primary"))

        return ToolDescriptor(
            name="calendar_delete",
            description="Delete a Google Calendar event by ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "event_id": _str_prop("Event ID from calendar_list"),
                    "calendar_id": _str_prop("Calendar ID (default: primary)"),
                },
                "required": ["event_id"],
            },
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Cron tools
    # ------------------------------------------------------------------

    def _cron_create(self, scheduler) -> ToolDescriptor:
        def execute(params: dict) -> str:
            try:
                return scheduler.create(
                    params["name"], params["prompt"],
                    params["schedule"], params.get("delivery", "speak"),
                )
            except ValueError as exc:
                return str(exc)

        return ToolDescriptor(
            name="cron_create",
            description=(
                "Schedule a recurring Aria task. "
                "Use for: 'every morning at 9, check my emails', "
                "'every Monday remind me to review PRs'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": _str_prop("Human label for the job"),
                    "prompt": _str_prop("What Aria should do when it fires"),
                    "schedule": _str_prop("Cron expression or natural: 'every day at 9am'"),
                    "delivery": _str_prop("speak | notify | silent (default: speak)"),
                },
                "required": ["name", "prompt", "schedule"],
            },
            execute=execute,
        )

    def _cron_list(self, scheduler) -> ToolDescriptor:
        def execute(params: dict) -> str:
            jobs = scheduler.list_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = []
            for j in jobs:
                status = "enabled" if j["enabled"] else "paused"
                lines.append(f"{j['id']} [{status}] — {j['prompt']} — {j['schedule_human']}")
            return "\n".join(lines)

        return ToolDescriptor(
            name="cron_list",
            description="List all scheduled Aria cron jobs.",
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )

    def _cron_delete(self, scheduler) -> ToolDescriptor:
        def execute(params: dict) -> str:
            return scheduler.delete(params["job_id"])

        return ToolDescriptor(
            name="cron_delete",
            description="Delete a scheduled cron job by ID.",
            input_schema={
                "type": "object",
                "properties": {"job_id": _str_prop("Job ID from cron_list")},
                "required": ["job_id"],
            },
            execute=execute,
        )

    def _cron_toggle(self, scheduler) -> ToolDescriptor:
        def execute(params: dict) -> str:
            return scheduler.toggle(params["job_id"], bool(params["enabled"]))

        return ToolDescriptor(
            name="cron_toggle",
            description="Enable or pause a scheduled cron job.",
            input_schema={
                "type": "object",
                "properties": {
                    "job_id": _str_prop("Job ID from cron_list"),
                    "enabled": _bool_prop("true to resume, false to pause"),
                },
                "required": ["job_id", "enabled"],
            },
            execute=execute,
        )
