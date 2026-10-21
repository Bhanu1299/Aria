"""plugins/apply/__init__.py — ApplyPlugin: job applications driven by the job sheet.

Tools:
  job_from_sheet — read the next not-yet-handled row (blank Status) from the job
                   sheet (MUSE_SHEET_ID / MUSE_SHEET_TAB) and remember it.
  apply_job      — fill the application for the next sheet job, a given sheet row,
                   or a URL, and stop for review (aria/features/applicator.py).
                   NEVER submits.
  job_outreach_draft — save a job outreach email as a draft in the JOBS Gmail
                   account (GOOGLE_JOBS_ACCOUNT). Never sends; never the personal account.
  job_outreach   — the outreach loop over the sheet (aria/features/outreach): people →
                   Coldstart emails → jobs-Gmail drafts or LinkedIn notes in the sheet.
                   NEVER sends. dry_run=true writes nothing and saves a report.

Every Google call here uses the jobs account (google_auth.JOBS), never personal.

Submitting is deliberately NOT a tool: only the user's own words ("submit it",
"I submitted it", "skip this job") reach applicator.handle_followup, a dispatcher
pre-handler, so no LLM can decide to send an application.
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse

import aria.core.plugin as _plugin_base
import aria.state.memory as memory
from aria.core.tool import ToolDescriptor, ToolRegistry
from aria.integrations import google_auth, sheets
from aria.observability import run_log

logger = logging.getLogger(__name__)

CURRENT_JOB_KEY = "sheet_job"


PICK_RULE = "first row from the top whose Status means not handled yet (blank / new / Not applied)"
_MAX_DECISIONS = 40


def pick_reasons(job: dict, picked: bool) -> list:
    """Why a sheet row was picked or passed over: status, fit score, deadline, the rule."""
    fields = job.get("fields") or {}
    deadline = next((v for k, v in fields.items() if "deadline" in k.lower() and v), "")
    reasons = [f"Status: {job.get('status') or 'blank'}",
               f"fit {job.get('alignment')}/5" if job.get("alignment") else "no fit score",
               f"deadline {deadline}" if deadline else "no deadline in the sheet"]
    reasons.append(f"picked: {PICK_RULE}" if picked else "skipped: Status says it's already handled")
    return reasons


def read_next_job(run: "run_log.Run") -> Optional[dict]:
    """Next sheet job, logging the read-sheet step and why each row above it was skipped."""
    with run.step("read sheet") as s:
        job = sheets.get_next_job()
        try:  # best effort: the reasons never block the apply
            rows = sheets.default_sheet().jobs()
        except Exception as exc:
            logger.warning("apply_job: couldn't list rows for pick reasons: %s", exc)
            rows = [job] if job else []
        cut = next((i for i, r in enumerate(rows) if job and r["row"] == job["row"]), len(rows) - 1)
        for r in rows[max(0, cut + 1 - _MAX_DECISIONS):cut + 1]:
            picked = bool(job) and r["row"] == job["row"]
            run.decision(r["company"], r["role"], r["row"], picked, pick_reasons(r, picked))
        s.detail = f"{len(rows)} rows; " + (f"picked row {job['row']}" if job else "no new rows")
    return job


def describe_job(job: dict) -> str:
    """One spoken sentence for a sheet row."""
    where = f" in {job['location']}" if job.get("location") else ""
    if job.get("url"):
        link = f"Link is on {urlparse(job['url']).netloc.replace('www.', '')}."
    elif job.get("apply_link"):
        link = f"No apply URL — the sheet says: {job['apply_link']}."
    else:
        link = "There's no apply link on that row."
    return f"Next job from your sheet: {job['role']} at {job['company']}{where}, row {job['row']}. {link}"


class ApplyPlugin(_plugin_base.PluginBase):

    def __init__(self, speaker=None, voice_capture=None, transcriber=None, keyterms_prompt: str = "") -> None:
        self._speaker = speaker
        self._voice_capture = voice_capture
        self._transcriber = transcriber
        self._keyterms_prompt = keyterms_prompt

    @classmethod
    def from_context(cls, ctx) -> "ApplyPlugin":
        return cls(ctx.speaker, ctx.voice_capture, ctx.transcriber, ctx.keyterms_prompt)

    def register(self, registry: ToolRegistry) -> None:
        registry.register(self._job_from_sheet_tool())
        registry.register(self._apply_job_tool())
        registry.register(self._job_outreach_draft_tool())
        registry.register(self._job_outreach_tool())

    # -- voice helpers -----------------------------------------------------

    def _progress_fn(self):
        last = [""]

        def say(msg: str) -> None:
            if self._speaker and msg and msg != last[0]:
                last[0] = msg
                try:
                    self._speaker.say(msg)
                except Exception:
                    pass
        return say

    def _voice_ask(self, field: str) -> str:
        if not (self._speaker and self._voice_capture and self._transcriber):
            return ""
        try:
            self._speaker.say(f"I need your answer for: {field}. Say it now.")
            wav = self._voice_capture.record_once(max_seconds=10)
            if wav is None:
                return ""
            return self._transcriber.transcribe(wav, initial_prompt=self._keyterms_prompt).strip()
        except Exception as exc:
            logger.warning("apply: voice ask failed: %s", exc)
            return ""

    # -- tools -------------------------------------------------------------

    def _job_from_sheet_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            try:
                job = sheets.get_next_job()
            except sheets.SheetError as exc:
                return str(exc)
            except Exception as exc:
                logger.error("job_from_sheet failed: %s", exc)
                return "I couldn't read your job sheet right now."
            if job is None:
                return "There are no new jobs in your sheet — every row already has a status."
            memory.set_persistent(CURRENT_JOB_KEY, job)
            return describe_job(job)

        return ToolDescriptor(
            name="job_from_sheet",
            description=(
                "Read out the next job in the user's job sheet (the Muse sheet): the first row "
                "whose Status is blank. Use for 'what's the next job from muse', 'next job on my "
                "list'. This only reads; to apply use apply_job."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            execute=execute,
        )

    def _apply_job_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features import applicator
            url = str(params.get("url") or "").strip()
            row: Optional[int] = None
            try:
                row = int(params["row"]) if params.get("row") not in (None, "") else None
            except (TypeError, ValueError):
                return "That row number didn't make sense."
            run = run_log.start("apply_job")
            try:
                if url:
                    job = {"url": url, "company": "", "role": "", "row": None, "fields": {}}
                    run.skip("read sheet", "you gave a URL")
                elif row is not None:
                    with run.step("read sheet") as s:
                        job = sheets.get_job(row)
                        s.detail = f"row {row} " + ("found" if job else "not found")
                    if job is None:
                        run.finish("error", f"row {row} not found")
                        return f"Row {row} isn't a job in your sheet."
                    run.decision(job["company"], job["role"], row, True, pick_reasons(job, True)[:3] + ["picked: you asked for this row"])
                else:
                    job = read_next_job(run)
                    if job is None:
                        run.finish("no change", "no new rows")
                        return "There are no new jobs in your sheet — every row already has a status."
            except sheets.SheetError as exc:
                run.finish("error", str(exc))
                return str(exc)
            except Exception as exc:
                logger.error("apply_job: reading the sheet failed: %s", exc)
                run.finish("error", f"reading the sheet failed: {exc}")
                return "I couldn't read your job sheet right now."
            memory.set_persistent(CURRENT_JOB_KEY, job)
            return applicator.run_application(job, progress_fn=self._progress_fn(), voice_ask_fn=self._voice_ask,
                                              run=run)

        return ToolDescriptor(
            name="apply_job",
            description=(
                "Fill out a job application and STOP for the user's review — it never submits. "
                "With no arguments it takes the next new job from the user's job sheet (the Muse "
                "sheet). Use for 'apply to the next job', 'apply to row 7', or 'apply to <url>'. "
                "The user submits afterwards by saying 'submit it'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "description": "Sheet row number (optional)"},
                    "url": {"type": "string", "description": "Application URL (optional)"},
                },
                "required": [],
            },
            execute=execute,
        )

    def _job_outreach_draft_tool(self) -> ToolDescriptor:
        from aria.plugins.productivity.gmail import GmailClient
        client = GmailClient(google_auth.JOBS)

        def execute(params: dict) -> str:
            return client.create_draft(
                params.get("to", ""), params.get("subject", ""), params["body"],
                params.get("cc", ""), params.get("reply_to_id", ""),
            )

        return ToolDescriptor(
            name="job_outreach_draft",
            description=(
                "Save a job-search outreach email as a DRAFT in the user's JOBS Gmail account — "
                "never sends. Use for emails to recruiters, hiring managers or referrals about a "
                "job or application. Personal email goes to gmail_create_draft instead."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body text"},
                    "cc": {"type": "string", "description": "CC addresses (optional)"},
                    "reply_to_id": {"type": "string",
                                    "description": "Message ID in the jobs inbox to reply to (optional)"},
                },
                "required": ["body"],
            },
            execute=execute,
        )

    def _job_outreach_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.outreach import pipeline
            try:
                return pipeline.run_outreach(progress_fn=self._progress_fn(), dry_run=bool(params.get("dry_run")))
            except Exception as exc:
                logger.error("job_outreach failed: %s", exc)
                return "Outreach didn't run — something went wrong reading the sheet."

        return ToolDescriptor(
            name="job_outreach",
            description=(
                "Run job-search outreach over the user's job sheet: for jobs with fit 3+ and no "
                "contact, find recruiters / hiring managers, look up their emails, and write Gmail "
                "DRAFTS (jobs account) or LinkedIn notes into the sheet. Never sends anything. Use for "
                "'do my outreach', 'find recruiters for my jobs', 'draft outreach for the sheet'. "
                "dry_run=true does everything but write drafts, the sheet or Coldstart feedback, and "
                "saves a report instead ('dry run my outreach', 'preview outreach')."
            ),
            input_schema={
                "type": "object",
                "properties": {"dry_run": {"type": "boolean", "description": "Preview only; write nothing"}},
                "required": [],
            },
            execute=execute,
        )
