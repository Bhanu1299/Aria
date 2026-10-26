"""plugins/apply/__init__.py — ApplyPlugin: job applications driven by the job sheet.

Tools:
  job_from_sheet — read the next not-yet-handled row (blank Status) from the job
                   sheet (MUSE_SHEET_ID / MUSE_SHEET_TAB) and remember it.
  job_find       — find jobs in both tabs by whatever the user calls them (company, role,
                   place, ID, misheard names): ranked, with tab, ID and status.
  apply_job      — fill the application for a job the user names (any order), the next
                   sheet job, a sheet row, or a URL, and stop for review
                   (aria/features/applicator.py). NEVER submits.
  job_status     — set a sheet row's Status (Not applied / In review / Skipped / Not gonna
                   apply) through the Sheets API. Never "Applied": that comes only from the
                   user's own words ("mark row 19 as applied") via the applying pre-handler.
  job_outreach_draft — save a job outreach email as a draft in the JOBS Gmail
                   account (GOOGLE_JOBS_ACCOUNT). Never sends; never the personal account.
  job_outreach   — the outreach loop over the sheet (aria/features/outreach): people →
                   Coldstart emails → jobs-Gmail drafts or LinkedIn notes in the sheet.
                   NEVER sends. dry_run=true writes nothing and saves a report.
  job_contacts   — the Contacts tab (features/outreach/contacts_tab.py): Coldstart emails
                   over HTTP → jobs-Gmail drafts. Runs in the background; NEVER sends.

Every Google call here uses the jobs account (google_auth.JOBS), never personal.

  start_applications — the batch run (aria.features.applying.session): up to N jobs, best fit
                   first, each in its own tab, never stopping at a sign-in or check; one report
                   at the end. Only when the user asks to apply to several / the next jobs.
  run_job_pipeline — the whole automation (aria.features.applying.job_pipeline): bring-up, the
                   application run and the Contacts emails → drafts at once, one report at the end.
                   Only when the user asks for the pipeline / the whole thing.
  application_continue — pick one application back up in its tab (after the user signed in,
                   or the site's next step, e.g. Handshake's "Step 2: apply on the employer's
                   site"). Never submits.
  application_action — submit / I submitted / skip / close / mark a job, for wordings the
                   exact phrases miss. Submit, I submitted, close and mark Applied are only
                   STAGED and read back; they run on the user's next "yes"
                   (aria.features.applying.commands.request / _confirm).

No LLM can send an application: exact phrases ("submit it", "I submitted it") go straight to
the aria.features.applying.commands pre-handler, and everything the agent understands in other
words still waits for the user's own yes.
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse

import aria.core.config as config
import aria.core.plugin as _plugin_base
import aria.state.memory as memory
from aria.core.tool import ToolDescriptor, ToolRegistry
from aria.integrations import google_auth, sheets
from aria.observability import run_log

logger = logging.getLogger(__name__)

CURRENT_JOB_KEY = "sheet_job"
JOB_STATUS_CHOICES = ("Not applied", "In review", "Skipped", "Not gonna apply")


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
    ref = job.get("job_id") or f"row {job['row']}"
    return f"Next job from your sheet: {job['role']} at {job['company']}{where} ({ref}). {link}"


def _ref(job: dict) -> str:
    """How the agent names a job back to apply_job: its ID, else its sheet row."""
    return job.get("job_id") or f"row {job['row']}"


def _match_line(job: dict) -> str:
    where = "Applied tab" if job.get("tab") == config.MUSE_APPLIED_TAB else "jobs tab"
    return (f"{job.get('role') or '?'} at {job.get('company') or '?'}"
            + (f" in {job['location']}" if job.get("location") else "")
            + f" — {_ref(job)}, {where}, Status {job.get('status') or 'blank'}")


_OVERVIEW_WORDS = frozenset("all every everything list jobs left remaining many how count open status overview "
                           "summary what whats s which have has had still not yet are is was were there do does "
                           "did show tell give need needs got any".split())


def _overview() -> str:
    """The sheet at a glance: counts by Status in each tab, then the open jobs with their IDs."""
    _add_ids()
    try:
        jobs = sheets.default_sheet().jobs()
    except sheets.SheetError as exc:
        return str(exc)
    except Exception as exc:
        logger.error("job overview failed: %s", exc)
        return "I couldn't read your job sheet right now."
    try:
        applied = len(sheets.tab_sheet(config.MUSE_APPLIED_TAB).jobs())
    except Exception:
        applied = None
    counts: dict = {}
    for j in jobs:
        key = j.get("status") or "blank"
        counts[key] = counts.get(key, 0) + 1
    open_jobs = [j for j in jobs if (j.get("status") or "").strip().lower() in sheets._NEW_ALIASES]
    out = (f"Jobs tab: {len(jobs)} rows — " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: -kv[1]))
           + (f". Applied tab: {applied} jobs." if applied is not None else "."))
    if open_jobs:
        out += f" {len(open_jobs)} still to apply to: " + "; ".join(
            f"{j['role']} at {j['company']} ({_ref(j)})" for j in open_jobs[:12]) + ("…" if len(open_jobs) > 12 else ".")
    return out


def _add_ids() -> None:
    """Best effort: new rows get their ID before a lookup (read-only access still finds by name)."""
    try:
        sheets.ensure_ids()
    except Exception as exc:
        logger.warning("couldn't add job IDs: %s", exc)


def resolve_named_job(said: str):
    """(job, None) when the user's words name exactly one open job; (None, reply-for-the-agent)
    otherwise — ask which, say it's already applied, or say nothing fits. Never guesses."""
    _add_ids()
    got = sheets.resolve_job(said)
    jobs, kind = got["jobs"], got["kind"]
    if kind == "one" and got["sure"]:
        return jobs[0], None
    if kind == "none":
        return None, (f"No job in the sheet matches {said!r}. Tell the user and offer to read what's "
                      "there (job_find with a company or role word); don't apply to a different job.")
    lines = "; ".join(_match_line(j) for j in jobs[:5])
    if not got["sure"]:
        return None, (f"Not sure which job {said!r} means — closest: {lines}. Ask the user if that's the "
                      f"one; on yes call apply_job with job={_ref(jobs[0])!r}.")
    if kind == "applied":
        return None, (f"Already applied: {lines}. Tell the user it's already in the Applied tab; don't fill "
                      "it again.")
    return None, (f"Several jobs fit {said!r}: {lines}. Ask the user which one (by role or place), then call "
                  "apply_job with that job's ID.")


def _terminal_progress(msg: str) -> None:
    """Fill progress goes to the terminal / log, never the speaker: an application run speaks
    only for walls and once when it's done (and never reads your answers aloud)."""
    if msg:
        print(f"[APPLY] {msg}", flush=True)
        logger.info("apply progress: %s", msg)


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
        registry.register(self._job_find_tool())
        registry.register(self._apply_job_tool())
        registry.register(self._job_status_tool())
        registry.register(self._application_action_tool())
        registry.register(self._start_applications_tool())
        registry.register(self._run_job_pipeline_tool())
        registry.register(self._open_dashboard_tool())
        registry.register(self._application_continue_tool())
        registry.register(self._job_outreach_draft_tool())
        registry.register(self._draft_followups_tool())
        registry.register(self._do_not_apply_tool())
        registry.register(self._check_job_replies_tool())
        registry.register(self._job_outreach_tool())
        registry.register(self._job_contacts_tool())

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

    def _job_find_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            said = str(params.get("query") or "").strip()
            if not [t for t in sheets._tokens(said) if t not in sheets._FILLER | _OVERVIEW_WORDS]:
                return _overview()                  # "how many are left", "what's on my list"
            _add_ids()
            try:
                found = sheets.find_jobs(said)
            except sheets.SheetError as exc:
                return str(exc)
            except Exception as exc:
                logger.error("job_find failed: %s", exc)
                return "I couldn't read your job sheet right now."
            if not found:
                return f"No job in either tab matches {said!r}."
            more = f" (+{len(found) - 8} more)" if len(found) > 8 else ""
            maybe = "" if found[0]["sure"] else " These are sound-alike matches only: check with the user."
            return "Matches, best first: " + "; ".join(_match_line(j) for j in found[:8]) + more + "." + maybe

        return ToolDescriptor(
            name="job_find",
            description=(
                "Look up jobs in the user's job sheet (the jobs tab AND the Applied tab) by whatever "
                "they call them: company, role, place, job ID ('J0018'), even misheard or misspelled "
                "('magnet' finds Magna). Returns matches with ID, tab and Status. Use for 'do I have "
                "an RTX job', 'which Deere roles are there', 'did I apply to Magna'. With an empty query "
                "it returns the overview: how many jobs are open / applied / skipped and the open ones "
                "with their IDs ('how many jobs are left', 'what's on my list')."
            ),
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string",
                                         "description": "The user's words for the job(s); empty for the overview"}},
                "required": [],
            },
            execute=execute,
        )

    def _apply_job_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features import applicator
            url = str(params.get("url") or "").strip()
            said = str(params.get("job") or "").strip()
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
                elif said:
                    with run.step("read sheet") as s:
                        job, reply = resolve_named_job(said)
                        s.detail = f"{said!r} → " + (_match_line(job) if job else "not one clear job")
                    if job is None:
                        run.finish("no change", f"{said!r} isn't one clear job")
                        return reply
                    run.decision(job["company"], job["role"], job["row"], True,
                                 pick_reasons(job, True)[:3] + [f"picked: you named it ({said!r})"])
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
            from aria.features.jobprep import blocklist
            listed = blocklist.hit(job) if not params.get("still_apply") else None
            if listed is not None:
                name = job.get("company") or listed["label"]
                run.finish("no change", f"{name} is on the do-not-apply list — asked the user")
                return (f"{name} is on your do-not-apply list"
                        + (f" ({listed['reason']})" if listed.get("reason") else "")
                        + " — still apply? (If the user says yes, call apply_job again with still_apply true.)")
            memory.set_persistent(CURRENT_JOB_KEY, job)
            return applicator.run_application(job, progress_fn=_terminal_progress, voice_ask_fn=self._voice_ask,
                                              run=run)

        return ToolDescriptor(
            name="apply_job",
            description=(
                "Fill out a job application and STOP for the user's review — it never submits. "
                "When the user names a job in ANY words ('the RTX onsite one', 'the Michigan job', "
                "'J0018', a misheard name), pass their words as `job` — never swap in the next job. "
                "With no arguments it takes the next new job from the user's job sheet (the Muse "
                "sheet). Also 'apply to row 7' (row) or 'apply to <url>' (url). If the reply says it "
                "isn't sure or several fit, ask the user, then call again with the job's ID. "
                "The user submits afterwards by saying 'submit it'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "job": {"type": "string",
                            "description": "The user's words for the job, or its ID like J0018 (optional)"},
                    "row": {"type": "integer", "description": "Sheet row number (optional)"},
                    "url": {"type": "string", "description": "Application URL (optional)"},
                    "still_apply": {"type": "boolean",
                                    "description": "True only after the user said yes to applying to a company "
                                                   "on their do-not-apply list"},
                },
                "required": [],
            },
            execute=execute,
        )

    def _draft_followups_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.jobprep import followups
            try:
                _sync_followups()
                return followups.draft_due()
            except Exception as exc:
                logger.error("draft_followups failed: %s", exc)
                return "I couldn't draft your follow-ups right now."

        return ToolDescriptor(
            name="draft_followups",
            description=(
                "Draft follow-up emails for applications whose follow-up is due (7 business days after "
                "applying, then 7 more; at most 2). Makes jobs-Gmail DRAFTS only — never sends — to the "
                "job's contact from the Contacts tab, else as a reply in the application's email thread. "
                "Use for 'draft my follow-ups', 'any follow-ups due?'."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            execute=execute,
        )

    def _do_not_apply_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.jobprep import blocklist
            action = str(params.get("action") or "add").lower()
            what = str(params.get("company") or "").strip()
            try:
                if action == "list":
                    got = blocklist.entries()
                    return ("Your do-not-apply list: " + ", ".join(e["label"] for e in got) + "."
                            if got else "Your do-not-apply list is empty.")
                if action == "remove":
                    return (f"Took {what} off your do-not-apply list." if blocklist.remove(what)
                            else f"{what} wasn't on your do-not-apply list.")
                e = blocklist.add(what, str(params.get("reason") or ""))
                return f"Got it — I'll never apply to {e['label']}. Batch runs skip it, and I'll ask before a single apply."
            except ValueError as exc:
                return str(exc)
            except Exception as exc:
                logger.error("do_not_apply failed: %s", exc)
                return "I couldn't update your do-not-apply list right now."

        return ToolDescriptor(
            name="do_not_apply",
            description=(
                "The user's do-not-apply list: companies (or domains like acme.com) Aria never applies to. "
                "'never apply to Acme' → add; 'take Acme off my do-not-apply list' → remove; "
                "'who's on my do-not-apply list' → list."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "remove", "list"]},
                    "company": {"type": "string", "description": "Company name or domain"},
                    "reason": {"type": "string", "description": "Why, if the user said (optional)"},
                },
                "required": [],
            },
            execute=execute,
        )

    def _check_job_replies_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.jobprep import replies
            try:
                return replies.scan_gmail(_sheet_jobs())
            except Exception as exc:
                logger.error("check_job_replies failed: %s", exc)
                return "I couldn't check for replies right now."

        return ToolDescriptor(
            name="check_job_replies",
            description=(
                "Check the jobs Gmail for employer replies (interview, rejection, offer) to applications and "
                "suggest a sheet status for each. Nothing is written to the sheet until the user approves it "
                "in the dashboard. Use for 'did anyone reply', 'check for replies from companies'."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            execute=execute,
        )

    def _job_status_tool(self) -> ToolDescriptor:
        allowed = {s.lower(): s for s in JOB_STATUS_CHOICES}

        def execute(params: dict) -> str:
            try:
                row = int(params.get("row"))
            except (TypeError, ValueError):
                return "Which row? Give the sheet row number."
            status = str(params.get("status") or "").strip()
            if status.lower() == sheets.STATUS_APPLIED.lower():
                return (f"Applied is set only from the user's own words. Tell them to say "
                        f"'mark row {row} as applied'.")
            if status.lower() not in allowed:
                return f"Status must be one of: {', '.join(JOB_STATUS_CHOICES)}."
            try:
                job = sheets.get_job(row)
                if job is None:
                    return f"Row {row} isn't a job in your sheet."
                written = sheets.set_status(row, allowed[status.lower()], f"{allowed[status.lower()]} (asked by voice)",
                                            company=job["company"] or None, role=job["role"] or None,
                                            job_id=job.get("job_id") or None)
            except sheets.SheetError as exc:
                return str(exc)
            except Exception as exc:
                logger.error("job_status failed: %s", exc)
                return "I couldn't update your job sheet right now."
            return f"Row {written} ({job['role']} at {job['company']}) is now {allowed[status.lower()]}."

        return ToolDescriptor(
            name="job_status",
            description=(
                "Set the Status of one row in the user's job sheet (the Muse sheet) through the "
                "Sheets API. Use this — never the browser — for 'mark row 12 as skipped', 'set "
                "John Deere back to not applied'. It can't set Applied; for that the user says "
                "'mark row N as applied'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "description": "Sheet row number"},
                    "status": {"type": "string", "enum": list(JOB_STATUS_CHOICES)},
                },
                "required": ["row", "status"],
            },
            execute=execute,
        )

    def _application_action_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.applying import commands
            try:
                return commands.request(str(params.get("action") or ""), str(params.get("job") or ""),
                                        str(params.get("status") or ""))
            except Exception as exc:
                logger.error("application_action failed: %s", exc)
                return "That didn't work — nothing was submitted or changed."

        return ToolDescriptor(
            name="application_action",
            description=(
                "Act on a job application the user is talking about, in whatever words they use "
                "('hit submit', 'send the RTX one in', 'I already sent it', 'drop the Deere one', "
                "'close that tab', 'put Magna down as applied'). action: submit (click submit), "
                "i_submitted (the user submitted it themselves), skip, close (the tab), mark (set the "
                "sheet Status; give status). Submit, i_submitted, close and mark-applied are NOT done "
                "by this tool: it returns a question — ask the user exactly that and stop; their yes "
                "does it. job = the user's words for the job; leave empty for 'it' / 'this one'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["submit", "i_submitted", "skip", "close", "mark"]},
                    "job": {"type": "string", "description": "The user's words for the job (optional)"},
                    "status": {"type": "string", "enum": ["applied", "not applied", "skipped", "not gonna apply"],
                               "description": "For action=mark"},
                },
                "required": ["action"],
            },
            execute=execute,
        )

    def _start_applications_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.applying import commands
            try:
                limit = int(params["limit"]) if params.get("limit") not in (None, "") else None
            except (TypeError, ValueError):
                limit = None
            try:
                return commands.start_run(limit)
            except Exception as exc:
                logger.error("start_applications failed: %s", exc)
                return "The application run didn't start."

        return ToolDescriptor(
            name="start_applications",
            description=(
                "Start the application run: fill up to `limit` new jobs from the sheet (best fit first, "
                "default 10), each in its own tab, WITHOUT stopping at sign-ins or checks — Aria tells the "
                "user at the end what's ready and what's stuck. Never submits. Use when the user asks to apply "
                "to several jobs or keep going ('start applying to the next jobs', 'do 10 of them', 'keep "
                "going with the applications'). Never on your own initiative. For ONE named job use apply_job."
            ),
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "How many jobs (optional, max 10)"}},
                "required": [],
            },
            execute=execute,
        )

    def _run_job_pipeline_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.applying import commands
            try:
                return commands.start_pipeline()
            except Exception as exc:
                logger.error("run_job_pipeline failed: %s", exc)
                return "The job pipeline didn't start."

        return ToolDescriptor(
            name="run_job_pipeline",
            description=(
                "Run the WHOLE job automation at once: bring everything up, fill the next new jobs in their "
                "own tabs (never submits) AND look up contact emails and write Gmail drafts (never sends), "
                "one report at the end. Only when the user asks for the pipeline or the whole thing ('run "
                "the job pipeline', 'do the whole job thing'). Applying only → start_applications; drafting "
                "emails only → job_contacts; the dashboard only → open_dashboard."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            execute=execute,
        )

    def _open_dashboard_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.applying import startup
            try:
                return startup.open_dashboard()
            except Exception as exc:
                logger.error("open_dashboard failed: %s", exc)
                return "The dashboard didn't open."

        return ToolDescriptor(
            name="open_dashboard",
            description=(
                "Open the user's job dashboard (applications, the live sheet, outreach, costs) in a tab "
                "of Aria's Chrome; starts it if it's down. Use for 'show my job tracker', 'pull up my "
                "applications', 'open the dashboard'. Never open Google Sheets for this."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            execute=execute,
        )

    def _application_continue_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.applying import commands
            try:
                return commands.continue_application(str(params.get("job") or ""))
            except Exception as exc:
                logger.error("application_continue failed: %s", exc)
                return "I couldn't pick that application back up."

        return ToolDescriptor(
            name="application_continue",
            description=(
                "Pick an application back up in its own tab and keep filling: after the user signed in or "
                "cleared a check, or when the site has a next step ('click the step 2 external apply', "
                "'continue the RTX one', 'I logged in, carry on'). Reads the page itself — never use "
                "screen_look or browser_task for application tabs. Never submits. job = the user's words "
                "for it; empty for the one in front."
            ),
            input_schema={
                "type": "object",
                "properties": {"job": {"type": "string", "description": "The user's words for the job (optional)"}},
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

    def _job_contacts_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.outreach import contacts_tab
            from aria.web import websearch
            if not contacts_tab.start_background(done_fn=self._progress_fn(),
                                                 search_fn=lambda q: websearch.search(q, max_results=8)):
                return "I'm already working through the Contacts tab."
            return "Finding emails for the Contacts tab in the background. I'll tell you when the drafts are ready."

        return ToolDescriptor(
            name="job_contacts",
            description=(
                "THE outreach-email tool. Works through the Contacts tab of the user's job sheet in the "
                "background: first researches applied jobs Muse left blank, then looks up emails for "
                "pending people with Coldstart and writes Gmail DRAFTS (jobs account) from the research. "
                "Never sends. Use for any request to email or draft outreach for jobs the user applied "
                "to: 'draft emails to the jobs I applied to (today)', 'find emails for my contacts', "
                "'run Coldstart', 'draft emails for my contacts'."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
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
                "The OLD sheet outreach loop: for job-sheet rows with fit 3+ and no contact, find "
                "recruiters / hiring managers and write LinkedIn notes into the sheet. It does NOT "
                "draft emails and does not cover applied jobs — emails for applied jobs are "
                "job_contacts. Use only for 'do my outreach', 'find recruiters for my jobs', "
                "'LinkedIn notes for the sheet'. "
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


def _sheet_jobs() -> list:
    """Every job in both tabs of the Muse sheet (the Applied tab too: replies come after applying)."""
    import aria.core.config as config
    jobs = list(sheets.default_sheet().jobs())
    if config.MUSE_APPLIED_TAB:
        try:
            jobs += sheets.tab_sheet(config.MUSE_APPLIED_TAB).jobs()
        except Exception as exc:
            logger.warning("apply: Applied tab read failed: %s", exc)
    return jobs


def _sync_followups() -> None:
    """Applied rows in the sheet → follow-ups scheduled (idempotent)."""
    from aria.features.jobprep import followups
    applied = [(j, None) for j in _sheet_jobs() if str(j.get("status") or "").strip().lower().startswith("applied")]
    followups.sync(applied)
