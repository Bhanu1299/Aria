"""applying/commands.py — the user's words about applications (dispatcher pre-handler).

Pure code, zero tokens, never an agent tool: no LLM can submit or close anything.

  run the (job) pipeline          the whole automation: bring-up, the application run and the
                                  Contacts emails → drafts at once, one report at the end
  start applications / let's apply to some jobs   bring everything up (startup), re-attach tabs, start the run
  open my dashboard               start the job dashboard if it's down and show it (no run)
  pause / resume / stop           only while a run is active (else not claimed: music keeps them)
  submit it | submit <company>    the tab in front | that company's tab; unclear → ask
  I submitted it | … <company>    the user submitted by hand
  skip this job | skip <company>  sheet Skipped; the tab stays open
  mark it|row N|<company> as applied / not applied / skipped    that sheet Status, now (the
                                  tab in front, else the job Aria last filled; anywhere in the sentence)
  close it | close <company> | close all     the only way an application tab closes
  applications are done           Applied for every submitted tab (confirmation seen or the
                                  user said so); then asks per filled-but-unconfirmed tab:
                                  "applied" / "not applied". Every tab stays open.

A company form ("submit <company>") is claimed only when it names an open application
tab, so "submit the report" or "skip this song" fall through to normal routing.

Any other phrasing reaches the agent, whose application_action tool calls request(): skip and
"mark not applied / skipped" act at once; submit, "I submitted", close and "mark applied" are
only staged and read back ("Submit Software Engineer I at RTX?"). They run when the user's NEXT
words are a yes (confirm handler at the top of handle()); anything else cancels them. So the LLM
can understand any wording, but only the user's own yes sends, closes or marks Applied.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional

import aria.web.agent_browser as agent_browser
from aria.dispatch.dispatcher import Partial
from aria.dispatch.parse import normalize
from aria.features import applicator
from aria.features.applying import control, job_pipeline, session, startup, tabs
from aria.features.ats import submit as ats_submit

logger = logging.getLogger(__name__)

_say: Callable[[str], None] = lambda _msg: None
_voice_ask: Optional[Callable[[str], str]] = None
_asking: list = []          # entry ids still to ask about after "applications are done"
_pending: dict = {}         # an action the agent staged; runs only on the user's next "yes"
PENDING_SECS = 120


def configure(say: Callable[[str], None], voice_ask: Optional[Callable[[str], str]] = None) -> None:
    global _say, _voice_ask
    _say, _voice_ask = say, voice_ask


_APPS = r"(?: (?:the |my )?applications?| applying)?"
_START = re.compile(
    r"start (?:the |my )?(?:job )?applications?|start applying(?: (?:to|for) (?:some )?jobs)?|"
    r"(?:let'?s|let us|i want to|time to)(?: go)?(?: start)? apply(?:ing)?(?: it)?(?: (?:to|for) (?:some |a few |a couple of |my )?jobs)?(?: now)?|"
    r"apply (?:it )?(?:to|for) (?:some|a few|a couple of) jobs")      # Whisper: "apply it to some jobs"
# Whisper puts stray words in front ("here, let us start applying to jobs"): allow two,
# never a negation or a question word ("don't start applying", "why did you start…")
_START_LEAD = re.compile(
    r"(?:(?!(?:don'?t|do|not|never|no|stop|why|how|when|what|should|did|didn'?t)\b)[\w']+ ){0,2}"
    r"(?:" + _START.pattern + r")")
# "run the job pipeline": the whole automation (job_pipeline: applications + contact drafts).
# Only "pipeline" words start it, so it's never confused with "start applying" or "draft emails".
_PIPELINE = re.compile(
    r"(?:(?:can you|could you|please|go|let'?s|let us|now|ok|okay) )*"
    r"(?:run|start|kick off|launch|do|begin|fire up)(?: up)? (?:the |my )?(?:whole |full |entire )?"
    r"(?:jobs? |job search |application )?pipe ?(?:line|layer)(?: (?:for|on) (?:my |the )?jobs)?(?: now)?(?: please)?")
# Any negation anywhere blocks a start: only an explicit positive command opens a run.
_NEGATION = re.compile(
    r"\b(?:not|no|never|nope|cannot|stop|dont|wont|cant|shouldnt|wouldnt|couldnt|isnt|arent|wasnt|"
    r"didnt|doesnt|mustnt|havent|hasnt)\b|\w+n't\b")
# "open my dashboard", "show the job dashboard", "open the dashboard for job applications"
_DASHBOARD = re.compile(
    r"(?:open|show|pull up|bring up)(?: me)?(?: the| my)?(?: jobs?| job applications?| applications?| application)?"
    r" dashboard(?: (?:for|of) (?:my |the )?(?:jobs?|job applications?|applications?))?")
_PAUSE = re.compile(r"pause" + _APPS + r"|hold on|wait")
_RESUME = re.compile(r"(?:resume|continue|keep going|go on|unpause)" + _APPS)
_STOP = re.compile(r"stop" + _APPS + r"|stop the run")
_STOP_EXPLICIT = re.compile(r"stop (?:the |my )?applications?|stop applying")
# "submit" only — "send it" is an email phrase and must never submit an application.
_SUBMIT_IT = re.compile(
    r"(?:yes |ok |okay )?(?:go ahead and )?submit(?: it| that| this| the application| my application|"
    r" the form| this application| this one)?(?: now)?")
_SUBMIT_CO = re.compile(r"(?:go ahead and )?submit (?:the )?(?P<co>.+?)(?:'s)?(?: application| form| one| job)?")
_BY_HAND_IT = re.compile(
    r"i(?: have|'ve)? (?:already |just )?(?:submitted|sent)(?: it| the application| my application| this one)?|"
    r"i applied(?: to it)?|mark (?:it|this|the job) (?:as )?applied")
_BY_HAND_CO = re.compile(r"i(?: have|'ve)? (?:already |just )?(?:submitted|sent|applied to) (?:the )?(?P<co>.+?)(?: application| form| one| job)?")
_SKIP_IT = re.compile(
    r"skip (?:this |the )?(?:job|application|role|one)|don'?t apply(?: to (?:it|this(?: one)?))?|"
    r"not applying(?: to (?:it|this(?: one)?))?|pass on (?:it|this(?: one| job)?)")
# skip / close by company need the job word ("skip the Spotify application", "close the Spotify
# tab"), so "skip ad" or "close spotify" stay with music and apps.
_SKIP_CO = re.compile(r"skip (?:the )?(?P<co>.+?)(?:'s)? (?:application|job|role)")
_CLOSE_IT = re.compile(r"close (?:it|this|this one|this tab|the tab|this application)")
_CLOSE_ALL = re.compile(r"close (?:all|everything|all (?:the |my )?(?:tabs|applications|application tabs))")
_CLOSE_CO = re.compile(r"close (?:the )?(?P<co>.+?)(?:'s)? (?:tab|application|job)")
_DONE = re.compile(r"(?:the |my )?applications? (?:are|is) (?:done|finished)|i'?m done with (?:the |my )?applications|"
                   r"done with (?:the |my )?applications|finish (?:up )?(?:the |my )?applications")
# "mark it as applied", "mark row 19 as applied", "set John Deere to not applied" — anywhere in
# the sentence ("yeah I've submitted, you can mark the field as applied"). Writes the sheet now.
_MARK = re.compile(r"\b(?:mark|set|change|move) (?P<what>.+?)(?: as| to)? "
                   r"(?P<st>not applied|not gonna apply|not going to apply|applied|skipped)\b")
_QUESTION = re.compile(r"(?:what|which|why|how|when|did|do|does|have|has|is|was|were|can you tell)\b")
_CONDITION = re.compile(r"\b(?:when|once|after|if|later|tomorrow|tonight|until|unless)\b")
_MARK_STATUS = {"applied": "Applied", "not applied": "Not applied", "skipped": "Skipped",
                "not gonna apply": "Not gonna apply", "not going to apply": "Not gonna apply"}
_MARK_IT = re.compile(r"(?:it|this|that|this one|that one|the (?:field|row|status|job|application|one|cell|sheet)|"
                      r"this (?:job|application|row)|(?:the )?status(?: of (?:it|this|that))?)(?: in the sheet)?")
_MARK_ROW = re.compile(r"(?:(?:row|number|job|line) (?:number )?)?(?P<n>\d{1,4})")
_YES = re.compile(r"(?:yes|yeah|yep)(?: i (?:did|applied))?|applied|i applied(?: to it)?|i did")
# A confirmed action's own words after the yes ("yes go ahead and submit it") aren't a second request.
_RESTATED = {"submit": _SUBMIT_IT, "i_submitted": _BY_HAND_IT, "close": _CLOSE_IT}
# Confirming a staged action: a yes-word first and no negation anywhere ("yeah go ahead",
# "sure, submit it"); a no-word first cancels ("no wait", "don't", "cancel").
_CONFIRM_YES = re.compile(r"(?:yes|yeah|yep|yup|ya|sure|ok|okay|alright|all right|do it|go ahead|go for it|"
                          r"confirm|confirmed|correct|right|please|definitely|absolutely|that'?s right)\b")
# "yes please and start applying to the next jobs": the yes is the first clause, the rest routes on.
_CONFIRM_SPLIT = re.compile(r"\s+(?:and|also|then|plus|after that)\s+")
_CONFIRM_NO = re.compile(r"(?:no|nope|nah|don'?t|do not|wait|hold on|cancel|stop|never ?mind|not yet|"
                         r"not now)\b")
_NO = re.compile(r"(?:no|nope|not yet)(?: i didn'?t)?|not applied|i didn'?t(?: apply)?|didn'?t apply")


def _run_active() -> Optional[control.RunControl]:
    ctl = control.current
    return ctl if ctl is not None and ctl.active else None


def _label(e: dict) -> str:
    return e.get("company") or e.get("role") or "that job"


def _list(entries: list) -> str:
    names = [_label(e) for e in entries]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _open_tabs() -> list:
    return [e for e in tabs.live()]


def _target(co: Optional[str], verb: str):
    """(entry, None) when the target is clear; (None, question) when it isn't; (None, None)
    when the words don't name an application tab (not ours — let routing continue)."""
    live = _open_tabs()
    if not live:
        return None, None
    if co is not None:
        hits = tabs.by_company(co)
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, f"Which one? {_list(hits)} each have a tab. Say {verb} and the role."
        return None, None
    if len(live) == 1:
        return live[0], None                    # one open application: "it" can only mean that one
    try:
        front = agent_browser.front_tabs([e["id"] for e in live])
    except Exception as exc:
        logger.warning("couldn't read the front tab: %s", exc)
        front = []
    if len(front) == 1:
        return next(e for e in live if e["id"] == front[0]), None
    return None, f"Which one? Open applications: {_list(live)}. Say {verb} and the company."


def _company(regex, q: str) -> Optional[str]:
    m = regex.fullmatch(q)
    return m.group("co").strip() if m else None


def _targeted(q: str, it_re, co_re, verb: str, act: Callable[[dict], str],
              allow_filling: bool = False) -> Optional[str]:
    if it_re.fullmatch(q):
        entry, ask = _target(None, verb)
    else:
        co = _company(co_re, q)
        if co is None:
            return None
        entry, ask = _target(co, verb)
    if entry is None:
        return ask
    if entry.get("state") == tabs.FILLING and not allow_filling:
        return f"{_label(entry)} is still filling. Say pause or wait for it to finish."
    return act(entry)


def _submit(entry: dict) -> str:
    if entry.get("state") == tabs.SUBMITTED:
        return f"{_label(entry)} is already submitted."
    return applicator.submit_entry(entry)


def _close(entry: dict) -> str:
    agent_browser.close_tab(entry["id"])
    tabs.mark_closed(entry["id"])
    return f"Closed the {_label(entry)} tab."


def _close_all() -> str:
    live = _open_tabs()
    if not live:
        return "There are no application tabs open."
    filling = [e for e in live if e.get("state") == tabs.FILLING]
    if filling:
        return f"{_label(filling[0])} is still filling. Say stop first, then close all."
    for e in live:
        _close(e)
    return f"Closed {len(live)} application tab{'s' if len(live) != 1 else ''}."


# -- applications are done ------------------------------------------------------

def _ask_next() -> str:
    while _asking:
        e = tabs.get(_asking[0])
        if e and e.get("state") != tabs.CLOSED:
            return (f"{e.get('role') or 'The role'} at {_label(e)} is filled but I never saw a confirmation. "
                    "Did you apply? Say applied or not applied.")
        _asking.pop(0)
    return "That's all of them. All your tabs stay open."


def _done() -> str:
    # Submitted tabs the user already closed still get their Applied write.
    closed_submitted = [dict(e, state=tabs.SUBMITTED) for e in tabs.all_entries()
                        if e.get("state") == tabs.CLOSED and e.get("closed_from") == tabs.SUBMITTED]
    live = [e for e in _open_tabs() + closed_submitted if not e.get("sheet_marked")]
    if not live:
        return "No application tabs are waiting on the sheet. Every tab stays open."
    marked, errors, unconfirmed, filling = [], [], [], []
    for e in live:
        if e.get("state") == tabs.FILLING:
            filling.append(e)
            continue
        if e.get("state") != tabs.SUBMITTED and not e.get("skipped") and ats_submit.page_confirms(e["id"]):
            e = tabs.update(e["id"], state=tabs.SUBMITTED, submitted_by="user", note="confirmation page seen")
        if e.get("state") == tabs.SUBMITTED:
            note = ("submitted by Aria after your review, confirmation seen" if e.get("submitted_by") == "aria"
                    else e.get("note") or "you submitted it")
            err = applicator.mark_applied(e, note)
            (errors if err else marked).append(e)
        elif not e.get("skipped"):
            unconfirmed.append(e)
    parts = []
    if marked:
        parts.append(f"Marked {len(marked)} Applied in the sheet: {_list(marked)}.")
    if errors:
        parts.append(f"I couldn't update the sheet for {_list(errors)}; I'll try again next time you say done.")
    if filling:
        parts.append(f"{_list(filling)} is still filling.")
    _asking[:] = [e["id"] for e in unconfirmed]
    if unconfirmed:
        parts.append(_ask_next())
    else:
        parts.append("Every tab stays open.")
    return " ".join(parts) if parts else "Nothing to mark. Every tab stays open."


def _answer(q: str) -> Optional[str]:
    if not _asking:
        return None
    if _YES.fullmatch(q):
        e = tabs.get(_asking.pop(0))
        if e:
            tabs.update(e["id"], state=tabs.SUBMITTED, submitted_by="user")
            err = applicator.mark_applied(e, "you said you applied (no confirmation page seen)")
            lead = (f"I couldn't update the sheet for {_label(e)}: {err}. " if err
                    else f"Marked {_label(e)} Applied. ")
        else:
            lead = ""
        return lead + _ask_next()
    if _NO.fullmatch(q):
        e = tabs.get(_asking.pop(0))
        lead = f"OK, {_label(e)} stays as it is. " if e else ""
        return lead + _ask_next()
    _asking.clear()          # moved on: a later "yes" must never mark a job Applied
    return None


# -- mark <job> as <status> -------------------------------------------------------

def _last_job() -> Optional[dict]:
    """The job Aria last filled outside a run (apply_job remembers it)."""
    import aria.state.memory as memory
    from aria.plugins.apply import CURRENT_JOB_KEY
    job = memory.get_persistent(CURRENT_JOB_KEY)
    return job if isinstance(job, dict) and job.get("row") else None


def _mark_target(what: str):
    """(job dict, entry | None, None) or (None, None, question)."""
    from aria.integrations import sheets
    m = _MARK_ROW.fullmatch(what)
    if m:
        job = sheets.get_job(int(m.group("n")))
        return (job, None, None) if job else (None, None, f"Row {m.group('n')} isn't a job in your sheet.")
    if _MARK_IT.fullmatch(what):
        live = _open_tabs()
        if live:
            entry, ask = (live[0], None) if len(live) == 1 else _target(None, "mark it")
            if entry is not None:
                return applicator._job_of(entry), entry, None
            if ask and _last_job() is None:
                return None, None, ask
        job = _last_job()
        if job:
            return job, None, None
        return None, None, "Which job? Say mark row 19 as applied, or the company's name."
    co = re.sub(r"^(?:the )?(.+?)(?: (?:job|application|one|row|role))?$", r"\1", what).strip()
    if len(co) < 3:
        return None, None, None
    word = re.compile(r"\b" + re.escape(co), re.I)
    hits = [j for j in sheets.default_sheet().jobs() if word.search(j["company"] or "")]
    if len(hits) == 1:
        entry = next((e for e in tabs.by_company(co)), None)
        return hits[0], entry, None
    # Several jobs, two companies ("John Deere and eBay"), or words that aren't a job ("the
    # meeting"): not one clear job — not ours to answer; the agent reads the whole sentence.
    return None, None, None


def _mark(q: str) -> Optional[str]:
    from aria.integrations import sheets
    m = _MARK.search(q)
    if not m or _NEGATION.search(q[:m.start("st")] + q[m.end("st"):]) or _QUESTION.match(q):
        return None
    if _CONDITION.search(q[m.end("st"):]):
        return None                        # "mark it as applied when I say so": not now
    status = _MARK_STATUS[m.group("st")]
    try:
        job, entry, ask = _mark_target(m.group("what").strip())
        if job is None:
            return ask
    except sheets.SheetError as exc:
        return str(exc)
    except Exception as exc:
        logger.error("mark status failed: %s", exc)
        return "I couldn't update the sheet right now."
    return _write_mark(job, entry, status)


def _write_mark(job: dict, entry: Optional[dict], status: str) -> str:
    """Sheet Status for this job, now (the tab's Applied bookkeeping too). Never raises."""
    from aria.integrations import sheets
    try:
        company, role = job.get("company") or "that job", job.get("role") or ""
        if entry is not None and status == sheets.STATUS_APPLIED:
            if entry.get("state") != tabs.SUBMITTED:
                tabs.update(entry["id"], state=tabs.SUBMITTED, submitted_by="user", note="you said you applied")
            err = applicator.mark_applied(entry, "you said you applied")
            if err:
                return f"I couldn't update the sheet: {err}"
            row = job.get("row")
        else:
            row = sheets.set_status(int(job["row"]), status, f"you said {status.lower()} (voice)",
                                    company=job.get("company") or None, role=job.get("role") or None,
                                    job_id=job.get("job_id") or None)
            if status == sheets.STATUS_APPLIED:
                applicator.schedule_followup(job)
    except sheets.SheetError as exc:
        return str(exc)
    except Exception as exc:
        logger.error("mark status failed: %s", exc)
        return "I couldn't update the sheet right now."
    what = f"{role} at {company}" if role else company
    ref = job.get("job_id") or (f"row {row}" if row else "")
    return f"Marked {what} {status} in the sheet" + (f" ({ref})." if ref else ".")


# -- any wording, via the agent: request() stages, the user's yes runs -------------

ACTIONS = ("submit", "i_submitted", "skip", "close", "mark")
_CONFIRMED = {"submit", "i_submitted", "close"}          # + mark Applied


def _entry_named(said: str) -> Optional[dict]:
    """The one open application tab the words name (company / role, sound-alikes allowed)."""
    from aria.integrations import sheets
    words = [t for t in sheets._tokens(said) if t not in sheets._FILLER]
    scored = sorted(((sheets._score(words, e), e) for e in _open_tabs()), key=lambda x: -x[0][0])
    scored = [(s, e) for s, e in scored if s[0] >= 1.6]
    if len(scored) == 1 or (len(scored) > 1 and scored[0][0][0] > scored[1][0][0] * 1.2):
        return scored[0][1]
    return None


def _request_target(said: str):
    """(job, entry, None) or (None, None, reply). Named words first, else the tab in front,
    else the job Aria last filled."""
    from aria.integrations import sheets
    if said:
        entry = _entry_named(said)
        if entry is not None:
            return applicator._job_of(entry), entry, None
        got = sheets.resolve_job(said)
        if got["kind"] == "one" and got["sure"]:
            job = got["jobs"][0]
            entry = next((e for e in _open_tabs() if job.get("job_id") and e.get("job_id") == job["job_id"]), None)
            return job, entry, None
        if got["kind"] == "applied" and got["sure"]:
            return None, None, f"{sheets.describe_match(got['jobs'][0])} is already in the Applied tab."
        if got["jobs"]:
            names = "; ".join(sheets.describe_match(j) for j in got["jobs"][:4])
            return None, None, f"Not sure which job {said!r} means. Closest: {names}. Ask the user which one."
        return None, None, f"No job matches {said!r} in the sheet or the open application tabs."
    live = _open_tabs()
    if live:
        entry, ask = (live[0], None) if len(live) == 1 else _target(None, "that")
        if entry is not None:
            return applicator._job_of(entry), entry, None
        if ask:
            return None, None, ask
    job = _last_job()
    if job:
        return job, None, None
    return None, None, "No application is open. Ask the user which job they mean."


def request(action: str, said: str = "", status: str = "") -> str:
    """The agent's way in (application_action tool), for any wording the exact phrases miss.
    Returns text for the agent: a result, a question to ask, or the read-back to confirm."""
    from aria.integrations import sheets
    action = (action or "").strip().lower()
    if action not in ACTIONS:
        return f"action must be one of: {', '.join(ACTIONS)}."
    st = _MARK_STATUS.get((status or "").strip().lower()) if action == "mark" else None
    if action == "mark" and st is None:
        return "status must be applied, not applied, skipped or not gonna apply."
    try:
        job, entry, reply = _request_target(said.strip())
    except sheets.SheetError as exc:
        return str(exc)
    except Exception as exc:
        logger.error("application_action %s failed: %s", action, exc)
        return "I couldn't read the sheet or the tabs right now."
    if job is None:
        return reply
    what = sheets.describe_match(job)
    if action in ("submit", "i_submitted", "close") and entry is None:
        return (f"{what} has no open application tab, so there's nothing to {action.replace('_', ' ')}. "
                "Tell the user.")
    if entry is not None and entry.get("state") == tabs.FILLING and action in ("submit", "close"):
        return f"{_label(entry)} is still filling. Tell the user to wait or say pause."
    if action == "skip":
        return applicator.skip_entry(entry) if entry is not None else _write_mark(job, None, sheets.STATUS_SKIPPED)
    if action == "mark" and st != sheets.STATUS_APPLIED:
        return _write_mark(job, entry, st)
    question = {"submit": f"Submit the {what} application?",
                "i_submitted": f"You submitted {what} yourself — should I record it as submitted?",
                "close": f"Close the {what} tab?",
                "mark": f"Mark {what} as Applied in the sheet?"}[action]
    _pending.clear()
    _pending.update(action=action, status=st, job=job, entry_id=(entry or {}).get("id", ""),
                    question=question, at=time.time())
    return (f"Waiting for the user's yes. Ask exactly: {question!r} Their next words decide; "
            "do not call application_action again for this.")


def _run_pending(p: dict) -> str:
    from aria.integrations import sheets
    entry = tabs.get(p["entry_id"]) if p.get("entry_id") else None
    if p["entry_id"] and (entry is None or entry.get("state") == tabs.CLOSED):
        return "That application tab isn't open anymore, so I didn't do anything."
    if p["action"] == "submit":
        return _submit(entry)
    if p["action"] == "i_submitted":
        return _by_hand(entry)
    if p["action"] == "close":
        return _close(entry)
    return _write_mark(p["job"], entry, p["status"] or sheets.STATUS_APPLIED)


def _confirm(text: str) -> Optional[str]:
    """A staged action runs only on the user's very next words, and only if they are a yes."""
    if not _pending:
        return None
    p = dict(_pending)
    _pending.clear()
    if time.time() - p["at"] > PENDING_SECS:
        return None
    s = re.sub(r"[^\w' ]+", " ", (text or "").lower()).strip()
    s = re.sub(r"\s+", " ", re.sub(r"^(?:(?:hey )?aria )", "", s))
    parts = _CONFIRM_SPLIT.split(s, maxsplit=1)
    head, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    if not (_CONFIRM_NO.match(head) or _CONFIRM_YES.match(head)):
        head, rest = s, ""                            # no yes/no up front: judge the whole thing
    restated = _RESTATED.get(p["action"])
    if rest and restated is not None and restated.fullmatch(normalize(rest)):
        rest = ""                                     # "yes go ahead and submit it": already done
    if _CONFIRM_NO.match(head):
        return Partial("OK, I won't.", rest)
    if _CONFIRM_YES.match(head) and not _NEGATION.search(head):
        try:
            return Partial(_run_pending(p), rest)
        except Exception as exc:
            logger.error("confirmed %s failed: %s", p["action"], exc)
            return "Something went wrong doing that. Check the browser before trying again."
    return None                                      # moved on: never run it later


# -- start a run / continue one application (also the agent's tools) -------------

def start_run(limit: Optional[int] = None) -> str:
    """The batch run: bring everything up, re-attach tabs, fill up to `limit` jobs (best fit
    first, at most 10 tabs), never stopping at a wall; one report at the end."""
    ok, message = startup.bring_up()
    if not ok:
        return message
    session.reattach()
    return session.start(_say, _voice_ask, limit=limit)


def start_pipeline() -> str:
    """The whole automation: bring-up, the application run and the Contacts drafts at once."""
    return job_pipeline.start(_say, _voice_ask)


def continue_application(said: str = "") -> str:
    """Pick an application back up in ITS tab — after the user cleared a sign-in, or when the
    site has a next step (Handshake's "Step 2: apply on the employer's site"). Re-runs the
    site's adapter from the job page; signed-in sessions carry over. Never submits."""
    from aria.integrations import sheets
    try:
        job, entry, reply = _request_target(said.strip())
    except sheets.SheetError as exc:
        return str(exc)
    if job is None:
        return reply
    if entry is not None and entry.get("state") == tabs.FILLING:
        return f"{_label(entry)} is still filling."
    job = dict(job, url=job.get("url") or (entry or {}).get("url", ""))
    return applicator.run_application(job, entry=entry)


def _handshake_step_two(entry: dict) -> None:
    """After "I submitted it" on Handshake: a two-step posting now shows "Step 2 — apply on the
    employer's site". Fill that part too, in the same tab, in the background; speak once."""
    import threading
    def run() -> None:
        try:
            out = applicator.run_application(applicator._job_of(entry), entry=entry)
            _say(f"Step 2 for {_label(entry)}: {out}")
        except Exception as exc:
            logger.error("handshake step 2 failed: %s", exc)
    threading.Thread(target=run, daemon=True, name="aria-handshake-step2").start()


def _by_hand(entry: dict) -> str:
    out = applicator.mark_submitted_by_hand(entry)
    try:
        from aria.features.ats.adapters import handshake
        if entry.get("ats") == "handshake" and handshake.has_step_two(entry["id"]):
            _handshake_step_two(entry)
            out += (" Handshake has a step 2 on the employer's site, so I'm filling that now in the same tab. "
                    "Nothing gets submitted.")
    except Exception as exc:
        logger.warning("step 2 check failed: %s", exc)
    return out


# -- entry point ----------------------------------------------------------------

def handle(text: str) -> Optional[str]:
    confirmed = _confirm(text)
    if confirmed is not None:
        return confirmed
    q = normalize(text)
    if not q:
        return None
    answered = _answer(q)
    if answered is not None:
        return answered

    if _DASHBOARD.fullmatch(q):
        return startup.open_dashboard()

    if _PIPELINE.fullmatch(q) and not _NEGATION.search(q):
        return start_pipeline()

    if _START_LEAD.fullmatch(q) and not _NEGATION.search(q):
        return start_run()

    ctl = _run_active()
    if ctl is not None:
        if _PAUSE.fullmatch(q):
            ctl.pause()
            return "Paused. Say resume when you're ready."
        if _RESUME.fullmatch(q):
            ctl.resume()
            return "Resuming."
        if _STOP.fullmatch(q):
            ctl.stop()
            return "Stopping the run. All your tabs stay open."
    elif _STOP_EXPLICIT.fullmatch(q):
        return "No application run is going. Your tabs are untouched."

    if _DONE.fullmatch(q):
        return _done()
    marked = _mark(q)
    if marked is not None:
        return marked
    if _CLOSE_ALL.fullmatch(q):
        return _close_all() if _open_tabs() else None

    for it_re, co_re, verb, act in (
        (_SUBMIT_IT, _SUBMIT_CO, "submit", _submit),
        (_BY_HAND_IT, _BY_HAND_CO, "I submitted", _by_hand),
        (_SKIP_IT, _SKIP_CO, "skip", applicator.skip_entry),
        (_CLOSE_IT, _CLOSE_CO, "close", _close),
    ):
        out = _targeted(q, it_re, co_re, verb, act)
        if out is not None:
            return out
    return None
