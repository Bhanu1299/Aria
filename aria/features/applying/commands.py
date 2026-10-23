"""applying/commands.py — the user's words about applications (dispatcher pre-handler).

Pure code, zero tokens, never an agent tool: no LLM can submit or close anything.

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
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import aria.web.agent_browser as agent_browser
from aria.dispatch.parse import normalize
from aria.features import applicator
from aria.features.applying import control, session, startup, tabs
from aria.features.ats import submit as ats_submit

logger = logging.getLogger(__name__)

_say: Callable[[str], None] = lambda _msg: None
_voice_ask: Optional[Callable[[str], str]] = None
_asking: list = []          # entry ids still to ask about after "applications are done"


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
_MARK_STATUS = {"applied": "Applied", "not applied": "Not applied", "skipped": "Skipped",
                "not gonna apply": "Not gonna apply", "not going to apply": "Not gonna apply"}
_MARK_IT = re.compile(r"(?:it|this|that|this one|that one|the (?:field|row|status|job|application|one|cell|sheet)|"
                      r"this (?:job|application|row)|(?:the )?status(?: of (?:it|this|that))?)(?: in the sheet)?")
_MARK_ROW = re.compile(r"(?:(?:row|number|job|line) (?:number )?)?(?P<n>\d{1,4})")
_YES = re.compile(r"(?:yes|yeah|yep)(?: i (?:did|applied))?|applied|i applied(?: to it)?|i did")
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
        return None, None, "Which job? Say mark row 19 as applied, or the company's name."
    word = re.compile(r"\b" + re.escape(co), re.I)
    hits = [j for j in sheets.default_sheet().jobs() if word.search(j["company"] or "")]
    if len(hits) == 1:
        entry = next((e for e in tabs.by_company(co)), None)
        return hits[0], entry, None
    if hits:
        rows = ", ".join(str(j["row"]) for j in hits[:6])
        return None, None, f"{hits[0]['company']} has rows {rows}. Say mark row and the number."
    return None, None, f"I couldn't find {co} in your sheet. Say mark row and the number."


def _mark(q: str) -> Optional[str]:
    from aria.integrations import sheets
    m = _MARK.search(q)
    if not m or _NEGATION.search(q[:m.start("st")] + q[m.end("st"):]) or _QUESTION.match(q):
        return None
    status = _MARK_STATUS[m.group("st")]
    try:
        job, entry, ask = _mark_target(m.group("what").strip())
        if job is None:
            return ask
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
                                    company=job.get("company") or None, role=job.get("role") or None)
    except sheets.SheetError as exc:
        return str(exc)
    except Exception as exc:
        logger.error("mark status failed: %s", exc)
        return "I couldn't update the sheet right now."
    what = f"{role} at {company}" if role else company
    return f"Marked {what} {status} in the sheet" + (f", row {row}." if row else ".")


# -- entry point ----------------------------------------------------------------

def handle(text: str) -> Optional[str]:
    q = normalize(text)
    if not q:
        return None
    answered = _answer(q)
    if answered is not None:
        return answered

    if _DASHBOARD.fullmatch(q):
        return startup.open_dashboard()

    if _START_LEAD.fullmatch(q) and not _NEGATION.search(q):
        ok, message = startup.bring_up()
        if not ok:
            return message
        session.reattach()
        return session.start(_say, _voice_ask)

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
        (_BY_HAND_IT, _BY_HAND_CO, "I submitted", applicator.mark_submitted_by_hand),
        (_SKIP_IT, _SKIP_CO, "skip", applicator.skip_entry),
        (_CLOSE_IT, _CLOSE_CO, "close", _close),
    ):
        out = _targeted(q, it_re, co_re, verb, act)
        if out is not None:
            return out
    return None
