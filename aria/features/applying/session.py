"""applying/session.py — the application run: every new sheet job in its own visible tab.

start(say, voice_ask, limit=None) → spoken line; the run goes on in a background thread
  1. jobs   sheet rows whose Status is new (blank / "Not applied" …) and that aren't
            already open in a tab, best fit first (fit desc, then row), at most `limit`
            and MAX_OPEN_TABS minus the application tabs already open
  2. each   fill_one: new tab → open the posting → applicator.run_application in that
            tab → never submits. A wall (CAPTCHA / sign-in / email code) does NOT stop the
            run: it's noted on the tab and the run moves on (RunControl.wait_walls = False).
            A single fill outside a run still waits for the user (wait_walls = True).
  3. guard  caffeinate for the whole run; pause / stop via control.current
  4. voice  nothing is spoken while filling (progress goes to the terminal / log). Once,
            at the end, run_report: how many are ready for review, which need answers,
            which stopped and why (sign-in, closed posting, …)

reattach() — after a restart, bind live registry entries to the tabs still open in
Chrome; entries whose tab is gone become closed.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable, Optional

import aria.system.prevent_sleep as prevent_sleep
import aria.web.agent_browser as agent_browser
import aria.web.browser_profile as browser_profile
from aria.features import applicator
from aria.features.applying import control, tabs, walls
from aria.features.ats import corrections
from aria.integrations import sheets
from aria.ui import notifier
from aria.observability import run_log

logger = logging.getLogger(__name__)

MAX_OPEN_TABS = 10
WALL_POLL_SECS = 3.0
WALL_TIMEOUT_SECS = 15 * 60

_thread: Optional[threading.Thread] = None
_NEW = {"", "new", "not applied", "not yet applied", "to apply"}


def fit(job: dict) -> float:
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(job.get("alignment") or ""))
    return float(m.group(1)) if m else 0.0


def _is_new(job: dict) -> bool:
    return re.sub(r"\s+", " ", str(job.get("status") or "")).strip().lower() in _NEW


def select_jobs(rows: list, limit: int) -> list:
    picked = [r for r in rows if _is_new(r) and not tabs.is_open_for(r)]
    picked.sort(key=lambda r: (-fit(r), r.get("row") or 0))
    return picked[:max(0, limit)]


def _names(jobs: list) -> str:
    names = [j.get("company") or "?" for j in jobs]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def start(say: Callable[[str], None], voice_ask: Optional[Callable[[str], str]] = None,
          limit: Optional[int] = None) -> str:
    global _thread
    if control.current is not None and control.current.active:
        return "Applications are already running. Say pause or stop."
    room = MAX_OPEN_TABS - len(tabs.live())
    if limit is not None and limit > 0:
        room = min(room, limit)
    if room <= 0:
        return f"You have {MAX_OPEN_TABS} application tabs open. Close some first: say close it or close all."
    try:
        jobs = select_jobs(sheets.default_sheet().jobs(), room)
    except sheets.SheetError as exc:
        return str(exc)
    except Exception as exc:
        logger.error("start applications: sheet read failed: %s", exc)
        return "I couldn't read your job sheet right now."
    if not jobs:
        return "There are no new jobs in your sheet that aren't already open in a tab."

    ctl = control.RunControl()
    ctl.wait_walls = False
    control.current = ctl
    _thread = threading.Thread(target=_run, args=(jobs, say, voice_ask, ctl), daemon=True, name="aria-apply-run")
    _thread.start()
    n = len(jobs)
    return (f"Starting {n} application{'s' if n != 1 else ''}, best fit first: {_names(jobs)}. "
            "I won't stop for sign-ins or checks; I'll tell you what's ready and what's stuck at the end. "
            "Nothing gets submitted.")


def _note(msg: str) -> None:
    """Progress while filling is never spoken — it goes to the terminal / log, the tabs and the dashboard."""
    logger.info(msg)
    print(f"[Aria] {msg}")


def _speak(say, msg: str) -> None:
    _note(msg)
    try:
        say(msg)
    except Exception as exc:
        logger.warning("apply run: speaking failed: %s", exc)


def finished_line(done: int, missing: list) -> str:
    """The one spoken line at the end of a run: done, nothing submitted, what's missing."""
    out = f"Done. {done} application{'s' if done != 1 else ''} filled, nothing submitted."
    if missing:
        n = len(missing)
        out += (f" {n} answer{'s are' if n != 1 else ' is'} missing from your answer bank, "
                "so those tabs need you. They're listed on the dashboard.")
    return out


_STOP_WORDS = (
    (re.compile(r"sign-?in|log ?in|account", re.I), "stopped at sign-in"),
    (re.compile(r"captcha", re.I), "a CAPTCHA"),
    (re.compile(r"verification|email code", re.I), "an email code"),
    (re.compile(r"closed|no longer|expired|gone", re.I), "the posting looks closed"),
    (re.compile(r"already applied", re.I), "says you already applied"),
    (re.compile(r"by email", re.I), "email-only"),
    (re.compile(r"assessment", re.I), "starts with an assessment"),
    (re.compile(r"no application link|no application url", re.I), "no application link"),
    (re.compile(r"find the form|form fields|no form", re.I), "couldn't find the form"),
)


def _why(note: str) -> str:
    for pattern, short in _STOP_WORDS:
        if pattern.search(note or ""):
            return short
    first = re.split(r"(?<=[.;])\s", (note or "").strip())[0]
    return (first[:60].rstrip(".") or "stopped")


def _list_names(names: list, cap: int = 5) -> str:
    if len(names) > cap:
        names = names[:cap] + [f"{len(names) - cap} more"]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def run_report(ids: list, done: int) -> str:
    """The one spoken line at the end of a run: what's ready, what needs answers, what stopped."""
    out = f"Done with {done} job{'s' if done != 1 else ''}, nothing submitted."
    ready, needs, stopped = [], [], []
    for tab in ids:
        e = tabs.get(tab) or {}
        name = e.get("company") or e.get("role") or "one job"
        if e.get("state") in (tabs.FILLED, tabs.SUBMITTED) and e.get("needs_me"):
            needs.append(f"{name} ({len(e['needs_me'])})")
        elif e.get("state") in (tabs.FILLED, tabs.SUBMITTED):
            ready.append(name)
        elif e:
            stopped.append(f"{name}: {_why(e.get('note', ''))}")
    if ready:
        out += f" {len(ready)} ready for your review: {_list_names(ready)}."
    if needs:
        out += f" {len(needs)} need{'s' if len(needs) == 1 else ''} answers I don't have: {_list_names(needs)}."
    if stopped:
        out += f" {len(stopped)} stopped — {'; '.join(stopped[:5])}" + (" and more." if len(stopped) > 5 else ".")
    if ready or needs or stopped:
        out += " It's all on the dashboard."
    return out


def _missing(ids: list) -> list:
    seen: list = []
    for tab in ids:
        for label in (tabs.get(tab) or {}).get("needs_me") or []:
            if label not in seen:
                seen.append(label)
    return seen


def _run(jobs: list, say, voice_ask, ctl: control.RunControl) -> None:
    prevent_sleep.start()
    done = 0
    ids: list = []
    try:
        for job in jobs:     # open + fill quietly; results live in the tab, review file and dashboard
            ctl.checkpoint()
            _note(fill_one(job, say, None, ctl, ids=ids))
            done += 1
        _speak(say, run_report(ids, done))
    except control.Stopped:
        _speak(say, f"Stopped after {done} application{'s' if done != 1 else ''}, nothing submitted. "
                    "Every tab stays open.")
    except Exception as exc:
        logger.error("application run failed: %s", exc)
        _speak(say, f"The application run hit an error and stopped after {done} application"
                    f"{'s' if done != 1 else ''}, nothing submitted. Every tab stays open.")
    finally:
        prevent_sleep.stop()
        ctl.active = False
        if control.current is ctl:
            control.current = None


def _skip_wall(entry: dict, kind: str) -> None:
    """Batch run: note the wall (correction row + log), don't speak, don't wait."""
    label = walls.NAMES.get(kind, "something only you can clear")
    job = {"url": entry.get("url", ""), "company": entry.get("company", ""), "row": entry.get("row")}
    corrections.record(job, kind, "wall", f"{label} — skipped for now (batch run)", ats=entry.get("ats", ""))
    _note(f"{entry.get('company') or 'A job'} is showing {label}; moving on, it's in the end-of-run report.")


def wait_wall(entry: dict, kind: str, say, ctl: Optional[control.RunControl],
              cleared: Optional[Callable[[], bool]] = None) -> bool:
    """Say what's blocking (voice + a notification naming the company's tab) and wait until
    it's gone: cleared() if given (a step walls.check can't see, e.g. Workday's terms box),
    else no wall on the page. False on timeout. The tab is NOT raised: Aria's Chrome never
    takes focus; the user switches to it when they're ready."""
    tab, company = entry["id"], entry.get("company") or "This job"
    label = walls.NAMES.get(kind, "something only you can clear")
    job = {"url": entry.get("url", ""), "company": entry.get("company", ""), "row": entry.get("row")}
    run_tab = lambda fn: agent_browser.run(fn, tab=tab)          # noqa: E731
    corrections.record(job, kind, "wall", f"{label} — waiting for you", ats=entry.get("ats", ""), run=run_tab)
    notifier.send_notification(f"Aria — {company}", f"{label[0].upper()}{label[1:]} in its tab in Aria's Chrome. "
                                                     "Clear it and I'll carry on.")
    _speak(say, f"{company} is showing {label}. Please clear it in its tab in Aria's Chrome; I'll wait.")
    deadline = time.time() + WALL_TIMEOUT_SECS
    while time.time() < deadline:
        if ctl is not None:
            ctl.wait(WALL_POLL_SECS)
        else:
            time.sleep(WALL_POLL_SECS)
        try:
            done = cleared() if cleared is not None else walls.check(tab) is None
        except Exception as exc:
            logger.debug("wall cleared-check failed: %s", exc)
            done = False
        if done:
            return True
    corrections.record(job, "timeout", "wall", f"{label} not cleared in {WALL_TIMEOUT_SECS // 60} min",
                       ats=entry.get("ats", ""))
    _note(f"I waited 15 minutes on {company}. Leaving that tab open and moving on.")
    return False


def fill_one(job: dict, say, voice_ask=None, ctl: Optional[control.RunControl] = None,
             ids: Optional[list] = None) -> str:
    """One job in its own new tab. Never submits; never closes the tab. control.Stopped
    propagates (the tab stays open). `ids` collects the tab ids this run opened."""
    gate = ctl.checkpoint if ctl is not None else None
    waits = ctl.wait_walls if ctl is not None else True
    entry = tabs.add(job)
    tab = entry["id"]
    if ids is not None:
        ids.append(tab)
    run = run_log.start("start applications")
    try:
        if job.get("url"):
            with agent_browser.use_tab(tab, gate=gate):
                agent_browser.navigate(job["url"], settle_secs=2.0)
            kind = walls.check(tab)
            if kind and not (waits and wait_wall(entry, kind, say, ctl)):
                if not waits:
                    _skip_wall(entry, kind)
                tabs.update(tab, state=tabs.OPEN, note=f"blocked by {walls.NAMES[kind]}")
                run.finish("error", f"blocked by {walls.NAMES[kind]}")
                return f"{job.get('company')} is still blocked, so I left its tab open unfilled."
        timed_out: set = set()          # one wall timeout, one announcement: never wait twice

        def wait_once(kind: str, cleared: Optional[Callable[[], bool]] = None) -> bool:
            if not waits:
                _skip_wall(entry, kind)
                return False
            if kind in timed_out:
                return False
            if wait_wall(entry, kind, say, ctl, cleared):
                return True
            timed_out.add(kind)
            return False
        out = applicator.run_application(job, progress_fn=None, voice_ask_fn=voice_ask, run=run,
                                         entry=entry, gate=gate, wait_fn=wait_once)
        kind = walls.check(tab) if job.get("url") and waits else None
        if kind and wait_once(kind):
            out += " It asked for a check partway, so look over the fields."
        return out
    except control.Stopped:
        if (tabs.get(tab) or {}).get("state") == tabs.FILLING:
            tabs.update(tab, state=tabs.OPEN, note="stopped before the fill")
        run.finish("stopped", "you said stop; the tab stays open")
        raise
    except Exception as exc:
        if (tabs.get(tab) or {}).get("state") == tabs.FILLING:
            tabs.update(tab, state=tabs.OPEN, note=f"error: {exc}")
        run.finish("error", str(exc))
        raise


def reattach() -> int:
    """Bind live entries to their open tabs after a restart. Entries whose tab is gone →
    closed; a fill cut off by a crash → open. Chrome unreachable → nothing changes."""
    live = tabs.live()
    wanted = {e["id"]: e.get("form_url") or e.get("url") for e in live if e.get("form_url") or e.get("url")}
    if not wanted or not browser_profile.cdp_up():     # never launch Chrome just to look
        return 0
    try:
        bound = agent_browser.attach(wanted)
    except Exception as exc:
        logger.warning("reattach: Chrome not reachable: %s", exc)
        return 0
    for e in live:
        if e["id"] not in bound:
            tabs.mark_closed(e["id"])
        elif e.get("state") == tabs.FILLING and not (control.current and control.current.active):
            tabs.update(e["id"], state=tabs.OPEN, note="Aria restarted mid-fill")
    return len(bound)
