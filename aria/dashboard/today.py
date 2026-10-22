"""dashboard/today.py — Tab 1: what to do now, from the pipeline model (read-only)."""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Optional

from aria.dashboard import model as m
from aria.dashboard.render import _date, _down, _no_data, _thread_url, e, label, pill, row

INTERVIEW = re.compile(r"(?i)interview|schedul|availability|phone screen|next steps|chat with|call with")
WINDOW = timedelta(days=m.NEEDS_ME_WINDOW_DAYS)


def bounce_fix(t: dict, cs: dict) -> str:
    """A suggestion for a bounced address, from Coldstart's cache (read-only)."""
    dom = t["domain"]
    contacts = [c for c in cs.get("contacts", []) if c["domain"] == dom and c["email"] != t["to"]]
    good = [c for c in contacts if c.get("status") == "valid"]
    if good:
        c = good[0]
        return f"Coldstart has a verified address at {dom}: {c['email']} ({c.get('name') or 'no name'})."
    if contacts:
        c = max(contacts, key=lambda c: c.get("confidence") or 0)
        return f"Coldstart's best guess at {dom}: {c['email']} ({c.get('status')}, {c.get('confidence')})."
    from aria.dashboard.sources import coldstart_domain
    pattern = (cs.get("domains", {}).get(dom) or coldstart_domain(dom)).get("inferred_pattern")
    if pattern:
        return f"Coldstart's format for {dom} is {pattern}; rebuild the address with it, or reach out on LinkedIn."
    return (f"No other address known. Run Coldstart (python -m coldstart find {dom} \"First Last\") "
            "or send a LinkedIn note instead.")


def build(mdl: dict) -> dict:
    now, jobs = mdl["now"], mdl["jobs"]
    cs = mdl.get("coldstart") or {}
    recent = lambda when: when is not None and now - when <= WINDOW
    replies = [t for t in mdl["threads"] if t["awaiting_me"] and recent(t["last"])]
    interviews = [{"job": j, "why": f"sheet says “{j.status or j.outcome}”"} for j in jobs
                  if re.search(r"(?i)interview", f"{j.status} {j.outcome}")]
    interviews += [{"job": t.get("job"), "thread": t, "why": f"reply mentions it: “{t['reply_subjects'][-1][:60]}”"}
                   for t in mdl["threads"] if t["replied"] and recent(t["replied"])
                   and any(INTERVIEW.search(s) for s in t["reply_subjects"])]
    bounces = [dict(t, fix=bounce_fix(t, cs)) for t in mdl["threads"] if t["bounced"] and recent(t["bounced"])]
    follow_ups = [t for t in mdl["threads"] if t.get("job") and not t["replied"] and not t["bounced"]
                  and timedelta(days=m.FOLLOW_UP_DAYS) < now - t["sent"] <= WINDOW]
    drafts = [d for d in mdl["drafts"] if timedelta(days=m.DRAFT_STALE_DAYS) < now - d["created"] <= WINDOW]
    finish = [j for j in jobs if not m.is_closed(j) and
              (j.status.lower() == "needs human" or j.stage_label == "Form filled (in review)")]
    linkedin = sorted((j for j in jobs if not j.threads and not m.is_closed(j) and (j.linkedin_note or (
                       "linkedin" in j.channels and not j.email and m.has_applied(j)))),
                      key=lambda j: -(float(j.fit) if (j.fit or "").replace(".", "").isdigit() else 0))
    deadlines = sorted((j for j in jobs if j.deadline and now <= j.deadline <= now + timedelta(days=3)
                        and not m.is_closed(j)), key=lambda j: j.deadline)
    return {"replies": replies, "interviews": interviews, "bounces": bounces, "follow_ups": follow_ups,
            "drafts": drafts, "finish": finish, "linkedin": linkedin, "deadlines": deadlines}


def _fit(j) -> float:
    try:
        return float(j.fit)
    except (TypeError, ValueError):
        return 0.0


def parts(d: dict, questions: Optional[int] = None) -> list:
    """[(count, singular, plural, link)] in order of urgency; the first non-zero one is Today's headline."""
    return [(len(d["interviews"]), "interview", "interviews", ""),
            (len(d["replies"]), "reply to answer", "replies to answer", "#outreach"),
            (len(d["finish"]), "application to finish", "applications to finish", "#applications"),
            (len(d["deadlines"]), "deadline in 3 days", "deadlines in 3 days", ""),
            (questions or 0, "question needs an answer", "questions need answers", "#answers"),
            (len(d["bounces"]), "bounce", "bounces", "#outreach"),
            (len(d["follow_ups"]), "follow-up due", "follow-ups due", "#outreach"),
            (len(d["drafts"]), "draft to send", "drafts to send", "#outreach"),
            (len(d["linkedin"]), "LinkedIn note to send", "LinkedIn notes to send", "#outreach")]


def summary_line(d: dict, questions: Optional[int] = None) -> str:
    out = [f"{n} {one if n == 1 else many}" for n, one, many, _ in parts(d, questions) if n]
    return " · ".join(out) if out else "Nothing needs you right now."


SHOWN = 5        # rows per group before "show more" / "view all"


def group(title: str, items: list, item, hint: str = "", down: str = "", shown: int = SHOWN,
          more_href: str = "", more_label: str = "") -> str:
    """One card of rows; empty groups render nothing. Past `shown` rows: a link to the full tab
    (more_href) or an inline "Show N more"."""
    if down:
        return f'<section class="card">{label(title)}{_no_data(down)}</section>'
    if not items:
        return ""
    rows = lambda xs: "".join(item(i) for i in xs)
    more = ""
    if len(items) > shown:
        more = (f'<p class="more"><a href="{e(more_href)}">{e(more_label)} →</a></p>' if more_href else
                f'<details class="showmore"><summary>Show {len(items) - shown} more</summary>'
                f'<div class="rows">{rows(items[shown:])}</div></details>')
    return f'<section class="card">{label(title, hint, len(items))}<div class="rows">{rows(items[:shown])}</div>{more}</section>'


# What each source's outage hides on Today and Outreach.
OUTAGE = {"gmail": "replies, bounces, follow-ups and drafts hidden",
          "sheet": "applications, LinkedIn notes and deadlines hidden",
          "aria_db": "filled forms and missing-answer counts hidden"}


def outage_bar(srcs: dict) -> str:
    """One amber bar for every source that's down, never a scatter of "no data" lines."""
    down = [n for n in OUTAGE if _down(srcs, n)]
    if not down:
        return ""
    lines = "<br>".join(f"{e(_down(srcs, n))} — {e(OUTAGE[n])}" for n in down)
    return f'<div class="alertbar" role="status"><div>{lines}</div><a href="#health">Data health →</a></div>'


def hero(now, counts: list, outage: bool) -> str:
    live = [c for c in counts if c[0]]
    if live:
        n, one, many, _ = live[0]
        head = f'<span class="n">{n}</span> {e(one if n == 1 else many)}'
        rest = " · ".join((f'<a href="{href}">{n} {e(one if n == 1 else many)}</a>' if href else
                           f'{n} {e(one if n == 1 else many)}') for n, one, many, href in live[1:])
    else:
        head = "No to-dos in the sources that loaded" if outage else "Nothing needs you right now."
        rest = ""
    sub = f'<p class="subline">{rest}</p>' if rest else '<p class="subline"></p>'
    return (f'<section class="card hero"><div class="dlabel">// Today · {now:%a, %b %d}</div>'
            f'<h1 class="headline">{head}</h1>{sub}<div class="actions"><a class="btn primary" href="#applications">'
            'Applications →</a><a class="btn" href="#outreach">Outreach →</a></div></section>')


def questions_card(n: int) -> str:
    if not n:
        return f'<section class="card">{label("Questions waiting")}<p class="muted">No questions waiting.</p></section>'
    return (f'<section class="card">{label("Questions waiting")}<div class="bignum"><span class="v">{n}</span>'
            f'<span class="t">question{"s" if n != 1 else ""} from your application forms need{"" if n != 1 else "s"} '
            'your answer</span><a class="btn" href="#answers">Answer them →</a></div></section>')


def render(mdl: dict, questions: Optional[int] = None) -> str:
    """questions: the "Needs your answer" count from My answers (None = answer bank unreadable)."""
    srcs = mdl["sources"]
    g = srcs.get("gmail")
    me = (g.data or {}).get("me", "") if g is not None and g.ok else ""
    d = build(mdl)
    link = lambda t: (f'<a href="{e(_thread_url(me, t["id"]))}" target="_blank" rel="noopener">open thread</a>'
                      if me and t.get("id") else "")
    missing = {id(i["job"]): i["count"] for i in m.needs_me(mdl)["fields"]} if srcs.get("aria_db") and srcs["aria_db"].ok else {}

    def finish(j) -> str:
        human = j.status.lower() == "needs human"
        state = pill("Needs you", "bad") if human else pill("Filled, not submitted", "warn")
        n = missing.get(id(j), 0)
        gap = f'<a href="#answers">{n} answer{"s" if n != 1 else ""} missing</a>' if n else ""
        return row(j.company, j.role, f"fit {e(j.fit)}/5" if j.fit else "", state + gap)

    def interview(i) -> str:
        j = i["job"]
        return row(j.company if j else "", j.role if j else "", e(i["why"]), link(i["thread"]) if i.get("thread") else "")

    finish_rows = sorted(d["finish"], key=lambda j: -_fit(j))        # stable: no fit → sheet order
    outage = any(_down(srcs, n) for n in OUTAGE)
    return "".join([
        hero(mdl["now"], parts(d, questions), outage),
        outage_bar(srcs),
        group("Applications to finish", finish_rows, finish,
              "each is open in its own Chrome tab; answer what's missing, then say “submit it”",
              more_href="#applications", more_label="View all in Applications"),
        questions_card(questions) if questions is not None else "",
        group("Interviews", d["interviews"], interview),
        group("Deadlines in the next 3 days", d["deadlines"],
              lambda j: row(j.company, j.role, right=pill(_date(j.deadline), "warn"))),
    ])
