"""dashboard/outreach.py — Tab "Outreach": Muse's Contacts pipeline (Companies + Contacts tabs:
lookups, drafts to review, waves, replies, what's held and why), LinkedIn notes to send and the
email threads waiting on you. Read-only: links open Gmail / LinkedIn, nothing is sent; cells are
edited on the Sheet tab."""
from __future__ import annotations

import re

from aria.dashboard import model as m
from aria.dashboard import today
from aria.dashboard.render import _date, _down, _thread_url, e, label, pill, row

SHOWN_NOTES = 10


def linkedin_row(j) -> str:
    who, _, where = (j.linkedin_to or j.contact).partition(" — ")
    profile = f'<a href="{e(where)}" target="_blank" rel="noopener">profile</a>' if where.startswith("http") else ""
    fit = pill(f"fit {j.fit or '?'}/5", "accent" if today._fit(j) >= 4 else "")
    note = (f'<div class="lnote"><span class="muted">Note (Aria, not sent):</span> {e(j.linkedin_note)}</div>'
            if j.linkedin_note else "")
    return row(j.company, j.role, e(who), fit + profile, note)


def email_card(mdl: dict, d: dict, me: str) -> str:
    """Replies · Follow-ups · Drafts · Bounced as underlined sub-tabs; one muted card when Gmail is down."""
    down = _down(mdl["sources"], "gmail")
    if down:
        return (f'<section class="card">{label("Email")}<p class="muted">{e(down)} — replies, follow-ups, drafts '
                'and bounces can’t be read until it’s back.</p></section>')
    link = lambda tid: (f'<a href="{e(_thread_url(me, tid))}" target="_blank" rel="noopener">open thread</a>'
                        if me and tid else "")
    job = lambda x: (x["job"].company, x["job"].role) if x.get("job") else ("", "")
    panes = [
        ("replies", "Replies to answer", d["replies"],
         lambda t: row(*job(t), f'{e(t["to"])} · last {_date(t["last"])}', link(t["id"]))),
        ("follow_ups", "Follow-ups due", d["follow_ups"],
         lambda t: row(*job(t), f'{e(t["to"])} · sent {_date(t["sent"])} · no reply in {m.FOLLOW_UP_DAYS}+ days',
                       link(t["id"]))),
        ("drafts", "Drafts waiting", d["drafts"],
         lambda x: row(*job(x), f'{e(x["to"] or "no recipient")} · “{e(x["subject"][:50])}” · '
                                f'{m.DRAFT_STALE_DAYS}+ days old', link(x["thread_id"]))),
        ("bounces", "Bounced", d["bounces"],
         lambda t: row(*job(t), f'{e(t["to"])} bounced {_date(t["bounced"])}', link(t["id"]),
                       f'<div class="fix">{e(t["fix"])}</div>')),
    ]
    first = next((k for k, _, items, _ in panes if items), panes[0][0])
    bar = "".join(f'<button type="button" class="utab{" on" if k == first else ""}" data-p="{k}">{e(name)} '
                  f'<span class="count">{len(items)}</span></button>' for k, name, items, _ in panes)
    bodies = "".join(
        f'<div class="upane" data-p="{k}"{"" if k == first else " hidden"}>'
        + (f'<div class="rows">{"".join(fn(i) for i in items)}</div>' if items
           else f'<p class="muted">No {e(name.lower())} right now.</p>') + '</div>'
        for k, name, items, fn in panes)
    return (f'<section class="card">{label("Email", f"Gmail (jobs), last {m.NEEDS_ME_WINDOW_DAYS} days")}'
            f'<div class="utabs" data-panes>{bar}</div>{bodies}</section>')


_TEAM = ("—", "–", "-")
_DONE = ("drafted", "approved", "sent", "replied", "no reply")
_NEEDS = re.compile(r"(?i)needs research|no job id|isn't on companies|llm draft failed|draft not created|"
                    r"no domain|coldstart error|found no address")


def _st(rec: dict, name: str) -> str:
    return (rec.get(name) or "").strip().lower()


def contact_state(c: dict) -> str:
    """Where one Contacts row is: replied · sent · approved · review (a draft) · needs (you / Muse) ·
    held (Companies says not now) · lookup (Coldstart next) · closed · waiting (an earlier wave first)."""
    ds = _st(c, "Draft status")
    if ds in ("replied", "sent", "approved"):
        return ds
    if ds == "drafted":
        return "review"
    if ds == "no reply":
        return "closed"
    if _st(c["company"], "Outreach status") != "active" or not c["company"]:
        return "needs" if not c["company"] else "held"
    last = (c.get("Notes") or "").strip().splitlines()[-1:] or [""]
    if _st(c, "Find status") == "error" or _st(c, "Confidence") == "not found" or _NEEDS.search(last[0]):
        return "needs"
    if not (c.get("Email") or "").strip() and _st(c, "Find status") in ("", "pending"):
        return "lookup"
    return "waiting"


def _who(c: dict) -> str:
    who = (c.get("Person") or "").strip()
    if who in _TEAM:
        return "team inbox"
    title = (c.get("Title") or "").strip()
    return e(who) + (f' <span class="muted">· {e(title[:60])}</span>' if title else "")


def contact_row(c: dict, why: str = "") -> str:
    co = c["company"]
    wave = (c.get("Wave") or "").strip()
    email = (c.get("Email") or "").strip()
    conf = (c.get("Confidence") or "").strip()
    meta = " · ".join(x for x in [
        f"wave {e(wave)}" if wave else "", _who(c),
        f"{e(email)}" + (f" ({e(conf)})" if conf and conf.lower() != "not found" else "") if email else "",
        e(c.get("job_id") or "")] if x)
    link = (c.get("Gmail draft link") or "").strip()
    tone = {"replied": "ok", "sent": "warn", "approved": "accent", "drafted": "accent", "no reply": "gray"}
    ds = (c.get("Draft status") or "").strip()
    right = pill(ds, tone.get(ds.lower(), "")) if ds and ds.lower() != "none" else ""
    if link.startswith("https://"):
        right += f'<a href="{e(link)}" target="_blank" rel="noopener">open draft</a>'
    note = why or ((c.get("Notes") or "").strip().splitlines()[-1:] or [""])[0]
    below = f'<div class="fix">{e(note[:160])}</div>' if note else ""
    return row(c.get("Company") or co.get("Company", ""), co.get("Role", ""), meta, right, below)


def _held_row(co: dict, people: list) -> str:
    st = (co.get("Outreach status") or "").strip() or "blank"
    blocker = (co.get("Blocker") or "").strip()
    names = ", ".join((p.get("Person") or "").strip() for p in people)
    return row(co.get("Company", ""), co.get("Role", ""),
               f"{len(people)} contact{'s' if len(people) != 1 else ''}" + (f": {e(names[:120])}" if names else ""),
               pill(st, "gray" if st == "skip" else "warn"), f'<div class="fix">{e(blocker)}</div>' if blocker else "")


def contacts_card(mdl: dict) -> str:
    """Muse's Contacts tab, joined with Companies: one sub-tab per state, plus held jobs."""
    down = _down(mdl["sources"], "sheet")
    if down:
        return f'<section class="card">{label("Contacts")}<p class="muted">{e(down)}</p></section>'
    rows = mdl.get("contacts") or []
    if not rows:
        return (f'<section class="card">{label("Contacts")}<p class="muted">No Contacts tab rows in the Muse '
                'sheet yet.</p></section>')
    by = {}
    for c in rows:
        by.setdefault(contact_state(c), []).append(c)
    held: dict = {}
    for c in by.get("held", []):
        held.setdefault(c["job_id"] or c.get("Company", ""), (c["company"], []))[1].append(c)
    why = lambda c: "" if c["company"] else (f"Job ID {c['job_id']} isn't on Companies" if c.get("Job ID")
                                              else "no Job ID: add one, or a Company that's on Companies")
    panes = [
        ("review", "Drafts to review", by.get("review", []), "Gmail drafts Aria wrote: read, then set Draft "
         "status to approved / sent on the Sheet tab"),
        ("approved", "Approved", by.get("approved", []), "yours to send from Gmail"),
        ("sent", "Sent", by.get("sent", []), "the next wave waits 5 business days for a reply"),
        ("replied", "Replied", by.get("replied", []), "the job's other contacts are closed"),
        ("needs", "Needs you", by.get("needs", []), "research, a Job ID, or an address Coldstart couldn't find"),
        ("lookup", "To look up", by.get("lookup", []), "Coldstart looks these up on the next run "
         "(\u201cfind emails for my contacts\u201d)"),
        ("waiting", "Later waves", by.get("waiting", []), "found; drafted when their wave comes up"),
    ]
    first = next((k for k, _, items, _ in panes if items), panes[0][0])
    bar = "".join(f'<button type="button" class="utab{" on" if k == first else ""}" data-p="{k}">{e(name)} '
                  f'<span class="count">{len(items)}</span></button>' for k, name, items, _ in panes)
    bar += (f'<button type="button" class="utab{" on" if first == "held" else ""}" data-p="held">Held '
            f'<span class="count">{len(held)}</span></button>')
    bodies = "".join(
        f'<div class="upane" data-p="{k}"{"" if k == first else " hidden"}><p class="muted small">{e(hint)}</p>'
        + (f'<div class="rows">{"".join(contact_row(c, why(c)) for c in items)}</div>' if items
           else f'<p class="muted">None right now.</p>') + '</div>'
        for k, name, items, hint in panes)
    bodies += ('<div class="upane" data-p="held" hidden><p class="muted small">Companies with Outreach status '
               'other than active: nobody there is looked up or written to.</p>'
               + (f'<div class="rows">{"".join(_held_row(co, ps) for co, ps in held.values())}</div>' if held
                  else '<p class="muted">None.</p>') + '</div>')
    closed = len(by.get("closed", []))
    hint = f"Muse sheet, Contacts + Companies · {len(rows)} people" + (f" · {closed} closed" if closed else "")
    return f'<section class="card">{label("Contacts", hint)}<div class="utabs" data-panes>{bar}</div>{bodies}</section>'


def review_count(mdl: dict) -> int:
    """Drafts waiting for the user's read (the sidebar badge)."""
    return sum(1 for c in mdl.get("contacts") or [] if contact_state(c) == "review")


def render(mdl: dict) -> str:
    srcs = mdl["sources"]
    g = srcs.get("gmail")
    me = (g.data or {}).get("me", "") if g is not None and g.ok else ""
    d = today.build(mdl)
    notes = today.group("LinkedIn notes to send", d["linkedin"], linkedin_row,
                        "from the sheet's outreach columns; Aria can't see LinkedIn, so some may already be sent",
                        down=_down(srcs, "sheet"), shown=SHOWN_NOTES)
    if not notes:
        notes = f'<section class="card">{label("LinkedIn notes to send")}<p class="muted">None to send.</p></section>'
    return contacts_card(mdl) + notes + email_card(mdl, d, me)
