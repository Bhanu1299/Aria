"""dashboard/outreach.py — Tab "Outreach": LinkedIn notes to send and the email threads waiting
on you (moved here from Today). Read-only: links open Gmail / LinkedIn, nothing is sent."""
from __future__ import annotations

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
    return notes + email_card(mdl, d, me)
