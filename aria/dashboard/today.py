"""dashboard/today.py — Tab 1: what to do now, from the pipeline model (read-only)."""
from __future__ import annotations

import re
from datetime import timedelta
from urllib.parse import quote

from aria.dashboard import model as m
from aria.dashboard.render import _date, _down, _no_data, _thread_url, e

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
    linkedin = sorted((j for j in jobs if "linkedin" in j.channels and not j.threads and not j.email
                       and m.has_applied(j) and not m.is_closed(j)),
                      key=lambda j: -(float(j.fit) if (j.fit or "").replace(".", "").isdigit() else 0))
    deadlines = sorted((j for j in jobs if j.deadline and now <= j.deadline <= now + timedelta(days=3)
                        and not m.is_closed(j)), key=lambda j: j.deadline)
    return {"replies": replies, "interviews": interviews, "bounces": bounces, "follow_ups": follow_ups,
            "drafts": drafts, "finish": finish, "linkedin": linkedin, "deadlines": deadlines}


def summary_line(d: dict) -> str:
    parts = [(len(d["replies"]), "reply to answer", "replies to answer"),
             (len(d["interviews"]), "interview", "interviews"), (len(d["bounces"]), "bounce", "bounces"),
             (len(d["follow_ups"]), "follow-up due", "follow-ups due"),
             (len(d["finish"]), "application to finish", "applications to finish"),
             (len(d["drafts"]), "draft to send", "drafts to send"),
             (len(d["linkedin"]), "LinkedIn note to send", "LinkedIn notes to send"),
             (len(d["deadlines"]), "deadline in 3 days", "deadlines in 3 days")]
    out = [f"{n} {one if n == 1 else many}" for n, one, many in parts if n]
    return " · ".join(out) if out else "Nothing needs you right now."


def render(mdl: dict) -> str:
    srcs = mdl["sources"]
    g = srcs.get("gmail")
    me = (g.data or {}).get("me", "") if g is not None and g.ok else ""
    d = build(mdl)
    link = lambda t: f' <a href="{e(_thread_url(me, t["id"]))}" target="_blank" rel="noopener">thread</a>' if me and t.get("id") else ""
    job = lambda j: f"{e(j.company)} — {e(j.role)}" if j is not None else "(no matching job)"

    def card(title, items, item, down="", empty="Nothing."):
        body = _no_data(down) if down else ('<p class="muted">' + e(empty) + '</p>' if not items else
                                            "<ul>" + "".join(f"<li>{item(i)}</li>" for i in items) + "</ul>")
        return f'<div class="card"><h3>{e(title)} <span class="count">{len(items)}</span></h3>{body}</div>'

    gm = _down(srcs, "gmail")
    cards = [
        card("Replies to answer", d["replies"], lambda t: f'{job(t.get("job"))} · {e(t["to"])} — last {_date(t["last"])}{link(t)}', gm),
        card("Interviews", d["interviews"], lambda i: f'{job(i["job"])} <span class="muted small">{e(i["why"])}</span>'
             + (link(i["thread"]) if i.get("thread") else "")),
        card("Bounces", d["bounces"], lambda t: f'{job(t.get("job"))} · {e(t["to"])} bounced {_date(t["bounced"])}{link(t)}'
             f'<div class="fix small">Fix: {e(t["fix"])}</div>', gm),
        card(f"Follow-ups due (no reply in {m.FOLLOW_UP_DAYS}+ days)", d["follow_ups"],
             lambda t: f'{job(t["job"])} · {e(t["to"])} — sent {_date(t["sent"])}{link(t)}', gm),
        card("Applications to finish", d["finish"], lambda j: f'{job(j)} <span class="muted small">{e(j.stage_label if j.status.lower() != "needs human" else "Needs human")}</span>'),
        card(f"Drafts to send ({m.DRAFT_STALE_DAYS}+ days old)", d["drafts"],
             lambda x: f'{job(x.get("job"))} · {e(x["to"] or "no recipient")} <span class="muted small">“{e(x["subject"][:50])}”</span>'
             + link({"id": x["thread_id"]}), gm),
        card("LinkedIn notes to send", d["linkedin"],
             lambda j: f'{job(j)} <span class="muted small">fit {e(j.fit or "?")}/5{(" · " + e(j.contact)) if j.contact else ""}</span>',
             empty="Nothing.") .replace("</h3>", '</h3><p class="muted small">From the sheet’s Channel column; Aria can’t see LinkedIn, so some may already be sent.</p>', 1),
        card("Deadlines in the next 3 days", d["deadlines"], lambda j: f'{job(j)} — {_date(j.deadline)}',
             empty=("None." if mdl.get("has_deadline_col") else "Your sheet has no Deadline column, so there are no deadlines to show.")),
    ]
    return f'<p class="summary">{e(summary_line(d))}</p><div class="cards">{"".join(cards)}</div>'
