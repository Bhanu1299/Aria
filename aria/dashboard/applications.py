"""dashboard/applications.py — Tab 3: one row per company, roles nested, with a timeline per role."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from aria.dashboard import model as m
from aria.dashboard.render import _date, _no_data, _thread_url, e, initials, pill

FILTERS = [("all", "All"), ("needs_action", "Needs action"), ("waiting", "Waiting"), ("no_contact", "No contact yet"),
           ("replied", "Replied"), ("closed", "Closed")]


def _fit(j: m.Job) -> float:
    try:
        return float(j.fit)
    except (TypeError, ValueError):
        return 0.0


def timeline(j: m.Job) -> list:
    """[(date | None, what, link | None)] in pipeline order."""
    items = [(j.found_date, "Found by Muse" + ("" if j.found_date else " (no date in the sheet)"), j.link or None)]
    for d in sorted(j.applied_on):
        items.append((d, "Applied", None))
    if m.has_applied(j) and not j.applied_on:
        items.append((None, "Applied (sheet Status; no date recorded)", None))
    for r in j.runs:
        if r.get("stage") == "filled":
            items.append((m._ts(r.get("created_at")), f"Form filled by Aria ({r.get('needs_me', 0)} fields needed you)", None))
    if j.contact:
        src = next((ev.source for ev in j.evidence if ev.stage == m.CONTACT), "")
        items.append((None, f"Contact found: {j.contact}{(' (' + j.contact_title + ')') if j.contact_title else ''} — {src}", None))
    for x in j.drafts:
        items.append((x["created"], f"Draft ready: “{x['subject'][:60]}”", ("thread", x["thread_id"])))
    for t in j.threads:
        items.append((t["sent"], f"Sent to {t['to']}", ("thread", t["id"])))
        if t["bounced"]:
            items.append((t["bounced"], f"Bounced: {t['to']}", ("thread", t["id"])))
        if t["replied"]:
            items.append((t["replied"], "Reply received", ("thread", t["id"])))
    if j.stage == m.OUTCOME:
        items.append((None, f"Outcome: {j.stage_label}{(' — ' + j.outcome) if j.outcome else ''}", None))
    return items


def build(mdl: dict) -> list:
    now = mdl["now"]
    groups: dict = {}
    for j in mdl["jobs"]:
        groups.setdefault(m.company_slug(j.company) or j.company.lower(), []).append(j)
    companies = []
    for slug, jobs in groups.items():
        sent_roles = [j.role for j in jobs if j.threads]
        roles = []
        for j in sorted(jobs, key=lambda j: (-j.stage, j.role)):
            elsewhere = next((r for r in sent_roles if r != j.role), "") if not j.threads else ""
            action, bucket = m.next_action(j, now, elsewhere)
            if bucket == "no_contact" and not m.has_applied(j):
                bucket = "todo"
            roles.append({"job": j, "action": action, "bucket": bucket,
                          "sent": min((t["sent"] for t in j.threads), default=None)})
        companies.append({"company": jobs[0].company, "roles": roles, "dup_outreach": len(sent_roles) > 1,
                          "best_fit": max(_fit(j) for j in jobs),
                          "last": max((j.last_update for j in jobs if j.last_update), default=None)})
    companies.sort(key=lambda c: (c["last"] or datetime.min), reverse=True)
    return companies


# Next action: what kind of to-do it is. Response: how it went.
ACTION_TONE = {"needs_action": "bad", "replied": "bad", "waiting": "warn", "no_contact": "warn", "todo": ""}
SORTS = [("recent", "Recent activity"), ("fit", "Fit"), ("applied", "Applied date"), ("company", "Company")]


def response_pill(text: str) -> str:
    low = text.lower()
    tone = ("ok" if low.startswith("replied") or low == "offer" else "bad" if low == "bounced"
            else "warn" if low == "interview" else "gray")
    return pill(text, tone)


def _role_row(r: dict, me: str) -> str:
    j = r["job"]
    conf = "" if j.email_confidence is None else f" · {j.email_confidence:.2f}"
    channel = " / ".join(sorted(j.channels)) or "—"
    contact = e(j.contact) + (f'<div class="muted small">{e(j.contact_title)}</div>' if j.contact_title else "")
    tl = "".join(
        f'<li><span class="muted small">{e(_date(d) or "—")}</span> {e(what)}'
        + (f' <a href="{e(_thread_url(me, ln[1]))}" target="_blank" rel="noopener">thread</a>' if ln and ln[0] == "thread" and me
           else f' <a href="{e(ln)}" target="_blank" rel="noopener">posting</a>' if isinstance(ln, str) else "") + "</li>"
        for d, what, ln in timeline(j))
    return (f'<tr class="role" data-bucket="{e(r["bucket"])}" data-fit="{_fit(j)}">'
            f'<td><details><summary>{e(j.role)}</summary><ul class="timeline">{tl}</ul></details></td>'
            f'<td>{e(_date(m.applied_date(j)) or ("yes" if m.has_applied(j) else ""))}</td><td>{contact}</td>'
            f'<td>{e(channel)}</td><td>{e(j.email)}<span class="muted small">{e(conf)}</span></td>'
            f'<td>{e(_date(r["sent"]))}</td><td>{response_pill(m.response(j))}</td>'
            f'<td>{pill(r["action"], ACTION_TONE.get(r["bucket"], ""))}</td></tr>')


def render(mdl: dict) -> str:
    if not mdl["jobs"]:
        return _no_data("no jobs (Muse sheet unavailable?)")
    g = mdl["sources"].get("gmail")
    me = (g.data or {}).get("me", "") if g is not None and g.ok else ""
    companies = build(mdl)
    counts = {k: sum(1 for c in companies for r in c["roles"] if k == "all" or r["bucket"] == k) for k, _ in FILTERS}
    tabs = "".join(f'<button type="button" class="utab filter{" on" if k == "all" else ""}" data-f="{k}">{e(v)} '
                   f'<span class="count">{counts[k]}</span></button>' for k, v in FILTERS)
    sorts = "".join(f'<option value="{k}">{e(v)}</option>' for k, v in SORTS)
    rows = []
    for c in companies:
        warn = f' {pill("outreach sent for more than one role", "warn")}' if c["dup_outreach"] else ""
        applied = max((m.applied_date(r["job"]) for r in c["roles"] if m.applied_date(r["job"])), default=None)
        rows.append(f'<tr class="company" data-fit="{c["best_fit"]}" data-applied="{e(_date(applied))}" '
                    f'data-name="{e(c["company"].lower())}"><td colspan="8"><span class="av" aria-hidden="true">'
                    f'{e(initials(c["company"]))}</span><b>{e(c["company"])}</b>'
                    f' <span class="muted small">{len(c["roles"])} role{"s" if len(c["roles"]) != 1 else ""}</span>{warn}</td></tr>')
        rows.extend(_role_row(r, me) for r in c["roles"])
    head = "".join(f"<th>{e(h)}</th>" for h in ("Role", "Applied", "Contact", "Channel", "Email · confidence",
                                                  "Sent", "Response", "Next action"))
    return (f'<section class="card"><div class="utabs filters">{tabs}</div>'
            '<div class="toolbar"><input id="appq" type="search" placeholder="Search company, role, contact…" autocomplete="off">'
            f'<label>Sort <select id="appsort">{sorts}</select></label></div>'
            '<p class="muted small" id="nocontact-note" hidden>Applied jobs with nobody to reach out to, best fit first.</p>'
            f'<div class="tablewrap"><table id="apps"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="muted small" id="apps-empty" hidden>No roles match.</p>'
            '<p class="muted small">Click a role for its timeline. One company row: outreach goes to one person per company.</p>'
            '</section>')


# Filter (tab) + search + sort, all client-side over the rows already on the page.
JS = r"""
(function(){var t=document.getElementById('apps');if(!t)return;var tb=t.tBodies[0];var note=document.getElementById('nocontact-note');
var q=document.getElementById('appq'),sortSel=document.getElementById('appsort'),empty=document.getElementById('apps-empty');
var groups=[],cur=null,f='all';
Array.prototype.forEach.call(tb.rows,function(r,i){if(r.classList.contains('company')){cur={head:r,roles:[],i:i};groups.push(cur)}else if(cur)cur.roles.push(r)});
function key(g,s){var d=g.head.dataset;if(s==='fit')return -parseFloat(d.fit||0);if(s==='applied')return d.applied?-Date.parse(d.applied):Infinity;
if(s==='company')return d.name;return g.i}
function cmp(a,b){return a<b?-1:a>b?1:0}
function hit(r,g,s){return !s||(g.head.dataset.name+' '+r.textContent.toLowerCase()).indexOf(s)>=0}
function apply(){var s=q.value.trim().toLowerCase(),shown=0;note.hidden=f!=='no_contact';tb.textContent='';
if(f==='no_contact'){var flat=[];groups.forEach(function(g){g.roles.forEach(function(r){if(r.dataset.bucket==='no_contact'&&hit(r,g,s))flat.push(r)})});
flat.sort(function(a,b){return parseFloat(b.dataset.fit)-parseFloat(a.dataset.fit)});flat.forEach(function(r){r.style.display='';tb.appendChild(r)});
empty.hidden=flat.length>0;return}
var so=sortSel.value,gs=groups.slice().sort(function(a,b){return cmp(key(a,so),key(b,so))||a.i-b.i});
gs.forEach(function(g){var any=false;tb.appendChild(g.head);g.roles.forEach(function(r){var on=(f==='all'||r.dataset.bucket===f)&&hit(r,g,s);
r.style.display=on?'':'none';if(on){any=true;shown++}tb.appendChild(r)});g.head.style.display=any?'':'none'});empty.hidden=shown>0}
document.querySelectorAll('.filter').forEach(function(b){b.onclick=function(){document.querySelectorAll('.filter').forEach(function(x){x.classList.remove('on')});
b.classList.add('on');f=b.dataset.f;apply()}});q.addEventListener('input',apply);sortSel.addEventListener('change',apply);apply()})();
"""
