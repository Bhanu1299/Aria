"""dashboard/jobprep.py — job prep on Today: follow-ups due, employer replies waiting for your
Approve, reposted postings, and the do-not-apply list (docs/job-prep-design.md §6–7).

    render(mdl) → HTML cards (read: aria.db via features/jobprep)
    act(body) → dict          POST /api/jobprep {op, ...}:
        draft_followups                       jobs-Gmail drafts for every due follow-up (never sends)
        scan_replies                          classify new replies in the jobs Gmail
        approve_reply {id}                    the suggested Status → the sheet (the ONLY sheet write here)
        dismiss_reply {id}
        dna_add {text, reason}  dna_remove {text}
        save_base {json}                      the base resume (My answers), validated
        rebuild_base                          rebuild it from the resume PDF (old one → resume_base.prev.json)
    applied_from(mdl) → [(job dict, applied date)]   the model's Applied jobs, for followups.sync

Rendering never raises out (the page guards it); act raises JobPrepError for bad input.
"""
from __future__ import annotations

import logging

from aria.dashboard import model as m
from aria.dashboard.render import e, label, pill, row

logger = logging.getLogger(__name__)

SHOWN = 8


class JobPrepError(Exception):
    pass


def applied_from(mdl: dict) -> list:
    out = []
    for j in mdl.get("jobs", []):
        if m.has_applied(j) and not m.is_closed(j):
            when = m.applied_date(j)
            out.append(({"job_id": j.job_id, "company": j.company, "role": j.role}, when))
    return out


def _sheet_jobs(mdl: dict) -> list:
    return [{"job_id": j.job_id, "company": j.company, "role": j.role, "row": None} for j in mdl.get("jobs", [])]


def _followups_card(today) -> str:
    from aria.features.jobprep import followups
    due = followups.due(today)
    nxt = followups.upcoming(3, today)
    if not due:
        return ""
    rows = "".join(row(r["company"], r["role"], f"follow-up {r['n']} of 2 · applied {e(r['applied_at'])}",
                       pill(f"due {r['next_due']}", "warn")) for r in due[:SHOWN])
    later = "".join(row(r["company"], r["role"], f"follow-up {r['n']} of 2", pill(f"due {r['next_due']}", "gray"))
                    for r in nxt)
    button = ('<div class="lactions"><button class="btn primary jp" data-op="draft_followups">Draft my follow-ups</button>'
              '<span class="result small"></span></div>') if due else ""
    return (f'<section class="card" id="jp-followups">{label("Application follow-ups due", "drafts only — never sent", len(due))}'
            f'<div class="rows">{rows}</div>'
            + (f'<details class="showmore"><summary>Coming up</summary><div class="rows">{later}</div></details>'
               if later else "") + f"{button}</section>")


def _replies_card() -> str:
    from aria.features.jobprep import replies
    got = replies.pending()
    tone = {"interview": "accent", "offer": "ok", "rejected": "gray"}

    def item(s) -> str:
        right = (pill(s["kind"], tone.get(s["kind"], "")) +
                 f'<button class="btn jp" data-op="approve_reply" data-id="{int(s["id"])}">Set Status → '
                 f'{e(s["suggested_status"])}</button>'
                 f'<button class="btn jp" data-op="dismiss_reply" data-id="{int(s["id"])}">Dismiss</button>'
                 '<span class="result small"></span>')
        return row(s["company"], s["role"], f'“{e(s["subject"][:90])}” · {e(s["sender"][:60])} · by {e(s["by"])}', right)

    rows = "".join(item(s) for s in got[:SHOWN])
    scan = ('<div class="lactions"><button class="btn jp" data-op="scan_replies">Check the jobs Gmail for replies</button>'
            '<span class="result small"></span></div>')
    return (f'<section class="card" id="jp-replies">{label("Employer replies", "nothing goes to the sheet until you approve", len(got))}'
            f'<div class="rows">{rows or "<p class=muted>No replies waiting for you.</p>"}</div>{scan}</section>')


def _reposts_card(now) -> str:
    from aria.features.jobprep import reposts
    got = reposts.reposts(now)
    if not got:
        return ""
    rows = "".join(row(r["company"], r["role"], f'{r["count"]} postings · first seen {", ".join(r["dates"][:4])}',
                       pill("Repost", "warn")) for r in got[:SHOWN])
    return (f'<section class="card" id="jp-reposts">{label("Reposted postings", "same role, new link — worth a second look", len(got))}'
            f'<div class="rows">{rows}</div></section>')


def _dna_card() -> str:
    from aria.features.jobprep import blocklist
    got = blocklist.entries()
    items = "".join(f'<div class="kv"><span>{e(x["label"])}</span><span class="muted small">{e(x["scope"])}'
                    f'{" · " + e(x["reason"]) if x["reason"] else ""}</span>'
                    f'<a href="#" class="jp" data-op="dna_remove" data-text="{e(x["label"])}">remove</a></div>' for x in got)
    return (f'<section class="card" id="jp-dna">{label("Do-not-apply list", "batch runs skip these; a single apply asks first", len(got))}'
            f'{items or "<p class=muted>Empty. Say “never apply to Acme” or add one here.</p>"}'
            '<div class="lactions"><input class="dna-text" placeholder="Company or domain (acme.com)">'
            '<input class="dna-reason" placeholder="Why (optional)">'
            '<button class="btn jp" data-op="dna_add">Add</button><span class="result small"></span></div></section>')


def base_resume_card() -> str:
    """My answers: the base resume every tailored resume starts from, as editable JSON."""
    import json
    from aria.features.resume import model as rm
    d = rm.load_base()
    body = json.dumps(rm.strip_ids(d), indent=2, ensure_ascii=False) if d else ""
    hint = ("every tailored resume starts here; employers, dates and numbers are never changed per job"
            if d else "not built yet — it's made from your resume PDF the first time a job is prepared")
    return (f'<section class="card" id="jp-base">{label("Base resume", hint)}'
            f'<textarea class="base-json" rows="18" spellcheck="false" style="width:100%;font:12px/1.45 var(--mono)">'
            f'{e(body)}</textarea><div class="lactions">'
            '<button class="btn primary jp" data-op="save_base">Save base resume</button>'
            '<button class="btn jp" data-op="rebuild_base">Rebuild from my resume PDF</button>'
            '<span class="result small"></span></div></section>')


def _save_base(text: str) -> dict:
    import json
    from aria.features.resume import model as rm
    try:
        got = json.loads(text or "")
    except ValueError as exc:
        raise JobPrepError(f"That isn't valid JSON: {exc}")
    d = rm.normalize(got)
    if not d["name"] or not (d["experience"] or d["projects"]):
        raise JobPrepError("A base resume needs a name and at least one job or project.")
    rm.save_base(d)
    return {"message": "Saved. Jobs prepared from now on use it."}


def _rebuild_base() -> dict:
    from aria.features.ats import planner
    from aria.features.resume import model as rm
    pdf = planner.resume_path()
    if not pdf:
        raise JobPrepError("No resume_path in your answer bank (or the file is missing).")
    if rm.BASE_PATH.is_file():
        rm.BASE_PATH.replace(rm.BASE_PATH.with_name("resume_base.prev.json"))
    got = rm.ensure_base(pdf)
    if not got:
        raise JobPrepError("Couldn't read that PDF (is poppler installed?).")
    return {"message": f"Rebuilt from {pdf}: {len(got['experience'])} jobs, {len(got['projects'])} projects. "
                       "Reload to see it."}


def render(mdl: dict) -> str:
    from aria.features.jobprep import followups
    parts = []
    try:
        followups.sync(applied_from(mdl))
    except Exception as exc:
        logger.warning("dashboard: follow-up sync failed: %s", exc)
    for name, fn in (("follow-ups", lambda: _followups_card(mdl["now"].date())), ("replies", _replies_card),
                     ("reposts", lambda: _reposts_card(mdl["now"])), ("do-not-apply", _dna_card)):
        try:
            parts.append(fn())
        except Exception as exc:
            logger.exception("dashboard: job prep %s card failed", name)
            parts.append(f'<section class="card">{label(name.capitalize())}<p class="nodata muted">No data — '
                         f'{e(type(exc).__name__)}</p></section>')
    return "".join(parts)


def act(body: dict) -> dict:
    from aria.features.jobprep import blocklist, followups, replies
    op = str(body.get("op") or "")
    if op == "draft_followups":
        return {"message": followups.draft_due()}
    if op == "scan_replies":
        from aria.dashboard import sources
        return {"message": replies.scan_gmail(_jobs_for_replies(sources))}
    if op in ("approve_reply", "dismiss_reply"):
        try:
            sid = int(body.get("id"))
        except (TypeError, ValueError):
            raise JobPrepError("bad id")
        if op == "dismiss_reply":
            replies.dismiss(sid)
            return {"message": "Dismissed."}
        try:
            return {"message": replies.approve(sid)}
        except ValueError as exc:
            raise JobPrepError(str(exc))
    if op == "save_base":
        return _save_base(str(body.get("json") or ""))
    if op == "rebuild_base":
        return _rebuild_base()
    if op == "dna_add":
        try:
            got = blocklist.add(str(body.get("text") or ""), str(body.get("reason") or ""))
        except ValueError as exc:
            raise JobPrepError(str(exc))
        return {"message": f"Added {got['label']}. Reload to see it."}
    if op == "dna_remove":
        ok = blocklist.remove(str(body.get("text") or ""))
        return {"message": "Removed." if ok else "It wasn't on the list."}
    raise JobPrepError("unknown op")


def _jobs_for_replies(sources) -> list:
    src = sources.load_all()
    mdl = m.build(src)
    return _sheet_jobs(mdl)


CSS = """
#jp-dna .kv{display:flex;gap:10px;align-items:center;padding:4px 0}#jp-dna .kv a{margin-left:auto}
#jp-dna input{font:inherit;font-size:13px;padding:6px 9px;border:1px solid var(--line);border-radius:8px;background:var(--input);color:var(--ink)}
#jp-replies .btn{margin-left:6px}
"""

JS = r"""
(function(){var tok=document.querySelector('meta[name=dashboard-token]');if(!tok)return;tok=tok.content;
document.addEventListener('click',function(ev){var t=ev.target.closest('.jp');if(!t)return;ev.preventDefault();
var body={op:t.dataset.op};if(t.dataset.id)body.id=parseInt(t.dataset.id,10);if(t.dataset.text)body.text=t.dataset.text;
var card=t.closest('section');if(body.op==='dna_add'){body.text=card.querySelector('.dna-text').value;body.reason=card.querySelector('.dna-reason').value}
if(body.op==='save_base'){body.json=card.querySelector('.base-json').value}
var out=(t.closest('.row,.lactions,.kv')||card).querySelector('.result')||card.querySelector('.result');t.disabled=true;
if(out){out.textContent='Working…';out.className='result small'}
fetch('/api/jobprep',{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':tok},body:JSON.stringify(body)})
.then(function(r){return r.json().then(function(d){d._ok=r.ok;return d})}).then(function(d){t.disabled=false;
if(out){out.textContent=d.message||d.error||'done';out.className='result small '+(d._ok?'ok':'bad')}
if(d._ok&&(body.op==='dismiss_reply'||body.op==='approve_reply'||body.op==='dna_remove')){var r=t.closest('.row,.kv');if(r)r.style.opacity=.45}})
.catch(function(){t.disabled=false;if(out){out.textContent='failed';out.className='result small bad'}})})})();
"""
