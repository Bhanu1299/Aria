"""dashboard/render.py — model → one HTML page. Each section renders on its own: if its
source is down (or it fails), that section says "no data" and the rest still shows."""
from __future__ import annotations

import html
import logging
import re
import time
from datetime import datetime
from typing import Callable, Optional
from urllib.parse import quote

from aria.dashboard import model as m

logger = logging.getLogger(__name__)

SOURCE_LABELS = {"sheet": "Muse sheet", "gmail": "Gmail (jobs)", "aria_db": "aria.db",
                 "coldstart": "Coldstart", "reviews": "Review files"}


def e(value) -> str:
    return html.escape("" if value is None else str(value))


def _date(d: Optional[datetime]) -> str:
    return d.strftime("%Y-%m-%d") if d else ""


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.0%}"


def _money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"${x:.6f}" if 0 < x < 0.001 else f"${x:.4f}" if x < 1 else f"${x:.2f}"


def _thread_url(me: str, thread_id: str) -> str:
    return f"https://mail.google.com/mail/?authuser={quote(me)}#all/{quote(thread_id)}"


def _no_data(why: str) -> str:
    return f'<p class="nodata">No data — {e(why)}</p>'


def _down(srcs: dict, *names: str) -> str:
    bad = [f"{SOURCE_LABELS.get(n, n)} unavailable" for n in names if not srcs.get(n) or not srcs[n].ok]
    return "; ".join(bad)


def _section(title: str, body: Callable[[], str], sub: str = "", sid: str = "") -> str:
    try:
        inner = body()
    except Exception as exc:
        logger.exception("dashboard: section %s failed", title)
        inner = _no_data(f"this section failed ({type(exc).__name__}: {exc})")
    sub_html = f'<span class="sub">{e(sub)}</span>' if sub else ""
    return f'<section id="{e(sid)}"><h2>{e(title)} {sub_html}</h2>{inner}</section>'


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _sources_bar(srcs: dict) -> str:
    now = time.time()
    chips = []
    for name, label in SOURCE_LABELS.items():
        s = srcs.get(name)
        if s is None:
            continue
        age = int(now - s.loaded_at)
        state = (f"ok · read {age // 60}m ago" if age >= 60 else "ok · fresh") if s.ok else "down"
        title = e(s.error) if not s.ok else ""
        chips.append(f'<span class="chip {"ok" if s.ok else "bad"}" title="{title}">{e(label)}: {state}</span>')
    return f'<div class="chips">{"".join(chips)}</div>'


def _funnel(mdl: dict) -> str:
    srcs = mdl["sources"]
    if not mdl["jobs"]:
        return _no_data(_down(srcs, "sheet") or "no jobs found")
    rows = []
    top = max(1, mdl["jobs"] and len(mdl["jobs"]))
    for r in m.funnel(mdl["jobs"]):
        width = int(100 * r["reached"] / top)
        drop = "" if r["drop"] is None else (f"−{r['drop']}" if r["drop"] else "0")
        extra = f' <span class="muted">({r["skipped"]} skipped)</span>' if r.get("skipped") else ""
        rows.append(
            f'<tr><td>{e(r["stage"])}</td>'
            f'<td class="num">{r["reached"]}</td>'
            f'<td class="bar"><span style="width:{width}%"></span></td>'
            f'<td class="num">{r["at"]}{extra}</td><td class="num drop">{drop}</td>'
            f'<td class="muted small">{e(", ".join(r["sources"])) or "—"}</td></tr>')
    note = _down(srcs, "gmail", "aria_db", "coldstart")
    warn = f'<p class="warn">Partial: {e(note)} — later stages may be undercounted.</p>' if note else ""
    return (warn + '<table class="funnel"><thead><tr><th>Stage</th><th class="num">Reached</th><th></th>'
            '<th class="num">Now at</th><th class="num">Drop</th><th>Evidence from</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _job_label(job: m.Job) -> str:
    return f"{e(job.company)} — {e(job.role)}"


def _needs_me(mdl: dict) -> str:
    srcs, me = mdl["sources"], (mdl["sources"].get("gmail").data or {}).get("me", "") if mdl["sources"].get("gmail") and mdl["sources"]["gmail"].ok else ""
    n = m.needs_me(mdl)
    blocks = []

    def block(title: str, items: list, render_item: Callable, down: str = "") -> None:
        if down:
            body = _no_data(down)
        elif not items:
            body = '<p class="muted">Nothing.</p>'
        else:
            body = "<ul>" + "".join(f"<li>{render_item(i)}</li>" for i in items) + "</ul>"
        blocks.append(f'<div class="card"><h3>{e(title)} <span class="count">{len(items)}</span></h3>{body}</div>')

    def field_item(i):
        # Machine names like cards[uuid][field0] mean nothing to a reader: count them instead.
        named = [f for f in i["fields"] if not re.match(r"^[\w-]+\[", f)]
        unnamed = len(i["fields"]) - len(named)
        names = ", ".join(named[:6]) + (" …" if len(named) > 6 else "")
        if unnamed:
            names += f"{', ' if names else ''}{unnamed} unlabeled field{'s' if unnamed != 1 else ''}"
        review = f'<div class="muted small">{e(i["review"])}</div>' if i["review"] else ""
        return f'{_job_label(i["job"])}: {i["count"]} field(s){(" — " + e(names)) if names else ""}{review}'

    def thread_item(t, what):
        link = f'<a href="{e(_thread_url(me, t["id"]))}" target="_blank" rel="noopener">thread</a>' if me else ""
        job = f'{_job_label(t["job"])} · ' if t.get("job") else ""
        who = e(t["to"]) if t.get("to") else '<span class="muted">no recipient</span>'
        subject = f' <span class="muted small">“{e(t["subject"][:60])}”</span>' if t.get("subject") and not t.get("job") else ""
        return f'{job}{who}{subject} — {what} {link}'

    block("Form fields marked needs_me", n["fields"], field_item, _down(srcs, "aria_db"))
    block("Rows marked Needs human", n["needs_human"],
          lambda i: f'{_job_label(i["job"])} <span class="muted small">({e(i["where"])})</span>', _down(srcs, "sheet"))
    block(f"Drafts not sent after {m.DRAFT_STALE_DAYS} days (last {m.NEEDS_ME_WINDOW_DAYS})", n["stale_drafts"],
          lambda d: thread_item(d | {"id": d["thread_id"]}, f'draft from {_date(d["created"])}'), _down(srcs, "gmail"))
    block(f"Sent, no reply after {m.FOLLOW_UP_DAYS} days — follow-up due (last {m.NEEDS_ME_WINDOW_DAYS})", n["follow_ups"],
          lambda t: thread_item(t, f'sent {_date(t["sent"])}'), _down(srcs, "gmail"))
    block("Bounced emails", n["bounces"], lambda t: thread_item(t, f'bounced {_date(t["bounced"])}'),
          _down(srcs, "gmail"))
    return f'<div class="cards">{"".join(blocks)}</div>'


def _jobs_table(mdl: dict) -> str:
    if not mdl["jobs"]:
        return _no_data(_down(mdl["sources"], "sheet") or "no jobs found")
    g = mdl["sources"].get("gmail")
    me = (g.data or {}).get("me", "") if g is not None and g.ok else ""
    rows = []
    for j in sorted(mdl["jobs"], key=lambda j: (-j.stage, j.company.lower())):
        conf = "" if j.email_confidence is None else f" · {j.email_confidence:.2f}"
        status = f" · {e(j.email_status)}" if j.email_status else ""
        links = []
        if j.link:
            links.append(f'<a href="{e(j.link)}" target="_blank" rel="noopener">posting</a>')
        for t in j.threads[:2]:
            if me:
                links.append(f'<a href="{e(_thread_url(me, t["id"]))}" target="_blank" rel="noopener">thread</a>')
        evidence = "\n".join(f"{m.STAGES[ev.stage]}: {ev.source}{(' — ' + ev.detail) if ev.detail else ''}"
                             for ev in sorted(j.evidence, key=lambda ev: ev.stage))
        rows.append(
            f'<tr><td>{e(j.company)}</td><td>{e(j.role)}</td>'
            f'<td class="num" data-v="{e(j.fit or 0)}">{e(j.fit)}</td>'
            f'<td data-v="{j.stage}"><span class="stage s{j.stage}" title="{e(evidence)}">{e(j.stage_label)}</span></td>'
            f'<td>{e(j.contact)}</td><td>{e(j.email)}<span class="muted small">{e(conf)}{status}</span></td>'
            f'<td data-v="{e(_date(j.last_update))}">{e(_date(j.last_update))}</td>'
            f'<td class="num" data-v="{j.cost:.6f}">{e(_money(j.cost) if j.cost else "")}</td>'
            f'<td>{" · ".join(links)}</td></tr>')
    head = "".join(f'<th data-k="{i}"{" class=num" if num else ""}>{e(name)}</th>' for i, (name, num) in enumerate(
        [("Company", 0), ("Role", 0), ("Fit /5", 1), ("Stage", 0), ("Contact", 0), ("Email · confidence", 0),
         ("Last update", 0), ("Cost", 1), ("Links", 0)]))
    unmatched = ""
    if mdl["unmatched"]:
        items = "".join(
            f'<li><span class="muted small">{e(u["source"])}</span> {e(u["what"])} <span class="muted small">{e(u["detail"])}</span>'
            + (f' <a href="{e(_thread_url(me, u["thread_id"]))}" target="_blank" rel="noopener">thread</a>'
               if me and u.get("thread_id") else "") + "</li>"
            for u in mdl["unmatched"])
        unmatched = (f'<details class="unmatched"><summary>{len(mdl["unmatched"])} records matched no job</summary>'
                     f'<ul>{items}</ul></details>')
    return (f'<input id="q" type="search" placeholder="Search company, role, contact, stage…" autocomplete="off">'
            f'<div class="tablewrap"><table id="jobs"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody>'
            f'</table></div><p class="muted small">Hover a stage for its evidence. Click a header to sort.</p>{unmatched}')


def _week(mdl: dict) -> str:
    w = m.this_week(mdl)
    srcs = mdl["sources"]
    stat = lambda label, value, note="": (f'<div class="stat"><div class="v">{value}</div><div class="l">{e(label)}</div>'
                                         f'{f"<div class=muted small>{e(note)}</div>" if note else ""}</div>')
    gm = srcs.get("gmail") is not None and srcs["gmail"].ok
    db = srcs.get("aria_db") is not None and srcs["aria_db"].ok
    found_note = f"{w['found_undated']} jobs have no found date" if w["found_undated"] else ""
    return (f'<p class="muted small">Since {w["start"]:%a %b %d}.</p><div class="stats">'
            + stat("jobs found", w["found"] if srcs.get("sheet") and srcs["sheet"].ok else "—", found_note)
            + stat("applied (by Aria)", w["applied"] if db else "—", "" if db else "aria.db unavailable")
            + stat("outreach sent", w["sent"] if gm else "—", "" if gm else "Gmail unavailable")
            + stat("replies", w["replies"] if gm else "—")
            + stat("reply rate this week", _pct(w["reply_rate_week"]) if gm else "—")
            + stat("reply rate, all time", _pct(w["reply_rate_all"]) if gm else "—",
                   f"{w['sent_all']} outreach threads" if gm else "")
            + "</div>")


def _cost(mdl: dict) -> str:
    if not mdl["sources"].get("aria_db") or not mdl["sources"]["aria_db"].ok:
        return _no_data(_down(mdl["sources"], "aria_db"))
    c = m.cost(mdl)
    return (f'<div class="stats"><div class="stat"><div class="v">{_money(c["total"])}</div>'
            f'<div class="l">total LLM spend</div><div class="muted small">{c["runs"]} pipeline runs</div></div>'
            f'<div class="stat"><div class="v">{_money(c["per_applied"])}</div><div class="l">per applied job</div>'
            f'<div class="muted small">{c["applied"]} applied jobs</div></div>'
            f'<div class="stat"><div class="v">{_money(c["per_applied_by_aria"])}</div>'
            f'<div class="l">per job Aria worked on</div><div class="muted small">{c["applied_by_aria"]} jobs with runs</div>'
            f'</div></div>')


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--accent:#2f5d8a;
--ok:#2e7d4f;--bad:#b3261e;--warn:#8a5a00;--bar:#c9d8e8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;
--muted:#9b9a94;--line:#33322f;--accent:#8ab4e0;--ok:#6cc08b;--bad:#ef8a80;--warn:#e0b25c;--bar:#2c4258}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{padding:20px 24px 8px;max-width:1280px;margin:0 auto}h1{margin:0 0 6px;font-size:20px}
main{max-width:1280px;margin:0 auto;padding:0 24px 40px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:14px 0}
h2{font-size:15px;margin:0 0 12px}h3{font-size:13px;margin:0 0 6px}.sub{font-weight:400;color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;white-space:nowrap}#jobs th{cursor:pointer;user-select:none}
#jobs th.asc:after{content:" ▲"}#jobs th.desc:after{content:" ▼"}
.num{text-align:right;font-variant-numeric:tabular-nums}.muted{color:var(--muted)}.small{font-size:12px}
.bar{width:28%}.bar span{display:block;height:10px;background:var(--bar);border-radius:3px;min-width:2px}
.drop{color:var(--bad)}.nodata{color:var(--warn);margin:0}.warn{color:var(--warn);margin:0 0 8px}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{font-size:12px;padding:2px 8px;border-radius:10px;border:1px solid var(--line)}
.chip.ok{color:var(--ok)}.chip.bad{color:var(--bad);border-color:var(--bad)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-wrap:anywhere;min-width:0}
td{overflow-wrap:anywhere}.card ul{margin:0;padding-left:18px}
.card li{margin:3px 0}.count{color:var(--muted);font-weight:400}
.stats{display:flex;flex-wrap:wrap;gap:12px}.stat{min-width:150px;border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.stat .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}.stat .l{color:var(--muted);font-size:12px}
#q{width:100%;max-width:420px;padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);margin-bottom:10px}
.tablewrap{overflow-x:auto}.stage{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;background:var(--bar);white-space:nowrap}
.stage.s8,.stage.s9{background:var(--ok);color:var(--panel)}a{color:var(--accent)}
details.unmatched{margin-top:12px}details.unmatched ul{padding-left:18px}
@media (max-width:720px){header,main{padding-left:16px;padding-right:16px}.bar{display:none}}
"""

JS = """
(function(){var t=document.getElementById('jobs');if(!t)return;var tb=t.tBodies[0];
var q=document.getElementById('q');q.addEventListener('input',function(){var s=q.value.toLowerCase();
Array.prototype.forEach.call(tb.rows,function(r){r.style.display=r.textContent.toLowerCase().indexOf(s)>=0?'':'none'})});
Array.prototype.forEach.call(t.tHead.rows[0].cells,function(th,i){th.addEventListener('click',function(){
var asc=!th.classList.contains('asc');Array.prototype.forEach.call(t.tHead.rows[0].cells,function(h){h.classList.remove('asc','desc')});
th.classList.add(asc?'asc':'desc');var rows=Array.prototype.slice.call(tb.rows);
rows.sort(function(a,b){var x=a.cells[i].dataset.v||a.cells[i].textContent.trim(),y=b.cells[i].dataset.v||b.cells[i].textContent.trim();
var nx=parseFloat(x),ny=parseFloat(y);var c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c});
rows.forEach(function(r){tb.appendChild(r)})})})})();
"""


TAB_NAMES = [("today", "Today"), ("answers", "My answers"), ("applications", "Applications"), ("live", "Live run"),
             ("costs", "Costs"), ("settings", "Settings"), ("health", "Data health")]
TAB_SUBS = {"today": "what needs you now", "answers": "everything Aria fills into forms — edits save to your answer bank",
            "applications": "one row per company, roles nested; click a role for its timeline",
            "live": "current or last pipeline run, from Aria's run log",
            "costs": "from GateLLM's per-call log + Aria's flight log",
            "settings": "LLM providers and tier models",
            "health": "sources, unmatched records and gaps in the data"}


def page(mdl: dict, tabs: Optional[dict] = None, token: str = "") -> str:
    """tabs: {key: html} for every TAB_NAMES key (each already guarded)."""
    from aria.dashboard import tabs as tabmod
    srcs = mdl["sources"]
    tabs = tabs or {}
    down = [SOURCE_LABELS.get(k, k) for k, v in srcs.items() if v is not None and not v.ok]
    warn = (f' <a class="downwarn" href="#health">{len(down)} source{"s" if len(down) != 1 else ""} down: '
            f'{e(", ".join(down))}</a>') if down else ""
    nav = "".join(f'<a href="#{k}">{e(v)}</a>' for k, v in TAB_NAMES)
    panes = []
    for key, label in TAB_NAMES:
        content = tabs.get(key, _no_data("not loaded"))
        panes.append(f'<div class="tab" id="{key}-tab"><section><h2>{e(label)} <span class="sub">{e(TAB_SUBS[key])}</span></h2>'
                     f'{content}</section></div>')
    meta = f'<meta name="dashboard-token" content="{e(token)}">' if token else ""
    from aria.dashboard import answers as ans, applications as apps
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">{meta}<title>Job Pipeline</title>'
            f'<style>{CSS}{tabmod.TABS_CSS}</style></head><body><header><h1>Job pipeline</h1>'
            f'<p class="muted small">Loaded {mdl["now"]:%Y-%m-%d %H:%M}. Gmail and the sheet are cached for 5 minutes. '
            f'Writes only: your answer bank (My answers) and GateLLM keys (Settings).{warn}</p>'
            f'<nav class="tabs">{nav}</nav></header><main>{"".join(panes)}</main>'
            f'<script>{tabmod.TABS_JS}{JS}{tabmod.LIVE_JS}{tabmod.SETTINGS_JS}{ans.JS}{apps.JS}</script></body></html>')


def today_extras(mdl: dict) -> str:
    """Funnel + this week, kept under Today."""
    return (_section("Funnel", lambda: _funnel(mdl), "jobs that reached each stage; a job sits at its furthest stage with evidence", "funnel")
            + _section("This week", lambda: _week(mdl), "", "week"))


def all_jobs_table(mdl: dict) -> str:
    return ('<details class="flat"><summary>Flat table of all jobs (searchable, sortable)</summary>'
            + _section("All jobs", lambda: _jobs_table(mdl), f"{len(mdl['jobs'])} jobs", "jobs-section") + "</details>")


def pipeline_cost(mdl: dict) -> str:
    return _section("Pipeline runs (aria.db)", lambda: _cost(mdl), "LLM spend recorded per application run", "cost")
