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
    return f'<p class="nodata muted">No data — {e(why)}</p>'


def _down(srcs: dict, *names: str) -> str:
    bad = [f"{SOURCE_LABELS.get(n, n)} unavailable" for n in names if not srcs.get(n) or not srcs[n].ok]
    return "; ".join(bad)


def _section(title: str, body: Callable[[], str], sub: str = "", sid: str = "") -> str:
    try:
        inner = body()
    except Exception as exc:
        logger.exception("dashboard: section %s failed", title)
        inner = _no_data(f"this section failed ({type(exc).__name__}: {exc})")
    return f'<section class="card" id="{e(sid)}">{label(title, sub)}{inner}</section>'


# ---------------------------------------------------------------------------
# building blocks shared by every tab
# ---------------------------------------------------------------------------

def label(title: str, hint: str = "", count: Optional[int] = None) -> str:
    """A section label: small caps title, optional count, then a muted hint after a ·"""
    n = f' <span class="count">{count}</span>' if count is not None else ""
    h = f'<span class="hint"> · {e(hint)}</span>' if hint else ""
    return f'<h3>{e(title)}{n}{h}</h3>'


def initials(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name or "")
    if not words:
        return "?"
    return (words[0][:2] if len(words) == 1 else words[0][0] + words[1][0]).upper()


def pill(text: str, tone: str = "") -> str:
    """tone: bad (needs you) · warn (in progress / waiting) · ok (done) · gray (closed) · accent."""
    return f'<span class="pill {tone}">{e(text)}</span>' if text else ""


def row(company: str, role: str = "", meta: str = "", right: str = "", below: str = "", attrs: str = "") -> str:
    """One list row: initials badge · Company · role, a muted meta line, pills/links on the right.
    meta/right/below are HTML; company and role are escaped here."""
    co = (f'<b class="co" title="{e(company)}">{e(company)}</b>' if company
          else '<b class="co muted">No matching job</b>')
    role_html = f'<span class="rrole">{e(role)}</span>' if role else ""
    meta_html = f'<div class="rmeta">{meta}</div>' if meta else ""
    right_html = f'<div class="rright">{right}</div>' if right else ""
    return (f'<div class="row"{attrs}><span class="av" aria-hidden="true">{e(initials(company))}</span>'
            f'<div class="rmain"><div class="rtitle">{co}{role_html}</div>{meta_html}{below}</div>{right_html}</div>')


def stat(value, name: str, note: str = "") -> str:
    note_html = f'<div class="note">{e(note)}</div>' if note else ""
    return f'<div class="stat"><div class="v">{value}</div><div class="l">{e(name)}</div>{note_html}</div>'


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
        chips.append(f'<span class="chip {"ok" if s.ok else "bad"}" title="{title}"><i class="dot"></i>{e(label)}: {state}</span>')
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
    gm = srcs.get("gmail") is not None and srcs["gmail"].ok
    db = srcs.get("aria_db") is not None and srcs["aria_db"].ok
    found_note = f"{w['found_undated']} jobs have no found date" if w["found_undated"] else ""
    return ('<div class="stats">'
            + stat(w["found"] if srcs.get("sheet") and srcs["sheet"].ok else "—", "jobs found", found_note)
            + stat(w["applied"] if db else "—", "applied (by Aria)", "" if db else "aria.db unavailable")
            + stat(w["sent"] if gm else "—", "outreach sent", "" if gm else "Gmail unavailable")
            + stat(w["replies"] if gm else "—", "replies")
            + stat(_pct(w["reply_rate_week"]) if gm else "—", "reply rate this week")
            + stat(_pct(w["reply_rate_all"]) if gm else "—", "reply rate, all time",
                   f"{w['sent_all']} outreach threads" if gm else "")
            + "</div>")


def _cost(mdl: dict) -> str:
    if not mdl["sources"].get("aria_db") or not mdl["sources"]["aria_db"].ok:
        return _no_data(_down(mdl["sources"], "aria_db"))
    c = m.cost(mdl)
    return ('<div class="stats">' + stat(_money(c["total"]), "total LLM spend", f'{c["runs"]} pipeline runs')
            + stat(_money(c["per_applied"]), "per applied job", f'{c["applied"]} applied jobs')
            + stat(_money(c["per_applied_by_aria"]), "per job Aria worked on", f'{c["applied_by_aria"]} jobs with runs')
            + '</div>')


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

# Dark is the default; [data-theme=light] is the toggle (remembered per browser in localStorage).
CSS = """
:root{color-scheme:dark;--bg:#0b0b0c;--panel:#141416;--soft:#1b1b1e;--ink:#ececec;--muted:#8a8a8f;--line:#232326;
--accent:#f0803c;--on-accent:#141414;--ok:#3fb950;--bad:#e5484d;--warn:#f5a524;--gray:#8a8a8f;--input:#0f0f11;
--serif:"Instrument Serif",ui-serif,Georgia,serif;--sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
:root[data-theme=light]{color-scheme:light;--bg:#f6f6f4;--panel:#ffffff;--soft:#f1f1ef;--ink:#17171a;--muted:#67676e;
--line:#e3e3e0;--accent:#d2621a;--on-accent:#ffffff;--ok:#2b8a3e;--bad:#d1343a;--warn:#a86400;--gray:#74747a;--input:#ffffff}
*{box-sizing:border-box}html,body{background:var(--bg)}
body{margin:0;color:var(--ink);font:14px/1.5 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.shell{display:grid;grid-template-columns:220px minmax(0,1fr);min-height:100vh}
.side{position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:2px;padding:20px 12px 16px;
border-right:1px solid var(--line);background:var(--bg)}
.logo{display:flex;align-items:center;gap:10px;padding:0 10px 18px;font:400 23px/1 var(--serif);color:var(--ink)}
.logo .mark{width:10px;height:10px;border-radius:3px;background:var(--accent);flex:none}
nav.tabs{display:flex;flex-direction:column;gap:2px}
nav.tabs a{position:relative;display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;color:var(--muted);
font-weight:500;text-decoration:none}
nav.tabs a:hover{color:var(--ink);background:var(--soft)}
nav.tabs a.on{color:var(--ink);background:var(--soft)}nav.tabs a.on svg{color:var(--accent)}
svg.i{width:16px;height:16px;flex:none;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.nb{margin-left:auto;font:600 11px/1 var(--mono);color:var(--accent);padding:3px 7px;border-radius:999px;
background:color-mix(in srgb,var(--accent) 15%,transparent)}
.sidefoot{margin-top:auto;padding:14px 10px 0;border-top:1px solid var(--line);font-size:12px;color:var(--muted);
display:flex;flex-direction:column;gap:10px}
.srcstate{display:flex;align-items:center;color:var(--muted)}a.srcstate{color:var(--bad)}
i.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);flex:none;margin-right:7px}
a.srcstate i.dot,.chip.bad i.dot{background:var(--bad)}
button.themebtn{justify-content:flex-start;background:none;color:var(--muted);font-size:12px;padding:7px 10px}
main{min-width:0;padding:32px 32px 72px}
.tab{display:none;max-width:860px;margin:0 auto}.tab.wide{max-width:1080px}.tab.on{display:block}
.phead{margin:0 0 22px}.ptitle{font:400 38px/1.1 var(--serif);margin:0;letter-spacing:-.01em}
.psub{margin:6px 0 0;color:var(--muted)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:0 0 16px;min-width:0;
overflow-wrap:anywhere}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin:0 0 16px}.cards>.card{margin:0}
h3{font:500 11px/1.5 var(--mono);text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:0 0 12px}
h3:before{content:"// ";color:var(--accent)}h3 .count{color:var(--accent);font-weight:600}
h3 .hint{text-transform:none;letter-spacing:0;font:400 12.5px/1.5 var(--sans)}h4{font-size:14px;font-weight:600;margin:4px 0}
.count{color:var(--accent);font:600 11px/1 var(--mono)}
.muted{color:var(--muted)}.small{font-size:12.5px}.num{text-align:right;font-variant-numeric:tabular-nums}
.nodata{margin:0}.warn{color:var(--warn);margin:0 0 10px}.drop{color:var(--bad)}.allclear{color:var(--ok);margin:0}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}
th{font:500 10.5px/1.4 var(--mono);text-transform:uppercase;letter-spacing:.08em;color:var(--muted);white-space:nowrap}
tbody tr:last-child td{border-bottom:none}.tablewrap{overflow-x:auto}
#jobs th{cursor:pointer;user-select:none}#jobs th.asc:after{content:" ▲"}#jobs th.desc:after{content:" ▼"}
.bar{width:28%}.bar span{display:block;height:8px;border-radius:4px;min-width:2px;
background:color-mix(in srgb,var(--accent) 60%,transparent)}
button,.btn{display:inline-flex;align-items:center;gap:6px;font:500 13px/1.2 var(--sans);padding:8px 14px;border-radius:8px;
border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer;text-decoration:none}
button:hover,.btn:hover{border-color:var(--muted);text-decoration:none}
button.primary,.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
button.primary:hover,.btn.primary:hover{filter:brightness(1.08)}
td button{padding:5px 10px;font-size:12px}
input,select{font:inherit;font-size:13px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--input);
color:var(--ink)}
input:focus,select:focus{outline:none;border-color:var(--accent)}input[type=checkbox]{padding:0;accent-color:var(--accent)}
details{border-top:1px solid var(--line);padding:8px 0}summary{cursor:pointer;color:var(--muted)}
details.flat{border:none;padding:0;margin:0 0 16px}details.flat>summary{padding:4px 0 10px}
.pill{display:inline-block;font:500 11.5px/1.6 var(--sans);padding:1px 9px;border-radius:999px;white-space:nowrap;
color:var(--gray);background:color-mix(in srgb,var(--gray) 16%,transparent)}
.pill.bad{color:var(--bad);background:color-mix(in srgb,var(--bad) 15%,transparent)}
.pill.warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.pill.ok{color:var(--ok);background:color-mix(in srgb,var(--ok) 15%,transparent)}
.pill.accent{color:var(--accent);background:color-mix(in srgb,var(--accent) 15%,transparent)}
.stage{display:inline-block;font-size:11.5px;padding:1px 9px;border-radius:999px;white-space:nowrap;color:var(--gray);
background:color-mix(in srgb,var(--gray) 16%,transparent)}
.stage.s8,.stage.s9{color:var(--ok);background:color-mix(in srgb,var(--ok) 15%,transparent)}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{display:inline-flex;align-items:center;font-size:12px;padding:3px 10px;
border-radius:999px;border:1px solid var(--line);background:var(--soft)}.chip.bad{color:var(--bad)}
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.stat{border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:var(--soft);min-width:0}
.stat .v{font:400 32px/1.1 var(--serif);font-variant-numeric:tabular-nums}.stat .l{font-size:12.5px;margin-top:4px}
.stat .note{color:var(--muted);font-size:11.5px;margin-top:2px}
.rows{display:flex;flex-direction:column}
.row{display:flex;align-items:center;gap:12px;padding:11px 0;border-top:1px solid var(--line);min-width:0}
.rows>.row:first-child{border-top:none;padding-top:0}
.av{width:32px;height:32px;border-radius:8px;display:grid;place-items:center;flex:none;font:600 11px/1 var(--mono);
color:var(--muted);background:var(--soft);border:1px solid var(--line)}
.rmain{flex:1;min-width:0}.rtitle{display:flex;align-items:baseline;gap:6px;min-width:0}
.co{flex:0 1 auto;min-width:2.5em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}
.rrole{flex:0 0 auto;max-width:65%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rrole:before{content:"· ";color:var(--muted)}
.rmeta{color:var(--muted);font-size:12.5px;margin-top:1px;overflow-wrap:anywhere}
.rright{display:flex;align-items:center;justify-content:flex-end;flex-wrap:wrap;gap:8px;flex:none;max-width:45%;font-size:12.5px}
.fix{color:var(--warn);font-size:12.5px;margin-top:4px}
.lnote{margin-top:8px;padding:8px 10px;border-radius:8px;background:var(--soft);font-size:12.5px}
.more{margin-top:10px;font-size:13px}details.showmore{border:none;padding:0}details.showmore>summary{padding:10px 0 0;font-size:13px}
.hero{padding:26px 28px}.dlabel{font:500 11px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.headline{font:400 42px/1.1 var(--serif);margin:14px 0 8px;letter-spacing:-.01em}.headline .n{color:var(--accent)}
.subline{color:var(--muted);margin:0 0 20px}.subline a{color:var(--muted)}.subline a:hover{color:var(--ink)}
.actions{display:flex;flex-wrap:wrap;gap:8px}
.alertbar{display:flex;gap:10px;align-items:flex-start;padding:12px 16px;margin:0 0 16px;border-radius:12px;font-size:13px;
color:var(--warn);background:color-mix(in srgb,var(--warn) 11%,transparent);border:1px solid color-mix(in srgb,var(--warn) 35%,transparent)}
.alertbar a{color:inherit;text-decoration:underline;margin-left:auto;white-space:nowrap}
.bignum{display:flex;align-items:center;gap:16px;flex-wrap:wrap}.bignum .v{font:400 44px/1 var(--serif);color:var(--accent)}
.bignum .t{flex:1;min-width:160px}
.utabs{display:flex;flex-wrap:wrap;gap:2px 20px;border-bottom:1px solid var(--line);margin:0 0 14px}
button.utab{background:none;border:none;border-bottom:2px solid transparent;border-radius:0;padding:8px 0;margin-bottom:-1px;
color:var(--muted)}
button.utab:hover{color:var(--ink)}button.utab.on{color:var(--ink);border-bottom-color:var(--accent)}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:0 0 12px}
.toolbar input[type=search]{flex:1;min-width:200px;max-width:380px}
.toolbar label{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12.5px}
#q{width:100%;max-width:420px;margin-bottom:10px}
details.unmatched{margin-top:12px}details.unmatched ul{padding-left:18px}
.subsec{scroll-margin-top:24px}
@media (max-width:900px){.shell{grid-template-columns:64px minmax(0,1fr)}.side{padding:16px 8px}
.logo{justify-content:center;padding:0 0 18px}.logo .t,nav.tabs .lbl,.sidefoot .txt{display:none}
nav.tabs a{justify-content:center;padding:10px 0}.nb{position:absolute;top:1px;right:1px;padding:2px 4px;font-size:9px}
.sidefoot{padding:12px 0 0;align-items:center}i.dot{margin:0}button.themebtn{justify-content:center;padding:7px}
main{padding:24px 16px 56px}.ptitle{font-size:32px}.headline{font-size:34px}.bar{display:none}}
@media (max-width:560px){.row{flex-wrap:wrap}.rright{max-width:none;justify-content:flex-start;padding-left:44px}}
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

# Underlined sub-tabs (Outreach → Email): a .utabs[data-panes] bar switches the .upane siblings in its card.
SUBTABS_JS = """
(function(){document.querySelectorAll('.utabs[data-panes]').forEach(function(bar){var box=bar.parentNode;
bar.addEventListener('click',function(ev){var b=ev.target.closest('.utab');if(!b)return;
bar.querySelectorAll('.utab').forEach(function(x){x.classList.toggle('on',x===b)});
box.querySelectorAll('.upane').forEach(function(p){p.hidden=p.dataset.p!==b.dataset.p})})})})();
"""

# Runs in <head> so the saved theme applies before the first paint.
THEME_BOOT = ("try{var t=localStorage.getItem('dash.theme');if(t==='light'||t==='dark')"
              "document.documentElement.setAttribute('data-theme',t)}catch(e){}")

THEME_JS = """
(function(){var b=document.getElementById('themebtn');if(!b)return;var r=document.documentElement;
function cur(){return r.getAttribute('data-theme')||'dark'}
function paint(){var dark=cur()==='dark';b.querySelector('.txt').textContent=dark?'Light theme':'Dark theme';
b.title='Switch to '+(dark?'light':'dark')+' theme'}
b.onclick=function(){var n=cur()==='dark'?'light':'dark';r.setAttribute('data-theme',n);
try{localStorage.setItem('dash.theme',n)}catch(e){}paint()};paint()})();
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600'
         '&family=Instrument+Serif&family=JetBrains+Mono:wght@500;600&display=swap">')

TAB_NAMES = [("today", "Today"), ("applications", "Applications"), ("outreach", "Outreach"), ("answers", "My answers"),
             ("costs", "Costs"), ("system", "System")]
TAB_SUBS = {"today": "what needs you now",
            "applications": "one row per company, roles nested; click a role for its timeline",
            "outreach": "LinkedIn notes to send and email threads waiting on you",
            "answers": "what Aria fills into application forms",
            "costs": "LLM spend, from GateLLM's per-call log + Aria's flight log",
            "system": "the run log, settings and data sources"}
WIDE = {"applications", "costs", "system"}          # tables need more than the 860px column
# The System tab holds what used to be separate tabs; old links (#live …) still land on them.
SYSTEM_PARTS = [("live", "Live run", "current or last pipeline run, from Aria's run log"),
                ("settings", "Settings", "LLM providers and tier models"),
                ("health", "Data health", "sources, unmatched records and gaps in the data")]
ICONS = {
    "today": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2'
             'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "applications": '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M3 13h18"/>',
    "outreach": '<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/>',
    "answers": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M8 9h8M8 13h5"/>',
    "costs": '<path d="M12 2v20M17 6H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "system": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "theme": '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
}


def icon(name: str) -> str:
    return f'<svg class="i" viewBox="0 0 24 24" aria-hidden="true">{ICONS[name]}</svg>'


def page(mdl: dict, tabs: Optional[dict] = None, token: str = "", badges: Optional[dict] = None) -> str:
    """tabs: {key: html} for every TAB_NAMES key but system, plus every SYSTEM_PARTS key.
    badges: {tab key: count} shown in the sidebar (zero/missing = no badge)."""
    from aria.dashboard import tabs as tabmod
    srcs = mdl["sources"]
    tabs = dict(tabs or {})
    badges = badges or {}
    down = [SOURCE_LABELS.get(k, k) for k, v in srcs.items() if v is not None and not v.ok]
    tabs["system"] = "".join(
        f'<section class="card subsec" id="{key}-sub">{label(name, sub)}{tabs.get(key, _no_data("not loaded"))}</section>'
        for key, name, sub in SYSTEM_PARTS)
    nav = "".join(f'<a href="#{k}" title="{e(v)}">{icon(k)}<span class="lbl">{e(v)}</span>'
                  + (f'<span class="nb">{badges[k]}</span>' if badges.get(k) else "") + '</a>' for k, v in TAB_NAMES)
    panes = []
    for key, name in TAB_NAMES:
        content = tabs.get(key, _no_data("not loaded"))
        head = ("" if key == "today" else
                f'<header class="phead"><h1 class="ptitle">{e(name)}</h1><p class="psub">{e(TAB_SUBS[key])}</p></header>')
        panes.append(f'<div class="tab{" wide" if key in WIDE else ""}" id="{key}-tab">{head}{content}</div>')
    if down:
        state = (f'<a class="srcstate" href="#health" title="{e(", ".join(down))}"><i class="dot"></i><span class="txt">'
                 f'{len(down)} source{"s" if len(down) != 1 else ""} down: {e(", ".join(down))}</span></a>')
    else:
        state = '<span class="srcstate" title="All sources loaded"><i class="dot"></i><span class="txt">All sources fine</span></span>'
    foot = (f'<div class="sidefoot"><span class="txt" title="Gmail and the sheet refresh every 5 minutes.">'
            f'Loaded {mdl["now"]:%b %d, %H:%M}</span>{state}'
            f'<button id="themebtn" class="themebtn" type="button">{icon("theme")}<span class="txt">Light theme</span></button></div>')
    meta = f'<meta name="dashboard-token" content="{e(token)}">' if token else ""
    from aria.dashboard import answers as ans, applications as apps
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="color-scheme" content="dark light">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">{meta}<title>Job Pipeline</title>'
            f'<script>{THEME_BOOT}</script>{FONTS}<style>{CSS}{tabmod.TABS_CSS}{ans.CSS}</style></head><body>'
            f'<div class="shell"><aside class="side"><div class="logo"><span class="mark"></span><span class="t">Job pipeline</span></div>'
            f'<nav class="tabs">{nav}</nav>{foot}</aside><main>{"".join(panes)}</main></div>'
            f'<script>{tabmod.TABS_JS}{JS}{SUBTABS_JS}{THEME_JS}{tabmod.LIVE_JS}{tabmod.SETTINGS_JS}{ans.JS}{apps.JS}'
            '</script></body></html>')


def today_extras(mdl: dict) -> str:
    """This week's numbers, under Today's to-do list."""
    return _section("This week", lambda: _week(mdl), f"since {m.week_start(mdl['now']):%a %b %d}", "week")


def funnel_details(mdl: dict) -> str:
    return ('<details class="flat"><summary>Funnel: how far each job got</summary>'
            + _section("Funnel", lambda: _funnel(mdl), "jobs that reached each stage; a job sits at its furthest stage with evidence", "funnel")
            + "</details>")


def all_jobs_table(mdl: dict) -> str:
    return ('<details class="flat"><summary>Flat table of all jobs (searchable, sortable)</summary>'
            + _section("All jobs", lambda: _jobs_table(mdl), f"{len(mdl['jobs'])} jobs", "jobs-section") + "</details>")


def pipeline_cost(mdl: dict) -> str:
    return _section("Pipeline runs (aria.db)", lambda: _cost(mdl), "LLM spend recorded per application run", "cost")
