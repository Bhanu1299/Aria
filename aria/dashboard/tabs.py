"""dashboard/tabs.py — the Live run, LLM costs and Settings tabs (HTML + their JS)."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from aria.dashboard.render import _date, _money, _no_data, e


def _pct(part: float, whole: float) -> str:
    return f"{part / whole:.0%}" if whole else "—"


def _ts(ts: Optional[float]) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""


# ---------------------------------------------------------------------------
# Live run — rendered client-side from /api/runs (polled every 2 s)
# ---------------------------------------------------------------------------

def live_tab(runs: list, error: str = "") -> str:
    if error:
        return _no_data(error)
    data = json.dumps(runs).replace("</", "<\\/")
    return ('<div class="live"><div class="runlist" id="runlist"></div><div class="rundetail" id="rundetail"></div></div>'
            '<p class="muted small" id="livestatus">Updates every 2 s from ~/.aria/run_log.jsonl. Read-only.</p>'
            f'<script id="runs-data" type="application/json">{data}</script>')


LIVE_JS = r"""
(function(){var list=document.getElementById('runlist');if(!list)return;
var detail=document.getElementById('rundetail'),status=document.getElementById('livestatus');
var runs=JSON.parse(document.getElementById('runs-data').textContent),sel=null;
function el(t,c,txt){var n=document.createElement(t);if(c)n.className=c;if(txt!=null)n.textContent=txt;return n}
function fmt(ts){if(!ts)return'';var d=new Date(ts*1000);return d.toLocaleString()}
function dur(ms){if(ms==null)return'';return ms<1000?ms+' ms':(ms/1000).toFixed(1)+' s'}
function title(r){var j=r.job;return j?((j.company||'?')+' — '+(j.role||'?')+(j.row!=null?' · row '+j.row:'')):(r.trigger||'run')}
function draw(){list.textContent='';if(!runs.length){list.appendChild(el('p','muted','No pipeline runs yet. Say "apply to the next job from Muse".'));detail.textContent='';return}
if(!sel||!runs.some(function(r){return r.id===sel}))sel=runs[0].id;
runs.forEach(function(r){var b=el('button','run'+(r.id===sel?' on':''));b.appendChild(el('div','rt',title(r)));
b.appendChild(el('div','muted small',r.trigger+' · '+fmt(r.started)));var s=el('span','badge st-'+r.status.replace(/\s/g,'-'),r.status);b.appendChild(s);
b.onclick=function(){sel=r.id;draw()};list.appendChild(b)});
var r=runs.filter(function(x){return x.id===sel})[0];detail.textContent='';
detail.appendChild(el('h3',null,title(r)));
var meta=el('p','muted small',r.trigger+' · started '+fmt(r.started)+(r.ended?' · took '+dur((r.ended-r.started)*1000):' · in progress'));detail.appendChild(meta);
if(r.summary)detail.appendChild(el('p',null,r.summary));
var t=el('table','steps');var h=el('tr');['Step','Status','Time','Detail'].forEach(function(x){h.appendChild(el('th',null,x))});t.appendChild(h);
r.steps.forEach(function(s){var tr=el('tr','st-'+s.status.replace(/\s/g,'-'));tr.appendChild(el('td',null,s.step));
tr.appendChild(el('td',null,s.status));tr.appendChild(el('td','num',dur(s.ms)));tr.appendChild(el('td','small',s.detail||''));t.appendChild(tr)});
detail.appendChild(t);
if(r.decisions.length){detail.appendChild(el('h3',null,'Why this job'));var u=el('ul','decisions');
r.decisions.slice().reverse().forEach(function(d){var li=el('li',d.picked?'picked':'');li.appendChild(el('b',null,(d.picked?'Picked: ':'Skipped: ')+(d.company||'?')+' — '+(d.role||'?')+' (row '+d.row+')'));
li.appendChild(el('div','muted small',(d.reasons||[]).join(' · ')));u.appendChild(li)});detail.appendChild(u)}}
draw();
function poll(){if(location.hash!=='#live'){return}fetch('/api/runs',{cache:'no-store'}).then(function(r){return r.json()}).then(function(d){runs=d.runs;draw();
status.textContent='Updated '+new Date().toLocaleTimeString()+' · polling every 2 s · read-only'}).catch(function(){status.textContent='Live updates paused (dashboard server not reachable).'})}
setInterval(poll,2000)})();
"""


# ---------------------------------------------------------------------------
# LLM costs
# ---------------------------------------------------------------------------

def costs_tab(summary: Optional[dict], via: str, error: str = "") -> str:
    if summary is None:
        return _no_data(error or "cost data unavailable")
    s = summary["spend"]
    src = {"endpoint": "GateLLM /v1/calls", "file": "GateLLM call log file (GateLLM not running)",
           "none": "no GateLLM call log yet"}.get(via, via)
    since = f" since {_date(s['first_call'])}" if s["first_call"] else ""
    stat = lambda v, l, n="": (f'<div class="stat"><div class="v">{v}</div><div class="l">{e(l)}</div>'
                               f'{f"<div class=muted small>{e(n)}</div>" if n else ""}</div>')
    out = [f'<p class="muted small">Per-call data: {e(src)}{e(since)}, {s["calls"]} calls ({s["errors"]} failed). '
           f'Older spend from the flight log ({s["history_entries"]} entries, totals only).</p>',
           '<div class="stats">', stat(_money(s["today"]), "today"), stat(_money(s["week"]), "this week"),
           stat(_money(s["all"]), "all time"),
           stat(_money(summary["per_applied"]), "per applied job",
                f"{summary['applied_jobs']} applied · {_money(summary['job_total'])} on jobs"), '</div>']

    def job_line(r):
        return f'{e(r["job"])} — {_money(r["total"])}'
    if summary["priciest"]:
        out.append('<div class="cards" style="margin-top:12px">'
                   f'<div class="card"><h3>Most expensive jobs</h3><ul>{"".join(f"<li>{job_line(r)}</li>" for r in summary["priciest"])}</ul></div>'
                   f'<div class="card"><h3>Cheapest jobs</h3><ul>{"".join(f"<li>{job_line(r)}</li>" for r in summary["cheapest"])}</ul></div></div>')

    # per task
    if summary["tasks"]:
        rows = "".join(
            f'<tr><td>{e(t["task"])}</td><td class="num">{t["count"]}</td>'
            f'<td>{e(", ".join(f"{m} ×{n}" for m, n in t["models"].most_common(3)))}</td>'
            f'<td class="num">{_money(t["avg"])}</td><td class="num">{_money(t["cost"])}</td></tr>'
            for t in summary["tasks"])
        out.append('<h3 class="sub3">Per task</h3><div class="tablewrap"><table><thead><tr><th>Task</th>'
                   '<th class="num">Calls</th><th>Models</th><th class="num">Avg cost</th><th class="num">Total</th>'
                   f'</tr></thead><tbody>{rows}</tbody></table></div>')
    else:
        out.append('<h3 class="sub3">Per task</h3><p class="muted">No per-call data yet — it starts with the next LLM call '
                   'through the updated GateLLM.</p>')

    # escalations
    if summary["escalations"]:
        rows = "".join(
            f'<tr><td class="small">{e(_ts(x["ts"]))}</td><td>{e(x["kind"])}</td><td>{e(x["task"] or "")}</td>'
            f'<td>{e(x["model"] or "")}</td><td class="small">{e(x["why"])}</td><td class="num">{_money(x["cost_usd"])}</td></tr>'
            for x in summary["escalations"][:50])
        out.append('<h3 class="sub3">Escalations</h3><div class="tablewrap"><table><thead><tr><th>When</th><th>Kind</th>'
                   '<th>Task</th><th>Served by</th><th>Why</th><th class="num">Cost</th></tr></thead>'
                   f'<tbody>{rows}</tbody></table></div>')
    else:
        out.append('<h3 class="sub3">Escalations</h3><p class="muted">None recorded.</p>')

    # per job
    if summary["jobs"]:
        blocks = []
        for r in summary["jobs"]:
            calls = "".join(
                f'<tr><td class="small">{e(_ts(c["ts"]))}</td><td>{e(c.get("task") or "")}</td><td>{e(c.get("tier") or "")}</td>'
                f'<td>{e(c.get("model") or "")}</td><td class="num">{int(c.get("tokens_in") or 0):,} / {int(c.get("tokens_out") or 0):,}</td>'
                f'<td class="num">{_money(float(c.get("cost_usd") or 0))}</td></tr>' for c in r["calls"])
            steps = "".join(
                f'<tr><td class="small">{e(st["date"])}</td><td>{e(st["stage"])} (flight log)</td><td></td>'
                f'<td class="muted">{st["calls"]} calls</td><td class="num">{st["tokens_in"]:,} / {st["tokens_out"]:,}</td>'
                f'<td class="num">{_money(st["cost_usd"])}</td></tr>' for st in r["steps"])
            n = len(r["calls"]) + len(r["steps"])
            blocks.append(f'<details><summary>{e(r["job"])} — <b>{_money(r["total"])}</b> '
                          f'<span class="muted small">({n} row{"s" if n != 1 else ""})</span></summary>'
                          '<div class="tablewrap"><table><thead><tr><th>When</th><th>Task</th><th>Tier</th><th>Model</th>'
                          f'<th class="num">Tokens in / out</th><th class="num">Cost</th></tr></thead><tbody>{calls}{steps}'
                          '</tbody></table></div></details>')
        out.append('<h3 class="sub3">Per job</h3>' + "".join(blocks))
    return "".join(out)


# ---------------------------------------------------------------------------
# Settings — the only writer
# ---------------------------------------------------------------------------

def settings_tab(providers: Optional[list], tiers: Optional[dict], error: str = "") -> str:
    if providers is None:
        return _no_data(error or "GateLLM settings unavailable")
    rows = []
    for p in providers:
        state = (f'<code>{e(p["masked"])}</code>' if p["has_key"] else '<span class="muted">no key</span>')
        rows.append(
            f'<tr data-provider="{e(p["name"])}"><td><b>{e(p["label"])}</b><div class="muted small">{e(p["env"])}</div></td>'
            f'<td class="keystate">{state}</td>'
            f'<td><input type="password" class="keyin" placeholder="{"Replace key" if p["has_key"] else "Paste API key"}" '
            'autocomplete="off" spellcheck="false"> <button class="save">Save</button>'
            + (' <button class="remove">Remove</button>' if p["has_key"] else "") +
            ' <button class="test">Test</button></td><td class="result small"></td></tr>')
    out = ['<p class="muted small">Keys are written only to GateLLM’s <code>.env</code> (chmod 600), then GateLLM reloads. '
           'The browser only ever sees a masked key. Not supported by GateLLM yet: Google (no adapter).</p>',
           '<div class="tablewrap"><table id="providers"><thead><tr><th>Provider</th><th>Key</th><th></th><th></th></tr>'
           f'</thead><tbody>{"".join(rows)}</tbody></table></div>']
    if tiers:
        models = tiers["models"]
        opts = "".join(f'<option value="{e(m["id"])}">{e(m["provider"])}</option>' for m in models)
        trows = "".join(
            f'<tr data-tier="{e(t)}"><td><b>{e(t)}</b></td><td><input class="tierin" value="{e(", ".join(chain))}" '
            f'list="model-ids" spellcheck="false"><div class="muted small">first model serves; the rest are fallbacks in order</div></td>'
            '<td><button class="savetier">Save</button></td><td class="result small"></td></tr>'
            for t, chain in tiers["tiers"].items())
        out.append(f'<h3 class="sub3">Tier models</h3><datalist id="model-ids">{opts}</datalist>'
                   '<div class="tablewrap"><table id="tiers"><thead><tr><th>Tier</th><th>Models (comma-separated)</th><th></th><th></th>'
                   f'</tr></thead><tbody>{trows}</tbody></table></div>'
                   f'<p class="muted small">Known models: {e(", ".join(m["id"] for m in models))}</p>')
    return "".join(out)


SETTINGS_JS = r"""
(function(){var tok=document.querySelector('meta[name=dashboard-token]');if(!tok)return;tok=tok.content;
function post(url,body){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':tok},
body:JSON.stringify(body)}).then(function(r){return r.json().then(function(d){d._ok=r.ok;return d})})}
function show(row,msg,ok){var c=row.querySelector('.result');c.textContent=msg;c.className='result small '+(ok?'ok':'bad')}
document.querySelectorAll('#providers tr[data-provider]').forEach(function(row){var p=row.dataset.provider,inp=row.querySelector('.keyin');
row.querySelector('.save').onclick=function(){var k=inp.value.trim();if(!k){show(row,'Paste a key first.',false);return}
show(row,'Saving…',true);post('/api/settings/key',{provider:p,key:k}).then(function(d){inp.value='';
if(d._ok){row.querySelector('.keystate').textContent=d.masked;show(row,'Saved. '+d.reload,true)}else show(row,d.error,false)})};
var rm=row.querySelector('.remove');if(rm)rm.onclick=function(){show(row,'Removing…',true);
post('/api/settings/key',{provider:p,key:null}).then(function(d){if(d._ok){row.querySelector('.keystate').textContent='no key';show(row,'Removed. '+d.reload,true)}else show(row,d.error,false)})};
row.querySelector('.test').onclick=function(){show(row,'Testing…',true);post('/api/settings/test',{provider:p}).then(function(d){
show(row,(d.model?d.model+': ':'')+(d.message||d.error),d._ok&&d.ok)})}});
document.querySelectorAll('#tiers tr[data-tier]').forEach(function(row){row.querySelector('.savetier').onclick=function(){
var models=row.querySelector('.tierin').value.split(',').map(function(s){return s.trim()}).filter(Boolean);show(row,'Saving…',true);
post('/api/settings/tier',{tier:row.dataset.tier,models:models}).then(function(d){show(row,d._ok?('Saved. '+d.reload):d.error,d._ok)})}})})();
"""

TABS_CSS = """
nav.tabs{display:flex;gap:4px;margin-top:12px;border-bottom:1px solid var(--line)}
nav.tabs a{padding:7px 14px;text-decoration:none;color:var(--muted);border:1px solid transparent;border-bottom:none;border-radius:6px 6px 0 0}
nav.tabs a.on{color:var(--ink);background:var(--panel);border-color:var(--line);margin-bottom:-1px}
.tab{display:none}.tab.on{display:block}
.live{display:grid;grid-template-columns:minmax(220px,300px) 1fr;gap:14px}
.runlist{display:flex;flex-direction:column;gap:6px}.run{text-align:left;border:1px solid var(--line);background:var(--panel);
color:var(--ink);border-radius:6px;padding:8px 10px;cursor:pointer;font:inherit}.run.on{border-color:var(--accent)}
.rt{font-weight:600;font-size:13px}.badge{display:inline-block;margin-top:4px;font-size:11px;padding:1px 7px;border-radius:9px;background:var(--bar)}
.st-error td:nth-child(2),.badge.st-error{color:var(--bad)}.st-running td:nth-child(2),.badge.st-running{color:var(--accent)}
.st-not-in-this-run td{color:var(--muted)}.steps{margin-bottom:12px}
ul.decisions{padding-left:18px}ul.decisions li{margin:4px 0}ul.decisions li.picked b{color:var(--ok)}
h3.sub3{margin:18px 0 8px}details{border-top:1px solid var(--line);padding:6px 0}summary{cursor:pointer}
#providers input,#tiers input{padding:5px 8px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--ink);width:240px;max-width:100%}
#tiers input{width:100%;min-width:260px}button{font:inherit;font-size:12px;padding:4px 10px;border:1px solid var(--line);
border-radius:5px;background:var(--panel);color:var(--ink);cursor:pointer}button:hover{border-color:var(--accent)}
.result.ok{color:var(--ok)}.result.bad{color:var(--bad)}code{font-size:12px}
@media (max-width:820px){.live{grid-template-columns:1fr}}
"""

TABS_JS = r"""
(function(){function show(){var h=location.hash||'#pipeline';if(!document.querySelector(h+'-tab'))h='#pipeline';
document.querySelectorAll('.tab').forEach(function(t){t.classList.toggle('on','#'+t.id===h+'-tab')});
document.querySelectorAll('nav.tabs a').forEach(function(a){a.classList.toggle('on',a.getAttribute('href')===h)})}
window.addEventListener('hashchange',show);show()})();
"""
