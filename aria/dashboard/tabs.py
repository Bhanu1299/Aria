"""dashboard/tabs.py — the Live run, LLM costs and Settings tabs (HTML + their JS)."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from aria.dashboard.render import _date, _money, _no_data, e, label, stat


def _pct(part: float, whole: float) -> str:
    return f"{part / whole:.0%}" if whole else "—"


def _ts(ts: Optional[float]) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""


# ---------------------------------------------------------------------------
# Live run — rendered client-side from /api/runs (polled every 2 s)
# ---------------------------------------------------------------------------

def live_tab(runs: list, error: str = "", apps: Optional[list] = None) -> str:
    """apps: the open application tabs (aria.features.applying.tabs), shown above the runs."""
    if error:
        return _no_data(error)
    data = json.dumps(runs).replace("</", "<\\/")
    apps_data = json.dumps(apps or []).replace("</", "<\\/")
    return ('<div id="apptabs"></div>'
            '<div class="live"><div class="runlist" id="runlist"></div><div class="rundetail" id="rundetail"></div></div>'
            '<p class="muted small" id="livestatus">Updates every 2 s from ~/.aria/run_log.jsonl and '
            '~/.aria/apply_tabs.json. Read-only.</p>'
            f'<script id="runs-data" type="application/json">{data}</script>'
            f'<script id="apps-data" type="application/json">{apps_data}</script>')


LIVE_JS = r"""
(function(){var list=document.getElementById('runlist');if(!list)return;
var detail=document.getElementById('rundetail'),status=document.getElementById('livestatus');
var runs=JSON.parse(document.getElementById('runs-data').textContent),sel=null;
var apps=JSON.parse(document.getElementById('apps-data').textContent),appbox=document.getElementById('apptabs');
function el(t,c,txt){var n=document.createElement(t);if(c)n.className=c;if(txt!=null)n.textContent=txt;return n}
var TONE={error:'bad',open:'bad',running:'accent',filling:'accent',filled:'warn',skipped:'',ok:'ok',submitted:'ok'};
function pill(s){return el('span','pill '+(TONE[s]||''),s)}
function drawApps(){appbox.textContent='';appbox.appendChild(el('h3',null,'Open application tabs ('+apps.length+')'));
if(!apps.length){appbox.appendChild(el('p','muted','None open. Say "start applications".'));return}
var t=el('table','steps apps');var h=el('tr');['Company','Role','State','Fields needing you'].forEach(function(x){h.appendChild(el('th',null,x))});t.appendChild(h);
apps.forEach(function(a){var tr=el('tr','st-'+a.state);tr.appendChild(el('td',null,a.company||'?'));
tr.appendChild(el('td',null,(a.role||'?')+(a.row!=null?' · row '+a.row:'')));var st=el('td');st.appendChild(pill(a.state));tr.appendChild(st);
var need=(a.needs_me||[]).join(', ')||(a.state==='filled'?'none':'');tr.appendChild(el('td','small',need+(a.note?(need?' · ':'')+a.note:'')));t.appendChild(tr)});
appbox.appendChild(t)}
function fmt(ts){if(!ts)return'';var d=new Date(ts*1000);return d.toLocaleString()}
function dur(ms){if(ms==null)return'';return ms<1000?ms+' ms':(ms/1000).toFixed(1)+' s'}
function title(r){var j=r.job;return j?((j.company||'?')+' — '+(j.role||'?')+(j.row!=null?' · row '+j.row:'')):(r.trigger||'run')}
function draw(){list.textContent='';if(!runs.length){list.appendChild(el('p','muted','No pipeline runs yet. Say "apply to the next job from Muse".'));detail.textContent='';return}
if(!sel||!runs.some(function(r){return r.id===sel}))sel=runs[0].id;
runs.forEach(function(r){var b=el('button','run'+(r.id===sel?' on':''));b.appendChild(el('div','rt',title(r)));
b.appendChild(el('div','muted small',r.trigger+' · '+fmt(r.started)));var s=pill(r.status);s.classList.add('badge');b.appendChild(s);
b.onclick=function(){sel=r.id;draw()};list.appendChild(b)});
var r=runs.filter(function(x){return x.id===sel})[0];detail.textContent='';
detail.appendChild(el('h3',null,title(r)));
var meta=el('p','muted small',r.trigger+' · started '+fmt(r.started)+(r.ended?' · took '+dur((r.ended-r.started)*1000):' · in progress'));detail.appendChild(meta);
if(r.summary)detail.appendChild(el('p',null,r.summary));
var t=el('table','steps');var h=el('tr');['Step','Status','Time','Detail'].forEach(function(x){h.appendChild(el('th',null,x))});t.appendChild(h);
r.steps.forEach(function(s){var tr=el('tr','st-'+s.status.replace(/\s/g,'-'));tr.appendChild(el('td',null,s.step));
var sc=el('td');sc.appendChild(pill(s.status));tr.appendChild(sc);tr.appendChild(el('td','num',dur(s.ms)));tr.appendChild(el('td','small',s.detail||''));t.appendChild(tr)});
detail.appendChild(t);
if(r.decisions.length){detail.appendChild(el('h3',null,'Why this job'));var u=el('ul','decisions');
r.decisions.slice().reverse().forEach(function(d){var li=el('li',d.picked?'picked':'');li.appendChild(el('b',null,(d.picked?'Picked: ':'Skipped: ')+(d.company||'?')+' — '+(d.role||'?')+' (row '+d.row+')'));
li.appendChild(el('div','muted small',(d.reasons||[]).join(' · ')));u.appendChild(li)});detail.appendChild(u)}}
draw();drawApps();
function poll(){if(!document.getElementById('system-tab').classList.contains('on')){return}fetch('/api/runs',{cache:'no-store'}).then(function(r){return r.json()}).then(function(d){runs=d.runs;draw();apps=d.apps||[];drawApps();
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
    out = ['<div class="stats" style="margin-bottom:10px">', stat(_money(s["today"]), "today"),
           stat(_money(s["week"]), "this week"), stat(_money(s["all"]), "all time"),
           stat(_money(summary["per_applied"]), "per applied job",
                f"{summary['applied_jobs']} applied · {_money(summary['job_total'])} on jobs"), '</div>',
           f'<p class="muted small" style="margin:0 0 16px">Per-call data: {e(src)}{e(since)}, {s["calls"]} calls '
           f'({s["errors"]} failed). Older spend from the flight log ({s["history_entries"]} entries, totals only).</p>']

    def job_list(items):
        return '<div class="rows">' + "".join(
            f'<div class="row"><div class="rmain">{e(r["job"])}</div><div class="rright"><b>{_money(r["total"])}</b></div></div>'
            for r in items) + '</div>'
    if summary["priciest"]:
        out.append(f'<div class="cards"><div class="card">{label("Most expensive jobs")}{job_list(summary["priciest"])}</div>'
                   f'<div class="card">{label("Cheapest jobs")}{job_list(summary["cheapest"])}</div></div>')

    # per task
    if summary["tasks"]:
        rows = "".join(
            f'<tr><td>{e(t["task"])}</td><td class="num">{t["count"]}</td>'
            f'<td>{e(", ".join(f"{m} ×{n}" for m, n in t["models"].most_common(3)))}</td>'
            f'<td class="num">{_money(t["avg"])}</td><td class="num">{_money(t["cost"])}</td></tr>'
            for t in summary["tasks"])
        out.append(f'<section class="card">{label("Per task")}<div class="tablewrap"><table><thead><tr><th>Task</th>'
                   '<th class="num">Calls</th><th>Models</th><th class="num">Avg cost</th><th class="num">Total</th>'
                   f'</tr></thead><tbody>{rows}</tbody></table></div></section>')
    else:
        out.append(f'<section class="card">{label("Per task")}<p class="muted">No per-call data yet — it starts with the '
                   'next LLM call through the updated GateLLM.</p></section>')

    # escalations
    if summary["escalations"]:
        rows = "".join(
            f'<tr><td class="small">{e(_ts(x["ts"]))}</td><td>{e(x["kind"])}</td><td>{e(x["task"] or "")}</td>'
            f'<td>{e(x["model"] or "")}</td><td class="small">{e(x["why"])}</td><td class="num">{_money(x["cost_usd"])}</td></tr>'
            for x in summary["escalations"][:50])
        out.append(f'<section class="card">{label("Escalations", count=len(summary["escalations"]))}<div class="tablewrap">'
                   '<table><thead><tr><th>When</th><th>Kind</th><th>Task</th><th>Served by</th><th>Why</th>'
                   f'<th class="num">Cost</th></tr></thead><tbody>{rows}</tbody></table></div></section>')
    else:
        out.append(f'<section class="card">{label("Escalations")}<p class="muted">None recorded.</p></section>')

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
        out.append(f'<section class="card">{label("Per job", count=len(blocks))}' + "".join(blocks) + '</section>')
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
           'The browser only ever sees a masked key. Test makes one tiny call on the provider’s cheapest model.</p>',
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
.live{display:grid;grid-template-columns:minmax(220px,280px) minmax(0,1fr);gap:16px}
.runlist{display:flex;flex-direction:column;gap:6px}button.run{display:block;width:100%;text-align:left;padding:10px 12px}
button.run.on{border-color:var(--accent);box-shadow:inset 2px 0 0 var(--accent)}
.rt{font-weight:600;font-size:13px}.rundetail h3:before{content:none}
.rundetail>h3{font:600 15px/1.4 var(--sans);text-transform:none;letter-spacing:0;color:var(--ink);margin:0 0 2px}
.badge{display:inline-block;margin-top:6px}.st-not-in-this-run td{opacity:.45}.steps{margin-bottom:14px}
#apptabs{margin-bottom:18px}
ul.decisions{padding-left:18px}ul.decisions li{margin:4px 0}ul.decisions li.picked b{color:var(--ok)}
h3.sub3{margin:22px 0 10px}
#providers input,#tiers input{width:240px;max-width:100%}#tiers input{width:100%;min-width:260px}
.result.ok{color:var(--ok)}.result.bad{color:var(--bad)}code{font:12px var(--mono)}
.filters{margin-bottom:12px}
#apps tr.company td{padding-top:16px;border-bottom:none}#apps tr.company .av{display:inline-grid;vertical-align:middle;margin-right:10px}
#apps td{overflow-wrap:break-word}#apps .pill{white-space:normal;line-height:1.45;padding:2px 9px}
#apps td:nth-child(5){max-width:200px;overflow-wrap:anywhere}#apps td:last-child{min-width:150px}#apps tr.role td:first-child{padding-left:54px;min-width:236px}#apps tr.company+tr.role td{border-top:none}
#apps details{border:none;padding:0}#apps summary{list-style:none;display:flex;align-items:center;gap:8px;color:var(--ink);font-weight:500}
#apps summary::-webkit-details-marker{display:none}
#apps summary:before{content:"";width:6px;height:6px;flex:none;border-right:1.5px solid var(--muted);border-bottom:1.5px solid var(--muted);
transform:rotate(-45deg);transition:transform .15s}#apps details[open] summary:before{transform:rotate(45deg)}
ul.timeline{margin:8px 0 4px;padding-left:16px}ul.timeline li{margin:3px 0}
input.ans{width:100%;min-width:180px}
table.answers td:first-child{width:28%}table.answers td:nth-child(2){width:32%}select.scope{max-width:260px;font-size:12px}
img.crop{display:block;max-width:420px;margin-top:6px;border:1px solid var(--line);border-radius:6px}.guess{white-space:pre-wrap}
@media (max-width:820px){.live{grid-template-columns:1fr}#apps tr.role td:first-child{padding-left:12px}}
"""

TABS_JS = r"""
(function(){var OLD={'#live':1,'#settings':1,'#health':1};
function saved(){try{return localStorage.getItem('dash.tab')}catch(e){return null}}
function show(){var h=location.hash||saved()||'#today',sub=null;
if(OLD[h]){sub=document.getElementById(h.slice(1)+'-sub');h='#system'}
if(!document.querySelector(h+'-tab'))h='#today';
try{localStorage.setItem('dash.tab',h)}catch(e){}
document.querySelectorAll('.tab').forEach(function(t){t.classList.toggle('on','#'+t.id===h+'-tab')});
document.querySelectorAll('nav.tabs a').forEach(function(a){a.classList.toggle('on',a.getAttribute('href')===h)});
if(sub)sub.scrollIntoView()}
window.addEventListener('hashchange',show);if(!location.hash&&saved())history.replaceState(null,'',saved());show()})();"""
