"""dashboard/sheet_tab.py — the Muse sheet itself, live and editable.

GET  /api/sheet?tab=jobs|applied   the whole tab, read fresh (the page polls it every POLL_SECS
                                   while the Sheet tab is open, so edits made in Google Sheets show up)
POST /api/sheet/move_applied       every Jobs row already marked Applied → the Applied tab
POST /api/sheet/cell               {tab, row, column, value, expect, company, role} → one cell;
                                   Status → Applied on the Jobs tab moves the row to the Applied tab

A write goes through the Sheets API (sheets.JobSheet.set_cell) — never the browser. It re-finds
the row by its company + role (rows move when the sheet is sorted) and refuses if the cell no
longer holds `expect`, the value the page showed, so an edit made in the sheet meanwhile is never
overwritten blind.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

POLL_SECS = 15
MAX_VALUE = 5000
TAB_KEYS = ("jobs", "applied")
# Status choices besides whatever the column already holds.
STATUSES = ("Not applied", "In review", "Applied", "Not gonna apply", "Skipped", "Needs human")


class SheetEditError(Exception):
    """A bad edit request; str(exc) is shown next to the cell."""


def _tab_name(key: str) -> str:
    import aria.core.config as config
    if key not in TAB_KEYS:
        raise SheetEditError(f"unknown tab {key!r}")
    return config.MUSE_SHEET_TAB if key == "jobs" else config.MUSE_APPLIED_TAB


def data(key: str, sheet=None) -> dict:
    """{"tab", "headers", "rows", "statuses"} — or {"error"} when the sheet can't be read."""
    from aria.integrations import sheets
    try:
        sh = sheet or sheets.tab_sheet(_tab_name(key))
        got = sh.table()
    except (SheetEditError, sheets.SheetError) as exc:
        return {"tab": key, "error": str(exc)}
    except Exception as exc:
        logger.error("dashboard: sheet read failed: %s", type(exc).__name__)
        return {"tab": key, "error": f"couldn't read the sheet ({type(exc).__name__})"}
    statuses = list(STATUSES)
    hdr = [h.strip().lower() for h in got["headers"]]
    if "status" in hdr:
        col = hdr.index("status")
        statuses += sorted({r["cells"][col].strip() for r in got["rows"]} - set(statuses) - {""})
    return {"tab": key, "headers": got["headers"], "rows": got["rows"], "statuses": statuses}


def _text(body: dict, name: str, required: bool = True) -> str:
    v = body.get(name)
    if v is None and not required:
        return ""
    if not isinstance(v, str):
        raise SheetEditError(f"{name} must be text")
    return v


def edit(body: dict, sheet=None, applied_sheet=None) -> dict:
    """Write one cell. Raises SheetEditError (bad request) or sheets.SheetError (sheet said no)."""
    from aria.dashboard import sources
    from aria.integrations import sheets
    row = body.get("row")
    if not isinstance(row, int) or isinstance(row, bool) or row < 2:
        raise SheetEditError("row must be a sheet row number")
    column, value = _text(body, "column"), _text(body, "value")
    if not column.strip():
        raise SheetEditError("column is required")
    if len(value) > MAX_VALUE:
        raise SheetEditError(f"value is longer than {MAX_VALUE} characters")
    expect = body.get("expect")
    if expect is not None and not isinstance(expect, str):
        raise SheetEditError("expect must be text")
    company, role = _text(body, "company", False), _text(body, "role", False)
    key = str(body.get("tab") or "")
    sh = sheet or sheets.tab_sheet(_tab_name(key))
    if key == "jobs" and column.strip().lower() == "status" and value.strip().lower() == "applied":
        if not (company or role):
            raise SheetEditError("this row has no company or role, so it can't be moved safely")
        dest = applied_sheet or sheets.tab_sheet(_tab_name("applied"))
        at = sh.move_row(row, dest, sheets.STATUS_APPLIED, company, role, expect=expect)
        sources.CACHE.drop("sheet")
        return {"ok": True, "moved": "applied", "row": at}
    written = sh.set_cell(row, column, value, expect=expect,
                          company=company if company or role else None, role=role if company or role else None)
    sources.CACHE.drop("sheet")          # the other tabs read the new value on their next load
    return {"ok": True, "row": written}


def move_applied(sheet=None, applied_sheet=None) -> dict:
    """Every Jobs row whose Status is already Applied → the Applied tab, one at a time (each
    re-found by company + role). {"moved": [names], "errors": [messages]}; never raises."""
    from aria.dashboard import sources
    from aria.integrations import sheets
    try:
        sh = sheet or sheets.tab_sheet(_tab_name("jobs"))
        dest = applied_sheet or sheets.tab_sheet(_tab_name("applied"))
        got = sh.table()
    except Exception as exc:
        return {"moved": [], "errors": [str(exc) if isinstance(exc, sheets.SheetError) else type(exc).__name__]}
    hdr = [h.strip().lower() for h in got["headers"]]
    if not {"status", "company", "role"} <= set(hdr):
        return {"moved": [], "errors": ["the Jobs tab needs Company, Role and Status columns"]}
    si, ci, ri = hdr.index("status"), hdr.index("company"), hdr.index("role")
    todo = [r for r in got["rows"] if r["cells"][si].strip().lower() == "applied"]
    moved, errors = [], []
    for r in todo:
        company, role = r["cells"][ci], r["cells"][ri]
        if not (company or role):
            errors.append(f"row {r['row']}: no company or role, left in place")
            continue
        try:
            sh.move_row(r["row"], dest, sheets.STATUS_APPLIED, company, role, expect=r["cells"][si])
            moved.append(company or role)
        except sheets.SheetError as exc:
            errors.append(f"{company or role}: {exc}")
        except Exception as exc:
            logger.error("dashboard: moving %s failed: %s", company, type(exc).__name__)
            errors.append(f"{company or role}: {type(exc).__name__}")
    if moved:
        sources.CACHE.drop("sheet")
    return {"moved": moved, "errors": errors}


def render() -> str:
    """The shell; the table is drawn by JS from GET /api/sheet (also how it refreshes)."""
    return ('<section class="card sheetcard" id="sheet-root" data-poll="' + str(POLL_SECS) + '">'
            '<div class="toolbar"><div class="utabs"><button type="button" class="utab shtab on" data-t="jobs">Jobs</button>'
            '<button type="button" class="utab shtab" data-t="applied">Applied</button></div>'
            '<input id="shq" type="search" placeholder="Filter rows…" autocomplete="off">'
            '<button type="button" class="btn" id="shrefresh">Refresh</button>'
            '<button type="button" class="btn" id="shmove" style="display:none"></button>'
            '<span class="muted small" id="shstate">Loading the sheet…</span></div>'
            '<div class="tablewrap shwrap"><table id="sheet"><thead></thead><tbody></tbody></table></div>'
            '<p class="muted small">Click a cell to edit it; Enter saves, Shift+Enter is a new line, Esc cancels. '
            'Saved straight to Google Sheets. Setting a job to Applied moves it to the Applied tab. '
            f'Changes made in the sheet show up here within {POLL_SECS} seconds, or press Refresh.</p></section>')


CSS = r"""
.tab.full{max-width:none}.sheetcard .toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.shwrap{max-height:calc(100vh - 230px);overflow:auto;border:1px solid var(--line);border-radius:10px;margin-top:10px}
table#sheet{font-size:12.5px;width:max-content;min-width:100%}
table#sheet th{position:sticky;top:0;z-index:2;background:var(--soft);text-align:left;white-space:nowrap}
table#sheet th:first-child,table#sheet td:first-child{position:sticky;left:0;z-index:1;background:var(--panel);
color:var(--muted);font:500 11px/1.4 var(--mono);text-align:right}
table#sheet th:first-child{z-index:3;background:var(--soft)}
table#sheet th:first-child,table#sheet td:first-child{min-width:0;width:1%;padding-left:8px;padding-right:8px;cursor:default}
table#sheet td{vertical-align:top;max-width:320px;min-width:70px;cursor:text}
table#sheet td .cv{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;white-space:pre-wrap;
overflow-wrap:anywhere}
table#sheet td.saving{opacity:.55}table#sheet td.err{box-shadow:inset 0 0 0 1px var(--bad)}
table#sheet td.flash{background:color-mix(in srgb,var(--accent) 14%,transparent)}
table#sheet td textarea{width:100%;min-width:220px;min-height:80px;font:inherit;background:var(--input);color:var(--ink);
border:1px solid var(--accent);border-radius:6px;padding:6px}
table#sheet select{font:inherit;max-width:170px}#shstate.bad{color:var(--bad)}
"""

JS = r"""
(function(){var root=document.getElementById('sheet-root');if(!root)return;
var tokEl=document.querySelector('meta[name=dashboard-token]'),tok=tokEl?tokEl.content:'';
var table=document.getElementById('sheet'),thead=table.tHead,tbody=table.tBodies[0];
var state=document.getElementById('shstate'),q=document.getElementById('shq'),pane=document.getElementById('sheet-tab');
var poll=parseInt(root.dataset.poll,10)*1000,tab='jobs',cur=null,editing=null,busy=false,lastAt=0,note=null;
function say(t,bad){state.textContent=t;state.classList.toggle('bad',!!bad)}
function col(name){return cur?cur.headers.findIndex(function(h){return h.trim().toLowerCase()===name}):-1}
function ident(r){var c=col('company'),ro=col('role');return{company:c>=0?r.cells[c]:'',role:ro>=0?r.cells[ro]:''}}
function isLink(v){return /^https?:\/\/\S+$/i.test(v.trim())}
function cellBody(td,v){td.textContent='';var d=document.createElement('div');d.className='cv';
if(isLink(v)){var a=document.createElement('a');a.href=v.trim();a.target='_blank';a.rel='noopener';
try{a.textContent=new URL(v.trim()).hostname}catch(e){a.textContent=v}d.appendChild(a)}else d.textContent=v;
td.appendChild(d);td.title=v.length>120?v:''}
function draw(prev){var s=col('status'),keep={};
if(prev)prev.rows.forEach(function(r){keep[r.row]=r.cells});
thead.textContent='';var tr=document.createElement('tr'),th=document.createElement('th');th.textContent='#';tr.appendChild(th);
cur.headers.forEach(function(h){var t=document.createElement('th');t.textContent=h;tr.appendChild(t)});thead.appendChild(tr);
tbody.textContent='';cur.rows.forEach(function(r){var tr=document.createElement('tr');tr.dataset.row=r.row;
var n=document.createElement('td');n.textContent=r.row;tr.appendChild(n);
r.cells.forEach(function(v,i){var td=document.createElement('td');td.dataset.c=i;
if(i===s){var sel=document.createElement('select');var opts=cur.statuses.slice();if(opts.indexOf(v)<0)opts.unshift(v);
opts.forEach(function(o){var op=document.createElement('option');op.value=o;op.textContent=o||'(blank)';sel.appendChild(op)});
sel.value=v;sel.dataset.was=v;td.appendChild(sel)}else cellBody(td,v);
if(prev&&keep[r.row]&&keep[r.row][i]!==v){td.classList.add('flash');setTimeout(function(){td.classList.remove('flash')},2500)}
tr.appendChild(td)});tbody.appendChild(tr)});filter();moveBtn()}
function moveBtn(){var b=document.getElementById('shmove'),s=col('status'),n=0;
if(cur&&tab==='jobs'&&s>=0)cur.rows.forEach(function(r){if(r.cells[s].trim().toLowerCase()==='applied')n++});
b.style.display=n?'':'none';b.textContent='Move '+n+' applied job'+(n===1?'':'s')+' to the Applied tab'}
function filter(){var s=q.value.trim().toLowerCase();Array.prototype.forEach.call(tbody.rows,function(r){
r.style.display=!s||r.textContent.toLowerCase().indexOf(s)>=0?'':'none'})}
function load(force){if(busy||editing||(!force&&(document.hidden||!pane.classList.contains('on'))))return;busy=true;
fetch('/api/sheet?tab='+tab,{cache:'no-store'}).then(function(r){return r.json()}).then(function(d){busy=false;
if(d.error){say(d.error,true);return}if(editing)return;var prev=cur&&cur.tab===d.tab?cur:null;cur=d;draw(prev);lastAt=Date.now();
if(note){say(note[0],note[1]);note=null}else say(d.rows.length+' rows · updated just now')}).catch(function(){busy=false;say('Dashboard server unreachable',true)})}
function ago(){if(!lastAt||state.classList.contains('bad')||/^(Moved|Saved)/.test(state.textContent)&&Date.now()-lastAt<15000)return;var s=Math.round((Date.now()-lastAt)/1000);
if(cur)say(cur.rows.length+' rows · updated '+(s<5?'just now':s+'s ago'))}
function rowOf(td){var n=parseInt(td.parentNode.dataset.row,10);return cur.rows.find(function(r){return r.row===n})}
function save(td,value,done){var r=rowOf(td),i=parseInt(td.dataset.c,10),was=r.cells[i];
if(value===was){done(false);return}td.classList.add('saving');td.classList.remove('err');
var id=ident(r);fetch('/api/sheet/cell',{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':tok},
body:JSON.stringify({tab:tab,row:r.row,column:cur.headers[i],value:value,expect:was,company:id.company,role:id.role})})
.then(function(x){return x.json().then(function(d){return{ok:x.ok,d:d}})}).then(function(res){td.classList.remove('saving');
if(res.ok&&res.d.moved){cur.rows=cur.rows.filter(function(x){return x!==r});td.parentNode.remove();done(true);
say('Moved '+(ident(r).company||'it')+' to the Applied tab (row '+res.d.row+')');lastAt=Date.now()}
else if(res.ok){r.cells[i]=value;done(true);say('Saved row '+res.d.row+' · '+cur.headers[i]);lastAt=Date.now()}
else{td.classList.add('err');done(false);say(res.d.error||'Save failed',true);setTimeout(function(){load(true)},1500)}})
.catch(function(){td.classList.remove('saving');td.classList.add('err');done(false);say('Save failed: server unreachable',true)})}
tbody.addEventListener('change',function(ev){var sel=ev.target;if(sel.tagName!=='SELECT')return;var td=sel.parentNode;
save(td,sel.value,function(ok){if(!ok)sel.value=sel.dataset.was;else sel.dataset.was=sel.value})});
tbody.addEventListener('click',function(ev){var td=ev.target.closest('td');
if(!td||ev.target.tagName==='A'||ev.target.tagName==='SELECT'||td.dataset.c===undefined||editing||td.querySelector('select'))return;
var r=rowOf(td),i=parseInt(td.dataset.c,10),ta=document.createElement('textarea');ta.value=r.cells[i];editing=td;
td.textContent='';td.appendChild(ta);ta.focus();var closed=false;
function close(commit){if(closed)return;closed=true;var v=ta.value;
var finish=function(){editing=null;cellBody(td,r.cells[i])};if(commit)save(td,v,finish);else finish()}
ta.addEventListener('keydown',function(k){if(k.key==='Escape'){k.preventDefault();close(false)}
else if(k.key==='Enter'&&!k.shiftKey){k.preventDefault();close(true)}});
ta.addEventListener('blur',function(){close(true)})});
root.querySelectorAll('.shtab').forEach(function(b){b.onclick=function(){if(editing)return;
root.querySelectorAll('.shtab').forEach(function(x){x.classList.toggle('on',x===b)});tab=b.dataset.t;cur=null;say('Loading…');load(true)}});
q.addEventListener('input',filter);
document.getElementById('shmove').onclick=function(){var b=this;if(editing||b.disabled)return;b.disabled=true;
say('Moving… (a couple of seconds per job)');busy=true;
fetch('/api/sheet/move_applied',{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':tok},body:'{}'})
.then(function(x){return x.json()}).then(function(d){busy=false;b.disabled=false;var m=(d.moved||[]).length,e=(d.errors||[]);
note=['Moved '+m+' to the Applied tab'+(e.length?' · '+e.length+' not moved: '+e.join('; '):''),e.length>0];load(true)})
.catch(function(){busy=false;b.disabled=false;say('Move failed: server unreachable',true)})};
document.getElementById('shrefresh').onclick=function(){if(editing)return;say('Refreshing…');load(true)};
window.addEventListener('hashchange',function(){setTimeout(function(){load(false)},0)});
document.addEventListener('visibilitychange',function(){load(false)});
setInterval(function(){load(false)},poll);setInterval(ago,5000);setTimeout(function(){load(false)},0)})();
"""
