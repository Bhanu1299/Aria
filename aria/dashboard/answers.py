"""dashboard/answers.py — Tab 2 "My answers": what Aria fills into forms, in your words.

Top to bottom:
  Needs your answer  one card per question still open in your reviews, deduped by meaning
                     (same bank field, or the same words), listing which jobs asked; yes/no
                     buttons for yes/no questions; one Save for every card. "only for <job>"
                     keeps an answer to that job. Unlabeled fields show only when Aria captured
                     their nearby text and a screenshot crop — otherwise there's nothing to go on.
  Check these        the LLM's guesses in your reviews (not from your bank): Keep / Change
  Your info          grouped, read-only; Edit per group (inputs + "always ask me"). Unset
                     fields first, in red.
  History            collapsed

Reads: the answer bank (seed file + aria.db learned answers + identity.json fallback) and
the review files. Writes: ONLY through aria.features.answer_bank (each change logged for undo).
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from aria.features import answer_bank as ab
from aria.features.applicator import is_unlabeled
from aria.dashboard.render import _no_data, e

GROUPS = OrderedDict([
    ("Identity", ["first_name", "last_name", "full_name", "legal_name", "preferred_name", "eeo.pronouns"]),
    ("Contact", ["email", "phone", "location", "address", "address_line1", "city", "state", "zip", "country"]),
    ("Education", ["school", "degree", "major", "gpa", "edu_start_month", "edu_start_year", "grad_date"]),
    ("Links", ["linkedin", "github", "website"]),
    ("Work preferences", ["work_authorization", "sponsorship", "relocation", "start_date", "years_experience",
                          "current_company", "hybrid_onsite_ok", "conflict_of_interest", "outside_business",
                          "referral_source"]),
    ("Demographic (optional)", ["eeo.gender", "eeo.race", "eeo.hispanic_latino", "eeo.veteran", "eeo.disability",
                                "eeo.transgender", "eeo.lgbtq"]),
])
# Shown for reference, never asked about: derived from other fields, or filled from the sheet.
_DERIVED = {"grad_month", "grad_year", "salary_rule"}
_OPTIONAL = {"gpa", "website", "address", "eeo.pronouns", "eeo.transgender", "eeo.lgbtq", "referral_source"}
# Needs-me notes that aren't questions the bank can answer (uploads, consents).
_NOT_A_QUESTION = re.compile(r"(?i)file upload|consent|checkbox|agreement|resume_path")


def _file_value(data: dict, key: str):
    node = data.get("answers", {})
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return None if isinstance(node, dict) else node


def _job(rv: dict) -> dict:
    job = {"company": rv.get("company", ""), "role": rv.get("role", "")}
    return {"company": job["company"] or "?", "role": job["role"], "row": rv.get("row"),
            "job_key": ab.job_key(job)}


def _add_job(card: dict, job: dict) -> None:
    if job["job_key"] not in [j["job_key"] for j in card["jobs"]]:
        card["jobs"].append(job)


# ---------------------------------------------------------------------------
# Needs your answer
# ---------------------------------------------------------------------------

def cards(bank: dict, reviews: dict) -> tuple:
    """(cards, skipped). A card: {id, key, title, questions, jobs, yes_no, unlabeled, nearby, crop}.
    Questions the bank (or an always-ask rule) already covers are left out."""
    out: "OrderedDict[str, dict]" = OrderedDict()
    skipped = 0
    for _path, rv in sorted(reviews.items(), key=lambda kv: -kv[1].get("mtime", 0)):
        job = _job(rv)
        captures = {c.get("name") or c.get("label"): c for c in (rv.get("unlabeled") or [])}
        for row in rv.get("sections", {}).get("needs_me", []):
            field = row["field"]
            if _NOT_A_QUESTION.search(row.get("note", "")):
                skipped += 1
                continue
            if is_unlabeled(field):
                cap = captures.get(field) or {}
                crop = Path(cap.get("crop") or "")
                if not (cap.get("nearby") and cap.get("crop") and crop.is_file()):
                    continue                          # nothing to tell you what the field is
                card = out.setdefault(f"u:{job['job_key']}:{field}", {
                    "key": "", "title": cap["nearby"][:160], "questions": [field], "jobs": [],
                    "yes_no": False, "unlabeled": True, "nearby": cap["nearby"][:240], "crop": crop.name})
                _add_job(card, job)
                continue
            q = ab.normalize(field)
            if not q or ab._lookup(bank, field, None, {**job, "source": ""}) or ab._always_ask(bank, q):
                continue
            key = ab.match_field(bank, q).key or ""
            if key in _DERIVED or key in ab.JOB_KEYS:      # computed, or from the sheet's source
                continue
            title = ab.key_title(key) if key and len(q.split()) <= 4 else field   # "Start date year" is ambiguous
            card = out.setdefault(f"k:{key}" if key else f"q:{q}", {
                "key": key, "title": title, "questions": [], "jobs": [], "yes_no": False,
                "unlabeled": False, "nearby": "", "crop": ""})
            if field not in card["questions"]:
                card["questions"].append(field)
            card["yes_no"] = card["yes_no"] or key in ab.YES_NO_KEYS or bool(ab._YES_NO_QUESTION.match(q))
            _add_job(card, job)
    result = list(out.values())
    for i, c in enumerate(result):
        c["id"] = i
    return result, skipped


def open_count(reviews_src) -> Optional[int]:
    """How many "Needs your answer" cards there are (Today and the sidebar badge); None if the bank is unreadable."""
    reviews = reviews_src.data if reviews_src is not None and reviews_src.ok else {}
    try:
        return len(cards(ab.load(), reviews)[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Check these (LLM guesses)
# ---------------------------------------------------------------------------

def check_these(bank: dict, reviews: dict) -> list:
    """LLM guesses in your reviews that your bank still doesn't answer."""
    out, seen = [], set()
    for rv in sorted(reviews.values(), key=lambda r: -r.get("mtime", 0)):
        job = _job(rv)
        for row in rv.get("sections", {}).get("llm_guess", []):
            mark = (ab.normalize(row["field"]), job["job_key"])
            if mark in seen or ab._lookup(bank, row["field"], None, job):
                continue
            seen.add(mark)
            out.append({"field": row["field"], "value": row["value"], **job})
    return out


# ---------------------------------------------------------------------------
# Your info
# ---------------------------------------------------------------------------

def info_groups(bank: dict, data: dict) -> list:
    """[{group, rows: [{key, label, value, always_ask, optional}]}], unset fields first."""
    ident = ab._identity_answers()
    always = set(bank.get("always_ask") or [])
    known = list(ab.DEFAULT_ALIASES) + [k for k in ab._answer_keys(data.get("answers", {}))
                                        if k not in ab.DEFAULT_ALIASES]
    grouped = {k for keys in GROUPS.values() for k in keys}
    other = [k for k in known if k not in grouped and k not in _DERIVED]
    out = []
    for group, keys in list(GROUPS.items()) + [("Other", other)]:
        rows = []
        for key in dict.fromkeys(keys):
            own = _file_value(data, key)
            own_s = "" if own is None else str(own).strip()
            if own_s and ab._PLACEHOLDER.match(own_s):
                own_s = ""
            value = own_s or ("" if "." in key else ident.get(key, ""))
            if not value and key in ("major",):
                value = ab._value(bank, key) or ""
            label = ab.key_title(key)
            if key == "resume_path" and value:
                value = Path(value).name
            rows.append({"key": key, "label": label, "value": value, "always_ask": key in always,
                         "optional": key in _OPTIONAL})
        rows.sort(key=lambda r: (bool(r["value"]) or r["optional"], bool(r["value"])))
        if rows:
            out.append({"group": group, "rows": rows})
    return out


# ---------------------------------------------------------------------------
# actions (POST /api/answers) — the only writes on this tab
# ---------------------------------------------------------------------------

class AnswerError(Exception):
    """Bad input; str(exc) is safe to show."""


def _answer_card(card: dict) -> list:
    value = str(card.get("value") or "").strip()
    questions = [str(q) for q in card.get("questions") or [] if str(q).strip()]
    key = str(card.get("key") or "")
    if not value or not questions:
        return []
    if card.get("scope") == "job":
        job = str(card.get("job") or "")
        return [ab.set_job_answer(job, q, value) for q in questions]
    key = key or ab.matched_key(questions[0]) or ""
    if key and key != "salary_rule":
        ids = [ab.set_answer(key, value)]
        ids += [cid for cid in (ab.add_alias(key, q) for q in questions) if cid]
        return ids
    return [ab.set_learned(q, value) for q in questions]


def act(body: dict) -> dict:
    op = str(body.get("op") or "")
    try:
        if op == "answer_cards":
            ids = [cid for card in body.get("cards") or [] for cid in _answer_card(card)]
            if not ids:
                raise AnswerError("Type or pick an answer first.")
            return {"ok": True, "changes": ids}
        if op == "set_group":
            ids = [ab.set_answer(k, v) for k, v in (body.get("values") or {}).items()]
            ids += [ab.set_always_ask(k, bool(on)) for k, on in (body.get("always_ask") or {}).items()]
            if "salary_floor" in body:
                ids.append(ab.set_salary_floor(int(body["salary_floor"] or 0)))
            return {"ok": True, "changes": ids}
        if op == "set_learned":
            cid = ab.set_learned(str(body.get("question") or ""), body.get("value"))
        elif op == "set_job":
            cid = ab.set_job_answer(str(body.get("job") or ""), str(body.get("question") or ""), body.get("value"))
        elif op == "keep_guess":
            ids = _answer_card({"questions": [body.get("question")], "value": body.get("value"), "scope": "all"})
            if not ids:
                raise AnswerError("Type an answer first.")
            return {"ok": True, "changes": ids}
        elif op == "undo":
            cid = ab.undo(int(body.get("id") or 0))
        else:
            raise AnswerError(f"unknown action {op!r}")
    except (ValueError, TypeError) as exc:
        raise AnswerError(str(exc)) from exc
    return {"ok": True, "change": cid}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _card_html(c: dict) -> str:
    jobs = " · ".join(e(j["company"]) for j in c["jobs"])
    also = (f'<div class="muted small">Also asked as: {e(" / ".join(c["questions"][1:]))}</div>'
            if len(c["questions"]) > 1 else "")
    if c["unlabeled"]:
        head = (f'<div class="q">A field with no label on {jobs}</div>'
                f'<div class="small">Next to it: “{e(c["nearby"])}”</div>'
                f'<img class="crop" src="/review-img?f={e(c["crop"])}" alt="the field">')
    else:
        head = f'<div class="q">{e(c["title"])}</div>{also}<div class="muted small">Asked by {jobs}</div>'
    if c["yes_no"]:
        field = '<span class="yn"><button type="button" data-v="Yes">Yes</button><button type="button" data-v="No">No</button></span>'
    else:
        field = '<input class="ans" placeholder="Your answer" spellcheck="false">'
    if c["unlabeled"]:
        scope = '<span class="muted small">for this job only</span>'
    else:
        scope = " ".join(f'<a href="#" class="onlyjob" data-job="{e(j["job_key"])}">only for {e(j["company"])}</a>'
                         for j in c["jobs"])
    fixed_job = c["jobs"][0]["job_key"] if c["unlabeled"] else ""
    return (f'<div class="card" data-key="{e(c["key"])}" data-qs="{e(json.dumps(c["questions"]))}"'
            f' data-fixedjob="{e(fixed_job)}">{head}<div class="cardrow">{field}<span class="scope small">{scope}</span>'
            '</div></div>')


def _group_html(g: dict, floor: Optional[int], ask_salary: bool) -> str:
    rows = g["rows"]
    view = []
    for r in rows:
        if r["value"]:
            view.append(f'<div class="kv"><span class="k">{e(r["label"])}</span><span class="v">{e(r["value"])}</span></div>')
        elif not r["optional"]:
            view.append(f'<div class="kv unset"><span class="k">{e(r["label"])}</span><span class="v">not set</span></div>')
    if floor is not None:
        view.append(f'<div class="kv"><span class="k">Expected salary</span><span class="v">the posting’s low end, '
                    f'never below ${floor:,}</span></div>')
    edit = "".join(
        f'<tr data-key="{e(r["key"])}"><td>{e(r["label"])}</td><td><input class="ans" value="{e(r["value"])}" spellcheck="false"'
        f'{" disabled" if r["key"] == "resume_path" else ""}></td><td><label class="small"><input type="checkbox" class="ask"'
        f'{" checked" if r["always_ask"] else ""}> always ask me</label></td></tr>' for r in rows)
    if floor is not None:
        edit += (f'<tr data-floor="1"><td>Expected salary floor</td><td><input class="ans" value="{floor}" inputmode="numeric"></td>'
                 f'<td><label class="small"><input type="checkbox" class="ask" data-key="salary_rule"'
                 f'{" checked" if ask_salary else ""}> always ask me</label></td></tr>')
    return (f'<section class="infogroup"><div class="ghead"><h4>{e(g["group"])}</h4>'
            f'<button class="editgroup">Edit</button></div><div class="view">{"".join(view) or "<span class=muted>Nothing here yet.</span>"}</div>'
            f'<div class="edit" hidden><table class="answers"><tbody>{edit}</tbody></table>'
            '<button class="savegroup primary">Save</button> <button class="cancelgroup">Cancel</button>'
            ' <span class="result small"></span></div><!--/edit--></section>')


def render(reviews_src) -> str:
    try:
        bank = ab.load()
        data = ab._read_file(ab.BANK_PATH)
    except Exception as exc:
        return _no_data(f"answer bank unreadable: {exc}")
    reviews = reviews_src.data if reviews_src is not None and reviews_src.ok else {}
    out = []

    open_cards, skipped = cards(bank, reviews)
    out.append(f'<h3 class="sub3">Needs your answer <span class="count">{len(open_cards)}</span></h3>')
    if open_cards:
        out.append(f'<div class="cards">{"".join(_card_html(c) for c in open_cards)}</div>'
                   '<div class="saveall"><button id="saveall" class="primary">Save answers</button> <span id="saveresult" class="result small"></span>'
                   '<span class="muted small"> Answers apply to every form unless you pick “only for”.</span></div>')
    else:
        out.append('<p class="muted">Nothing is waiting on you.</p>')
    if skipped:
        out.append(f'<p class="muted small">{skipped} file uploads and consent boxes are left out: those stay yours on each form.</p>')

    guesses = check_these(bank, reviews)
    if guesses:
        trs = "".join(
            f'<tr data-q="{e(g["field"])}"><td>{e(g["field"])}<div class="muted small">{e(g["company"])}</div></td>'
            f'<td><span class="guess">{e(g["value"][:300])}</span><input class="ans" value="{e(g["value"][:300])}" hidden></td>'
            '<td><button class="keepguess">Keep</button> <button class="changeguess">Change</button>'
            ' <span class="result small"></span></td></tr>' for g in guesses[:40])
        out.append(f'<h3 class="sub3">Check these <span class="count">{len(guesses)}</span></h3>'
                   '<p class="muted small">The LLM filled these, not your answers. Keep one to make it yours.</p>'
                   f'<div class="card"><div class="tablewrap"><table class="answers"><tbody>{trs}</tbody></table></div></div>')

    floor = int(bank.get("salary_rule", {}).get("floor") or ab.DEFAULT_SALARY_FLOOR)
    ask_salary = "salary_rule" in (bank.get("always_ask") or [])
    groups = info_groups(bank, data)
    learned = bank.get("learned") or {}
    jobs = bank.get("job_answers") or {}
    out.append('<h3 class="sub3">Your info</h3><div class="infogroups">'
               + "".join(_group_html(g, floor if g["group"] == "Work preferences" else None, ask_salary) for g in groups)
               + "</div>")
    if learned or jobs:
        items = "".join(f'<div class="kv" data-learned="{e(q)}"><span class="k">{e(q)}</span><span class="v">{e(str(v))}</span>'
                        f' <a href="#" class="forget small">forget</a></div>' for q, v in sorted(learned.items()))
        items += "".join(f'<div class="kv" data-jobq="{e(q)}" data-job="{e(jk)}"><span class="k">{e(q)}'
                         f' <span class="muted small">({e(jk)} only)</span></span><span class="v">{e(str(v))}</span>'
                         f' <a href="#" class="forgetjob small">forget</a></div>'
                         for jk, qa in jobs.items() for q, v in qa.items())
        out.append(f'<section class="infogroup"><div class="ghead"><h4>Other answers you gave</h4></div>'
                   f'<div class="view">{items}</div></section>')

    hist = ab.history(30)
    trs = "".join(
        f'<tr><td class="small">{e(h["ts"])}</td><td>{e(ab.key_title(h["key"]) if h["target"] == "answer" else h["key"])}'
        f'{(" · " + e(h["job"])) if h["job"] and h["target"] != "alias" else ""}</td>'
        f'<td class="small">{e(_short(h["old"]))} → {e(_short(h["new"]))}</td><td>'
        + ('<span class="muted small">undone</span>' if h["undone_by"] else f'<button class="undo" data-id="{h["id"]}">Undo</button>')
        + '</td></tr>' for h in hist)
    out.append(f'<details class="history"><summary>History ({len(hist)})</summary>'
               + (f'<div class="tablewrap"><table><tbody>{trs}</tbody></table></div>' if hist else
                  '<p class="muted">No changes yet.</p>') + '</details>')
    return "".join(out)


def _short(raw: Optional[str]) -> str:
    if raw is None:
        return "(unset)"
    return raw if len(raw) <= 40 else raw[:37] + "…"


CSS = r"""
.card .q{font-weight:600;margin-bottom:2px}.cardrow{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
.cardrow input.ans{flex:1;min-width:160px}.yn{display:inline-flex;gap:6px}
.yn button.on{border-color:var(--accent);color:var(--accent);background:color-mix(in srgb,var(--accent) 12%,transparent)}
.scope a{color:var(--muted);margin-right:8px}.scope a.on{color:var(--accent);font-weight:600}
.saveall{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 22px}
.infogroups{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;margin:0 0 22px}
.infogroup{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;min-width:0}
.ghead{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}.ghead button{padding:4px 10px;font-size:12px}
.kv{display:grid;grid-template-columns:minmax(110px,40%) minmax(0,1fr) auto;gap:12px;padding:7px 0;font-size:13px;
border-top:1px solid var(--line);overflow-wrap:anywhere}.view .kv:first-child{border-top:none}
.kv .k{color:var(--muted)}.kv.unset .k,.kv.unset .v{color:var(--bad);font-weight:600}
.infogroup.editing{grid-column:1/-1}.infogroup .edit label{white-space:nowrap}.infogroup .edit td:first-child{width:30%}
.infogroup .edit button{margin-top:10px}
details.history{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px}
details.history>summary{font:500 11px/1.5 var(--mono);text-transform:uppercase;letter-spacing:.09em}
"""

JS = r"""
(function(){var tok=document.querySelector('meta[name=dashboard-token]');if(!tok||!document.getElementById('answers-tab'))return;tok=tok.content;
function post(body){return fetch('/api/answers',{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':tok},
body:JSON.stringify(body)}).then(function(r){return r.json().then(function(d){d._ok=r.ok;return d})})}
function msg(el,text,ok){if(el){el.textContent=text;el.className='result small '+(ok?'ok':'bad')}}
var tab=document.getElementById('answers-tab');
tab.addEventListener('click',function(ev){var t=ev.target;
if(t.matches('.yn button')){t.parentNode.querySelectorAll('button').forEach(function(b){b.classList.toggle('on',b===t)});return}
if(t.matches('a.onlyjob')){ev.preventDefault();var was=t.classList.contains('on');t.parentNode.querySelectorAll('a').forEach(function(a){a.classList.remove('on')});
if(!was)t.classList.add('on');return}
if(t.id==='saveall'){var cards=[];tab.querySelectorAll('.card').forEach(function(c){var on=c.querySelector('.yn button.on');var inp=c.querySelector('input.ans');
var v=on?on.dataset.v:(inp?inp.value.trim():'');if(!v)return;var only=c.querySelector('a.onlyjob.on');var job=c.dataset.fixedjob||(only?only.dataset.job:'');
cards.push({key:c.dataset.key,questions:JSON.parse(c.dataset.qs),value:v,scope:job?'job':'all',job:job})});
post({op:'answer_cards',cards:cards}).then(function(d){msg(document.getElementById('saveresult'),d._ok?('Saved '+cards.length+' — used on the next fill. Reload to tidy the list.'):(d.error||'failed'),d._ok)});return}
var g=t.closest('.infogroup');
if(t.classList.contains('editgroup')){g.classList.add('editing');g.querySelector('.view').hidden=true;g.querySelector('.edit').hidden=false;t.hidden=true;return}
if(t.classList.contains('cancelgroup')){g.classList.remove('editing');g.querySelector('.view').hidden=false;g.querySelector('.edit').hidden=true;g.querySelector('.editgroup').hidden=false;return}
if(t.classList.contains('savegroup')){var values={},ask={},body={op:'set_group'};g.querySelectorAll('tr').forEach(function(r){var i=r.querySelector('input.ans');var c=r.querySelector('input.ask');
if(r.dataset.floor){body.salary_floor=i.value;if(c)ask.salary_rule=c.checked;return}if(!i||i.disabled)return;
if(i.value!==i.defaultValue)values[r.dataset.key]=i.value;if(c&&c.checked!==c.defaultChecked)ask[r.dataset.key]=c.checked});
body.values=values;body.always_ask=ask;post(body).then(function(d){msg(g.querySelector('.result'),d._ok?'Saved — reload to see it.':(d.error||'failed'),d._ok)});return}
var row=t.closest('tr');
if(t.classList.contains('keepguess')){var inp=row.querySelector('input.ans');post({op:'keep_guess',question:row.dataset.q,value:inp.hidden?row.querySelector('.guess').textContent:inp.value})
.then(function(d){msg(row.querySelector('.result'),d._ok?'Saved as your answer.':(d.error||'failed'),d._ok)});return}
if(t.classList.contains('changeguess')){row.querySelector('.guess').hidden=true;var i2=row.querySelector('input.ans');i2.hidden=false;i2.focus();t.hidden=true;
row.querySelector('.keepguess').textContent='Save';return}
if(t.matches('a.forget')){ev.preventDefault();var k=t.closest('.kv');post({op:'set_learned',question:k.dataset.learned,value:''}).then(function(d){if(d._ok)k.remove()});return}
if(t.matches('a.forgetjob')){ev.preventDefault();var k2=t.closest('.kv');post({op:'set_job',job:k2.dataset.job,question:k2.dataset.jobq,value:''}).then(function(d){if(d._ok)k2.remove()});return}
if(t.classList.contains('undo')){post({op:'undo',id:parseInt(t.dataset.id,10)}).then(function(d){if(d._ok)t.replaceWith(document.createTextNode('undone — reload to see'))})}})})();
"""
