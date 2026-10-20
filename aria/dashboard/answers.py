"""dashboard/answers.py — Tab 2 "My answers": everything Aria fills, editable.

Reads: the answer bank (seed file + aria.db learned answers + identity.json fallback),
review files (needs_me questions, LLM guesses, unlabeled-field captures).
Writes: ONLY through aria.features.answer_bank's edit functions (seed file + learned
store in aria.db, each change logged for undo). Nothing else on this tab writes.
"""
from __future__ import annotations

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
    ("Education", ["school", "degree", "grad_date"]),
    ("Links", ["linkedin", "github", "website"]),
    ("Work preferences", ["work_authorization", "sponsorship", "relocation", "start_date", "years_experience",
                          "current_company", "hybrid_onsite_ok", "conflict_of_interest", "outside_business"]),
    ("Demographic (EEO, optional)", ["eeo.gender", "eeo.race", "eeo.hispanic_latino", "eeo.veteran", "eeo.disability",
                                     "eeo.transgender", "eeo.lgbtq"]),
])
# Needs-me notes that aren't questions the bank can answer (uploads, consents).
_NOT_A_QUESTION = re.compile(r"(?i)file upload|consent|checkbox|agreement|resume_path")


def _file_value(data: dict, key: str):
    node = data.get("answers", {})
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return None if isinstance(node, dict) else node


def _label(key: str) -> str:
    return key.split(".")[-1].replace("_", " ").capitalize()


def answer_rows(bank: dict, data: dict) -> "OrderedDict[str, list]":
    """Group → [{key, label, value, source, always_ask}] for every known key."""
    ident = ab._identity_answers()
    always = set(bank.get("always_ask") or [])
    known = list(ab.DEFAULT_ALIASES) + [k for k in ab._answer_keys(data.get("answers", {}))
                                        if k not in ab.DEFAULT_ALIASES]
    placed, out = set(), OrderedDict()
    for group, keys in list(GROUPS.items()) + [("Other", [k for k in known if not any(k in v for v in GROUPS.values())])]:
        rows = []
        for key in keys:
            if key in placed or key == "salary_rule":
                continue
            placed.add(key)
            own = _file_value(data, key)
            own_s = "" if own is None else str(own).strip()
            if own_s and not ab._PLACEHOLDER.match(own_s):
                value, source = own_s, "answer bank"
            elif "." not in key and ident.get(key):
                value, source = ident[key], "identity.json"
            else:
                value, source = "", "not set — Aria asks you"
            rows.append({"key": key, "label": _label(key), "value": value, "source": source,
                         "always_ask": key in always})
        if rows:
            out[group] = rows
    return out


def _job_label(review: dict) -> str:
    row = f" (row {review['row']})" if review.get("row") else ""
    return f"{review.get('company') or '?'} — {review.get('role') or '?'}{row}"


def new_questions(bank: dict, reviews: dict) -> tuple:
    """(questions, unlabeled, skipped_count). Questions: needs_me fields still unanswered,
    deduped by normalized text, each with the jobs it came from."""
    qs: "OrderedDict[str, dict]" = OrderedDict()
    unlabeled, skipped = [], 0
    for path, rv in sorted(reviews.items(), key=lambda kv: -kv[1].get("mtime", 0)):
        captures = {c.get("name") or c.get("label"): c for c in (rv.get("unlabeled") or [])}
        for row in rv.get("sections", {}).get("needs_me", []):
            field = row["field"]
            if _NOT_A_QUESTION.search(row.get("note", "")):
                skipped += 1
                continue
            job = {"company": rv.get("company", ""), "role": rv.get("role", "")}
            if is_unlabeled(field):
                unlabeled.append({"field": field, "job": _job_label(rv), "job_key": ab.job_key(job),
                                  "capture": captures.get(field), "screenshot": rv.get("screenshot", ""),
                                  "has_captures": rv.get("unlabeled") is not None})
                continue
            q = ab.normalize(field)
            if not q or ab._lookup(bank, field, None, job) or ab._always_ask(bank, q):
                continue
            entry = qs.setdefault(q, {"question": field, "norm": q, "jobs": [], "key": ab._match(bank, q)[0]})
            if _job_label(rv) not in [j["label"] for j in entry["jobs"]]:
                entry["jobs"].append({"label": _job_label(rv), "job_key": ab.job_key(job)})
    return list(qs.values()), unlabeled, skipped


def llm_guesses(reviews: dict) -> list:
    out = []
    for rv in sorted(reviews.values(), key=lambda r: -r.get("mtime", 0)):
        job = {"company": rv.get("company", ""), "role": rv.get("role", "")}
        for src, kind in (("llm_guess", "LLM guess"), ("llm_draft", "LLM draft")):
            for row in rv.get("sections", {}).get(src, []):
                out.append({"field": row["field"], "value": row["value"], "kind": kind, "job": _job_label(rv),
                            "job_key": ab.job_key(job)})
    return out


# ---------------------------------------------------------------------------
# actions (POST /api/answers) — the only writes on this tab
# ---------------------------------------------------------------------------

class AnswerError(Exception):
    """Bad input; str(exc) is safe to show."""


def act(body: dict) -> dict:
    op = str(body.get("op") or "")
    try:
        if op == "set_answer":
            cid = ab.set_answer(str(body.get("key") or ""), body.get("value"))
        elif op == "set_learned":
            cid = ab.set_learned(str(body.get("question") or ""), body.get("value"))
        elif op == "set_job":
            cid = ab.set_job_answer(str(body.get("job") or ""), str(body.get("question") or ""), body.get("value"))
        elif op == "always_ask":
            cid = ab.set_always_ask(str(body.get("item") or ""), bool(body.get("on")))
        elif op == "salary_floor":
            cid = ab.set_salary_floor(int(body.get("value") or 0))
        elif op == "save_question":
            question, value = str(body.get("question") or ""), str(body.get("value") or "").strip()
            if not value:
                raise AnswerError("Type an answer first.")
            if body.get("scope") == "job":
                cid = ab.set_job_answer(str(body.get("job") or ""), question, value)
            else:
                key = ab.matched_key(question)
                cid = (ab.set_answer(key, value) if key and key != "salary_rule" and not key.startswith("learned")
                       else ab.set_learned(question, value))
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

def _input(value: str, attrs: str) -> str:
    return f'<input class="ans" value="{e(value)}" {attrs} spellcheck="false">'


_ANSWER_INPUT = _input("", 'placeholder="Your answer"')
_JOB_INPUT = _input("", 'placeholder="Answer for this job"')


def render(reviews_src) -> str:
    try:
        bank = ab.load()
        data = ab._read_file(ab.BANK_PATH)
    except Exception as exc:
        return _no_data(f"answer bank unreadable: {exc}")
    reviews = reviews_src.data if reviews_src is not None and reviews_src.ok else {}
    out = []
    backup = ab.backup_path()
    out.append(f'<p class="muted small">Edits save to {e(ab.BANK_PATH)} and the learned answers in aria.db; every change '
               f'is listed under History with undo. ' + (f'Backup of the original: {e(backup.name)}.' if backup
                                                         else 'A backup is made before the first save.') + '</p>')

    qs, unlabeled, skipped = new_questions(bank, reviews)
    rows = "".join(
        f'<tr data-q="{e(q["question"])}"><td><b>{e(q["question"])}</b>'
        f'<div class="muted small">from {e("; ".join(j["label"] for j in q["jobs"]))}'
        f'{(" · maps to " + e(q["key"])) if q["key"] else ""}</div></td>'
        f'<td>{_ANSWER_INPUT}</td>'
        f'<td><select class="scope"><option value="all">Save for all forms</option>'
        + "".join(f'<option value="job:{e(j["job_key"])}">This job only: {e(j["label"])}</option>' for j in q["jobs"])
        + '</select> <button class="saveq">Save</button></td><td class="result small"></td></tr>' for q in qs)
    out.append(f'<h3 class="sub3">New questions <span class="count">{len(qs)}</span></h3>'
               + (f'<div class="tablewrap"><table id="newq"><tbody>{rows}</tbody></table></div>' if qs else
                  '<p class="muted">No unanswered questions from your reviews.</p>')
               + (f'<p class="muted small">{skipped} file uploads / consent boxes left out — those stay yours on each form.</p>' if skipped else ""))

    if unlabeled:
        items = []
        for u in unlabeled:
            cap = u["capture"]
            if cap:
                crop = Path(cap.get("crop") or "").name
                img = f'<img class="crop" src="/review-img?f={e(crop)}" alt="field screenshot">' if crop else ""
                what = (f'<div class="small">type: <b>{e(cap.get("tag", ""))}{("/" + e(cap["type"])) if cap.get("type") else ""}</b>'
                        f'{(" · placeholder “" + e(cap["placeholder"]) + "”") if cap.get("placeholder") else ""}</div>'
                        f'<div class="small">nearby text: “{e(cap.get("nearby", "")[:240])}”</div>{img}')
            else:
                why = ("Aria didn't find this field on the page." if u["has_captures"] else
                       "This fill ran before Aria captured field context (from the next fill on, you'll see the field's "
                       "type, nearby text and a screenshot crop here).")
                shot = Path(u["screenshot"]).name if u["screenshot"] else ""
                what = f'<div class="muted small">{e(why)}</div>' + (
                    f'<div class="small">Full-page screenshot: <a href="/review-img?f={e(shot)}" target="_blank">{e(shot)}</a></div>' if shot else "")
            items.append(f'<tr data-q="{e(u["field"])}" data-job="{e(u["job_key"])}"><td><code>{e(u["field"])}</code>'
                         f'<div class="muted small">{e(u["job"])}</div>{what}</td>'
                         f'<td>{_JOB_INPUT}</td>'
                         '<td><button class="savejobq">Save for this job</button></td><td class="result small"></td></tr>')
        out.append(f'<h3 class="sub3">Unlabeled fields <span class="count">{len(unlabeled)}</span></h3>'
                   f'<div class="tablewrap"><table id="unlabeled"><tbody>{"".join(items)}</tbody></table></div>')

    for group, rows_ in answer_rows(bank, data).items():
        trs = "".join(
            f'<tr data-key="{e(r["key"])}"><td>{e(r["label"])}<div class="muted small"><code>{e(r["key"])}</code></div></td>'
            f'<td>{_input(r["value"], "")}</td><td class="muted small">{e(r["source"])}</td>'
            f'<td><button class="savekey">Save</button> <label class="small"><input type="checkbox" class="ask"'
            f'{" checked" if r["always_ask"] else ""}> always ask me</label></td><td class="result small"></td></tr>'
            for r in rows_)
        out.append(f'<h3 class="sub3">{e(group)}</h3><div class="tablewrap"><table class="answers">'
                   f'<tbody>{trs}</tbody></table></div>')

    floor = int(bank.get("salary_rule", {}).get("floor") or ab.DEFAULT_SALARY_FLOOR)
    ask_salary = "salary_rule" in (bank.get("always_ask") or [])
    out.append('<h3 class="sub3">Salary rule</h3><table class="answers"><tbody>'
               f'<tr data-key="salary_rule"><td>Expected salary<div class="muted small">the posting’s low end, never below the floor; '
               'no range in the posting → Aria asks you</div></td>'
               f'<td>{_input(str(floor), "inputmode=numeric id=floor")}</td><td class="muted small">answer bank (floor)</td>'
               f'<td><button class="savefloor">Save</button> <label class="small"><input type="checkbox" class="ask"'
               f'{" checked" if ask_salary else ""}> always ask me</label></td><td class="result small"></td></tr></tbody></table>')

    learned = bank.get("learned") or {}
    trs = "".join(
        f'<tr data-learned="{e(q)}"><td>{e(q)}</td><td>{_input(str(v), "")}</td><td class="muted small">learned</td>'
        f'<td><button class="savelearned">Save</button> <button class="dellearned">Forget</button> '
        f'<label class="small"><input type="checkbox" class="ask"{" checked" if ab._always_ask(bank, q) else ""}> always ask me</label></td>'
        '<td class="result small"></td></tr>' for q, v in sorted(learned.items()))
    out.append(f'<h3 class="sub3">Learned answers <span class="count">{len(learned)}</span></h3>'
               + (f'<div class="tablewrap"><table class="answers"><tbody>{trs}</tbody></table></div>' if learned else
                  '<p class="muted">None yet — answers you give in reviews or by voice show up here.</p>'))

    jobs = bank.get("job_answers") or {}
    if jobs:
        trs = "".join(
            f'<tr data-jobq="{e(q)}" data-job="{e(jk)}"><td>{e(q)}<div class="muted small">{e(jk)}</div></td>'
            f'<td>{_input(str(v), "")}</td><td class="muted small">this job only</td>'
            '<td><button class="savejob">Save</button> <button class="deljob">Remove</button></td><td class="result small"></td></tr>'
            for jk, qa in jobs.items() for q, v in qa.items())
        out.append(f'<h3 class="sub3">Answers for one job</h3><div class="tablewrap"><table class="answers"><tbody>{trs}</tbody></table></div>')

    guesses = llm_guesses(reviews)
    if guesses:
        trs = "".join(
            f'<tr data-q="{e(g["field"])}" data-job="{e(g["job_key"])}"><td>{e(g["field"])}<div class="muted small">{e(g["job"])}</div></td>'
            f'<td><div class="guess small">{e(g["value"][:400])}</div></td><td class="muted small">{e(g["kind"])}</td>'
            f'<td><button class="keepguess" data-scope="all">Save for all forms</button> '
            f'<button class="keepguess" data-scope="job">This job only</button></td><td class="result small"></td></tr>'
            for g in guesses[:40])
        out.append(f'<h3 class="sub3">LLM guesses in your reviews <span class="count">{len(guesses)}</span></h3>'
                   '<p class="muted small">Filled by the LLM, not from your bank. Save one to make it yours.</p>'
                   f'<div class="tablewrap"><table class="answers"><tbody>{trs}</tbody></table></div>')

    hist = ab.history(30)
    if hist:
        trs = "".join(
            f'<tr><td class="small">{e(h["ts"])}</td><td>{e(h["target"])}</td><td>{e(h["key"])}'
            f'{(" · " + e(h["job"])) if h["job"] else ""}</td><td class="small">{e(_short(h["old"]))} → {e(_short(h["new"]))}</td>'
            f'<td class="small">{e(h["source"])}</td><td>'
            + ('<span class="muted small">undone</span>' if h["undone_by"] else f'<button class="undo" data-id="{h["id"]}">Undo</button>')
            + '</td></tr>' for h in hist)
        out.append(f'<h3 class="sub3">History</h3><div class="tablewrap"><table><tbody>{trs}</tbody></table></div>')
    return "".join(out)


def _short(raw: Optional[str]) -> str:
    if raw is None:
        return "(unset)"
    return raw if len(raw) <= 40 else raw[:37] + "…"


JS = r"""
(function(){var tok=document.querySelector('meta[name=dashboard-token]');if(!tok||!document.getElementById('answers-tab'))return;tok=tok.content;
function post(body){return fetch('/api/answers',{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':tok},
body:JSON.stringify(body)}).then(function(r){return r.json().then(function(d){d._ok=r.ok;return d})})}
function show(row,msg,ok){var c=row.querySelector('.result');if(c){c.textContent=msg;c.className='result small '+(ok?'ok':'bad')}}
function done(row){return function(d){show(row,d._ok?'Saved — used on the next fill.':(d.error||'failed'),d._ok)}}
var tab=document.getElementById('answers-tab');
tab.addEventListener('click',function(ev){var b=ev.target;if(b.tagName!=='BUTTON')return;var row=b.closest('tr');var inp=row&&row.querySelector('input.ans');
var val=inp?inp.value:'';
if(b.classList.contains('saveq')){var s=row.querySelector('.scope').value;post({op:'save_question',question:row.dataset.q,value:val,
scope:s==='all'?'all':'job',job:s==='all'?'':s.slice(4)}).then(done(row))}
else if(b.classList.contains('savejobq')){post({op:'save_question',question:row.dataset.q,value:val,scope:'job',job:row.dataset.job}).then(done(row))}
else if(b.classList.contains('savekey')){post({op:'set_answer',key:row.dataset.key,value:val}).then(done(row))}
else if(b.classList.contains('savefloor')){post({op:'salary_floor',value:val}).then(done(row))}
else if(b.classList.contains('savelearned')){post({op:'set_learned',question:row.dataset.learned,value:val}).then(done(row))}
else if(b.classList.contains('dellearned')){post({op:'set_learned',question:row.dataset.learned,value:''}).then(function(d){show(row,d._ok?'Forgotten.':d.error,d._ok)})}
else if(b.classList.contains('savejob')){post({op:'set_job',job:row.dataset.job,question:row.dataset.jobq,value:val}).then(done(row))}
else if(b.classList.contains('deljob')){post({op:'set_job',job:row.dataset.job,question:row.dataset.jobq,value:''}).then(function(d){show(row,d._ok?'Removed.':d.error,d._ok)})}
else if(b.classList.contains('keepguess')){var g=row.querySelector('.guess').textContent;post({op:'save_question',question:row.dataset.q,value:g,
scope:b.dataset.scope,job:row.dataset.job}).then(done(row))}
else if(b.classList.contains('undo')){post({op:'undo',id:parseInt(b.dataset.id,10)}).then(function(d){if(d._ok){b.replaceWith(document.createTextNode('undone — reload to see'))}else alert0(d)})}});
function alert0(d){console.log(d.error)}
tab.addEventListener('change',function(ev){var c=ev.target;if(!c.classList.contains('ask'))return;var row=c.closest('tr');
var item=row.dataset.key||row.dataset.learned;post({op:'always_ask',item:item,on:c.checked}).then(function(d){
show(row,d._ok?(c.checked?'Aria will always ask you.':'Aria may fill this again.'):(d.error||'failed'),d._ok)})})})();
"""
