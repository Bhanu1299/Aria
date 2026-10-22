"""dashboard/health.py — Tab 7 "Data health": gaps and loose ends in the data (read-only)."""
from __future__ import annotations

from rapidfuzz import fuzz, process

from aria.dashboard import model as m
from aria.dashboard.render import _sources_bar, _thread_url, e, stat

LINK_NOTE = ("Linking is shown as a suggestion only: saving a link needs a place to store it, and the "
             "dashboard may only write to the answer bank and GateLLM keys.")


def suggest_job(record: dict, jobs: list):
    """The job a record most likely belongs to (company/domain similarity), or None."""
    what = record.get("what", "")
    head = what.split(" — ")[0]
    target = head.split("@")[1].split(".")[0] if "@" in head else head
    names = {i: m.company_slug(j.company) for i, j in enumerate(jobs) if j.company}
    if not target or not names:
        return None
    best = process.extractOne(m.company_slug(target), names, scorer=fuzz.ratio)
    return jobs[best[2]] if best and best[1] >= 70 else None


def render(mdl: dict) -> str:
    jobs = mdl["jobs"]
    g = mdl["sources"].get("gmail")
    me = (g.data or {}).get("me", "") if g is not None and g.ok else ""
    no_date = [j for j in jobs if not j.found_date]
    nameless = [j for j in jobs if j.email and not j.contact]
    cs = mdl.get("coldstart") or {}
    cs_nameless = [c for c in cs.get("contacts", []) if not (c.get("name") or "").strip()]
    no_conf = [j for j in jobs if j.email and j.email_confidence is None]
    out = [f'<h3 class="sub3">Sources</h3>{_sources_bar(mdl["sources"])}',
           '<div class="stats" style="margin-top:16px">'
           + stat(len(mdl["unmatched"]), "records matching no job") + stat(len(no_date), "jobs missing a found date")
           + stat(len(nameless) + len(cs_nameless), "contacts missing names") + stat(len(no_conf), "emails with no confidence")
           + '</div>']

    rows = []
    for u in mdl["unmatched"]:
        job = suggest_job(u, jobs)
        sug = f'{e(job.company)} — {e(job.role)}' if job else '<span class="muted">no likely job</span>'
        link = (f' <a href="{e(_thread_url(me, u["thread_id"]))}" target="_blank" rel="noopener">thread</a>'
                if me and u.get("thread_id") else "")
        rows.append(f'<tr><td class="small">{e(u["source"])}</td><td>{e(u["what"])}{link}'
                    f'<div class="muted small">{e(u["detail"])}</div></td><td>{sug}</td></tr>')
    out.append(f'<h3 class="sub3">Records that match no job <span class="count">{len(rows)}</span></h3>'
               f'<p class="muted small">{e(LINK_NOTE)}</p>'
               + (f'<div class="tablewrap"><table><thead><tr><th>Source</th><th>What it is</th><th>Likely job</th></tr></thead>'
                  f'<tbody>{"".join(rows)}</tbody></table></div>' if rows else '<p class="muted">None.</p>'))

    out.append(f'<h3 class="sub3">Jobs missing a found date <span class="count">{len(no_date)}</span></h3>'
               '<p class="muted small">The sheet has no found-date column; dates are read from Source / Updates text '
               '(“Oct 20”, “2026-10-20”). Without one, “jobs found this week” can’t count the job.</p>'
               + _job_list(no_date, 40))

    cs_items = "".join(f"<li>Coldstart contact {e(c['email'])} has no name</li>" for c in cs_nameless[:20])
    out.append(f'<h3 class="sub3">Contacts missing names <span class="count">{len(nameless) + len(cs_nameless)}</span></h3>'
               + _job_list(nameless, 40, lambda j: e(j.email))
               + (f"<ul>{cs_items}</ul>" if cs_items else ""))

    out.append(f'<h3 class="sub3">Emails with no confidence <span class="count">{len(no_conf)}</span></h3>'
               '<p class="muted small">The address came from the sheet or Gmail, not from Coldstart, so nothing scored it.</p>'
               + _job_list(no_conf, 40, lambda j: e(j.email) + f' <span class="muted small">'
                           f'{e(next((ev.source for ev in j.evidence if ev.stage == m.EMAIL), ""))}</span>'))
    return "".join(out)


def _job_list(jobs: list, limit: int, extra=None) -> str:
    if not jobs:
        return '<p class="muted">None.</p>'
    items = "".join(f'<li>{e(j.company)} — {e(j.role)}{(" · " + extra(j)) if extra else ""}</li>' for j in jobs[:limit])
    more = f'<p class="muted small">…and {len(jobs) - limit} more.</p>' if len(jobs) > limit else ""
    return f"<ul>{items}</ul>{more}"
