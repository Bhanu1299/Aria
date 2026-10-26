"""features/jobprep/reposts.py — postings Aria has seen, and which ones are reposts.

Every row Aria reads or prepares is recorded (table postings_seen, one row per URL). A
repost = the same company + normalized title seen under 2+ different URLs, first seen on
2+ distinct dates, within 90 days. Reposts are flagged (dashboard badge, run log), never
skipped automatically.

    record(job, when=None)              upsert by URL (last_seen bumped)
    record_many(jobs)                   the same for a sheet read; never raises
    is_repost(company, role) → dict | None   {"urls", "dates", "count"}
    reposts() → [{"company", "role", "urls", "dates", "count"}]

Never raises: a db error logs and reads as "not a repost".
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

from aria.features.jobprep import store
from aria.features.jobprep.blocklist import company_key

logger = logging.getLogger(__name__)

WINDOW_DAYS = 90
_LEVEL = re.compile(r"\b(?:senior|sr|junior|jr|i{1,3}|iv|\d)\b\.?")


def url_key(url: str) -> str:
    u = urlparse((url or "").strip() if "://" in (url or "") else f"https://{(url or '').strip()}")
    host = (u.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{u.path.rstrip('/')}".lower() if host else ""


def title_key(role: str) -> str:
    t = re.sub(r"\([^)]*\)", " ", (role or "").lower())
    t = re.split(r"\s+[-–—|]\s+", t)[0]
    return " ".join(re.sub(r"[^a-z0-9+#]+", " ", t).split())


def _url(job: dict) -> str:
    return job.get("url") or job.get("post_link") or job.get("apply_link") or job.get("link") or ""


def record(job: dict, when: Optional[datetime] = None) -> None:
    key = url_key(_url(job))
    if not key or not (job.get("company") or job.get("role")):
        return
    ts = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with store.conn() as c:
            c.execute("INSERT INTO postings_seen (company, role, url, url_key, first_seen, last_seen) "
                      "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(url_key) DO UPDATE SET last_seen=excluded.last_seen",
                      (job.get("company", ""), job.get("role", ""), _url(job), key, ts, ts))
    except Exception as exc:
        logger.warning("reposts: record failed: %s", exc)


def record_many(jobs: list) -> None:
    for j in jobs or []:
        record(j)


def _groups(now: Optional[datetime] = None) -> dict:
    since = ((now or datetime.now()) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        got = store.rows("SELECT company, role, url, first_seen FROM postings_seen WHERE first_seen >= ?", (since,))
    except Exception as exc:
        logger.warning("reposts: read failed: %s", exc)
        return {}
    groups: dict = {}
    for r in got:
        k = (company_key(r["company"]), title_key(r["role"]))
        g = groups.setdefault(k, {"company": r["company"], "role": r["role"], "urls": [], "dates": set()})
        g["urls"].append(r["url"])
        g["dates"].add(r["first_seen"][:10])
    return groups


def _flagged(g: dict) -> Optional[dict]:
    if len(set(g["urls"])) >= 2 and len(g["dates"]) >= 2:
        return {"company": g["company"], "role": g["role"], "urls": sorted(set(g["urls"])),
                "dates": sorted(g["dates"]), "count": len(set(g["urls"]))}
    return None


def is_repost(company: str, role: str, now: Optional[datetime] = None) -> Optional[dict]:
    g = _groups(now).get((company_key(company), title_key(role)))
    return _flagged(g) if g else None


def reposts(now: Optional[datetime] = None) -> list:
    return [f for f in (_flagged(g) for g in _groups(now).values()) if f]
