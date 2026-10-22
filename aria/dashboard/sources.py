"""dashboard/sources.py — read-only loaders. Each returns Source(ok, data, error), never raises.

  load_sheet()      Muse sheet tabs (current jobs + Applied), jobs Google account   cached 5 min
  load_gmail()      jobs Gmail: drafts + every thread with a sent message           cached 5 min
  load_aria_db()    aria.db application_runs + applications (opened read-only)
  load_reviews(p)   every review file (~/.aria/reviews/*.md + aria.db paths), parsed per
                    section, plus the unlabeled-field captures (<review>.fields.json)
  load_coldstart()  Coldstart contact cache (its SQLite opened read-only)

Nothing here writes: SQLite files are opened with mode=ro, Google APIs only get/list.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from email.utils import getaddresses, parseaddr
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CACHE_TTL = 300.0
# Gmail allows 250 quota units per user per second and threads.get costs 10: 10 reads per
# batch, then a pause, stays well under it (a batch of 50 was 500 units in one burst).
_BATCH = 10
_BATCH_PAUSE = 0.6


@dataclass
class Source:
    ok: bool
    data: Any = None
    error: str = ""
    loaded_at: float = field(default_factory=time.time)


def _guard(name: str, fn: Callable[[], Any]) -> Source:
    try:
        return Source(True, fn())
    except Exception as exc:
        logger.warning("dashboard: %s unavailable: %s", name, exc)
        return Source(False, None, f"{type(exc).__name__}: {exc}")


class TTLCache:
    """Keeps a successful Source for `ttl` seconds; failures are retried next load."""

    def __init__(self, ttl: float = CACHE_TTL, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl, self._clock = ttl, clock
        self._items: dict[str, tuple[float, Source]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, load: Callable[[], Source]) -> Source:
        with self._lock:
            hit = self._items.get(key)
            if hit and self._clock() - hit[0] < self.ttl:
                return hit[1]
        src = load()
        if src.ok:
            with self._lock:
                self._items[key] = (self._clock(), src)
        return src


CACHE = TTLCache()


def _ro_connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(str(path))
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Muse sheet
# ---------------------------------------------------------------------------

def _sheet_rows(values: list[list[str]]) -> list[dict]:
    from aria.integrations.sheets import dedupe_headers
    if not values:
        return []
    hdr = dedupe_headers(values[0])
    rows = []
    for i, r in enumerate(values[1:], start=2):
        if not any(str(c).strip() for c in r):
            continue
        row = {h: (str(r[j]).strip() if j < len(r) else "") for j, h in enumerate(hdr)}
        row["_row"] = i
        rows.append(row)
    return rows


def fetch_sheet(service=None) -> dict:
    import aria.core.config as config
    from aria.integrations import google_auth
    if not config.MUSE_SHEET_ID:
        raise RuntimeError("MUSE_SHEET_ID is not set")
    svc = service or google_auth.build_service("sheets", "v4", [google_auth.SPREADSHEETS],
                                               account=google_auth.JOBS)
    tabs = {"current": config.MUSE_SHEET_TAB, "applied": config.MUSE_APPLIED_TAB}
    ranges = ["'" + t.replace("'", "''") + "'" for t in tabs.values()]
    resp = svc.spreadsheets().values().batchGet(spreadsheetId=config.MUSE_SHEET_ID, ranges=ranges).execute()
    out = {"tabs": tabs}
    for key, vr in zip(tabs, resp.get("valueRanges", [])):
        out[key] = _sheet_rows(vr.get("values", []))
    return out


def load_sheet() -> Source:
    return CACHE.get("sheet", lambda: _guard("sheet", fetch_sheet))


# ---------------------------------------------------------------------------
# Gmail (jobs account)
# ---------------------------------------------------------------------------

_BOUNCE_FROM = re.compile(r"mailer-daemon|postmaster", re.I)
_BOUNCE_SUBJECT = re.compile(r"(?i)delivery status notification|undeliver|failure notice|"
                             r"delivery has failed|returned mail|address not found")
_GMAIL_HDRS = ["From", "To", "Cc", "Subject", "In-Reply-To", "References"]


def _headers(msg: dict) -> dict:
    return {h["name"]: h["value"] for h in (msg.get("payload") or {}).get("headers", [])}


_RETRY_WAITS = (2.0, 5.0, 10.0)


def _batch_get(svc, requests: list, sleep: Callable[[float], None] = time.sleep) -> list:
    """Run googleapiclient requests in batches. Failed items (rate limits) are retried
    with backoff; if any still fail, raise — partial Gmail data is never shown as complete."""
    out: list = [None] * len(requests)
    todo = list(range(len(requests)))
    errors: dict[int, Exception] = {}
    for wait in (0.0,) + _RETRY_WAITS:
        if not todo:
            break
        if wait:
            sleep(wait)
        errors = {}
        for start in range(0, len(todo), _BATCH):
            if start:
                sleep(_BATCH_PAUSE)
            batch = svc.new_batch_http_request()
            for i in todo[start:start + _BATCH]:
                def cb(_rid, resp, exc, i=i):
                    if exc is None:
                        out[i] = resp
                    else:
                        errors[i] = exc
                batch.add(requests[i], callback=cb)
            batch.execute()
        todo = sorted(errors)
    if todo:
        raise RuntimeError(f"{len(todo)} of {len(requests)} Gmail reads failed: {errors[todo[0]]}")
    return out


def _parse_thread(t: dict, me: str) -> dict:
    msgs = []
    for m in t.get("messages", []):
        h = _headers(m)
        frm = parseaddr(h.get("From", ""))[1].lower()
        msgs.append({
            "id": m["id"], "ts": int(m.get("internalDate", 0)) / 1000.0, "from": frm,
            "to": [a.lower() for _, a in getaddresses([h.get("To", ""), h.get("Cc", "")]) if a],
            "subject": h.get("Subject", ""), "is_reply": bool(h.get("In-Reply-To") or h.get("References")),
            "from_me": frm == me,
            "bounce": bool(_BOUNCE_FROM.search(frm) or (frm != me and _BOUNCE_SUBJECT.search(h.get("Subject", "")))),
        })
    msgs.sort(key=lambda m: m["ts"])
    return {"id": t["id"], "messages": msgs}


def _list_all(call: Callable[[Optional[str]], dict], key: str) -> list:
    items, tok = [], None
    while True:
        resp = call(tok)
        items += resp.get(key, [])
        tok = resp.get("nextPageToken")
        if not tok:
            return items


def fetch_gmail(service=None, days: int = 365, cache_path: Optional[Path] = None) -> dict:
    from aria.dashboard import gmail_cache
    from aria.integrations import google_auth
    svc = service or google_auth.build_service("gmail", "v1", [google_auth.GMAIL_MODIFY],
                                               account=google_auth.JOBS)
    users = svc.users()
    me = users.getProfile(userId="me").execute()["emailAddress"].lower()
    path = cache_path or gmail_cache.PATH
    cache = gmail_cache.read(path, me)

    listed = _list_all(lambda tok: users.threads().list(userId="me", q=f"in:sent newer_than:{days}d", maxResults=500,
                                                        pageToken=tok).execute(), "threads")
    threads = {t["id"]: cache["threads"][t["id"]] for t in listed
               if t["id"] in cache["threads"] and cache["threads"][t["id"]].get("h") == t.get("historyId")}
    stale = [t for t in listed if t["id"] not in threads]
    got = _batch_get(svc, [users.threads().get(userId="me", id=t["id"], format="metadata",
                                               metadataHeaders=_GMAIL_HDRS) for t in stale])
    for t, raw in zip(stale, got):
        if raw:
            threads[t["id"]] = {"h": t.get("historyId"), "t": _parse_thread(raw, me)}

    # A draft's message id changes whenever it's edited, so it's the cache key.
    listed_drafts = _list_all(lambda tok: users.drafts().list(userId="me", maxResults=500, pageToken=tok).execute(),
                              "drafts")
    drafts = {(d.get("message") or {}).get("id", ""): cache["drafts"][(d.get("message") or {}).get("id", "")]
              for d in listed_drafts if (d.get("message") or {}).get("id", "") in cache["drafts"]}
    new = [d for d in listed_drafts if (d.get("message") or {}).get("id", "") not in drafts]
    for d in _batch_get(svc, [users.drafts().get(userId="me", id=d["id"], format="metadata") for d in new]):
        if not d:
            continue
        m = d.get("message", {})
        h = _headers(m)
        drafts[m.get("id", "")] = {"id": d["id"], "thread_id": m.get("threadId", ""),
                                   "ts": int(m.get("internalDate", 0)) / 1000.0,
                                   "to": [a.lower() for _, a in getaddresses([h.get("To", ""), h.get("Cc", "")]) if a],
                                   "subject": h.get("Subject", "")}

    gmail_cache.write(path, {"me": me, "threads": threads, "drafts": drafts})   # only what's still listed
    logger.info("dashboard: Gmail read %d of %d threads, %d of %d drafts (rest cached)",
                len(stale), len(listed), len(new), len(listed_drafts))
    return {"me": me, "threads": [threads[t["id"]]["t"] for t in listed if t["id"] in threads],
            "drafts": list(drafts.values())}


def load_gmail() -> Source:
    return CACHE.get("gmail", lambda: _guard("gmail", fetch_gmail))


# ---------------------------------------------------------------------------
# aria.db + review files
# ---------------------------------------------------------------------------

def fetch_aria_db(path: Optional[Path] = None) -> dict:
    from aria.state import db
    conn = _ro_connect(Path(path or db.DB_PATH))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        runs = [dict(r) for r in conn.execute("SELECT * FROM application_runs ORDER BY created_at")] \
            if "application_runs" in tables else []
        apps = [dict(r) for r in conn.execute("SELECT * FROM applications ORDER BY applied_at")] \
            if "applications" in tables else []
    finally:
        conn.close()
    return {"runs": runs, "applications": apps}


def load_aria_db() -> Source:
    return _guard("aria.db", fetch_aria_db)


_SECTION_KEYS = {"Needs you": "needs_me", "Drafted by the LLM": "llm_draft", "Guessed by the LLM": "llm_guess",
                 "From your answer bank": "bank", "Left empty": "skipped"}
_TITLE = re.compile(r"^# Review: (.*) at (.*)$", re.M)
_META = re.compile(r"^- \*\*(Form|Sheet row|Screenshot|Problem):\*\* (.*)$", re.M)


def _cells(line: str) -> list[str]:
    parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in parts]


def parse_review(text: str) -> dict:
    """A review markdown file → {role, company, form, row, screenshot, sections: {source: [rows]}}.
    Rows: {field, value, required, note}; llm_draft rows carry the drafted text as value."""
    out: dict = {"role": "", "company": "", "form": "", "row": None, "screenshot": "", "problem": "", "sections": {}}
    m = _TITLE.search(text)
    if m:
        out["role"], out["company"] = m.group(1).strip(), m.group(2).strip()
    for k, v in _META.findall(text):
        key = {"Form": "form", "Sheet row": "row", "Screenshot": "screenshot", "Problem": "problem"}[k]
        out[key] = int(v) if key == "row" and v.strip().isdigit() else v.strip()
    current, draft = None, None
    for line in text.splitlines():
        if line.startswith("## "):
            title = re.sub(r"\s*\(\d+\)\s*$", "", line[3:]).split(":")[0].strip()
            current = next((v for k, v in _SECTION_KEYS.items() if title.startswith(k)), None)
            if current:
                out["sections"].setdefault(current, [])
            draft = None
            continue
        if current == "llm_draft":
            if line.startswith("### "):
                draft = {"field": line[4:].strip(), "value": "", "required": True, "note": ""}
                out["sections"]["llm_draft"].append(draft)
            elif draft is not None and line.startswith(">"):
                draft["value"] = (draft["value"] + "\n" + line[1:].strip()).strip()
            continue
        if current and line.startswith("|"):
            cells = _cells(line)
            if not cells or cells[0] in ("Field", "") or set(cells[0]) <= {"-"}:
                continue
            cells += [""] * (4 - len(cells))
            out["sections"][current].append({"field": cells[0].replace(" ✱", "").strip(), "value": cells[1],
                                             "required": cells[2] == "yes", "note": cells[3]})
    return out


def parse_review_needs_me(text: str) -> list[str]:
    """Field labels in a review file's "## Needs you (N)" table."""
    return [r["field"] for r in parse_review(text)["sections"].get("needs_me", [])]


def load_reviews(paths: list[str], reviews_dir: Optional[Path] = None) -> Source:
    def read() -> dict:
        if reviews_dir is None:
            from aria.features.applicator import REVIEWS_DIR
            folder = REVIEWS_DIR
        else:
            folder = reviews_dir
        every = set(paths) | ({str(p) for p in folder.glob("*.md")} if folder.exists() else set())
        out = {}
        for p in sorted(every):
            path = Path(p)
            try:
                parsed = parse_review(path.read_text())
            except OSError:
                continue
            parsed["path"], parsed["mtime"] = p, path.stat().st_mtime
            sidecar = path.with_suffix(".fields.json")
            try:
                parsed["unlabeled"] = json.loads(sidecar.read_text()) if sidecar.exists() else None
            except (OSError, ValueError):
                parsed["unlabeled"] = None
            out[p] = parsed
        return out
    return _guard("reviews", read)


# ---------------------------------------------------------------------------
# Coldstart
# ---------------------------------------------------------------------------

def fetch_coldstart(path: Optional[Path] = None) -> dict:
    if path is None:
        import coldstart.config as cs_config
        path = cs_config.db_path()
    conn = _ro_connect(Path(path))
    try:
        contacts = [dict(r) for r in conn.execute(
            "SELECT domain, name, title, email, status, confidence, source, checked_at FROM contacts "
            "WHERE email IS NOT NULL AND email != '' ORDER BY confidence DESC")]
        doms = sorted({c["domain"] for c in contacts})
        domains = {}
        if doms:
            q = "SELECT domain, provider, catch_all, inferred_pattern FROM domains WHERE domain IN (%s)"
            domains = {r["domain"]: dict(r) for r in conn.execute(q % ",".join("?" * len(doms)), doms)}
    finally:
        conn.close()
    return {"contacts": contacts, "domains": domains}


def coldstart_domain(domain: str, path: Optional[Path] = None) -> dict:
    """One domain's cached row (provider, pattern) from Coldstart, read-only; {} if unknown."""
    try:
        if path is None:
            import coldstart.config as cs_config
            path = cs_config.db_path()
        conn = _ro_connect(Path(path))
        try:
            row = conn.execute("SELECT domain, provider, catch_all, inferred_pattern FROM domains WHERE domain=?",
                               (domain,)).fetchone()
        finally:
            conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def load_coldstart() -> Source:
    return _guard("coldstart", fetch_coldstart)


def load_all() -> dict:
    """Every source, fresh (Gmail and sheet from the 5-minute cache)."""
    srcs = {"sheet": load_sheet(), "gmail": load_gmail(), "aria_db": load_aria_db(),
            "coldstart": load_coldstart()}
    paths = sorted({r["review_path"] for r in (srcs["aria_db"].data or {}).get("runs", [])
                    if r.get("review_path")}) if srcs["aria_db"].ok else []
    srcs["reviews"] = load_reviews(paths)
    return srcs
