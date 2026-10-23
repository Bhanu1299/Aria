"""features/ats/corrections.py — every apply failure gets a row, to read after a batch.

record(job, kind, step, detail, ats, page_url, run) → row id (0 on error). Never raises.
  kind: unknown_ats · apply_not_found · apply_click_nothing · unmapped_field · failed_field ·
        login_failed · captcha · login · email_code · agreement · timeout · fill_error
  run:  a tab-bound run(fn) → fn(page); when given, the page is saved as
        ~/.aria/corrections/<stem>.png + .html (snapshot column = the stem's path)
recent(days) → rows, newest first
python -m aria.features.ats.corrections [--days N]   prints them grouped by kind

Rows live in aria.db apply_corrections. Snapshots never hold a password: every password box
is emptied in the saved HTML (SNAPSHOT_JS) and scripts are dropped; the PNG shows only dots.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import aria.state.db as db
from aria.features.ats.detect import domain

logger = logging.getLogger(__name__)

CORRECTIONS_DIR = Path.home() / ".aria" / "corrections"
# The page's HTML with every password box emptied: React mirrors typed values into the
# value attribute, so a plain page.content() after a sign-in could carry the password.
SNAPSHOT_JS = """() => { const doc = document.documentElement.cloneNode(true);
  doc.querySelectorAll('input[type=password], input[autocomplete*=password]').forEach(e => { e.removeAttribute('value'); e.setAttribute('value', ''); });
  doc.querySelectorAll('script').forEach(e => e.remove());
  return '<!doctype html>' + doc.outerHTML; }"""


def _stem(job: dict, kind: str) -> str:
    who = re.sub(r"[^a-z0-9]+", "-", (job.get("company") or domain(job.get("url", "")) or "job").lower()).strip("-")
    return f"{datetime.now():%Y%m%d-%H%M%S}_{who[:40]}_{kind}"


def snapshot(run: Callable, stem: str) -> str:
    """Save the tab as <stem>.png + <stem>.html (owner-only). Returns the stem's path, '' on failure."""
    try:
        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        base = CORRECTIONS_DIR / stem
        html = run(lambda page: page.evaluate(SNAPSHOT_JS))
        _write(base.with_suffix(".html"), html.encode("utf-8", "replace"))
        try:
            png = run(lambda page: page.screenshot(timeout=8000))
            _write(base.with_suffix(".png"), png)
        except Exception as exc:          # background tabs sometimes don't paint; the html is enough
            logger.debug("correction screenshot failed: %s", exc)
        return str(base)
    except Exception as exc:
        logger.warning("correction snapshot failed: %s", exc)
        return ""


def _write(path: Path, data: bytes) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def record(job: Optional[dict], kind: str, step: str, detail: str = "", ats: str = "",
           page_url: str = "", run: Optional[Callable] = None) -> int:
    job = job or {}
    snap = snapshot(run, _stem(job, kind)) if run is not None else ""
    if run is not None and not page_url:
        try:
            page_url = run(lambda page: page.url) or ""
        except Exception:
            page_url = ""
    try:
        row = job.get("row")
        with db.get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO apply_corrections (job_url, company, row, ats, step, kind, detail, page_url, snapshot) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (job.get("url", ""), job.get("company", ""), int(row) if str(row or "").isdigit() else None,
                 ats, step, kind, detail[:500], page_url, snap))
            return int(cur.lastrowid or 0)
    except Exception as exc:
        logger.warning("correction not recorded (%s: %s): %s", kind, detail[:80], exc)
        return 0


def recent(days: int = 7) -> list:
    try:
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM apply_corrections WHERE created_at >= datetime('now', ?) ORDER BY id DESC",
                (f"-{int(days)} days",)).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("corrections unreadable: %s", exc)
        return []


def report(rows: list) -> str:
    if not rows:
        return "No apply failures recorded."
    out = []
    for kind in sorted({r["kind"] for r in rows}):
        group = [r for r in rows if r["kind"] == kind]
        out.append(f"\n{kind} ({len(group)})")
        for r in group:
            where = r["company"] or domain(r["job_url"])
            ats = f" [{r['ats']}]" if r["ats"] else ""
            out.append(f"  {r['created_at']}  {where}{ats}  {r['step']}: {r['detail']}")
            out.append(f"      {r['page_url'] or r['job_url']}")
            if r["snapshot"]:
                out.append(f"      snapshot: {r['snapshot']}.png / .html")
    return "\n".join(out).lstrip("\n")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aria.features.ats.corrections",
                                 description="Apply failures from the last N days, grouped by kind.")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args(argv)
    print(report(recent(args.days)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
