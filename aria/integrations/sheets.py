"""integrations/sheets.py — the job sheet ("Muse") that feeds the apply pipeline.

Reads the header row on every call — no hard-coded column letters, so columns can
be reordered or added in the sheet. Duplicate header names are disambiguated in
order: "Outreach", "Outreach" → "Outreach", "Outreach_2".

    get_next_job(status="new") → dict | None     first matching row, top to bottom
    get_job(row) → dict | None
    ensure_ids()       every job row in both tabs gets a stable ID ("J0042"), never rewritten
    find_jobs(query)   ranked fuzzy matches over both tabs ("the RTX onsite one", "magnet")
    resolve_job(query) → {"kind": one | many | applied | none, "jobs": [...]}
    set_status(row, status, note="", company=, role=, job_id=)
    set_fields(row, {"Who to contact": ...}, note="", company=, role=, job_id=)   outreach columns only

A blank Status cell counts as "new". Notes are appended to "Updates" as
"YYYY-MM-DD: note", joined with " | " — existing text is never overwritten.
Columns this module writes: ID (blank cells only), Status, Updates, and the outreach columns in
OUTREACH_FIELDS (set_fields refuses any other). Everything else is the user's.

Rows move when the sheet is sorted or a row moves to the Applied tab, so writes re-find
the job by its ID, then by (Company, Role), when the given row no longer holds it.

Raises SheetError (str is safe to speak) — callers turn it into a reply.

Always the jobs Google account (GOOGLE_JOBS_ACCOUNT) — never the personal one.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse
from datetime import date
from typing import Callable, Optional

import aria.core.config as config
from aria.integrations import google_auth

logger = logging.getLogger(__name__)

COL_ID = "ID"                  # stable job ID ("J0042"); added as the last column if missing
COL_COMPANY = "Company"
COL_ROLE = "Role"
COL_LOCATION = "Location"
COL_ALIGNMENT = "Profile alignment /5"
COL_STATUS = "Status"
COL_APPLY_LINK = "Apply link"
COL_POST_LINK = "Post link"
COL_UPDATES = "Updates"
COL_SOURCE = "Source"          # optional: how you found the job ("How did you hear about us")
COL_WHO = "Who to contact"
COL_OUTREACH = "Outreach_2"   # the second "Outreach" column: the planned message + channel
OUTREACH_FIELDS = (COL_WHO, COL_OUTREACH)

# Status values Aria writes. "Applied" / "Not gonna apply" are the user's own.
STATUS_NEW = "new"
STATUS_IN_REVIEW = "In review"
STATUS_APPLIED = "Applied"
STATUS_SKIPPED = "Skipped"
STATUS_NEEDS_HUMAN = "Needs human"
# Statuses that still mean "not handled yet" (the user's sheet defaults rows to "Not applied").
_NEW_ALIASES = {"", "not applied", "not yet applied", "to apply"}

_URL = re.compile(r"^https?://", re.I)
_JOB_ID = re.compile(r"j(\d{1,6})", re.I)


class SheetError(Exception):
    """The job sheet can't be read or written. str(exc) is safe to speak."""


def dedupe_headers(row: list) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for cell in row:
        name = str(cell).strip()
        n = seen.get(name.lower(), 0) + 1
        seen[name.lower()] = n
        out.append(name if n == 1 else f"{name}_{n}")
    return out


def col_letter(index: int) -> str:
    """0-based column index → A1 letters (0 → A, 25 → Z, 26 → AA)."""
    letters = ""
    n = index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


# Where a job was posted, by the Post link's site. Job boards / ATS pages say nothing
# about how you found it, so they give no source.
_SOURCE_SITES = {
    "instagram.com": "Instagram", "linkedin.com": "LinkedIn", "indeed.com": "Indeed",
    "glassdoor.com": "Glassdoor", "joinhandshake.com": "Handshake", "wellfound.com": "Wellfound",
    "angel.co": "Wellfound", "x.com": "Twitter", "twitter.com": "Twitter", "youtube.com": "YouTube",
    "tiktok.com": "TikTok", "facebook.com": "Facebook", "reddit.com": "Reddit", "builtin.com": "Built In",
    "ycombinator.com": "Y Combinator", "themuse.com": "The Muse", "dice.com": "Dice",
}


def source_from_link(url: str) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    return next((name for site, name in _SOURCE_SITES.items() if host == site or host.endswith("." + site)), "")


class JobSheet:
    def __init__(
        self,
        sheet_id: str,
        tab: str,
        service=None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.sheet_id = sheet_id
        self.tab = tab
        self._service = service
        self._today = today

    # -- I/O ---------------------------------------------------------------

    def _values_api(self):
        if not self.sheet_id:
            raise SheetError("No job sheet configured. Set MUSE_SHEET_ID in .env.")
        if self._service is None:
            try:
                self._service = google_auth.build_service(
                    "sheets", "v4", [google_auth.SPREADSHEETS], account=google_auth.JOBS)
            except google_auth.GoogleAuthError as exc:
                raise SheetError(str(exc)) from exc
        return self._service.spreadsheets().values()

    def _a1(self, cell: str = "") -> str:
        tab = "'" + self.tab.replace("'", "''") + "'"
        return f"{tab}!{cell}" if cell else tab

    def _read(self) -> list[list[str]]:
        try:
            resp = self._values_api().get(spreadsheetId=self.sheet_id, range=self._a1()).execute()
        except SheetError:
            raise
        except Exception as exc:
            logger.error("sheets: read failed: %s", exc)
            raise SheetError(f"I couldn't read the job sheet: {exc}") from exc
        return [[str(c) for c in r] for r in resp.get("values", [])]

    def _write(self, cells: dict[str, str]) -> None:
        """cells: {"E5": "Applied", ...} — RAW, so text is never parsed as a formula."""
        body = {
            "valueInputOption": "RAW",
            "data": [{"range": self._a1(a1), "values": [[v]]} for a1, v in cells.items()],
        }
        try:
            self._values_api().batchUpdate(spreadsheetId=self.sheet_id, body=body).execute()
        except SheetError:
            raise
        except Exception as exc:
            logger.error("sheets: write failed: %s", exc)
            if getattr(getattr(exc, "resp", None), "status", None) == 403:
                raise SheetError("Your jobs Google account can read the job sheet but can't edit it. "
                                 "Share the sheet with that account as Editor.") from exc
            raise SheetError(f"I couldn't update the job sheet: {exc}") from exc

    # -- layout --------------------------------------------------------------

    @staticmethod
    def _index(headers: list[str]) -> dict[str, int]:
        return {_norm(h): i for i, h in enumerate(headers) if h}

    def _require(self, index: dict[str, int], name: str) -> int:
        if _norm(name) not in index:
            raise SheetError(f"The job sheet has no {name!r} column.")
        return index[_norm(name)]

    def _row_dict(self, headers: list[str], values: list[str], row_number: int) -> dict:
        cells = list(values) + [""] * (len(headers) - len(values))
        fields = {h: cells[i].strip() for i, h in enumerate(headers) if h}
        get = lambda name: fields.get(next((h for h in fields if _norm(h) == _norm(name)), ""), "")
        apply_link, post_link = get(COL_APPLY_LINK), get(COL_POST_LINK)
        url = apply_link if _URL.match(apply_link) else (post_link if _URL.match(post_link) else "")
        return {
            "row": row_number,
            "job_id": get(COL_ID),
            "company": get(COL_COMPANY),
            "role": get(COL_ROLE),
            "location": get(COL_LOCATION),
            "alignment": get(COL_ALIGNMENT),
            "status": get(COL_STATUS),
            "apply_link": apply_link,
            "post_link": post_link,
            "url": url,
            "source": get(COL_SOURCE) or source_from_link(post_link),
            "fields": fields,
        }

    # -- public --------------------------------------------------------------

    def jobs(self) -> list[dict]:
        values = self._read()
        if not values:
            raise SheetError("The job sheet is empty.")
        headers = dedupe_headers(values[0])
        index = self._index(headers)
        for required in (COL_COMPANY, COL_ROLE):
            self._require(index, required)
        return [
            self._row_dict(headers, vals, i + 2)
            for i, vals in enumerate(values[1:])
            if any(str(v).strip() for v in vals)
        ]

    def get_next_job(self, status: str = STATUS_NEW) -> Optional[dict]:
        want = _norm(status)
        for job in self.jobs():
            have = _norm(job["status"])
            if have == want or (want == STATUS_NEW and have in _NEW_ALIASES):
                return job
        return None

    def get_job(self, row: int) -> Optional[dict]:
        return next((j for j in self.jobs() if j["row"] == row), None)

    def set_status(
        self,
        row: int,
        status: str,
        note: str = "",
        company: Optional[str] = None,
        role: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> int:
        """Write Status (and append note to Updates). Returns the row actually written."""
        return self._update_row(row, {COL_STATUS: status}, note, company, role, job_id)

    def set_fields(
        self,
        row: int,
        fields: dict[str, str],
        note: str = "",
        company: Optional[str] = None,
        role: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> int:
        """Write outreach columns (OUTREACH_FIELDS only) and append note to Updates."""
        bad = [k for k in fields if _norm(k) not in {_norm(f) for f in OUTREACH_FIELDS}]
        if bad:
            raise SheetError(f"Aria doesn't write the {bad[0]!r} column.")
        return self._update_row(row, fields, note, company, role, job_id)

    def table(self) -> dict:
        """The whole tab as it is: {"headers": [...], "rows": [{"row": n, "cells": [...]}]}.
        Headers are deduped ("Outreach", "Outreach_2") so each names one column; cells are padded."""
        values = self._read()
        if not values:
            return {"headers": [], "rows": []}
        headers = dedupe_headers(values[0])
        width = max([len(headers)] + [len(v) for v in values[1:]])
        headers += [""] * (width - len(headers))
        idc = self._index(headers).get(_norm(COL_ID))
        rows = [{"row": i, "id": str(vals[idc]).strip() if idc is not None and idc < len(vals) else "",
                 "cells": [str(c) for c in vals] + [""] * (width - len(vals))}
                for i, vals in enumerate(values[1:], start=2) if any(str(c).strip() for c in vals)]
        return {"headers": headers, "rows": rows}

    def set_cell(self, row: int, column: str, value: str, expect: Optional[str] = None,
                 company: Optional[str] = None, role: Optional[str] = None,
                 job_id: Optional[str] = None) -> int:
        """Write one existing cell, found by column header and by the row's company/role (rows move
        when the sheet is sorted). `expect` = the value the caller saw; if the sheet now holds
        something else, nothing is written. Returns the row written."""
        values = self._read()
        if not values:
            raise SheetError("The job sheet is empty.")
        headers = dedupe_headers(values[0])
        col = self._index(headers).get(_norm(column)) if column else None
        if col is None:
            raise SheetError(f"The job sheet has no {column!r} column.")
        row = self._resolve_row(values, headers, row, company, role, job_id)
        current = values[row - 1][col] if col < len(values[row - 1]) else ""
        if expect is not None and current.strip() != expect.strip():
            raise SheetError(f"That cell changed in the sheet since you loaded it (it now says {current[:60]!r}).")
        self._write({f"{col_letter(col)}{row}": value})
        logger.info("sheets: %s row %d %s set (%d chars)", self.tab, row, column, len(value))
        return row

    def _gid(self) -> int:
        """This tab's numeric sheetId (deleting a row needs it)."""
        self._values_api()
        try:
            meta = self._service.spreadsheets().get(spreadsheetId=self.sheet_id, fields="sheets.properties").execute()
        except Exception as exc:
            raise SheetError(f"I couldn't read the job sheet's tabs: {exc}") from exc
        for s in meta.get("sheets", []):
            if s.get("properties", {}).get("title") == self.tab:
                return int(s["properties"]["sheetId"])
        raise SheetError(f"The job sheet has no {self.tab!r} tab.")

    def move_row(self, row: int, dest: "JobSheet", status: str, company: str, role: str,
                 expect: Optional[str] = None, job_id: Optional[str] = None) -> int:
        """Move one job row to `dest` (the Applied tab) with Status = `status`: columns copied by
        header name, then the row deleted here. Already in `dest` (same company + role) → only its
        Status is set. The row is re-found by company + role before each step, so a sort in
        between never moves or deletes the wrong row. Returns the row number in `dest`."""
        values = self._read()
        if not values:
            raise SheetError("The job sheet is empty.")
        headers = dedupe_headers(values[0])
        row = self._resolve_row(values, headers, row, company, role, job_id)
        index = self._index(headers)
        cells = values[row - 1]
        job_id = job_id or (cells[index[_norm(COL_ID)]].strip()
                            if _norm(COL_ID) in index and index[_norm(COL_ID)] < len(cells) else "") or None
        if expect is not None:
            si = index.get(_norm(COL_STATUS))
            have = cells[si] if si is not None and si < len(cells) else ""
            if have.strip() != expect.strip():
                raise SheetError(f"That row's Status changed in the sheet since you loaded it (it now says {have[:40]!r}).")
        src = {_norm(h): (cells[i] if i < len(cells) else "") for i, h in enumerate(headers) if h}
        src[_norm(COL_STATUS)] = status

        dvalues = dest._read()
        if not dvalues:
            raise SheetError(f"The {dest.tab!r} tab has no header row.")
        dheaders = dedupe_headers(dvalues[0])
        try:
            at = dest._resolve_row(dvalues, dheaders, 0, company, role, job_id)
        except SheetError:
            at = 0
        if at:
            dest.set_cell(at, COL_STATUS, status, company=company, role=role, job_id=job_id)
        else:
            if job_id and _norm(COL_ID) not in {_norm(h) for h in dheaders}:
                dest._write({f"{col_letter(dest._width(dvalues))}1": COL_ID})   # the ID moves with the row
                dvalues = dest._read()
                dheaders = dedupe_headers(dvalues[0])
            line = [src.get(_norm(h), "") for h in dheaders]
            try:
                resp = dest._values_api().append(
                    spreadsheetId=dest.sheet_id, range=dest._a1(), valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS", body={"values": [line]}).execute()
            except Exception as exc:
                logger.error("sheets: append to %s failed: %s", dest.tab, exc)
                raise SheetError(f"I couldn't add the row to the {dest.tab} tab, so nothing was moved.") from exc
            m = re.search(r"!\$?[A-Z]+\$?(\d+)", (resp.get("updates") or {}).get("updatedRange", ""))
            at = int(m.group(1)) if m else len(dvalues) + 1

        values = self._read()
        row = self._resolve_row(values, dedupe_headers(values[0]), row, company, role, job_id)
        try:
            self._service.spreadsheets().batchUpdate(spreadsheetId=self.sheet_id, body={"requests": [
                {"deleteDimension": {"range": {"sheetId": self._gid(), "dimension": "ROWS",
                                               "startIndex": row - 1, "endIndex": row}}}]}).execute()
        except SheetError:
            raise
        except Exception as exc:
            logger.error("sheets: delete row %d from %s failed: %s", row, self.tab, exc)
            raise SheetError(f"It's in the {dest.tab} tab (row {at}) but I couldn't remove it from "
                             f"{self.tab}; delete row {row} there yourself.") from exc
        logger.info("sheets: moved %s row %d → %s row %d (%s)", self.tab, row, dest.tab, at, status)
        return at

    def _update_row(self, row: int, fields: dict[str, str], note: str,
                    company: Optional[str], role: Optional[str], job_id: Optional[str] = None) -> int:
        values = self._read()
        if not values:
            raise SheetError("The job sheet is empty.")
        headers = dedupe_headers(values[0])
        index = self._index(headers)
        cells: dict[str, str] = {}

        def column(name: str) -> int:
            if _norm(name) not in index:
                index[_norm(name)] = len(headers)
                cells[f"{col_letter(len(headers))}1"] = name.split("_")[0]
                headers.append(name)
            return index[_norm(name)]

        for name in fields:
            column(name)
        row = self._resolve_row(values, headers, row, company, role, job_id)
        for name, value in fields.items():
            cells[f"{col_letter(column(name))}{row}"] = value

        if note:
            updates_col = column(COL_UPDATES)
            current = values[row - 1] if row - 1 < len(values) else []
            existing = current[updates_col].strip() if updates_col < len(current) else ""
            entry = f"{self._today().isoformat()}: {note.strip()}"
            cells[f"{col_letter(updates_col)}{row}"] = f"{existing} | {entry}" if existing else entry

        self._write(cells)
        logger.info("sheets: row %d → %s%s", row, ", ".join(f"{k}={v[:40]!r}" for k, v in fields.items()),
                    f" ({note})" if note else "")
        return row

    @staticmethod
    def _width(values) -> int:
        """Columns in use: the header row or any longer data row (a new column goes after both)."""
        return max(len(v) for v in values) if values else 0

    def _resolve_row(self, values, headers, row, company, role, job_id=None) -> int:
        idc = self._index(headers).get(_norm(COL_ID))
        if job_id and idc is not None:
            at = lambda vals: vals[idc].strip().lower() if idc < len(vals) else ""
            if 2 <= row <= len(values) and at(values[row - 1]) == job_id.strip().lower():
                return row
            for i, vals in enumerate(values[1:], start=2):
                if at(vals) == job_id.strip().lower():
                    if row:
                        logger.info("sheets: job %s moved from row %d to row %d", job_id, row, i)
                    return i
        if company is None and role is None:
            if row < 2 or row > len(values):
                raise SheetError(f"Row {row} isn't a job row in the sheet.")
            return row
        index = self._index(headers)
        ci, ri = index.get(_norm(COL_COMPANY)), index.get(_norm(COL_ROLE))

        def holds(vals) -> bool:
            cell = lambda i: vals[i].strip() if i is not None and i < len(vals) else ""
            return ((company is None or _norm(cell(ci)) == _norm(company))
                    and (role is None or _norm(cell(ri)) == _norm(role)))

        if 2 <= row <= len(values) and holds(values[row - 1]):
            return row
        for i, vals in enumerate(values[1:], start=2):
            if holds(vals):
                logger.info("sheets: job moved from row %d to row %d", row, i)
                return i
        raise SheetError(f"I couldn't find {role or 'that job'} at {company or 'that company'} in the sheet anymore.")


def default_sheet() -> JobSheet:
    return JobSheet(config.MUSE_SHEET_ID, config.MUSE_SHEET_TAB)


def tab_sheet(tab: str) -> JobSheet:
    """The jobs tab or the Applied tab of the Muse sheet — no other tab."""
    if tab not in (config.MUSE_SHEET_TAB, config.MUSE_APPLIED_TAB):
        raise SheetError(f"{tab!r} isn't one of the job sheet's tabs.")
    return JobSheet(config.MUSE_SHEET_ID, tab)


def get_next_job(status: str = STATUS_NEW) -> Optional[dict]:
    return default_sheet().get_next_job(status)


def get_job(row: int) -> Optional[dict]:
    return default_sheet().get_job(row)


def set_status(row: int, status: str, note: str = "", company: Optional[str] = None,
               role: Optional[str] = None, job_id: Optional[str] = None) -> int:
    return default_sheet().set_status(row, status, note, company=company, role=role, job_id=job_id)


def set_fields(row: int, fields: dict[str, str], note: str = "", company: Optional[str] = None,
               role: Optional[str] = None, job_id: Optional[str] = None) -> int:
    return default_sheet().set_fields(row, fields, note, company=company, role=role, job_id=job_id)


def get_job_by_id(job_id: str) -> Optional[dict]:
    want = (job_id or "").strip().lower()
    return next((j for j in default_sheet().jobs() if want and j["job_id"].lower() == want), None)


# -- stable IDs -------------------------------------------------------------------

def _id_number(value: str) -> int:
    m = _JOB_ID.fullmatch((value or "").strip())
    return int(m.group(1)) if m else 0


def ensure_ids(tabs: Optional[list] = None) -> int:
    """Give every job row in the jobs tab and the Applied tab an ID ("J0042"), numbered after the
    highest ID in either tab, so a row keeps one ID wherever it moves. Adds the ID header (after
    the last used column) to a tab that has none. Existing IDs are never rewritten; a repeated ID
    (a copied row) is renumbered on its later copy. Returns how many IDs were written."""
    tabs = tabs if tabs is not None else [default_sheet(), tab_sheet(config.MUSE_APPLIED_TAB)]
    read = [(sh, sh._read()) for sh in tabs]
    top = 0
    for _, values in read:
        col = JobSheet._index(dedupe_headers(values[0])).get(_norm(COL_ID)) if values else None
        if col is not None:
            top = max([top] + [_id_number(vals[col]) for vals in values[1:] if col < len(vals)])
    seen: set = set()
    written = 0
    for sh, values in read:
        if not values:
            continue
        headers = dedupe_headers(values[0])
        col = sh._index(headers).get(_norm(COL_ID))
        cells: dict[str, str] = {}
        if col is None:
            col = sh._width(values)
            cells[f"{col_letter(col)}1"] = COL_ID
        for i, vals in enumerate(values[1:], start=2):
            if not any(str(v).strip() for v in vals):
                continue
            have = vals[col].strip() if col < len(vals) else ""
            if have and have.lower() not in seen:
                seen.add(have.lower())
                continue
            top += 1
            new = f"J{top:04d}"
            seen.add(new.lower())
            cells[f"{col_letter(col)}{i}"] = new
        if cells:
            sh._write(cells)
            ids = len([a1 for a1 in cells if a1 != f"{col_letter(col)}1"])
            written += ids
            logger.info("sheets: %s: %d IDs written", sh.tab, ids)
    return written


# -- find a job by what the user calls it ------------------------------------------

# Words that say nothing about WHICH job ("let's apply to the one from RTX").
_FILLER = frozenset(
    "a an the job jobs one ones role roles position posting opening application applications apply "
    "applying applied to for at from in on of with and or i me my let lets us please that this it is "
    "there sheet muse want wanna like company team new next open called named about can you do "
    "fill out start go head lets get".split())


def _tokens(text: str) -> list[str]:
    text = re.sub(r"(\w)-(\w)", r"\1\2", (text or "").lower())        # on-site → onsite
    return re.findall(r"[a-z0-9]+", text)


def _sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


def _score(query: list[str], job: dict) -> tuple:
    """(score, sure): how well the user's words name this job — company words count most, then
    role, then place. Misheard / misspelled words still count when close ("magnet" ≈ Magna), but
    only a word that matches (near-)exactly makes the match `sure`."""
    if any(_JOB_ID.fullmatch(t) and _id_number(t) == _id_number(job.get("job_id", "")) for t in query):
        return 100.0, True
    fields = ((3.0, _tokens(job.get("company", ""))), (2.0, _tokens(job.get("role", ""))),
              (1.0, _tokens(job.get("location", ""))))
    total, sure = 0.0, False
    for t in query:
        best = 0.0
        for weight, words in fields:
            for w in words:
                s = _sim(t, w)
                if s < 0.8 and not (s >= 0.7 and min(len(t), len(w)) >= 5):
                    continue                                          # long names may be misheard more
                sure = sure or (s >= 0.9 and weight >= 2.0 and min(len(t), len(w)) >= 3)
                s = s if s >= 0.8 else s * 0.8
                short = 0.5 if min(len(t), len(w)) <= 2 else 1.0      # "ai", "io" say little alone
                best = max(best, weight * s * short)
        total += best
    company = "".join(fields[0][1])
    for a, b in zip(query, query[1:]):                                # "single store" → SingleStore
        if len(company) >= 6 and _sim(a + b, company) >= 0.85:
            total, sure = total + 3.0, True
    return total, sure


def find_jobs(query: str, tabs: Optional[list] = None) -> list[dict]:
    """Jobs in both tabs ranked by how well `query` names them. Each match is the job dict plus
    "tab", "score" and "sure" (a word matched near-exactly, not only by sound-alike)."""
    words = [t for t in _tokens(query) if t not in _FILLER]
    if not words:
        return []
    tabs = tabs if tabs is not None else [default_sheet(), tab_sheet(config.MUSE_APPLIED_TAB)]
    out = []
    for sh in tabs:
        try:
            jobs = sh.jobs()
        except SheetError:
            if sh.tab == config.MUSE_SHEET_TAB:
                raise
            jobs = []                                                 # no Applied tab is fine
        for job in jobs:
            score, sure = _score(words, job)
            if score >= 1.6:
                out.append(dict(job, tab=sh.tab, score=round(score, 2), sure=sure))
    out.sort(key=lambda j: (-j["score"], j["tab"] != config.MUSE_SHEET_TAB, j["row"]))
    return out


def resolve_job(query: str, tabs: Optional[list] = None) -> dict:
    """What the user meant, as a decision code can act on:
      one      exactly one job in the jobs tab fits best          {"jobs": [job]}
      many     several fit about equally — ask which             {"jobs": [...]}
      applied  the best fits are only in the Applied tab         {"jobs": [...]}
      none     nothing fits                                      {"jobs": []}
    "sure" is False when the best fit came only from sound-alike words ("scale" ≈ Sallie): read
    it back as a question, never act on it. A bare "row 18" / "18" means that row of the jobs tab."""
    m = re.fullmatch(r"\s*(?:row|number|line)?\s*(\d{1,4})\s*", query or "", re.I)
    if m:
        job = (tabs[0] if tabs else default_sheet()).get_job(int(m.group(1)))
        return {"kind": "one", "sure": True, "jobs": [dict(job, tab=config.MUSE_SHEET_TAB, score=100.0, sure=True)]} \
            if job else {"kind": "none", "sure": False, "jobs": []}
    matches = find_jobs(query, tabs)
    if not matches:
        return {"kind": "none", "sure": False, "jobs": []}
    sure = matches[0]["sure"]
    near = [j for j in matches if j["score"] >= matches[0]["score"] * 0.85]
    open_ = [j for j in near if j["tab"] == config.MUSE_SHEET_TAB]
    if not open_:
        return {"kind": "applied", "sure": sure, "jobs": near}
    return {"kind": "one" if len(open_) == 1 else "many", "sure": sure, "jobs": open_}


def describe_match(job: dict) -> str:
    """'Software Engineer I (Onsite) at RTX (J0018)' — how a match is read back."""
    what = f"{job.get('role') or 'a role'} at {job.get('company') or 'that company'}"
    return what + (f" ({job['job_id']})" if job.get("job_id") else "")
