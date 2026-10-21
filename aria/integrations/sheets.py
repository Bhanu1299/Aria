"""integrations/sheets.py — the job sheet ("Muse") that feeds the apply pipeline.

Reads the header row on every call — no hard-coded column letters, so columns can
be reordered or added in the sheet. Duplicate header names are disambiguated in
order: "Outreach", "Outreach" → "Outreach", "Outreach_2".

    get_next_job(status="new") → dict | None     first matching row, top to bottom
    get_job(row) → dict | None
    set_status(row, status, note="", company=, role=)
    set_fields(row, {"Who to contact": ...}, note="", company=, role=)   outreach columns only

A blank Status cell counts as "new". Notes are appended to "Updates" as
"YYYY-MM-DD: note", joined with " | " — existing text is never overwritten.
Columns this module writes: Status, Updates, and the outreach columns in
OUTREACH_FIELDS (set_fields refuses any other). Everything else is the user's.

Rows move when the sheet is sorted, so set_status() re-finds the job by
(Company, Role) when the given row no longer holds it.

Raises SheetError (str is safe to speak) — callers turn it into a reply.

Always the jobs Google account (GOOGLE_JOBS_ACCOUNT) — never the personal one.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Callable, Optional

import aria.core.config as config
from aria.integrations import google_auth

logger = logging.getLogger(__name__)

COL_COMPANY = "Company"
COL_ROLE = "Role"
COL_LOCATION = "Location"
COL_ALIGNMENT = "Profile alignment /5"
COL_STATUS = "Status"
COL_APPLY_LINK = "Apply link"
COL_POST_LINK = "Post link"
COL_UPDATES = "Updates"
COL_WHO = "Who to contact"
COL_OUTREACH = "Outreach_2"   # the second "Outreach" column: the planned message + channel
OUTREACH_FIELDS = (COL_WHO, COL_OUTREACH)

# Status values Aria writes. "Applied" / "Not gonna apply" are the user's own.
STATUS_NEW = "new"
STATUS_IN_REVIEW = "In review"
STATUS_APPLIED = "Applied"
STATUS_SKIPPED = "Skipped"
STATUS_NEEDS_HUMAN = "Needs human"

_URL = re.compile(r"^https?://", re.I)


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
            "company": get(COL_COMPANY),
            "role": get(COL_ROLE),
            "location": get(COL_LOCATION),
            "alignment": get(COL_ALIGNMENT),
            "status": get(COL_STATUS),
            "apply_link": apply_link,
            "post_link": post_link,
            "url": url,
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
            if have == want or (want == STATUS_NEW and have == ""):
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
    ) -> int:
        """Write Status (and append note to Updates). Returns the row actually written."""
        return self._update_row(row, {COL_STATUS: status}, note, company, role)

    def set_fields(
        self,
        row: int,
        fields: dict[str, str],
        note: str = "",
        company: Optional[str] = None,
        role: Optional[str] = None,
    ) -> int:
        """Write outreach columns (OUTREACH_FIELDS only) and append note to Updates."""
        bad = [k for k in fields if _norm(k) not in {_norm(f) for f in OUTREACH_FIELDS}]
        if bad:
            raise SheetError(f"Aria doesn't write the {bad[0]!r} column.")
        return self._update_row(row, fields, note, company, role)

    def _update_row(self, row: int, fields: dict[str, str], note: str,
                    company: Optional[str], role: Optional[str]) -> int:
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
        row = self._resolve_row(values, headers, row, company, role)
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

    def _resolve_row(self, values, headers, row, company, role) -> int:
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


def get_next_job(status: str = STATUS_NEW) -> Optional[dict]:
    return default_sheet().get_next_job(status)


def get_job(row: int) -> Optional[dict]:
    return default_sheet().get_job(row)


def set_status(row: int, status: str, note: str = "", company: Optional[str] = None,
               role: Optional[str] = None) -> int:
    return default_sheet().set_status(row, status, note, company=company, role=role)


def set_fields(row: int, fields: dict[str, str], note: str = "", company: Optional[str] = None,
               role: Optional[str] = None) -> int:
    return default_sheet().set_fields(row, fields, note, company=company, role=role)
