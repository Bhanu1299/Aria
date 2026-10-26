"""integrations/header_sheet.py — any sheet tab read and written by header name, never by position.

Row 1 is the header. Every call re-reads it, so columns can be reordered, inserted or
renamed (case and spacing don't matter) without code changes.

    sheet = HeaderSheet(sheet_id, "Contacts", writable=CONTACTS_WRITABLE)
    sheet.records() → [{"_row": 2, "Company": "Acme", ...}]   blank rows skipped
    sheet.update(row, {"Email": "pat@acme.com"}, match={"Company": "Acme", "Person": "Pat Lee"})

update() refuses a header the sheet doesn't have (it never invents a column) and, when
`writable` is set, any header outside it. `match` guards against rows that moved since
they were read (the sheet was sorted or a row inserted): the row is re-found by those
cells, and SheetError is raised when nothing holds them anymore.

I/O, auth (the jobs Google account) and SheetError come from integrations/sheets.py.
"""
from __future__ import annotations

from typing import Iterable, Optional

from aria.integrations.sheets import JobSheet, SheetError, _norm, col_letter, dedupe_headers

ROW = "_row"


class HeaderSheet(JobSheet):
    def __init__(self, sheet_id: str, tab: str, service=None, writable: Optional[Iterable[str]] = None) -> None:
        super().__init__(sheet_id, tab, service=service)
        self.writable = {_norm(h) for h in writable} if writable is not None else None

    def _layout(self) -> tuple[list[str], list[list[str]]]:
        values = self._read()
        if not values:
            raise SheetError(f"The {self.tab} tab is empty.")
        return dedupe_headers(values[0]), values

    @staticmethod
    def _record(headers: list[str], vals: list[str], row: int) -> dict:
        cells = list(vals) + [""] * (len(headers) - len(vals))
        rec = {h: cells[i].strip() for i, h in enumerate(headers) if h}
        rec[ROW] = row
        return rec

    def headers(self) -> list[str]:
        return [h for h in self._layout()[0] if h]

    def records(self) -> list[dict]:
        headers, values = self._layout()
        return [self._record(headers, vals, i) for i, vals in enumerate(values[1:], start=2)
                if any(str(v).strip() for v in vals)]

    def update(self, row: int, fields: dict, match: Optional[dict] = None) -> int:
        """Write {header: value} into one row. Returns the row actually written."""
        if not fields:
            return row
        headers, values = self._layout()
        index = {_norm(h): i for i, h in enumerate(headers) if h}
        for name in list(fields) + list(match or {}):
            if _norm(name) not in index:
                raise SheetError(f"The {self.tab} tab has no {name!r} column.")
        if self.writable is not None:
            bad = [k for k in fields if _norm(k) not in self.writable]
            if bad:
                raise SheetError(f"Aria doesn't write the {bad[0]!r} column.")
        row = self._find(values, index, row, match)
        self._write({f"{col_letter(index[_norm(k)])}{row}": str(v) for k, v in fields.items()})
        return row

    @staticmethod
    def _find(values: list[list[str]], index: dict, row: int, match: Optional[dict]) -> int:
        def holds(vals) -> bool:
            cell = lambda i: vals[i].strip() if i < len(vals) else ""
            return all(_norm(cell(index[_norm(k)])) == _norm(str(v)) for k, v in match.items())

        if not match:
            if row < 2 or row > len(values):
                raise SheetError(f"Row {row} isn't a row in the sheet.")
            return row
        if 2 <= row <= len(values) and holds(values[row - 1]):
            return row
        for i, vals in enumerate(values[1:], start=2):
            if holds(vals):
                return i
        raise SheetError("That row isn't in the sheet anymore: " + ", ".join(f"{k}={v}" for k, v in match.items()))


def get(rec: dict, name: str) -> str:
    """rec[name], matching the header case- and space-insensitively. '' when absent."""
    if name in rec:
        return rec[name]
    want = _norm(name)
    return next((v for k, v in rec.items() if k != ROW and _norm(k) == want), "")
