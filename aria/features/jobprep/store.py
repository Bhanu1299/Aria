"""features/jobprep/store.py — one short-lived aria.db connection per call.

    with store.conn() as c:
        c.execute(...)          # committed on success, rolled back on error, always closed
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import aria.state.db as db


@contextlib.contextmanager
def conn() -> Iterator:
    c = db.get_connection()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def rows(sql: str, args: tuple = ()) -> list:
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]
