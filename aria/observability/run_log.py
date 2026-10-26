"""observability/run_log.py — step-by-step log of job-pipeline runs, for the live dashboard.

One JSON event per line in ~/.aria/run_log.jsonl:
  run_start  {run, trigger}
  job        {run, company, role, row}                    the job this run works on
  decision   {run, company, role, row, picked, reasons}   why a row was picked or skipped
  step       {run, step, status, ms, detail}              status: running | ok | error | skipped
  run_end    {run, status, summary}

Steps of a pipeline run, in order (STEPS). A run that never reaches a step leaves it
out; the dashboard shows it as not part of that run. Writing never raises.

    run = run_log.start("apply_job")
    with run.step("read sheet") as s:
        ...; s.detail = "row 7"
    run.finish("ok", "form filled, 2 fields need you")
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

LOG_PATH = Path.home() / ".aria" / "run_log.jsonl"
MAX_BYTES = 5 * 1024 * 1024
STEPS = ["read sheet", "prepare resume", "find contacts", "find email", "fill form", "draft", "update sheet"]

_lock = threading.Lock()


def _write(event: dict) -> None:
    try:
        with _lock:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if LOG_PATH.exists() and LOG_PATH.stat().st_size > MAX_BYTES:
                os.replace(LOG_PATH, LOG_PATH.with_name(LOG_PATH.name + ".1"))
            with LOG_PATH.open("a") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), **event}) + "\n")
    except Exception as exc:
        logger.warning("run_log: write failed: %s", exc)


class _Step:
    def __init__(self) -> None:
        self.detail = ""
        self.status = "ok"


class Run:
    def __init__(self, trigger: str) -> None:
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.trigger = trigger
        self.finished = False
        _write({"type": "run_start", "run": self.id, "trigger": trigger})

    def job(self, company: str, role: str, row) -> None:
        _write({"type": "job", "run": self.id, "company": company, "role": role, "row": row})

    def decision(self, company: str, role: str, row, picked: bool, reasons: list) -> None:
        _write({"type": "decision", "run": self.id, "company": company, "role": role, "row": row,
                "picked": bool(picked), "reasons": [str(r)[:200] for r in reasons]})

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[_Step]:
        """Times the block; an exception marks the step error (and propagates)."""
        s = _Step()
        _write({"type": "step", "run": self.id, "step": name, "status": "running"})
        start = time.monotonic()
        try:
            yield s
        except Exception as exc:
            s.status, s.detail = "error", s.detail or f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _write({"type": "step", "run": self.id, "step": name, "status": s.status,
                    "ms": int((time.monotonic() - start) * 1000), "detail": str(s.detail)[:300]})

    def note(self, name: str, status: str, detail: str) -> None:
        """A finished step in one line (no timing): status ok | error | skipped."""
        _write({"type": "step", "run": self.id, "step": name, "status": status, "ms": 0, "detail": detail[:300]})

    def skip(self, name: str, why: str) -> None:
        _write({"type": "step", "run": self.id, "step": name, "status": "skipped", "ms": 0, "detail": why[:300]})

    def finish(self, status: str, summary: str = "") -> None:
        if not self.finished:
            self.finished = True
            _write({"type": "run_end", "run": self.id, "status": status, "summary": summary[:300]})


def start(trigger: str) -> Run:
    return Run(trigger)


def job_label(company: str, role: str, row) -> str:
    """The label GateLLM's per-call log stores for this job (see aria.llm.context.job)."""
    return f"{company or '?'} | {role or '?'} | row {row if row not in (None, '') else '-'}"


def read_runs(limit: int = 10, path: Optional[Path] = None) -> list:
    """The last `limit` runs, newest first, rebuilt from the event log (read-only)."""
    path = path or LOG_PATH
    runs: dict = {}
    order: list = []
    for p in (path.with_name(path.name + ".1"), path):
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                rid = ev.get("run")
                if not rid:
                    continue
                if rid not in runs:
                    runs[rid] = {"id": rid, "trigger": "", "started": ev["ts"], "ended": None, "status": "running",
                                 "summary": "", "job": None, "decisions": [], "steps": {}, "updated": ev["ts"]}
                    order.append(rid)
                r = runs[rid]
                r["updated"] = ev["ts"]
                kind = ev.get("type")
                if kind == "run_start":
                    r["trigger"], r["started"] = ev.get("trigger", ""), ev["ts"]
                elif kind == "job":
                    r["job"] = {k: ev.get(k) for k in ("company", "role", "row")}
                elif kind == "decision":
                    r["decisions"].append({k: ev.get(k) for k in ("company", "role", "row", "picked", "reasons")})
                elif kind == "step":
                    st = r["steps"].setdefault(ev["step"], {"step": ev["step"], "status": "running", "ms": None,
                                                            "detail": "", "started": ev["ts"]})
                    st["status"] = ev.get("status", st["status"])
                    if ev.get("status") != "running":
                        st["ms"], st["detail"] = ev.get("ms"), ev.get("detail", "")
                elif kind == "run_end":
                    r["status"], r["summary"], r["ended"] = ev.get("status", ""), ev.get("summary", ""), ev["ts"]
    out = []
    for rid in reversed(order[-limit:]):
        r = runs[rid]
        known = [r["steps"][s] for s in STEPS if s in r["steps"]]
        extra = [v for k, v in r["steps"].items() if k not in STEPS]
        r["steps"] = known + extra + [{"step": s, "status": "not in this run", "ms": None, "detail": ""}
                                      for s in STEPS if s not in r["steps"]]
        r["steps"].sort(key=lambda s: STEPS.index(s["step"]) if s["step"] in STEPS else len(STEPS))
        out.append(r)
    return out
