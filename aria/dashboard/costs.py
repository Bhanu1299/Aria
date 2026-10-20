"""dashboard/costs.py — LLM spend from GateLLM's per-call log + Aria's flight log (read-only).

  GateLLM calls   GET /v1/calls when GateLLM is up, else its log file
                  (GATEWAY_DIR/logs/calls.jsonl): one entry per call with task / job /
                  tier / model / tokens / cost / attempts / escalated_from.
  flight log      ~/.aria/flight_log.jsonl: per-command and per-job-step totals. Used for
                  spend from before the call log existed (no per-call detail there).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

import aria.core.config as config

ESCALATION_TASK_HINT = "escalate"


def _read_jsonl(*paths: Path) -> list:
    out = []
    for p in paths:
        if p.exists():
            with p.open() as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    return out


def fetch_calls(base_url: Optional[str] = None, log_path: Optional[Path] = None) -> dict:
    """{"calls": [...], "via": "endpoint" | "file" | "none"} — never starts GateLLM."""
    url = (base_url or config.GATEWAY_URL).rstrip("/")
    try:
        r = httpx.get(url + "/v1/calls", params={"limit": 5000}, timeout=3)
        if r.status_code == 200:
            return {"calls": r.json().get("data", []), "via": "endpoint"}
    except httpx.HTTPError:
        pass
    path = log_path or Path(config.GATEWAY_DIR) / "logs" / "calls.jsonl"
    if path.exists() or path.with_name(path.name + ".1").exists():
        return {"calls": _read_jsonl(path.with_name(path.name + ".1"), path), "via": "file"}
    return {"calls": [], "via": "none"}


def fetch_flight(path: Optional[Path] = None) -> list:
    from aria.observability import flight_recorder
    return _read_jsonl(Path(path or flight_recorder._LOG_PATH))


def _cost(e: dict) -> float:
    return float(e.get("cost_usd") or 0.0)


def _job_key(label: str) -> str:
    return (label or "").strip() or "(no job)"


def summarize(calls: list, flight: list, now: datetime, applied_jobs: int = 0) -> dict:
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week0 = day0 - timedelta(days=now.weekday())
    ok = [c for c in calls if c.get("status") == "ok"]
    first_call = min((c["ts"] for c in calls), default=None)

    # Flight log usage from before the call log began (so nothing is counted twice).
    history = [e for e in flight if (e.get("usage") or {}).get("cost_usd")
               and (first_call is None or e.get("ts", 0) < first_call)]

    def spend_since(t0: Optional[datetime]) -> float:
        s = t0.timestamp() if t0 else 0
        return (sum(_cost(c) for c in ok if c["ts"] >= s)
                + sum(e["usage"]["cost_usd"] for e in history if e.get("ts", 0) >= s))

    spend = {"today": spend_since(day0), "week": spend_since(week0), "all": spend_since(None),
             "calls": len(ok), "errors": len(calls) - len(ok), "history_entries": len(history),
             "first_call": datetime.fromtimestamp(first_call) if first_call else None}

    # per job: calls from the call log, step totals from the flight log history
    jobs: dict = defaultdict(lambda: {"total": 0.0, "calls": [], "steps": []})
    for c in ok:
        if c.get("job"):
            j = jobs[_job_key(c["job"])]
            j["total"] += _cost(c)
            j["calls"].append(c)
    for e in history:
        if e.get("kind") == "job":
            label = f"{e.get('company') or '?'} | {e.get('role') or '?'} | row {e.get('row') if e.get('row') is not None else '-'}"
            j = jobs[label]
            j["total"] += e["usage"]["cost_usd"]
            j["steps"].append({"stage": e.get("stage"), "date": e.get("date"), "cost_usd": e["usage"]["cost_usd"],
                               "calls": e["usage"].get("calls", 0), "tokens_in": e["usage"].get("tokens_in", 0),
                               "tokens_out": e["usage"].get("tokens_out", 0)})
    job_rows = sorted(({"job": k, **v} for k, v in jobs.items()), key=lambda r: -r["total"])

    # per task
    tasks: dict = defaultdict(lambda: {"count": 0, "cost": 0.0, "models": Counter(), "tokens_in": 0, "tokens_out": 0})
    for c in ok:
        t = tasks[c.get("task") or "(unlabelled)"]
        t["count"] += 1
        t["cost"] += _cost(c)
        t["models"][c.get("model") or "?"] += 1
        t["tokens_in"] += int(c.get("tokens_in") or 0)
        t["tokens_out"] += int(c.get("tokens_out") or 0)
    task_rows = sorted(({"task": k, **v, "avg": v["cost"] / v["count"] if v["count"] else 0.0}
                        for k, v in tasks.items()), key=lambda r: -r["cost"])

    # escalations: Aria retried on a bigger tier (escalated_from), or GateLLM moved down the chain
    escalations = []
    for c in calls:
        failed = [a for a in c.get("attempts") or [] if a.get("outcome") not in ("ok", None)]
        if c.get("escalated_from"):
            escalations.append({"ts": c["ts"], "kind": "Aria: small → frontier", "task": c.get("task"),
                                "job": c.get("job"), "model": c.get("model"), "why": c["escalated_from"],
                                "cost_usd": _cost(c)})
        elif ESCALATION_TASK_HINT in (c.get("task") or ""):
            escalations.append({"ts": c["ts"], "kind": "Aria: explicit escalation model", "task": c.get("task"),
                                "job": c.get("job"), "model": c.get("model"),
                                "why": "the task's own escalation path (stronger model)", "cost_usd": _cost(c)})
        if failed or (c.get("fallbacks") or 0) > 0:
            why = "; ".join(f"{a.get('model')}: {a.get('outcome')}" for a in failed) or "fell back"
            escalations.append({"ts": c["ts"], "kind": "GateLLM fallback", "task": c.get("task"), "job": c.get("job"),
                                "model": c.get("model") or "(all failed)", "why": why, "cost_usd": _cost(c)})
    escalations.sort(key=lambda e: -e["ts"])

    priced = [r for r in job_rows if r["job"] != "(no job)" and r["total"] > 0]
    job_total = sum(r["total"] for r in priced)
    return {
        "spend": spend, "jobs": job_rows, "tasks": task_rows, "escalations": escalations,
        "cheapest": sorted(priced, key=lambda r: r["total"])[:3], "priciest": priced[:3],
        "job_total": job_total, "applied_jobs": applied_jobs,
        "per_applied": (job_total / applied_jobs) if applied_jobs else None,
    }
