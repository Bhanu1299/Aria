"""plugins/productivity/cron.py — CronScheduler: local voice-driven cron jobs."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_JOBS_DIR = Path.home() / ".aria" / "cron"

_NATURAL_TO_CRON = [
    (r"every\s+hour",                      "0 * * * *"),
    (r"every\s+(\d+)\s+minutes?",          lambda m: f"*/{m.group(1)} * * * *"),
    (r"every\s+day\s+at\s+(\d+)([ap]m)?", lambda m: _daily(m)),
    (r"every\s+morning\s+at\s+(\d+)",      lambda m: f"0 {m.group(1)} * * *"),
    (r"every\s+morning",                   "0 9 * * *"),
    (r"every\s+evening",                   "0 18 * * *"),
    (r"every\s+night",                     "0 21 * * *"),
    (r"every\s+monday\s+at\s+(\d+)([ap]m)?", lambda m: _weekly(m, "1")),
    (r"every\s+tuesday\s+at\s+(\d+)([ap]m)?", lambda m: _weekly(m, "2")),
    (r"every\s+wednesday\s+at\s+(\d+)([ap]m)?", lambda m: _weekly(m, "3")),
    (r"every\s+thursday\s+at\s+(\d+)([ap]m)?", lambda m: _weekly(m, "4")),
    (r"every\s+friday\s+at\s+(\d+)([ap]m)?", lambda m: _weekly(m, "5")),
    (r"every\s+saturday\s+at\s+(\d+)([ap]m)?", lambda m: _weekly(m, "6")),
    (r"every\s+sunday\s+at\s+(\d+)([ap]m)?", lambda m: _weekly(m, "0")),
]


def _to_24h(hour: int, ampm: str | None) -> int:
    if ampm == "pm" and hour < 12:
        return hour + 12
    if ampm == "am" and hour == 12:
        return 0
    return hour


def _daily(m) -> str:
    hour = _to_24h(int(m.group(1)), m.group(2))
    return f"0 {hour} * * *"


def _weekly(m, dow: str) -> str:
    hour = _to_24h(int(m.group(1)), m.group(2))
    return f"0 {hour} * * {dow}"


def parse_schedule(text: str) -> str | None:
    """
    Convert natural language schedule to cron expression.
    Returns the cron string, or None if the input is already a valid cron expression.
    Raises ValueError if the schedule can't be parsed.
    """
    from croniter import croniter
    text = text.strip()
    if croniter.is_valid(text):
        return text
    lower = text.lower()
    for pattern, replacement in _NATURAL_TO_CRON:
        m = re.search(pattern, lower)
        if m:
            result = replacement(m) if callable(replacement) else replacement
            if croniter.is_valid(result):
                return result
    raise ValueError(f"I didn't understand that schedule. Try 'every day at 9am'.")


def _schedule_human(cron_expr: str) -> str:
    """Return a readable description of a cron expression."""
    try:
        from croniter import croniter
        itr = croniter(cron_expr, datetime.now())
        nxt = itr.get_next(datetime)
        return f"next: {nxt.strftime('%a %b %d at %H:%M')}"
    except Exception:
        return cron_expr


class CronJob:
    def __init__(self, id: str, prompt: str, schedule: str,
                 delivery: str = "speak", enabled: bool = True,
                 created_at: str = "", last_run_at: str | None = None,
                 last_run_status: str | None = None) -> None:
        self.id = id
        self.prompt = prompt
        self.schedule = schedule
        self.delivery = delivery
        self.enabled = enabled
        self.created_at = created_at or datetime.now().isoformat()
        self.last_run_at = last_run_at
        self.last_run_status = last_run_status

    def to_dict(self) -> dict:
        return {
            "id": self.id, "prompt": self.prompt, "schedule": self.schedule,
            "delivery": self.delivery, "enabled": self.enabled,
            "created_at": self.created_at, "last_run_at": self.last_run_at,
            "last_run_status": self.last_run_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CronJob":
        return cls(**{k: d.get(k) for k in
                      ("id", "prompt", "schedule", "delivery", "enabled",
                       "created_at", "last_run_at", "last_run_status")
                      if d.get(k) is not None or k in
                      ("last_run_at", "last_run_status")})

    def is_due(self) -> bool:
        """Return True if the job is due to run within the last 60 seconds."""
        if not self.enabled:
            return False
        try:
            from croniter import croniter
            now = datetime.now()
            # Compare previous tick to now within a 60s window
            itr = croniter(self.schedule, now)
            prev = itr.get_prev(datetime)
            return (now - prev).total_seconds() < 60
        except Exception:
            return False

    def next_run(self) -> str:
        try:
            from croniter import croniter
            itr = croniter(self.schedule, datetime.now())
            return itr.get_next(datetime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "unknown"


class CronScheduler:
    def __init__(self, agent=None, speaker=None, notifier=None) -> None:
        self._agent = agent
        self._speaker = speaker
        self._notifier = notifier
        self._lock = threading.Lock()
        self._stop = threading.Event()
        _JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        t = threading.Thread(target=self._loop, daemon=True, name="cron-scheduler")
        t.start()

    def stop(self) -> None:
        self._stop.set()

    def create(self, name: str, prompt: str, schedule: str,
               delivery: str = "speak") -> str:
        from croniter import croniter
        cron_expr = parse_schedule(schedule)
        job_id = re.sub(r"[^a-z0-9-]", "-", name.lower())[:40]
        job = CronJob(id=job_id, prompt=prompt, schedule=cron_expr, delivery=delivery)
        self._save(job)
        human = _schedule_human(cron_expr)
        return f"Scheduled. Job ID: {job_id} ({human})"

    def list_jobs(self) -> list:
        jobs = self._load_all()
        return [
            {
                "id": j.id,
                "prompt": j.prompt,
                "schedule_human": _schedule_human(j.schedule),
                "next_run": j.next_run(),
                "enabled": j.enabled,
            }
            for j in jobs
        ]

    def delete(self, job_id: str) -> str:
        path = _JOBS_DIR / f"{job_id}.json"
        if not path.exists():
            return f"No job found with ID {job_id!r}."
        path.unlink()
        return "Deleted."

    def toggle(self, job_id: str, enabled: bool) -> str:
        path = _JOBS_DIR / f"{job_id}.json"
        if not path.exists():
            return f"No job found with ID {job_id!r}."
        job = self._load(path)
        if job is None:
            return f"Could not read job {job_id!r}."
        job.enabled = enabled
        self._save(job)
        return "Resumed." if enabled else "Paused."

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.warning("cron-scheduler tick error: %s", exc)
            self._stop.wait(60)

    def _tick(self) -> None:
        for job in self._load_all():
            if job.is_due():
                t = threading.Thread(
                    target=self._run_job,
                    args=(job,), daemon=True,
                    name=f"cron-{job.id}",
                )
                t.start()

    def _run_job(self, job: CronJob) -> None:
        logger.info("cron: running job %r", job.id)
        try:
            if self._agent is None:
                return
            result = self._agent.run(job.prompt)
            job.last_run_at = datetime.now().isoformat()
            job.last_run_status = "ok"
            self._save(job)
            self._deliver(result, job.delivery)
        except Exception as exc:
            logger.error("cron: job %r failed: %s", job.id, exc)
            job.last_run_at = datetime.now().isoformat()
            job.last_run_status = "failed"
            self._save(job)
            if self._notifier:
                try:
                    self._notifier(f"Cron job {job.id!r} failed: {exc}")
                except Exception:
                    pass

    def _deliver(self, result: str, delivery: str) -> None:
        if delivery == "silent" or not result:
            return
        if delivery == "notify" and self._notifier:
            try:
                self._notifier(result)
            except Exception:
                pass
        elif delivery == "speak" and self._speaker:
            try:
                self._speaker.say(result)
            except Exception:
                pass

    def _save(self, job: CronJob) -> None:
        path = _JOBS_DIR / f"{job.id}.json"
        path.write_text(json.dumps(job.to_dict(), indent=2))

    def _load(self, path: Path) -> CronJob | None:
        try:
            return CronJob.from_dict(json.loads(path.read_text()))
        except Exception as exc:
            logger.warning("cron: failed to load %s: %s", path, exc)
            return None

    def _load_all(self) -> list:
        jobs = []
        for path in _JOBS_DIR.glob("*.json"):
            job = self._load(path)
            if job:
                jobs.append(job)
        return jobs
