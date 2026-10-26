"""features/resume/prepare.py — the per-job folder Aria makes when it prepares an application.

~/.aria/jobs/<JobID>-<company-slug>/   (made by Aria, never the user)
    posting.txt     the JD as read (instruction-like sentences stripped, see untrusted.py)
    resume.json     the tailored resume (the base, unchanged, when the base already scores ≥ 95)
    resume.pdf      rendered in the base layout, one page
    changes.md      added skills (what to brush up on), every reworded bullet old → new,
                    dropped items, ATS score before / after
    checks.json     ATS breakdown before / after, fact check, injection flags, timings
    cover_letter.pdf  written only when a form asks for one (cover_letter())

    prepare(job, jd_text, title="", use_llm=True, base=None, run=None) → Prepared
        Reuses the folder when the posting hasn't changed. On ANY failure: Prepared with
        resume_pdf = the base resume path and `fallback` set — the run never stops here.
    cover_letter(prepared, job, draft_text="") → path | ""
    cleanup(statuses, days=60) → [folders deleted]   skipped / closed jobs older than 60 days;
                                                      applied jobs are kept

Never raises.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from aria.features.ats import untrusted
from aria.features.resume import ats_score, model, render, tailor

logger = logging.getLogger(__name__)

JOBS_DIR = Path.home() / ".aria" / "jobs"
GOOD_ENOUGH = 95
KEEP_DAYS = 60
_APPLIED = re.compile(r"(?i)\bapplied\b|interview|offer|submitted")


@dataclass
class Prepared:
    dir: str = ""
    resume_pdf: str = ""            # what the form uploads
    resume_json: str = ""
    score_before: Optional[int] = None
    score_after: Optional[int] = None
    used_base: bool = False         # the base resume, unchanged (already ≥ 95, or a fallback)
    fallback: str = ""              # why the base PDF is used instead of a tailored one
    added_skills: list = field(default_factory=list)
    injection_flags: list = field(default_factory=list)
    reused: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.dir) and not self.fallback


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:n] or "job"


def job_dir(job: dict) -> Path:
    jid = re.sub(r"[^A-Za-z0-9_-]+", "", str(job.get("job_id") or job.get("id") or ""))
    if not jid:
        jid = f"row{job['row']}" if job.get("row") else \
            hashlib.sha1((job.get("url") or job.get("role") or "").encode()).hexdigest()[:8]
    return JOBS_DIR / f"{jid}-{slug(job.get('company', ''))}"


def _digest(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def _changes_md(job: dict, t: Optional[tailor.Tailored], before: Optional[int], after: Optional[int],
                used_base: bool) -> str:
    out = [f"# {job.get('role') or 'Role'} — {job.get('company') or 'Company'}", ""]
    out.append(f"ATS score: {before if before is not None else '?'} before → "
               f"{after if after is not None else '?'} after" + (" (base resume used unchanged)" if used_base else ""))
    out.append("")
    if t is None or used_base:
        out.append("The base resume already scores ≥ 95 for this job: nothing was changed.")
        return "\n".join(out) + "\n"
    out.append("## Skills added (brush up on these)")
    out += [f"- {s}" for s in t.added_skills] or ["- none"]
    out += ["", "## Headline", f"- old: {t.headline_before}", f"- new: {t.resume['headline']}", "",
            "## Reworded bullets"]
    for c in t.changes:
        out += [f"- **{c['id']}** ({c['how']})", f"  - old: {c['old']}", f"  - new: {c['new']}"]
    if not t.changes:
        out.append("- none")
    out += ["", "## Dropped to stay on one page"]
    out += [f"- {d}" for d in t.dropped] or ["- none"]
    if t.notes:
        out += ["", "## Notes"] + [f"- {n}" for n in t.notes]
    return "\n".join(out) + "\n"


def _note(run, msg: str, status: str = "ok") -> None:
    """One line in the run log (if this prep belongs to a run) and the Python log."""
    if run is not None:
        try:
            run.note("prepare resume", status, msg)
        except Exception as exc:
            logger.debug("prepare: run log note failed: %s", exc)
    logger.info("prepare: %s", msg)


def _repost(job: dict) -> Optional[dict]:
    """Records the posting (postings_seen) and says whether it looks reposted. Never raises."""
    try:
        from aria.features.jobprep import reposts
        reposts.record(job)
        return reposts.is_repost(job.get("company", ""), job.get("role", ""))
    except Exception as exc:
        logger.warning("prepare: repost check failed: %s", exc)
        return None


def prepare(job: dict, jd_text: str, title: str = "", use_llm: bool = True, base: Optional[dict] = None,
            run=None, base_pdf: Optional[str] = None) -> Prepared:
    started = time.monotonic()
    out = Prepared()
    if base_pdf is None:
        try:
            from aria.features.ats import planner
            base_pdf = planner.resume_path()
        except Exception:
            base_pdf = None
    out.resume_pdf = base_pdf or ""
    try:
        base = base or model.ensure_base(base_pdf)
        if not base:
            out.fallback = "no base resume (~/.aria/resume_base.json and no readable resume PDF)"
            _note(run, f"fallback to the base resume: {out.fallback}", "error")
            return out
        title = title or job.get("role", "")
        repost = _repost(job)
        cleaned, flags = untrusted.clean(jd_text or "")
        out.injection_flags = flags
        folder = job_dir(job)
        folder.mkdir(parents=True, exist_ok=True)
        out.dir = str(folder)
        digest = _digest(cleaned, title, json.dumps(base, sort_keys=True))
        checks_path = folder / "checks.json"
        try:
            old = json.loads(checks_path.read_text())
        except (OSError, ValueError):
            old = {}
        if old.get("digest") == digest and (folder / "resume.pdf").is_file() and (folder / "resume.json").is_file():
            out.resume_pdf, out.resume_json = str(folder / "resume.pdf"), str(folder / "resume.json")
            out.score_before, out.score_after = old.get("score_before"), old.get("score_after")
            out.used_base, out.added_skills, out.reused = bool(old.get("used_base")), old.get("added_skills", []), True
            return out
        (folder / "posting.txt").write_text(cleaned)

        with tempfile.TemporaryDirectory() as tmp:
            base_check = Path(tmp) / "base.pdf"
            render.render(base, base_check)
            before = ats_score.score(base_check, base, cleaned, title)
        out.score_before = before["score"]
        t = None
        problems: list = []
        if before["score"] >= GOOD_ENOUGH:
            final, after, out.used_base = model.normalize(base), before, True
        else:
            t = tailor.tailor(base, cleaned, title, use_llm=use_llm)
            problems = tailor.verify(base, t.resume)
            if problems or not render.fits(t.resume):
                why = "; ".join(problems) or "tailored resume doesn't fit one page"
                out.fallback = f"tailoring discarded: {why}"
                final, after, out.used_base = model.normalize(base), before, True
            else:
                final = t.resume
                out.added_skills = list(t.added_skills)
        render.render(final, folder / "resume.pdf")
        model.save(final, folder / "resume.json")
        if not out.used_base:
            after = ats_score.score(folder / "resume.pdf", final, cleaned, title)
            if after["score"] < before["score"]:            # never worse than the base
                render.render(base, folder / "resume.pdf")
                model.save(base, folder / "resume.json")
                final, after, out.used_base = model.normalize(base), before, True
                out.fallback = "tailored resume scored lower than the base"
        out.score_after = after["score"]
        out.resume_pdf, out.resume_json = str(folder / "resume.pdf"), str(folder / "resume.json")
        (folder / "changes.md").write_text(_changes_md(job, t, out.score_before, out.score_after, out.used_base))
        checks = {"digest": digest, "job": {k: job.get(k, "") for k in ("job_id", "company", "role", "url", "row")},
                  "title": title, "score_before": out.score_before, "score_after": out.score_after,
                  "used_base": out.used_base, "fallback": out.fallback, "added_skills": out.added_skills,
                  "before": before, "after": after, "fact_check": problems,
                  "injection_flags": flags, "repost": repost, "llm_used": bool(t and t.llm_used),
                  "dropped": t.dropped if t else [], "seconds": round(time.monotonic() - started, 2),
                  "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        checks_path.write_text(json.dumps(checks, indent=1, default=str))
        if out.fallback:
            _note(run, f"fallback to the base resume: {out.fallback}", "error")
        if not out.fallback:
            _note(run, f"ATS {out.score_before} → {out.score_after}"
                       + (" (base unchanged)" if out.used_base else f", {len(out.added_skills)} skills added"))
        if repost:
            _note(run, f"looks reposted: {repost['count']} postings since {repost['dates'][0]}")
        if flags:
            _note(run, f"posting had {len(flags)} instruction-like sentence(s), stripped: {flags[0][:120]}")
        return out
    except Exception as exc:
        logger.error("prepare(%s) failed: %s", job.get("company"), exc)
        out.resume_pdf = base_pdf or ""
        out.fallback = f"prepare failed: {exc}"
        _note(run, f"fallback to the base resume: {out.fallback}", "error")
        return out


def cover_letter(prepared: Prepared, job: dict, draft_text: str = "") -> str:
    """cover_letter.pdf in the job's folder: written from the tailored resume + posting, else
    rendered from `draft_text`. "" when neither worked."""
    from aria.features.resume import cover_letter as letter
    if not prepared or not prepared.dir:
        return ""
    folder = Path(prepared.dir)
    out = folder / "cover_letter.pdf"
    if out.is_file():
        return str(out)
    try:
        resume = model.load(prepared.resume_json) if prepared.resume_json else None
        if not resume:
            return ""
        jd = (folder / "posting.txt").read_text() if (folder / "posting.txt").is_file() else ""
        if letter.write(resume, jd, job, out) or (draft_text and letter.from_text(resume, draft_text, out)):
            return str(out)
    except Exception as exc:
        logger.warning("prepare: cover letter failed: %s", exc)
    return ""


def load_checks(folder) -> dict:
    try:
        return json.loads((Path(folder) / "checks.json").read_text())
    except (OSError, ValueError):
        return {}


def cleanup(statuses: dict, days: int = KEEP_DAYS, now: Optional[float] = None) -> list:
    """Deletes job folders older than `days` whose job isn't applied. statuses: {folder name
    or job id: sheet status}; a folder whose job isn't listed counts as closed."""
    now = now or time.time()
    gone = []
    if not JOBS_DIR.is_dir():
        return gone
    for folder in JOBS_DIR.iterdir():
        try:
            if not folder.is_dir() or now - folder.stat().st_mtime < days * 86400:
                continue
            status = statuses.get(folder.name, "")
            if not status:                                   # keyed by job id: "<id>-<company-slug>"
                status = next((v for k, v in statuses.items() if k and folder.name.startswith(f"{k}-")), "")
            if _APPLIED.search(status or ""):
                continue
            shutil.rmtree(folder)
            gone.append(folder.name)
        except OSError as exc:
            logger.warning("prepare.cleanup: %s: %s", folder, exc)
    return gone


def as_dict(p: Prepared) -> dict:
    return asdict(p)
