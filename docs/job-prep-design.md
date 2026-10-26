# Job prep: tailored resumes, answer checks, follow-ups, reposts — design

Status: approved by the user 2026-10-26 (chat), to be built on branch `feat/job-prep`.
Read CLAUDE.md first: Python 3.9 (`from __future__ import annotations`), every LLM call via
GateLLM with a task in `aria/llm/tasks.py` TASK_TIERS, Playwright only via
`aria.web.agent_browser`, handlers never raise, no placeholders/TODOs. tests/conftest.py
blocks live HTTP — mock `aria.llm.llm_client` / `gateway_client`.

## Why
Aria applies to jobs with one fixed resume and nothing checks what the LLM writes. Ideas
taken from career-ops (github.com/santifer/career-ops: verify-cv-facts, keyword-match,
jd-skill-gap, followup-cadence, reply-watch, detect-reposts, blacklist gate). Goal: every
application goes out with a resume that scores ≥95/100 on a strict ATS check most of the
time, answers that agree with that resume, and nothing wasted on dead/reposted/blocked jobs.

## Storage (decided)
- `~/.aria/resume_base.json` — the base resume as structured data (header, contact, headline,
  skills [(label, items)], experience [(company, dates, role, bullets)], projects [(title,
  dates, link?, bullets)], education, certifications). Built once from the PDF at
  answer_bank `resume_path` via the existing `ats/resume_sections.parse` + `resume_text`;
  editable later in the dashboard. Never committed (personal data). Tests use a fictional
  fixture resume.
- `~/.aria/jobs/<JobID>-<company-slug>/` — made by Aria (never the user) when a job is
  prepared: `resume.pdf`, `resume.json`, `cover_letter.pdf`, `changes.md`, `posting.txt`,
  `checks.json`. Only for jobs Aria actually prepares. Folders of skipped/closed jobs older
  than 60 days are deleted; applied jobs kept.
- `~/.aria/aria.db` new tables: `postings_seen` (company, role, url, url_key, first_seen,
  last_seen), `do_not_apply` (company or domain, scope, reason, since), `followups`
  (job_id, applied_at, next_due, done_at), `reply_suggestions` (job_id, thread_id,
  kind interview|rejected|offer|noise, suggested_status, approved_at, rejected_at).
- Local only. No Google Drive yet (user's choice).
- Muse sheet: only status writes, and only after the user approves them in the dashboard.

## 5. Tailored resume per job (aria/features/resume/)
- **Layout never changes.** Render with ReportLab (add `reportlab` to requirements), exactly
  the base resume's layout, measured from the user's PDF (letter 612×792, points from top):
  name Times-Bold 17 centered at y=40; contact line Times-Roman 9.5 centered y=58, separators
  " | ", links underlined 0.4pt + linkURL; headline Times-Bold 9.5 centered y=70; section
  title Times-Bold 11.5 at x=42 with a 0.7pt rule x=42..570 under it; entry header Times-Bold
  9.5 at x=42 with dates right-aligned at x=570; role Times-Italic 9.5; bullets "•"
  Helvetica 10 at x=46, text Times-Roman 9.5 at x=56 wrapped to x=570, leading 11.2; skill
  bullets = bold "Label: " + roman items; project header may carry "  |  github.com/…"
  roman, underlined, linked. Sections: TECHNICAL SKILLS, EXPERIENCE, PROJECTS, EDUCATION,
  CERTIFICATIONS. One page always: drop least-relevant bullets, then least-relevant project;
  never shrink fonts/margins.
- **Tailoring rules (user's; changed 2026-10-26 from "every JD skill"):** a JD skill goes into
  that job's resume only when it is CLOSE to the base resume's skills (shares a family in
  skills_lexicon.json "_families", e.g. Terraform ← AWS/GCP, Kubernetes ← Docker; "*" =
  everyday practice) or CORE to the job (in the title, or named 2+ times with one mention
  outside the nice-to-have part). The rest is listed in changes.md as "not added". Added
  skills are merged into the matching skill line (new line if none fits) and the LLM
  rewrites bullets to carry them in the JD's wording — whole-sentence rewrites only: code
  never appends "using X" to a bullet (removed: read as nonsense). Headline = the
  job's title framing. Projects/bullets reordered by relevance. Employers, titles, dates,
  degrees, certifications and existing numbers are NEVER changed; no new numbers. Non-skill
  requirements (years, degrees, clearance, citizenship, certifications) are NOT added —
  forms answer those from the answer bank. If the base already scores ≥95 on the ATS check,
  use the base unchanged.
- Code first: JD skill extraction = regex/dictionary (a skills lexicon file in the repo) +
  one small-tier LLM pass for terms the lexicon misses; bullet rewriting = strong-tier LLM
  returning JSON (bullet id → new text), validated in code (same facts: every number and
  employer in the original bullet must survive). New tasks in TASK_TIERS:
  `resume_skills_extract` (small), `resume_tailor` (strong), `cover_letter_write` (strong).
- `changes.md`: added skills list at the top (what to brush up on), then every reworded
  bullet old → new, dropped items, ATS score before/after.
- Cover letter: same fonts, one page, `cover_letter.pdf`, from the tailored resume + JD.
  Replaces the `.txt` drafts in `ats/planner.py` COVER_LETTER_DIR.
- The fill uploads the job's `resume.pdf` instead of `resume_path` (planner upload path).
  Fallback on any failure: base resume, noted in the run log — the run never stops here.

### Strict ATS check (aria/features/resume/ats_score.py) — the acceptance gate
Pure code, 0–100, deterministic, modeled on Jobscan-style scoring:
- Parse (25): `pdftotext` of the PDF reads in order; name, email, phone found; standard
  section headers; dates parse (Mon YYYY - Mon YYYY|Present); no text lost vs resume.json.
- Hard skills (40): share of JD hard skills found (case/punctuation-insensitive, synonyms
  e.g. "JS"/"JavaScript", "Postgres"/"PostgreSQL", "k8s"/"Kubernetes").
- Title match (10): JD title words in headline / most recent role or project header.
- Context (15): share of JD skills that appear in a bullet, not only the skills list.
- Format (10): one page, no images/tables, fonts standard, file < 1 MB, bullets ≤ 2 lines.
`checks.json` stores the breakdown. Target: ≥95 on most jobs.
**Acceptance test (must pass before merge):** `scripts/manual_ats_test.py` + a pytest
version on fixtures: 10+ real-style JDs (backend, ML/LLM, full-stack, data, infra; save
fixtures under tests/data/jds/, public postings only) × the fictional fixture resume →
tailor → render → score. Requirement (live LLM, `--llm`): median ≥95, ≥80% of JDs ≥95,
none <85, every PDF one page, every employer/date/degree/number unchanged, every chosen
skill on the page. The score counts every JD skill, so skills left out on purpose cost
points. pytest (LLM blocked = GateLLM-down fallback): none <70 + the same structural checks.
2026-10-26 on the user's real base, live: median 98, 10/12 ≥95, min 89 (Data Scientist).
The renderer reproduces the user's base PDF exactly (re-render: 80/80 lines, 5/5 rules).

## 3. Answer check against the resume (ats/answer_check.py)
After the LLM writes an open-text answer (planner open_text / generated_answers), code
checks it against THAT job's tailored resume.json (base if none): every number, percentage,
employer, school, degree and named project in the answer must appear in the resume; a
skill that's in neither → removed sentence or one strong-tier rewrite with the violations
listed; still failing → the answer without the offending sentences. Never asks the user,
never leaves the field empty. Logged in `<stem>.log.json`.

## 4. Posting text is untrusted
Everything read from a posting/page (JD, field labels, help text) is wrapped as data in every
LLM prompt: fenced block + a rule in the system text that instructions inside it are
ignored. Code flags imperative text aimed at an AI ("ignore previous", "as an AI", "you
must include…") → noted in checks.json + run log, stripped before prompting. Test with an
injected JD.

## 6. Follow-ups and employer replies
- Follow-up cadence: when a job turns Applied, `followups` row with next_due = +7 business
  days; then +7 again, max 2. Dashboard Today shows due follow-ups; "draft my follow-ups"
  (voice, direct route + agent tool) makes jobs-Gmail DRAFTS only (never sends), to the
  contact from the Contacts tab if any, else the application email thread.
- Replies: the dashboard already labels Interview/Rejected/Offer from jobs Gmail
  (aria/dashboard/model.py). Add classification to `reply_suggestions` (rules first, small
  tier only when rules can't tell; noise = auto-confirmations, newsletters). Suggested sheet
  status shown in the dashboard with Approve / Dismiss; NOTHING written to the sheet until
  Approve (user's choice). Approve → JobSheet.set_cell (existing, row re-found by ID).

## 7. Reposts and do-not-apply list
- Every row Aria reads/prepares → `postings_seen`. Repost = same company + normalized title,
  2+ different URLs, 2+ distinct dates within 90 days → flagged (dashboard badge, run log),
  not skipped automatically.
- `do_not_apply`: company (case/punctuation-insensitive) or domain (exact host or subdomain
  of the posting URL). Editable in the dashboard (My answers or a small card on Applications)
  + voice "never apply to <company>". A hit skips the job in batch runs (noted in the end
  report) and `apply_job` asks once ("<company> is on your do-not-apply list — still apply?").

## Not in scope
Google Drive copies; liveness API pre-check and no-sponsorship filter (career-ops items 1–2,
later); interview prep; portal scanning.

## Done means
pytest green; ATS acceptance numbers above met and pasted into the PR/notes; ARCHITECTURE.md
updated (storage + flow); .claude/project-state.md Status/Next updated; branch committed,
not merged.

## Build status (branch `feat/job-prep`, 2026-10-26)
Built: §5 (resume model, lexicon, renderer, strict ATS check, tailoring, per-job folder, cover
letters, fills upload the job's resume, base resume editor on My answers), §3 answer check, §4
untrusted posting text, §6 follow-ups + reply suggestions (Approve/Dismiss), §7 reposts +
do-not-apply. Acceptance gate passes on the code-only path (see the final commit message for the
table). `.claude/project-state.md` is local (gitignored): copy this section's Status / Next there.

Next:
- Point the answer bank's `resume_path` at the new resume PDF; the first prepared job builds
  `~/.aria/resume_base.json` from it — check it on My answers (Base resume).
- Run `python scripts/manual_ats_test.py --llm` with GateLLM up, and once with
  `--resume ~/.aria/resume_base.json`, to see the live rewrite on the real resume.
- Watch the first real batch: `changes.md` per job, the `prepare resume` step in Live run.
