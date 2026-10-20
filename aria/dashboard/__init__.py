"""dashboard — local, read-only job pipeline page: every job and its stage, end to end.

    python -m aria.dashboard [--port 8787] [--no-browser]

Serves on 127.0.0.1 only. Each page load reads every source fresh (Gmail and the
sheet are cached for 5 minutes) and writes nothing anywhere.

  sources.py  one loader per source (Muse sheet, aria.db, review files, Coldstart,
              Gmail) — each returns Source(ok, data, error) and never raises
  model.py    pure: match records into jobs, stage per job, funnel, needs-me, week, cost
  render.py   HTML; a section whose source is down says "no data" instead of failing
"""
