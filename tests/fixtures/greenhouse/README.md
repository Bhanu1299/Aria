Saved public Greenhouse job pages (fetched 2026-10-19) for parser tests — no network in tests.

- conviva_embed.html       embed URL (`/embed/job_app?for=conviva&token=…`), required single-selects, required textarea, EEOC
- discord_jobboards.html   job-boards page with demographic_questions (single + multi select) and Location (City) autocomplete
- faraday_jobboards.html   minimal form: text fields, resume + cover letter only
- anthropic_jobboards.html pay_ranges ($190,000–$230,000) and "Why Anthropic?" textarea

Refresh: curl -sL -A "<browser UA>" -o <file> <url>
