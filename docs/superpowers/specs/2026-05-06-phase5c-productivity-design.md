# Phase 5C — Productivity Integrations Design

**Date:** 2026-05-06  
**Status:** Approved  
**Depends on:** Phase 5A (core engine)  
**Unlocks:** Gmail, Google Calendar, recurring scheduled tasks

---

## Goal

Aria reads and sends Gmail, reads and creates Google Calendar events, and runs scheduled tasks on a cron schedule — all by voice.

---

## Voice examples

- "What emails did I get from recruiters today?"
- "Reply to the Amazon recruiter saying I'm very interested"
- "What's on my calendar tomorrow?"
- "Block 2pm to 3pm tomorrow as deep work"
- "Every morning at 9, check my emails and tell me what's urgent"
- "List my cron jobs"

---

## Architecture

### Plugin structure

```
plugins/
  productivity/
    __init__.py          ← ProductivityPlugin(PluginBase)
    gmail.py             ← GmailClient
    gcal.py              ← CalendarClient
    cron.py              ← CronScheduler, CronJob
    setup.py             ← OAuth CLI helper
```

---

## Gmail

### Mechanism

`google-api-python-client` + `google-auth-oauthlib`. OAuth2 credentials stored at `~/.aria/credentials/google.json`. Scopes: `gmail.modify` (read + send + mark read) + `gmail.labels`.

### Setup (one-time)

```bash
python -m aria.setup gmail
```

Opens browser to Google OAuth consent screen. After approval, tokens saved locally. Refresh token auto-renews silently — user never needs to re-auth unless they explicitly revoke access.

### Tools registered

**`gmail_list`**
```
params:
  query: str            # Gmail search syntax: "from:recruiter is:unread"
  limit: int            # default 5, max 20
returns:
  list of {id, from, subject, date, snippet, is_unread}
```

**`gmail_read`**
```
params:
  email_id: str         # from gmail_list result
returns:
  {from, to, subject, date, body_text}   # plain text only, no HTML
```

**`gmail_send`**
```
params:
  to: str
  subject: str
  body: str
  cc: str               # optional
returns:
  "Sent." or error
```

**`gmail_reply`**
```
params:
  email_id: str         # reply threads correctly (In-Reply-To header)
  body: str
returns:
  "Sent." or error
```

### Smart query translation

The agent translates natural language to Gmail query syntax:
- "emails from recruiters today" → `from:recruiter after:{today} OR subject:opportunity after:{today}`
- "unread emails" → `is:unread`
- "emails about the Amazon interview" → `amazon interview`

The LLM handles this translation — `gmail_list` accepts raw Gmail syntax.

---

## Google Calendar

### Mechanism

Same OAuth credentials as Gmail. Additional scope: `calendar`. Uses `google-api-python-client` calendar v3 API.

### Tools registered

**`calendar_list`**
```
params:
  time_min: str         # ISO 8601 or natural: "today", "tomorrow", "this week"
  time_max: str         # optional
  calendar_id: str      # default "primary"
returns:
  list of {id, title, start, end, location, description}
```

**`calendar_create`**
```
params:
  title: str
  start: str            # ISO 8601
  end: str              # ISO 8601
  description: str      # optional
  location: str         # optional
returns:
  {event_id, url} or error
```

**`calendar_delete`**
```
params:
  event_id: str
returns:
  "Deleted." or error
```

### Time parsing

Agent converts "tomorrow at 3pm" to ISO 8601 before calling `calendar_create`. User's timezone read from system (`/etc/localtime` symlink) at startup, stored in `identity.json` as `timezone`.

---

## Cron Scheduler

### Mechanism

Pure local — no auth, no network. `croniter` parses cron expressions. Jobs stored as JSON files at `~/.aria/cron/<job-id>.json`. Background daemon thread in `CronScheduler` wakes every 60 seconds, checks due jobs, fires `agent.run(prompt)` in a separate daemon thread.

### CronJob schema

```json
{
  "id": "morning-email-check",
  "prompt": "Check my emails and tell me what's urgent",
  "schedule": "0 9 * * *",
  "delivery": "speak",
  "enabled": true,
  "created_at": "2026-05-06T10:00:00",
  "last_run_at": null,
  "last_run_status": null
}
```

`delivery` options:
- `"speak"` — result spoken via `speaker.say()`
- `"notify"` — macOS notification via `notifier.py`
- `"silent"` — run but don't surface result (background maintenance tasks)

### Tools registered

**`cron_create`**
```
params:
  name: str             # human label, used as job ID
  prompt: str           # what Aria should do when it fires
  schedule: str         # cron expression OR natural: "every day at 9am"
  delivery: str         # "speak" | "notify" | "silent"
returns:
  "Scheduled. Job ID: {id}" or error
```

**`cron_list`**
```
params: none
returns:
  list of {id, prompt, schedule_human, next_run, enabled}
```

**`cron_delete`**
```
params:
  job_id: str
returns:
  "Deleted." or error
```

**`cron_toggle`**
```
params:
  job_id: str
  enabled: bool
returns:
  "Paused." or "Resumed."
```

### Natural schedule parsing

"every morning at 9" → `0 9 * * *`  
"every Monday at 8am" → `0 8 * * 1`  
"every hour" → `0 * * * *`  
"every 30 minutes" → `*/30 * * * *`  

Agent translates natural language to cron expression before calling `cron_create`. `CronScheduler` accepts only valid cron expressions — validation via `croniter.is_valid()`.

### Isolated execution

Each cron job fires in a daemon thread with its own agent run — it does not share state with the main command loop. Result is delivered per the job's `delivery` setting. If the main loop is busy (speaking), cron result queues and delivers after current speech ends.

---

## Not-yet-setup behavior

If `~/.aria/credentials/google.json` doesn't exist, all Gmail and Calendar tools return:
`"Google isn't connected yet. Run: python -m aria.setup gmail"`

Cron tools always work — no auth required.

---

## Error handling

| Error | Response |
|---|---|
| OAuth token expired | Auto-refresh silently. If refresh fails: "Google auth expired. Run: python -m aria.setup gmail" |
| Gmail quota exceeded | "Gmail is rate-limited. Try again in a minute." |
| Event conflict | Return existing events in that slot, let user decide |
| Invalid cron expression | "I didn't understand that schedule. Try 'every day at 9am'." |
| Cron job fires, agent errors | Log error, set last_run_status="failed", notify via notifier.py |

---

## Testing

- `tests/test_gmail.py` — list, read, send, reply with mocked Google API
- `tests/test_gcal.py` — list, create, delete with mocked API + timezone handling
- `tests/test_cron.py` — job CRUD, schedule parsing, execution firing, delivery routing

---

## Dependencies

```
google-api-python-client>=2.100.0
google-auth-oauthlib>=1.1.0
google-auth-httplib2>=0.1.1
croniter>=1.4.1
```

---

## Definition of done

- [ ] `GmailClient` with list, read, send, reply
- [ ] `CalendarClient` with list, create, delete + timezone handling
- [ ] `CronScheduler` background thread + job CRUD + delivery routing
- [ ] One-time setup script `python -m aria.setup gmail`
- [ ] Natural schedule parsing (cron expression translation)
- [ ] `ProductivityPlugin` registered and loading in core engine
- [ ] Tests passing for Gmail, Calendar, Cron
