# Aria Phase 4b Design

**Date:** 2026-05-05  
**Status:** Approved  
**Branch:** main

## Overview

Phase 4b adds five capabilities to Aria: autonomous code execution (the anchor feature), Mac sleep prevention, contextual tips, away summary, and agent summary. Together they move Aria from a voice-controlled browser agent to a full voice-controlled coding + browsing agent.

---

## Feature 1: coder.py — Agentic Code Execution

### Purpose
User speaks a coding command. Aria plans and executes it fully autonomously — creates files, runs bash commands, edits code, installs dependencies, iterates on errors — then speaks a summary of what was done. Zero interruptions, zero confirmations.

### Architecture

**New file:** `coder.py`

**Tools available to Claude during execution:**

| Tool | Description |
|---|---|
| `bash(cmd)` | Run shell command in active project dir, capture stdout+stderr |
| `file_read(path)` | Read file content |
| `file_write(path, content)` | Create or overwrite file |
| `file_edit(path, old_str, new_str)` | Exact string replace (CC-style) |
| `glob(pattern)` | Find files matching pattern |
| `grep(pattern, path)` | Search file content |

**Execution loop:**
1. Receive transcribed voice command
2. Call Claude API (claude-sonnet-4-6) with tools + system prompt
3. On `tool_use` response: execute tool, update menubar label, feed result back
4. Loop until `stop_reason == "end_turn"`
5. `speaker.say()` final 1-2 sentence summary of what was accomplished

**Live menubar feedback** — label updates as each tool fires:
- `"CODING • reading files..."`
- `"CODING • running pip install..."`
- `"CODING • writing server.py..."`
- `"CODING • running tests..."`
- `"DONE"`

**Autonomy:** Fully autonomous. No mid-task confirmations. Claude recovers from errors by reading stderr and retrying.

**Limits:**
- 60s timeout per bash command
- 50 max tool calls per task (prevents infinite loops)
- No other restrictions — user is responsible for what they ask

### Project Management

**`PROJECTS_HOME`** in `config.py` — default: `~/Documents/trae_projects/`

**`active_project`** in `identity.json` — the directory all code tools operate in. Defaults to PROJECTS_HOME.

**`last_task_summary`** in `identity.json` — 1-sentence summary of the last completed task, written by agent_summary.py, read by away_summary.py on return.

**Intent: `"project"`** — handled by coder.py:
- "new project called X" → `mkdir PROJECTS_HOME/X`, set active_project = PROJECTS_HOME/X, say "Created project X and switched to it"
- "switch to X" → set active_project = PROJECTS_HOME/X, say "Switched to X"
- "list projects" → list dirs in PROJECTS_HOME, speak them

### Router changes
`router.py` gets two new intents:
- `"code"` — triggers `coder.handle()`
- `"project"` — triggers `coder.handle_project()`

---

## Feature 2: prevent_sleep.py

### Purpose
Prevent macOS from sleeping during long-running tasks (browser research, code execution).

### Architecture

**New file:** `prevent_sleep.py`

Uses macOS `caffeinate -i -t 300` subprocess:
- `start()` — spawns caffeinate, restarts every 4 min before timeout
- `stop()` — kills caffeinate process
- Reference-counted: multiple concurrent callers safe

**Wired in `main.py`:** `prevent_sleep.start()` before command processing, `prevent_sleep.stop()` in finally block.

---

## Feature 3: tips.py

### Purpose
After every 10th command Aria speaks a rotating helpful tip to teach the user new voice capabilities.

### Architecture

**New file:** `tips.py`

- `TIPS` — list of tip strings (e.g. "Try saying: new project called X")
- `maybe_speak_tip(command_count)` — speaks next tip if count % 10 == 0, cycles through list
- Command count sourced from `memory.py` `increment_command_count()` (already exists)

**Wired in `main.py`:** called async after `speaker.say()` completes.

---

## Feature 4: away_summary.py

### Purpose
When the user returns after 30+ minutes of inactivity, Aria speaks a 1-sentence recap of the last task before answering the new command.

### Architecture

**New file:** `away_summary.py`

- `identity.json` gets `last_active_at` (ISO timestamp)
- `check_and_speak(last_command_summary)` — if gap since `last_active_at` > 30 min, speak recap before processing new command
- `update_last_active()` — called after every command completes

**Wired in `main.py`:** `check_and_speak()` before routing, `update_last_active()` in finally block.

---

## Feature 5: agent_summary.py

### Purpose
After every completed task, generate and speak a natural 1-sentence summary of what was accomplished. Makes Aria feel more conversational and less mechanical.

### Architecture

**New file:** `agent_summary.py`

- `summarize_async(intent, raw_result)` — calls Claude API (small/fast model) with intent + raw result to produce a single clean spoken sentence
- Runs in daemon thread so it doesn't block next command
- Result spoken via `speaker.say()` — replaces the raw response

**Wired in `main.py`:** `agent_summary.summarize_async()` wraps every `speaker.say()` call post-command.

---

## Files Changed

| File | Change |
|---|---|
| `coder.py` | NEW — agentic code execution engine |
| `prevent_sleep.py` | NEW — caffeinate wrapper |
| `tips.py` | NEW — tip scheduler |
| `away_summary.py` | NEW — away recap |
| `agent_summary.py` | NEW — post-task summary |
| `config.py` | Add PROJECTS_HOME |
| `identity.json` | Add active_project, last_active_at |
| `router.py` | Add "code" and "project" intents |
| `main.py` | Wire all 5 features into command loop |

## Tests

Each new file gets a corresponding `tests/test_<file>.py` with 4-6 unit tests covering the core logic. `coder.py` tests mock the Claude API and tool execution.

---

## Success Criteria

- "Write a Flask hello world" → Aria creates the file, runs it, speaks "Done — created app.py and started Flask on port 5000"
- "New project called scraper" → folder created in PROJECTS_HOME, active_project updated, confirmed by voice
- Mac does not sleep during a 2-minute browser research task
- After 30-min gap: Aria recaps last task before answering new command
- Every 10th command: a tip is spoken
