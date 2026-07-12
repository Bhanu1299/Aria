# Extraction roadmap — Claude Code / OpenClaw patterns for Aria

Source trees studied 2026-07-08: `claude-code-main.zip` (full Claude Code src)
and `openclaw-main` (personal-assistant gateway). Ported so far (aria/core):
`tasklist.py` (TodoWrite), completion gate + budget wrap-up reminders +
tool-result truncation (agent.py), `subagent.py` (Task/AgentTool).

## Worth porting next (in rough value order)

1. **Commitments (OpenClaw `src/commitments/`)** — after each reply, a cheap
   background LLM pass extracts promises Aria made ("I'll remind you at 4",
   "I'll keep looking") into a store with due times; the heartbeat delivers
   them when due. Fits Aria: extractor piggybacks on memory_extractor's async
   path, delivery via existing cron plugin + speaker.
2. **Heartbeat (OpenClaw `src/infra/heartbeat-*`)** — periodic quiet agent run
   with cooldown + active-hours + "only speak if something needs attention"
   policy. Aria has cron; missing the visibility/cooldown policy layer.
3. **Todo-staleness reminder (CC `utils/attachments.ts` todo_reminder)** — if
   N assistant turns pass with an unfinished list and no update_tasks call,
   inject the rendered list as a system-reminder. Only matters if runs get
   longer than ~25 calls.
4. **Queued interruptions (CC queued messages)** — let a wake-word utterance
   land mid-run and be injected as a user message at the next loop iteration
   instead of being dropped.
5. **Skills-as-prompts (CC SkillTool)** — markdown playbooks (e.g. "apply to a
   job") loaded on demand as context, not code. Natural upgrade for
   aria/skills/skill_loader.py (currently orphaned).
6. **Hooks (CC PreToolUse/PostToolUse)** — user-defined shell/py callbacks
   around tool calls; useful for confirmation policies on messaging tools.

## Explicitly not porting

- Permission-mode UI, IDE/terminal rendering, MCP client, worktrees — terminal
  concerns with no voice equivalent.
- OpenClaw channel gateway (Telegram/WhatsApp) — Aria is local-first; the
  messaging plugins already cover sending.
