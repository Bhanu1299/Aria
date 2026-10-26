"""llm/tasks.py — which GateLLM tier every LLM call in Aria uses. Tune it here, only here.

Rule: no LLM call decides routing. Each call site names a TASK; this table maps the
task to a GateLLM tier — or an explicit GateLLM model id when one model is the right one.
GateLLM then owns fallback chains, retries and cost.

Tiers, cheapest first (chains live in ~/GateLLM/config.yaml):
  small     classification, matching, short extraction      Groq / Gemini Flash-Lite / DeepSeek Flash
  strong    writing and escalations from small               DeepSeek v4 Pro / Gemini Flash
  vision    reading screenshots                              Gemini Flash / Groq Qwen
  agent     the tool-loop agent: DeepSeek v4 Pro, Claude Sonnet as its fallback (chosen on the
            87-case stress test — scripts/stress_test.py --agent-model <id> to re-measure)
  frontier  Claude: only where a task needs it (coding, hard computer-use escalation)

    llm_client.complete(messages, task="job_query_parse", ...)
    gateway_client().chat.completions.create(model=tier_for("screen_qa"), ...)
    llm_client.complete_checked("form_short_answer", messages, validate=...)   # small → frontier

Escalation (complete_checked): run the task's tier, validate the output in code
(schema, non-empty, option in the allowed list, length); on failure retry ONCE one tier
up (ESCALATE_TO) and count it in aria.llm.meter (→ flight log).
"""
from __future__ import annotations

SMALL = "small"
STRONG = "strong"
AGENT = "agent"
FRONTIER = "frontier"
VISION = "vision"

# complete_checked's one retry: the next tier up, not straight to Claude.
ESCALATE_TO = {SMALL: STRONG, STRONG: FRONTIER, VISION: FRONTIER, AGENT: FRONTIER}

TASK_TIERS: dict[str, str] = {
    # conversation
    "agent": AGENT,                       # the tool-loop agent (core/agent.py)
    "coder": FRONTIER,                    # agentic coding loop (features/coder.py)
    "history_compact": SMALL,             # core/compact.py
    "agent_summary": SMALL,               # features/agent_summary.py
    "prompt_suggest": SMALL,              # features/prompt_suggester.py
    # memory
    "memory_extract": SMALL,              # state/memory_extractor.py
    "session_notes": SMALL,               # state/session_notes.py
    "away_summary": SMALL,                # state/away_summary.py
    "auto_dream": SMALL,                  # state/auto_dream.py
    # summaries / briefings
    "summarize": SMALL,                   # features/summarizer.py
    "briefing": SMALL,                    # features/briefing.py
    "media_intent": SMALL,                # features/media.py
    "mac_text": SMALL,                    # features/mac_controller.py
    # screen + browser
    "mac_vision": VISION,                 # features/mac_controller.py
    "screen_qa": VISION,                  # screen/screen_qa.py
    "page_vision": VISION,                # web/vision.py
    "browser_dom_decide": SMALL,          # web/computer_use.py — DOM-first next action
    "browser_vision_decide": VISION,      # web/computer_use.py — thin DOM, screenshot
    # Explicit model, not a tier: escalation needs a strong model that accepts images.
    "browser_vision_escalate": "claude-sonnet-5-5",
    # jobs
    "job_query_parse": SMALL,             # features/jobs.py
    "job_listing_vision": VISION,         # features/jobs.py — read listings off a screenshot
    "form_short_answer": SMALL,           # features/ats/answers.py — unknown short form fields
    "form_field_match": SMALL,            # features/answer_bank.py — which bank KEY a question means
    "apply_page_situation": SMALL,        # features/ats/situations.py — what a non-form page is, what to click
    "form_free_text": STRONG,             # features/ats/answers.py — "why us" answers
    "outreach_company_fact": SMALL,       # features/outreach/company_fact.py — what a company solves, from its pages
    "outreach_email": STRONG,             # features/outreach/writer.py — cold email body
    "outreach_linkedin_note": STRONG,     # features/outreach/writer.py — LinkedIn connection note
}


def tier_for(task: str) -> str:
    """GateLLM tier (or model id) for a task. Unknown task → KeyError: add it to TASK_TIERS."""
    return TASK_TIERS[task]
