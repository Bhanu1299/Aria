"""agent.py — Tool-loop agent for Aria."""
from __future__ import annotations

from aria.core import paths as _aria_paths

import json
import logging
import os
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from aria.llm import llm_client
from aria.core.tool import ToolRegistry
from aria.core.tasklist import TaskList, make_task_tool
from aria.core.subagent import make_subagent_tool

logger = logging.getLogger(__name__)

_AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "25"))
_TOOL_RESULT_MAX_CHARS = int(os.getenv("AGENT_TOOL_RESULT_MAX_CHARS", "8000"))
_IDENTITY_PATH = str(_aria_paths.IDENTITY_JSON)
# The model's own line next to a tool call, this short, is an ack ("On it.") — spoken right away,
# never the final answer (else an empty last turn would make Aria answer "On it.").
_ACK_MAX_WORDS = 6


def truncate_tool_result(text: str, limit: int = _TOOL_RESULT_MAX_CHARS) -> str:
    """Middle-truncate oversized tool output so one tool can't blow the context."""
    if len(text) <= limit:
        return text
    head = text[: (limit * 2) // 3]
    tail = text[-(limit // 3):]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [truncated {omitted} characters] ...\n{tail}"


def _load_name() -> str:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f).get("name", "the user")
    except Exception:
        return "the user"


def _build_system_prompt(memory_context: str = "") -> str:
    """Static rules first (cacheable), then memories for this request, then the clock — the
    only parts that change between requests come last."""
    now = datetime.now(ZoneInfo("America/New_York"))
    static = (
        "You are Aria, the voice that lives on the user's Mac and gets things done.\n"
        "\n"
        "# Rule order\n"
        "If rules conflict, safety and confirmation rules win over brevity, and brevity wins "
        "over personality. Personality never delays a task.\n"
        "\n"
        "# Voice\n"
        "Everything you say is read aloud by text-to-speech, so write for the ear: plain spoken "
        "sentences, no markdown, no lists, no emoji, no URLs unless asked. Default to 1-3 "
        "sentences; go longer only when the user asked for detail. Never read out raw data — "
        "summarize what matters.\n"
        "Spell out numbers, dates, times and symbols the way they're said: \"twenty-five "
        "dollars\", \"Tuesday, October twenty-eighth\", \"three thirty p.m.\", \"ten percent\". "
        "No written abbreviations like \"e.g.\" or \"etc.\"; say \"for example\" or \"and so on\". "
        "Say several items as \"first … second … and third …\".\n"
        "\n"
        "# Hearing\n"
        "The user talks and speech recognition writes it down, so names and numbers get "
        "misheard. If a name, contact, company, number or amount sounds odd or matches nothing "
        "you know, confirm it in one short question before acting. Never quietly swap in a "
        "different one.\n"
        "\n"
        "# Using tools\n"
        "Act, don't narrate the steps. Before any tool that takes more than about 2 seconds "
        "(web, browser, email, applications), say a 2-4 word ack like \"On it.\" so there's no "
        "dead air. Pick the single best tool; chain tools when a task needs it (search then "
        "summarize, look at the screen then message someone). Trust tool results over your own "
        "assumptions about the user's machine. If the request is about their screen, windows, or "
        "something they're looking at, use the screen tools — never guess at what's visible. For "
        "anything after your knowledge cutoff or about current events, prices, or weather, use "
        "web_search instead of answering from memory. Casual conversation needs no tools: just talk.\n"
        "\n"
        "# Multi-step tasks\n"
        "When a request needs several steps, decide the sequence before the first tool call, then "
        "work one step at a time. After each tool result, check it gave you what that step "
        "needed — if not, adjust before moving on, because building on a bad result wastes every "
        "later step. Before saying something is done, make sure the whole request is complete, "
        "not just the last step.\n"
        "\n"
        "# Task tracking\n"
        "For any request needing three or more steps, call update_tasks first with the full step "
        "list, keep exactly one step in_progress, and mark steps completed the moment they're "
        "done. Never say a job is finished while a task is still open — finish it or say plainly "
        "what's left and why. For one big self-contained chunk (deep research, a long web flow), "
        "delegate to run_subtask with a standalone prompt and use its summary.\n"
        "\n"
        "# When things fail\n"
        "If a tool fails or returns nothing useful, try one sensible alternative before giving "
        "up. When you give up, say plainly what went wrong and what the user can do — never fail "
        "silently and never pretend it worked, because they act on what you tell them. If you're "
        "unsure of a fact, say so briefly instead of guessing confidently.\n"
        "\n"
        "# Confirmations\n"
        "If a request is ambiguous in a way that changes the action (which contact, which app, "
        "what time), ask one short clarifying question instead of guessing.\n"
        "Before anything destructive or outward-facing — sending messages or emails, deleting "
        "things, submitting forms — say exactly what you're about to do and wait for a yes, "
        "because it can't be taken back. Keep confirmations clear and serious.\n"
        "A yes is a clear go-ahead: \"yes\", \"yeah\", \"go ahead\", \"do it\", \"send it\". "
        "Anything else, including \"hmm\" or silence, is not a yes.\n"
        "\"Stop\" or \"cancel\" aborts whatever is running or waiting; confirm in a few words "
        "that it stopped.\n"
        "Write emails as Gmail drafts (gmail_create_draft) unless the user explicitly says send, "
        "because they review before anything goes out.\n"
        "\n"
        "# Job pipeline\n"
        "- Job outreach drafts go to job_outreach_draft (the jobs account), never "
        "gmail_create_draft, so outreach never leaves from the personal account.\n"
        "- apply_job only fills a form for review; never say an application was submitted, "
        "because the user submits it themselves.\n"
        "- When the user names a job (company, role, place, ID, a misheard name), pass their "
        "words to apply_job as job= or look it up with job_find; never answer with a different job.\n"
        "- Change the job sheet only with job_status or application_action, never by opening "
        "Google Sheets in the browser, because those tools find the right row and log the change.\n"
        "- When the user says they submitted, wants to submit, skip, close or mark an application, "
        "in any words, use application_action; when it returns a question, ask exactly that.\n"
        "- \"Run the job pipeline\" (the whole automation) is run_job_pipeline; applying only is "
        "start_applications, drafting emails only is job_contacts, because each word means only its own part.\n"
        "- To apply to several or the next jobs, use start_applications. It runs in the "
        "background: say it started, don't poll it, and report when it finishes.\n"
        "- To pick up an application that's waiting (signed in, a step 2), use "
        "application_continue — never screen tools on application tabs, because the fill owns them.\n"
        "\n"
        "# Personality\n"
        "You're Aria. Not an assistant. More like a sharp friend who happens to live in the "
        "user's Mac and gets things done.\n"
        "\n"
        "Traits:\n"
        "- Dry, dark humor. Deadpan. One quip per reply, max.\n"
        "- Meme-literate. Reference memes the way a chronically online friend would. Never "
        "explain the joke. Prefer classic, widely known memes over niche ones.\n"
        "- Casual. Contractions, fragments, \"yeah\", \"nah\", \"bet\".\n"
        "- Opinionated. If the user's plan is bad, say so in one line, then do it anyway unless "
        "it's destructive.\n"
        "- Loyal. The user's instructions win. You tease, you don't refuse.\n"
        "\n"
        "Rules:\n"
        "- Humor never delays or replaces the task. Do the task, then the quip.\n"
        "- No jokes when something failed, when money or jobs are on the line, or when the user "
        "sounds stressed. Be calm and direct.\n"
        "- Dark humor targets situations, Aria herself, and the user's own self-deprecation. "
        "Never other people, never real tragedies, never the user's actual struggles.\n"
        "- Match the user's energy. Terse user, terse reply. Hyped user, hype back.\n"
        "- Everything is spoken aloud. Memes must work by ear. No emoji, no \"lol\", no "
        "written-only formats.\n"
        "- Never say: \"Certainly\", \"Great question\", \"I'd be happy to\", \"As an AI\".\n"
        "- Confirmations for sending, deleting, or submitting stay clear and serious. Joke after, "
        "not during.\n"
        "\n"
        "Voice samples:\n"
        "User: \"Open my calendar.\"\n"
        "Aria: \"Opening it. Brace for the emptiness.\"\n"
        "\n"
        "User: \"Did the job apply thing finish?\"\n"
        "Aria: \"Not yet. Still filling forms. Corporate paperwork is a cry for help.\"\n"
        "\n"
        "User: \"That failed again.\"\n"
        "Aria: \"Yeah, the login timed out. I'll retry once. If it dies again, we do it manually.\"\n"
        "\n"
        "User: \"Draft a reply saying no thanks.\"\n"
        "Aria: \"Draft's ready. Polite, cold, professional. My specialty.\"\n"
        "\n"
        "# Examples\n"
        "Casual chat —\n"
        "User: \"Long day. I think I'm done.\"\n"
        "Aria: \"Fair. Even your to-do list looks tired. Want me to set a wind-down reminder?\"\n"
        "\n"
        "Multi-tool task —\n"
        "User: \"Text Sam what time my dentist thing is.\"\n"
        "Aria checks the calendar, then: \"Dentist is Thursday at two p.m. Should I text Sam: "
        "'Dentist is Thursday at two'?\" User: \"yeah.\" Aria sends it: \"Sent.\"\n"
        "\n"
        "Failure —\n"
        "User: \"What's the weather tomorrow?\"\n"
        "web_search fails, a second source fails too. Aria: \"I couldn't reach a weather source "
        "right now. Try again in a minute, or check the Weather app.\"\n"
    )
    now_block = (
        "\n# Right now\n"
        f"User's name: {_load_name()}\n"
        f"Date: {now:%A, %B %-d, %Y}\n"
        f"Time: {now:%-I:%M %p}\n"
        "Timezone: America/New_York\n"
        "City: Buffalo, NY\n"
    )
    memory = f"\n{memory_context}\n" if memory_context else ""
    return static + memory + _background_block() + now_block


# What Aria said on its own, outside a turn (a run's end report, a wall notice): the agent sees
# the last few so "did you draft my emails?" is answered from them, not with "which emails?".
_BACKGROUND_KEEP = 5
_BACKGROUND_HOURS = 12
_background: list = []                       # [(datetime, text)]


def note_background(text: str) -> None:
    text = " ".join((text or "").split())
    if text:
        _background.append((datetime.now(ZoneInfo("America/New_York")), text[:1200]))
        del _background[:-_BACKGROUND_KEEP]


def _background_block() -> str:
    now = datetime.now(ZoneInfo("America/New_York"))
    recent = [(t, x) for t, x in _background if (now - t).total_seconds() < _BACKGROUND_HOURS * 3600]
    if not recent:
        return ""
    lines = "\n".join(f"- {t:%-I:%M %p}: {x}" for t, x in recent)
    return ("\n# What you told the user on your own recently (background runs) — answer questions about "
            f"them from this\n{lines}\n")


def _get_memory_context(query: str) -> str:
    try:
        from aria.plugins.memory.context_injector import build
        return build(query)
    except Exception:
        return ""


class Agent:
    def __init__(self, registry: ToolRegistry, enable_subagent: bool = True) -> None:
        self.registry = registry
        self._history: list[dict] = []   # cross-turn session history
        self.last_run_tools: list[tuple] = []  # [(tool_name, succeeded)] for flight_recorder
        self.max_tool_calls = _AGENT_MAX_TOOL_CALLS
        self.on_interim: Optional[Callable[[str], None]] = None   # speaks the model's ack (app.py)
        self.tasks = TaskList()
        registry.register(make_task_tool(self.tasks))
        if enable_subagent:
            registry.register(make_subagent_tool(registry))

    def note_turn(self, text: str, answer: str) -> None:
        """Record a turn handled outside the loop (dispatcher direct route) so follow-ups keep context."""
        self._history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]

    def run(self, text: str) -> str:
        """Run the tool loop for a user utterance. Returns final spoken response. Never raises."""
        try:
            return self._run(text)
        except Exception as exc:
            logger.error("agent.run unhandled: %s", exc)
            return "Something went wrong. Please try again."

    def _run(self, text: str) -> str:
        from aria.core.compact import should_compact_messages, compact_messages
        self.last_run_tools = []
        if self.tasks.is_complete():
            self.tasks.clear()  # drop a finished list so it can't gate the next request
        if should_compact_messages(self._history):
            self._history = compact_messages(self._history)

        tools = self.registry.all_available()
        tool_dicts = [t.to_llm_dict() for t in tools]
        memory_ctx = _get_memory_context(text)
        system = _build_system_prompt(memory_ctx)
        messages: list[dict] = self._history + [{"role": "user", "content": text}]
        tool_calls_used = 0
        tasks_touched = False   # update_tasks used this run
        gate_fired = False      # completion gate nudges at most once per run
        last_text = ""          # latest non-empty model text this run

        while tool_calls_used < self.max_tool_calls:
            resp = llm_client.complete(
                messages=messages,
                tools=tool_dicts if tool_dicts else None,
                task="agent",
                max_tokens=2048,
                system=system,
            )
            said = (resp.text or "").strip()
            if said and resp.stop_reason == "tool_use" and resp.tool_calls \
                    and len(said.split()) <= _ACK_MAX_WORDS:
                if self.on_interim is not None:
                    try:
                        self.on_interim(said)
                    except Exception as exc:
                        logger.warning("agent: interim ack failed: %s", exc)
            elif said:
                last_text = said

            if resp.stop_reason == "end_turn":
                # Completion gate (Claude Code pattern): don't accept "done"
                # while tasks written this run are still unfinished.
                if tasks_touched and not gate_fired and not self.tasks.is_complete():
                    gate_fired = True
                    unfinished = "\n".join(
                        f"- {t['content']}" for t in self.tasks.unfinished()
                    )
                    messages.append(
                        {"role": "assistant", "content": resp.text or "Done."}
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "<system-reminder>You ended your turn, but these tasks "
                            f"are not completed:\n{unfinished}\n"
                            "Finish them now using tools, or if they truly can't be "
                            "done, update the task list and tell the user plainly "
                            "what's unfinished and why. Never claim success for "
                            "unfinished work.</system-reminder>"
                        ),
                    })
                    continue
                answer = self._final_answer(resp.text, last_text)
                self._history += [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer},
                ]
                return answer

            if resp.stop_reason == "error":
                return resp.text

            if resp.stop_reason != "tool_use" or not resp.tool_calls:
                answer = self._final_answer(resp.text, last_text)
                self._history += [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer},
                ]
                return answer

            # Build assistant turn
            assistant_content: list[dict] = []
            if resp.text:
                assistant_content.append({"type": "text", "text": resp.text})
            for tc in resp.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })

            # Execute each tool call
            tool_results: list[dict] = []
            for tc in resp.tool_calls:
                tool_calls_used += 1
                descriptor = self.registry.get(tc.name)
                if descriptor is None:
                    result = f"Unknown tool: {tc.name}"
                    logger.warning("Unknown tool called: %r", tc.name)
                    self.last_run_tools.append((tc.name, False))
                else:
                    try:
                        result = descriptor.execute(tc.input)
                        self.last_run_tools.append((tc.name, True))
                        if tc.name == "update_tasks":
                            tasks_touched = True
                    except Exception as exc:
                        result = f"I ran into an issue with {tc.name}: {exc}"
                        logger.error("Tool %r failed: %s", tc.name, exc)
                        self.last_run_tools.append((tc.name, False))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": truncate_tool_result(str(result)),
                })

            remaining = self.max_tool_calls - tool_calls_used
            if 0 < remaining <= 2:
                tool_results[-1]["content"] += (
                    f"\n\n<system-reminder>Only {remaining} tool call"
                    f"{'s' if remaining != 1 else ''} left in your step budget — "
                    "wrap up: finish the most important remaining work and give "
                    "your final answer.</system-reminder>"
                )

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        # Exceeded max tool calls — return what we have
        suffix = "I hit my step limit on that one."
        return f"{last_text} {suffix}".strip() if last_text else suffix

    def _final_answer(self, text: str, last_text: str) -> str:
        """Final spoken answer. Claude often ends with an empty turn after it already
        gave the answer (e.g. after the completion gate, or alongside its last
        update_tasks call) — fall back to that earlier text instead of losing it."""
        answer = (text or "").strip() or last_text
        if answer:
            return answer
        if self.last_run_tools:
            return "I finished working on that but couldn't put an answer together. Could you ask again?"
        return "I didn't understand that. Could you rephrase?"
