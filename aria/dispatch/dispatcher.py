"""dispatch/dispatcher.py — the one path every command takes.

handle(text):
  1. pre-handlers   app-specific intercepts (e.g. jobs follow-ups), in order
  2. direct route   a RuleSet match → ToolDescriptor.execute(), 0 LLM tokens
  3. agent          everything else, and any direct route that can't run
                    (unparseable slot, missing tool, tool raised)

Direct turns are written into the agent's history so a follow-up like
"turn it back on" still has context. LLM usage is measured with aria.llm.meter
around the whole command, so calls made inside tools are counted too.
Never raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

from aria.dispatch.rules import Escalate, RuleMatch, RuleSet
from aria.llm import meter

logger = logging.getLogger(__name__)

PreHandler = Tuple[str, Callable[[str], Optional[str]]]


def _no_usage() -> dict:
    return {"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0}


@dataclass
class DispatchResult:
    answer: str
    route: str                  # "pre:<name>" | "direct:<rule_id>" | "agent"
    tools: list = field(default_factory=list)
    usage: dict = field(default_factory=_no_usage)
    escalated_from: str = ""    # rule id that matched but handed off to the agent


class Dispatcher:
    def __init__(
        self,
        registry,
        agent,
        rules: Optional[RuleSet],
        pre_handlers: Sequence[PreHandler] = (),
    ) -> None:
        self.registry = registry
        self.agent = agent
        self.rules = rules          # None = fast path off (ARIA_FASTPATH=0)
        self.pre_handlers = list(pre_handlers)

    def handle(self, text: str) -> DispatchResult:
        before = meter.snapshot()
        result = self._route(text)
        result.usage = meter.since(before)
        return result

    def _route(self, text: str) -> DispatchResult:
        for name, fn in self.pre_handlers:
            try:
                answer = fn(text)
            except Exception as exc:
                logger.error("pre-handler %s failed: %s", name, exc)
                answer = None
            if answer is not None:
                return DispatchResult(answer, f"pre:{name}")

        escalated_from = ""
        match = self._match(text)
        if isinstance(match, RuleMatch):
            answer = self._run_direct(match)
            if answer:
                self._note_turn(text, answer)
                return DispatchResult(answer, f"direct:{match.rule_id}", tools=[(match.tool, True)])
            escalated_from = match.rule_id
        elif isinstance(match, Escalate):
            escalated_from = match.rule_id
            logger.info("[dispatch] %s escalated: %s", match.rule_id, match.reason)

        answer = self.agent.run(text)
        return DispatchResult(
            answer,
            "agent",
            tools=list(getattr(self.agent, "last_run_tools", [])),
            escalated_from=escalated_from,
        )

    def _match(self, text: str):
        if self.rules is None:
            return None
        try:
            return self.rules.match(text)
        except Exception as exc:
            logger.error("rule match failed: %s", exc)
            return None

    def _run_direct(self, match: RuleMatch) -> str:
        tool = self.registry.get(match.tool)
        if tool is None or not tool.is_available():
            logger.warning("[dispatch] %s → tool %r unavailable, using agent", match.rule_id, match.tool)
            return ""
        try:
            return tool.execute(match.params) or ""
        except Exception as exc:
            logger.error("[dispatch] %s → %s raised: %s", match.rule_id, match.tool, exc)
            return ""

    def _note_turn(self, text: str, answer: str) -> None:
        note = getattr(self.agent, "note_turn", None)
        if note is not None:
            try:
                note(text, answer)
            except Exception as exc:
                logger.debug("note_turn failed: %s", exc)
