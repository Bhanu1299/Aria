"""dispatch/rules.py — regex rules that map a command straight to a tool call.

Modeled on Siri's Enigma fast path: whole-command regexes (fullmatch, after
normalize()) with named slots, a target tool, and typed params. Precision over
recall — a miss costs nothing because the agent still answers; a wrong match
does the wrong thing.

patterns.json:
  never: regexes that send a command to the agent no matter what
  rules: [{id, regex, tool, params, priority}] — higher priority is tried first
  params values: literal, or "{slot}" / "{slot:type}" with type in
    text | int | duration (→ seconds) | when (→ ISO datetime) |
    task_when (key "*": expands to task + optional due)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Union

from aria.dispatch.parse import SlotError, normalize, parse_duration, parse_when, split_task_when

PATTERNS_PATH = Path(__file__).with_name("patterns.json")
_TEMPLATE = re.compile(r"^\{(?P<slot>\w+)(?::(?P<type>\w+))?\}$")


@dataclass
class RuleMatch:
    rule_id: str
    tool: str
    params: dict


@dataclass
class Escalate:
    """A rule matched but a slot didn't parse — the agent should handle it."""
    rule_id: str
    reason: str


@dataclass
class _Rule:
    id: str
    regex: re.Pattern
    tool: str
    params: dict
    priority: int


class RuleSet:
    def __init__(self, data: dict, now: Callable[[], datetime] = datetime.now) -> None:
        self._now = now
        self._never = [re.compile(p) for p in data.get("never", [])]
        rules = [
            _Rule(
                id=r["id"],
                regex=re.compile(r["regex"]),
                tool=r["tool"],
                params=r.get("params", {}),
                priority=int(r.get("priority", 50)),
            )
            for r in data.get("rules", [])
        ]
        # Stable sort: equal priorities keep file order.
        self.rules = sorted(rules, key=lambda r: -r.priority)

    @classmethod
    def load(cls, path: Path = PATTERNS_PATH) -> "RuleSet":
        with open(path) as f:
            return cls(json.load(f))

    def match(self, text: str) -> Union[RuleMatch, Escalate, None]:
        s = normalize(text)
        if not s or any(rx.search(s) for rx in self._never):
            return None
        for rule in self.rules:
            m = rule.regex.fullmatch(s)
            if not m:
                continue
            try:
                return RuleMatch(rule.id, rule.tool, self._build_params(rule.params, m))
            except SlotError as exc:
                return Escalate(rule.id, str(exc))
        return None

    def _build_params(self, spec: dict, m: re.Match) -> dict:
        slots = {k: v for k, v in m.groupdict().items() if v is not None}
        params: dict = {}
        for key, value in spec.items():
            t = _TEMPLATE.match(value) if isinstance(value, str) else None
            if not t:
                params[key] = value
                continue
            raw = slots.get(t.group("slot"))
            if raw is None:
                continue
            converted = self._convert(raw.strip(), t.group("type") or "text")
            if key == "*":
                params.update(converted)
            else:
                params[key] = converted
        return params

    def _convert(self, raw: str, kind: str):
        if kind == "text":
            return raw
        if kind == "int":
            return int(raw)
        if kind == "duration":
            return parse_duration(raw)
        if kind == "when":
            return parse_when(raw, self._now()).isoformat(timespec="minutes")
        if kind == "task_when":
            task, due = split_task_when(raw, self._now())
            return {"task": task, **({"due": due.isoformat(timespec="minutes")} if due else {})}
        raise ValueError(f"unknown slot type {kind!r}")
