"""
plan_context.py — Aria agentic planner: PlanContext dataclass.

Holds the full context of a multi-step plan: goal, steps, results, progress.
Serializable to/from dict for memory persistence and inter-module communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanContext:
    """
    Represents a multi-step plan being executed by the agentic planner.

    Fields:
      goal (str)           — The user's high-level goal.
      steps (list[dict])   — List of step definitions, each with id, description, intent_type, params, result_key, depends_on.
      results (dict)       — Maps result_key → result value for each completed step.
      current_step (int)   — 0-indexed position of the next step to execute.
      retry_count (int)    — Number of times the current step has been retried.
    """

    goal: str
    steps: list[dict]
    results: dict[str, str] = field(default_factory=dict)
    current_step: int = 0
    retry_count: int = 0

    def to_dict(self) -> dict:
        """Serialize PlanContext to dict for storage/transmission."""
        return {
            "goal": self.goal,
            "steps": self.steps,
            "results": self.results,
            "current_step": self.current_step,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanContext":
        """Deserialize PlanContext from dict."""
        return cls(
            goal=d["goal"],
            steps=d["steps"],
            results=d.get("results", {}),
            current_step=d.get("current_step", 0),
            retry_count=d.get("retry_count", 0),
        )
