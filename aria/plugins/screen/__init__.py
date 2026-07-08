"""plugins/screen/__init__.py — ScreenPlugin: see and annotate the user's screen.

Tools:
  screen_look   — screenshot + vision answer ("what's on my screen", "explain
                  what's highlighted" — highlighted text is read exactly via
                  the Accessibility API and fed to the model)
  screen_point  — same, but also draws labeled boxes on the screen over the
                  relevant spots ("show me where X is", "point out the error")
"""
from __future__ import annotations

import logging

import aria.core.plugin as _plugin_base
from aria.core.tool import ToolDescriptor, ToolRegistry

logger = logging.getLogger(__name__)

_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The user's question about their screen, verbatim",
        },
    },
    "required": ["question"],
}


class ScreenPlugin(_plugin_base.PluginBase):

    def register(self, registry: ToolRegistry) -> None:
        registry.register(self._look_tool())
        registry.register(self._point_tool())

    def _look_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            try:
                import aria.screen.screen_qa as screen_qa
                return screen_qa.answer(params.get("question", ""))
            except Exception as exc:
                logger.error("screen_look failed: %s", exc)
                return "I couldn't look at the screen right now."

        return ToolDescriptor(
            name="screen_look",
            description=(
                "Look at the user's CURRENT screen and answer their question about it. "
                "Use for: 'what's on my screen', 'what does this error mean', 'read my screen', "
                "'explain what is highlighted', 'summarize this', 'what am I looking at', "
                "'translate the selected text'. If the question references highlighted or "
                "selected text, the exact selection is read and included automatically. "
                "Do NOT use for questions about the web or general knowledge."
            ),
            input_schema=_QUESTION_SCHEMA,
            execute=execute,
        )

    def _point_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            try:
                import aria.screen.screen_qa as screen_qa
                return screen_qa.explain_visual(params.get("question", ""))
            except Exception as exc:
                logger.error("screen_point failed: %s", exc)
                return "I couldn't draw on the screen right now."

        return ToolDescriptor(
            name="screen_point",
            description=(
                "Look at the user's screen, answer, AND draw labeled boxes directly on the "
                "screen over the relevant spots. Use when the user wants something located "
                "or pointed at visually: 'show me where X is', 'point out the error', "
                "'circle what I should click', 'where is the save button on my screen'. "
                "Prefer screen_look when they only want an explanation."
            ),
            input_schema=_QUESTION_SCHEMA,
            execute=execute,
        )
