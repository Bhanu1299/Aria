"""plugins/system/__init__.py — SystemPlugin: on-device actions that need no LLM.

These are the targets of the dispatcher's direct routes (aria/dispatch) and are
also ordinary agent tools, so a command runs the same code whichever path picks it.

Tools:
  system_control — volume, brightness, dark mode, screenshot, lock, wifi, bluetooth, quit app
  clock          — current time or date
  calculate      — arithmetic (aria.skills.calculate, AST-based, never eval)
  timer_set / timer_status / timer_cancel — countdown timers (live while Aria runs)
  reminder_add   — Reminders.app, optional due time
  agenda         — today's / tomorrow's events from Calendar.app
"""
from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime

import aria.core.plugin as _plugin_base
from aria.core.tool import ToolDescriptor, ToolRegistry
from aria.dispatch.parse import spoken_when
from aria.plugins.system.timers import TimerManager

logger = logging.getLogger(__name__)

_ACTIONS = (
    "volume_up", "volume_down", "volume_set", "mute", "unmute",
    "brightness_up", "brightness_down", "dark_mode_on", "dark_mode_off", "dark_mode_toggle",
    "screenshot", "lock_screen", "wifi_on", "wifi_off", "bluetooth_on", "bluetooth_off",
    "quit_app",
)


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _lock_screen() -> str:
    # Ctrl-Cmd-Q is the system lock shortcut; needs Accessibility. Fall back to
    # sleeping the display, which locks when "require password immediately" is on.
    r = subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "q" using {control down, command down}'],
        capture_output=True, timeout=5,
    )
    if r.returncode != 0:
        subprocess.run(["pmset", "displaysleepnow"], capture_output=True, timeout=5)
    return "Locking your Mac."


def reminder_script(task: str, due: datetime | None) -> str:
    """AppleScript for a new reminder in the default list.

    The due date is built from components, never parsed from a string, so it
    doesn't depend on the Mac's locale or date format.
    """
    props = f'name:"{_esc(task)}"'
    date_lines = ""
    if due is not None:
        secs = due.hour * 3600 + due.minute * 60
        date_lines = (
            "    set d to current date\n"
            "    set day of d to 1\n"  # avoid month overflow (Jan 31 → Feb) while assigning
            f"    set year of d to {due.year}\n"
            f"    set month of d to {due.month}\n"
            f"    set day of d to {due.day}\n"
            f"    set time of d to {secs}\n"
        )
        props += ", due date:d, remind me date:d"
    return (
        'tell application "Reminders"\n'
        f"{date_lines}"
        "    tell default list\n"
        f"        make new reminder with properties {{{props}}}\n"
        "    end tell\n"
        "end tell"
    )


def _prop(kind: str, desc: str, **extra) -> dict:
    return {"type": kind, "description": desc, **extra}


class SystemPlugin(_plugin_base.PluginBase):

    def __init__(self, speaker=None) -> None:
        self._speaker = speaker
        self.timers = TimerManager(on_fire=self._timer_done)

    @classmethod
    def from_context(cls, ctx: _plugin_base.PluginContext) -> "SystemPlugin":
        return cls(speaker=ctx.speaker)

    def register(self, registry: ToolRegistry) -> None:
        for tool in (
            self._system_control_tool(), self._clock_tool(), self._calculate_tool(),
            self._timer_set_tool(), self._timer_status_tool(), self._timer_cancel_tool(),
            self._reminder_tool(), self._agenda_tool(),
        ):
            registry.register(tool)

    # ------------------------------------------------------------------

    def _timer_done(self, label: str) -> None:
        from aria.ui.notifier import send_notification
        text = f"Your {label} timer is done." if label else "Time's up. Your timer is done."
        send_notification("Aria — Timer", text)
        if self._speaker is not None:
            try:
                self._speaker.say(text)
            except Exception as exc:
                logger.error("timer speak failed: %s", exc)

    def _system_control_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            import aria.features.mac_controller as mc
            action = params.get("action", "")
            simple = {
                "volume_up": mc._volume_up, "volume_down": mc._volume_down,
                "mute": mc._volume_mute, "unmute": mc._volume_unmute,
                "brightness_up": mc._brightness_up, "brightness_down": mc._brightness_down,
                "dark_mode_on": mc._dark_mode_on, "dark_mode_off": mc._dark_mode_off,
                "dark_mode_toggle": mc._dark_mode_toggle, "screenshot": mc._screenshot,
                "wifi_on": mc._wifi_on, "wifi_off": mc._wifi_off,
                "bluetooth_on": mc._bluetooth_on, "bluetooth_off": mc._bluetooth_off,
                "lock_screen": _lock_screen,
            }
            try:
                if action in simple:
                    return simple[action]()
                if action == "volume_set":
                    return mc._volume_set(max(0, min(100, int(params.get("level", 50)))))
                if action == "quit_app":
                    name = (params.get("app_name") or "").strip()
                    return mc._app_quit(name) if name else "Which app should I quit?"
            except Exception as exc:
                logger.error("system_control %s failed: %s", action, exc)
                return "I couldn't do that on your Mac."
            return f"I don't know the system action {action!r}."

        return ToolDescriptor(
            name="system_control",
            description=(
                "Instant Mac system actions: volume up/down/set/mute/unmute, brightness up/down, "
                "dark mode on/off/toggle, screenshot, lock screen, wifi on/off, bluetooth on/off, "
                "quit an app. Prefer this over app_control for these actions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": _prop("string", "The system action", enum=list(_ACTIONS)),
                    "level": _prop("integer", "Volume 0-100, for volume_set"),
                    "app_name": _prop("string", "App to quit, for quit_app"),
                },
                "required": ["action"],
            },
            execute=execute,
        )

    def _clock_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            now = datetime.now()
            if params.get("what") == "date":
                return f"Today is {now:%A, %B} {now.day}."
            return f"It's {now:%-I:%M %p}."

        return ToolDescriptor(
            name="clock",
            description="The current local time or today's date.",
            input_schema={
                "type": "object",
                "properties": {"what": _prop("string", "time or date", enum=["time", "date"])},
                "required": ["what"],
            },
            execute=execute,
        )

    def _calculate_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.skills.calculate import handle
            expr = params.get("expression", "")
            expr = re.sub(r"(?<=\d),(?=\d{3})", "", expr)           # 1,200 → 1200
            expr = re.sub(r"\s*(?:x|×)\s*", " times ", expr)
            expr = expr.replace("÷", " divided by ")
            return handle(f"calculate {expr}")

        return ToolDescriptor(
            name="calculate",
            description="Exact arithmetic: + - * / powers and percentages, e.g. '15 percent of 80'.",
            input_schema={
                "type": "object",
                "properties": {"expression": _prop("string", "Arithmetic expression, words or symbols")},
                "required": ["expression"],
            },
            execute=execute,
        )

    def _timer_set_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            seconds = int(params.get("seconds", 0) or 0)
            if seconds <= 0:
                return "How long should the timer be?"
            return self.timers.start(seconds, params.get("label", ""))

        return ToolDescriptor(
            name="timer_set",
            description="Start a countdown timer. Aria speaks and notifies when it ends (while Aria is running).",
            input_schema={
                "type": "object",
                "properties": {
                    "seconds": _prop("integer", "Timer length in seconds"),
                    "label": _prop("string", "Optional name, e.g. 'pasta'"),
                },
                "required": ["seconds"],
            },
            execute=execute,
        )

    def _timer_status_tool(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="timer_status",
            description="How much time is left on running timers.",
            input_schema={"type": "object", "properties": {}},
            execute=lambda params: self.timers.status(),
        )

    def _timer_cancel_tool(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="timer_cancel",
            description="Cancel a timer by label, or all timers if no label is given.",
            input_schema={
                "type": "object",
                "properties": {"label": _prop("string", "Timer label; omit to cancel all")},
            },
            execute=lambda params: self.timers.cancel(params.get("label", "")),
        )

    def _reminder_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            import aria.features.mac_controller as mc
            task = (params.get("task") or "").strip()
            if not task:
                return "What should I remind you about?"
            due = None
            if params.get("due"):
                try:
                    due = datetime.fromisoformat(params["due"])
                except ValueError:
                    return "I couldn't understand that time."
            ok, _ = mc._run_applescript(reminder_script(task, due))
            if not ok:
                return "I couldn't add the reminder. Reminders may need permission."
            if due is None:
                return f"Added {task} to your reminders."
            return f"Okay, I'll remind you to {task} {spoken_when(due)}."

        return ToolDescriptor(
            name="reminder_add",
            description="Add a reminder to Reminders.app, optionally with a due time (alerts at that time).",
            input_schema={
                "type": "object",
                "properties": {
                    "task": _prop("string", "What to be reminded about, e.g. 'call mom'"),
                    "due": _prop("string", "Optional local due time, ISO 8601 (YYYY-MM-DDTHH:MM)"),
                },
                "required": ["task"],
            },
            execute=execute,
        )

    def _agenda_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            import aria.features.mac_controller as mc
            return mc._calendar_read(params.get("day") or "today")

        return ToolDescriptor(
            name="agenda",
            description="Today's or tomorrow's events from the Mac's Calendar app (all calendars).",
            input_schema={
                "type": "object",
                "properties": {"day": _prop("string", "today or tomorrow", enum=["today", "tomorrow"])},
            },
            execute=execute,
        )
