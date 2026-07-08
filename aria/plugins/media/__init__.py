"""plugins/media/__init__.py — MediaPlugin: unified music control."""
from __future__ import annotations

import logging

import aria.core.plugin as _plugin_base
from aria.core.tool import ToolDescriptor, ToolRegistry

logger = logging.getLogger(__name__)

_NO_BACKEND_MSG = "No music backend is configured."


def _str_prop(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _int_prop(desc: str) -> dict:
    return {"type": "integer", "description": desc}


class MediaPlugin(_plugin_base.PluginBase):

    def register(self, registry: ToolRegistry) -> None:
        from aria.plugins.media.router import MediaRouter
        router = MediaRouter()

        registry.register(self._play_tool(router))
        registry.register(self._pause_tool(router))
        registry.register(self._resume_tool(router))
        registry.register(self._skip_tool(router))
        registry.register(self._previous_tool(router))
        registry.register(self._volume_tool(router))
        registry.register(self._now_playing_tool(router))

    def _play_tool(self, router) -> ToolDescriptor:
        def execute(params: dict) -> str:
            backend = router.get_backend()
            if backend is None:
                return _NO_BACKEND_MSG
            return backend.play(params.get("query", ""))

        return ToolDescriptor(
            name="music_play",
            description=(
                "Play music. Use for: 'play [artist/song/playlist]', 'play something chill', "
                "'put on some music'. Omit query to resume paused playback."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": _str_prop("Artist, song, album, or playlist name (optional)"),
                },
                "required": [],
            },
            execute=execute,
        )

    def _pause_tool(self, router) -> ToolDescriptor:
        def execute(params: dict) -> str:
            backend = router.get_backend()
            return backend.pause() if backend else _NO_BACKEND_MSG

        return ToolDescriptor(
            name="music_pause",
            description="Pause music playback.",
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )

    def _resume_tool(self, router) -> ToolDescriptor:
        def execute(params: dict) -> str:
            backend = router.get_backend()
            return backend.resume() if backend else _NO_BACKEND_MSG

        return ToolDescriptor(
            name="music_resume",
            description="Resume paused music.",
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )

    def _skip_tool(self, router) -> ToolDescriptor:
        def execute(params: dict) -> str:
            backend = router.get_backend()
            return backend.skip() if backend else _NO_BACKEND_MSG

        return ToolDescriptor(
            name="music_skip",
            description="Skip to the next track.",
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )

    def _previous_tool(self, router) -> ToolDescriptor:
        def execute(params: dict) -> str:
            backend = router.get_backend()
            return backend.previous() if backend else _NO_BACKEND_MSG

        return ToolDescriptor(
            name="music_previous",
            description="Go back to the previous track.",
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )

    def _volume_tool(self, router) -> ToolDescriptor:
        def execute(params: dict) -> str:
            backend = router.get_backend()
            if backend is None:
                return _NO_BACKEND_MSG
            direction = params.get("direction", "").lower()
            if direction == "up":
                level = _current_volume(backend) + 10
            elif direction == "down":
                level = _current_volume(backend) - 10
            else:
                level = int(params.get("level", 50))
            return backend.set_volume(max(0, min(100, level)))

        return ToolDescriptor(
            name="music_volume",
            description=(
                "Control music volume. Use for: 'turn it up', 'turn it down', "
                "'set volume to 70', 'volume up/down'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "level": _int_prop("Volume level 0–100"),
                    "direction": _str_prop("up | down (adjusts by 10)"),
                },
                "required": [],
            },
            execute=execute,
        )

    def _now_playing_tool(self, router) -> ToolDescriptor:
        def execute(params: dict) -> str:
            backend = router.get_backend()
            if backend is None:
                return _NO_BACKEND_MSG
            info = backend.now_playing()
            if isinstance(info, str):
                return info
            return f"{info['track']} by {info['artist']}."

        return ToolDescriptor(
            name="music_now_playing",
            description="Ask what's currently playing.",
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )


def _current_volume(backend) -> int:
    """Try to get current volume; fall back to 50."""
    try:
        if hasattr(backend, "_sp") and backend._sp:
            pb = backend._sp.current_playback()
            if pb and pb.get("device"):
                return pb["device"].get("volume_percent", 50)
    except Exception:
        pass
    return 50
