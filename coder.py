"""
coder.py — Aria Phase 4b: Agentic Code Execution Engine.

Handles "code" and "project" intents. Runs fully autonomously:
  voice command → Claude API (tool loop) → execute tools → speak result.

Tools available to Claude:
  bash, file_read, file_write, file_edit, glob, grep

Live menubar feedback updates as each tool fires.
Project management: new_project(), switch_project(), get_active_project().
"""
from __future__ import annotations

import glob as _glob_module
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Any

import config
from llm import llm_client

logger = logging.getLogger(__name__)

_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "identity.json")
_BASH_TIMEOUT = config.CODER_BASH_TIMEOUT
_MAX_TOOL_CALLS = config.CODER_MAX_TOOL_CALLS

# ---------------------------------------------------------------------------
# Claude tools definition
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "name": "bash",
        "description": "Run a shell command in the active project directory. Captures stdout and stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "file_read",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": "Create or overwrite a file with given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_edit",
        "description": "Replace an exact string in a file. Fails if old_str not found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "old_str": {"type": "string", "description": "Exact string to replace"},
                "new_str": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search file content for a pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex)"},
                "path": {"type": "string", "description": "File or directory to search"},
            },
            "required": ["pattern", "path"],
        },
    },
]

_SYSTEM_PROMPT = """\
You are Aria's coding engine — a fully autonomous senior software engineer with strong design taste.
Execute the user's request completely on your own using the tools available.
Work in the active project directory. Never ask for confirmation.
If a command fails, read the error and fix it. Iterate until done.

Code quality standards — always apply these:
- Web/UI work: produce polished, modern, professional results. Use clean typography (Google Fonts),
  thoughtful color palettes, proper spacing, smooth CSS transitions, and responsive layouts.
  A portfolio should look like it was made by a senior designer, not a student project.
  Dark themes: deep backgrounds (#0f0f0f–#1a1a2e), vibrant accents, glassmorphism cards.
  Light themes: crisp whites, subtle shadows, generous whitespace.
- HTML/CSS: semantic markup, CSS custom properties, flexbox/grid layouts, hover effects.
- Python: type hints, docstrings on public functions, proper error handling.
- Never produce placeholder lorem ipsum — infer realistic content from context.
  For a portfolio, include plausible skills, projects, and a bio based on any context available.

For web apps / servers:
- The bash tool auto-detects server commands and backgrounds them, then opens the browser.
- Just call bash with the normal run command (e.g. "python app.py") — do NOT add & yourself.
- After the server starts you will get back the localhost URL confirming it's running.

When finished, respond with a concise 1-2 sentence summary of what was accomplished.
"""

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Patterns that indicate a long-running server process
_SERVER_RE = re.compile(
    r"\b(?:python|python3|node|npm\s+(?:start|run)|uvicorn|gunicorn|flask|fastapi|http\.server|serve)\b"
    r".*(?:app\.py|server\.py|main\.py|index\.js|run|start|serve)\b"
    r"|python3?\s+-m\s+(?:http\.server|flask|uvicorn)",
    re.IGNORECASE,
)

# Ports to check after starting a server
_PORT_RE = re.compile(r"(?:port|:)\s*(\d{4,5})", re.IGNORECASE)


def _tool_bash(command: str, cwd: str | None = None) -> str:
    """Run shell command, return combined stdout+stderr. Never raises.

    If command looks like a blocking server, backgrounds it automatically,
    waits 2 seconds for startup, then opens the browser to localhost.
    """
    work_dir = cwd or get_active_project()
    # Strip trailing & if user already tried to background it
    cmd_stripped = command.rstrip("& \t")

    if _SERVER_RE.search(cmd_stripped) and "&" not in command:
        return _run_server(cmd_stripped, work_dir)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT,
        )
        output = result.stdout + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timeout after {_BASH_TIMEOUT}s"
    except Exception as exc:
        return f"Error: {exc}"


def _run_server(command: str, cwd: str) -> str:
    """Start a server process in the background, open browser to localhost."""
    try:
        # Detect port from command, default 5000
        m = _PORT_RE.search(command)
        port = int(m.group(1)) if m else 5000
        # flask default
        if "flask" in command.lower() or "app.py" in command.lower():
            port = 5000
        if "8000" in command:
            port = 8000
        if "8080" in command:
            port = 8080

        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)  # give server time to start
        if proc.poll() is not None:
            # Server already exited — grab output
            out, err = proc.communicate()
            return f"Server exited early:\n{(out + err).decode()[:500]}"

        # Open browser
        try:
            subprocess.Popen(["open", f"http://localhost:{port}"])
        except Exception:
            pass

        logger.debug("coder: server started on port %d (pid %d)", port, proc.pid)
        return f"Server started on http://localhost:{port} (pid {proc.pid}). Browser opened."
    except Exception as exc:
        return f"Error starting server: {exc}"


def _tool_file_read(path: str) -> str:
    """Read file content. Returns error string on failure."""
    try:
        path = _resolve_path(path)
        with open(path) as f:
            return f.read()
    except Exception as exc:
        return f"Error reading {path}: {exc}"


def _tool_file_write(path: str, content: str) -> str:
    """Write file, creating parent dirs as needed. Returns status string."""
    try:
        path = _resolve_path(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Written {path}"
    except Exception as exc:
        return f"Error writing {path}: {exc}"


def _tool_file_edit(path: str, old_str: str, new_str: str) -> str:
    """Replace exact string in file. Returns error if old_str not found."""
    try:
        path = _resolve_path(path)
        with open(path) as f:
            content = f.read()
        if old_str not in content:
            return f"Error: string not found in {path}"
        new_content = content.replace(old_str, new_str, 1)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Edited {path}"
    except Exception as exc:
        return f"Error editing {path}: {exc}"


def _tool_glob(pattern: str) -> str:
    """Find files matching pattern. Returns newline-separated list."""
    try:
        if not os.path.isabs(pattern):
            pattern = os.path.join(get_active_project(), pattern)
        matches = _glob_module.glob(pattern, recursive=True)
        return "\n".join(sorted(matches)) if matches else "(no matches)"
    except Exception as exc:
        return f"Error: {exc}"


def _tool_grep(pattern: str, path: str) -> str:
    """Search for pattern in path. Returns matching lines."""
    try:
        path = _resolve_path(path)
        result = subprocess.run(
            ["grep", "-rn", pattern, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "(no matches)"
    except Exception as exc:
        return f"Error: {exc}"


def _resolve_path(path: str) -> str:
    """Make relative paths absolute using active project as base."""
    if os.path.isabs(path):
        return path
    return os.path.join(get_active_project(), path)


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------

def _load_identity() -> dict:
    try:
        with open(_IDENTITY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_identity(identity: dict) -> None:
    try:
        dir_ = os.path.dirname(_IDENTITY_PATH)
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(identity, f, indent=2)
            tmp = f.name
        os.replace(tmp, _IDENTITY_PATH)
    except Exception as exc:
        logger.warning("coder: failed to save identity: %s", exc)


def get_active_project() -> str:
    """Return active project path. Falls back to PROJECTS_HOME."""
    identity = _load_identity()
    path = identity.get("active_project", "").strip()
    if path and os.path.isdir(path):
        return path
    return config.PROJECTS_HOME


def new_project(name: str) -> str:
    """Create new project directory and set it as active. Returns status string."""
    path = os.path.join(config.PROJECTS_HOME, name)
    os.makedirs(path, exist_ok=True)
    identity = _load_identity()
    identity["active_project"] = path
    _save_identity(identity)
    return path


def switch_project(name: str) -> str:
    """Switch active project to named directory. Returns path."""
    path = os.path.join(config.PROJECTS_HOME, name)
    identity = _load_identity()
    identity["active_project"] = path
    _save_identity(identity)
    return path


def list_projects() -> list[str]:
    """Return list of project directory names in PROJECTS_HOME."""
    try:
        return [
            d for d in os.listdir(config.PROJECTS_HOME)
            if os.path.isdir(os.path.join(config.PROJECTS_HOME, d))
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_tool(name: str, input_: dict) -> str:
    if name == "bash":
        return _tool_bash(input_["command"])
    if name == "file_read":
        return _tool_file_read(input_["path"])
    if name == "file_write":
        return _tool_file_write(input_["path"], input_["content"])
    if name == "file_edit":
        return _tool_file_edit(input_["path"], input_["old_str"], input_["new_str"])
    if name == "glob":
        return _tool_glob(input_["pattern"])
    if name == "grep":
        return _tool_grep(input_["pattern"], input_["path"])
    return f"Error: unknown tool {name!r}"


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """Inject user identity context into the system prompt."""
    try:
        import json as _json
        identity_path = os.path.join(os.path.dirname(__file__), "identity.json")
        with open(identity_path) as f:
            identity = _json.load(f)
        name = identity.get("name", "Bhanu Teja")
        bio = identity.get("bio", "MS CS graduate, Python and GenAI specialist, active job seeker")
        skills = identity.get("skills", "Python, GenAI, Full Stack, Machine Learning")
        ctx = f"\nUser context: Name={name}, Background={bio}, Skills={skills}"
    except Exception:
        ctx = "\nUser context: Name=Bhanu Teja, Background=MS CS graduate, Python and GenAI specialist, Skills=Python, GenAI, Full Stack, Machine Learning"
    return _SYSTEM_PROMPT + ctx


def run(command: str, menubar=None) -> str:
    """
    Execute voice command autonomously using the LLM tool loop.
    Returns final spoken summary. Never raises.
    """
    try:
        messages: list[dict] = [{"role": "user", "content": command}]
        tool_calls = 0
        system = _build_system_prompt()

        while tool_calls < _MAX_TOOL_CALLS:
            resp = llm_client.complete(
                messages=messages,
                tools=_TOOLS,
                tier="smart",
                max_tokens=4096,
                system=system,
            )

            if resp.stop_reason == "end_turn":
                return resp.text.strip() or "Done."

            if resp.stop_reason != "tool_use":
                return "Done."

            # Reconstruct assistant turn from LLMResponse
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

            # Process tool calls
            tool_results = []
            for tc in resp.tool_calls:
                tool_calls += 1

                if menubar is not None:
                    label = _menubar_label(tc.name, tc.input)
                    try:
                        menubar.set_state_label(f"CODING • {label}")
                    except Exception:
                        pass

                logger.debug("coder: tool %s input=%r", tc.name, tc.input)
                result = _dispatch_tool(tc.name, tc.input)
                logger.debug("coder: tool %s result=%r", tc.name, result[:200])

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        return "Done — reached maximum tool call limit."

    except Exception as exc:
        logger.error("coder.run failed: %s", exc)
        return f"Coding task failed: {exc}"


def _menubar_label(tool_name: str, tool_input: dict) -> str:
    if tool_name == "bash":
        cmd = tool_input.get("command", "")[:40]
        return f"running {cmd}..."
    if tool_name == "file_write":
        return f"writing {os.path.basename(tool_input.get('path', ''))}..."
    if tool_name == "file_read":
        return f"reading {os.path.basename(tool_input.get('path', ''))}..."
    if tool_name == "file_edit":
        return f"editing {os.path.basename(tool_input.get('path', ''))}..."
    if tool_name == "glob":
        return "finding files..."
    if tool_name == "grep":
        return "searching..."
    return f"{tool_name}..."


# ---------------------------------------------------------------------------
# Project intent handler
# ---------------------------------------------------------------------------

_NEW_PROJECT_RE = re.compile(
    r"\bnew\s+project\s+(?:called|named)?\s+([a-zA-Z0-9_\-]+)\b",
    re.IGNORECASE,
)
_SWITCH_RE = re.compile(
    r"\b(?:switch|go)\s+to\s+(?:project\s+)?([a-zA-Z0-9_\-]+)\b",
    re.IGNORECASE,
)


def handle_project(command: str) -> str:
    """Handle project management commands. Returns spoken response."""
    m = _NEW_PROJECT_RE.search(command)
    if m:
        name = m.group(1)
        new_project(name)
        return f"Created project {name} and switched to it."

    m = _SWITCH_RE.search(command)
    if m:
        name = m.group(1)
        switch_project(name)
        return f"Switched to {name}."

    projects = list_projects()
    if projects:
        return "Your projects: " + ", ".join(projects) + "."
    return f"No projects found in {config.PROJECTS_HOME}."
