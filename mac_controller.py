"""
mac_controller.py — Aria Phase 2C: Native Mac App Control

Three-layer fallback:
  Layer 1: AppleScript via osascript subprocess (primary)
  Layer 2: pyobjc Accessibility API (fallback when AppleScript errors)
  Layer 3: screencapture + Groq vision (last resort)

Public API:
  handle_app_command(transcript: str) -> str
    Groq sub-classifies the transcript, routes to appropriate handler,
    returns a spoken-word response string.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time as _time

from groq import Groq

import config

logger = logging.getLogger(__name__)

_CLIENT: Groq | None = None


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


# ---------------------------------------------------------------------------
# AppleScript runner
# ---------------------------------------------------------------------------

def _run_applescript(script: str) -> tuple[bool, str]:
    """
    Run an AppleScript string via osascript.

    Returns (success: bool, output: str).
    On failure, output contains stderr. Never raises.
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        err = result.stderr.strip() or result.stdout.strip()
        print(f"[mac_controller] osascript error (rc={result.returncode}): {err}")
        return False, err
    except subprocess.TimeoutExpired:
        print("[mac_controller] osascript timed out")
        return False, "timeout"
    except Exception as exc:
        print(f"[mac_controller] osascript exception: {exc}")
        return False, str(exc)


# ---------------------------------------------------------------------------
# Layer 1 — AppleScript handlers
# ---------------------------------------------------------------------------

def _spotify_play(query: str) -> str:
    """Search and play a song/artist on Spotify."""
    if not query:
        ok, _ = _run_applescript('tell application "Spotify" to play')
        return "Resuming Spotify." if ok else _applescript_fail("Spotify")

    script = f'''tell application "Spotify"
    activate
    search for "{query.replace('"', chr(92) + '"')}"
    delay 1
    play
end tell'''
    ok, out = _run_applescript(script)
    if ok:
        return f"Playing {query} on Spotify."
    return _layer2_fallback("Spotify", f"play {query}") or _applescript_fail("Spotify")


def _spotify_pause() -> str:
    ok, _ = _run_applescript('tell application "Spotify" to pause')
    return "Paused Spotify." if ok else _applescript_fail("Spotify")


def _spotify_resume() -> str:
    ok, _ = _run_applescript('tell application "Spotify" to play')
    return "Resumed Spotify." if ok else _applescript_fail("Spotify")


def _spotify_skip() -> str:
    ok, _ = _run_applescript('tell application "Spotify" to next track')
    return "Skipped to next track." if ok else _applescript_fail("Spotify")


def _spotify_whats_playing() -> str:
    script = '''tell application "Spotify"
    set t to name of current track
    set a to artist of current track
    return t & " by " & a
end tell'''
    ok, out = _run_applescript(script)
    if ok and out:
        return f"Now playing: {out}."
    return "I couldn't get the current track from Spotify."


def _volume_set(level: int) -> str:
    """Set output volume 0-100."""
    level = max(0, min(100, level))
    ok, _ = _run_applescript(f"set volume output volume {level}")
    return f"Volume set to {level}." if ok else "I couldn't change the volume."


def _volume_mute() -> str:
    ok, _ = _run_applescript("set volume with output muted")
    return "Muted." if ok else "I couldn't mute the volume."


def _volume_unmute() -> str:
    ok, _ = _run_applescript("set volume without output muted")
    return "Unmuted." if ok else "I couldn't unmute the volume."


def _volume_up() -> str:
    script = """set v to output volume of (get volume settings)
set volume output volume (v + 10)"""
    ok, _ = _run_applescript(script)
    return "Volume up." if ok else "I couldn't increase the volume."


def _volume_down() -> str:
    script = """set v to output volume of (get volume settings)
set volume output volume (v - 10)"""
    ok, _ = _run_applescript(script)
    return "Volume down." if ok else "I couldn't decrease the volume."


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------

def _brightness_up() -> str:
    # F2 key = brightness up on MacBook keyboards (key code 144)
    ok, _ = _run_applescript(
        'tell application "System Events" to key code 144'
    )
    if ok:
        return "Brightness up."
    # Fallback: brightness CLI (brew install brightness)
    if shutil.which("brightness"):
        r = subprocess.run(["brightness", "-a", "0.1"], capture_output=True, timeout=5)
        return "Brightness up." if r.returncode == 0 else "Couldn't change brightness."
    return "Couldn't change brightness. Run: brew install brightness"


def _brightness_down() -> str:
    ok, _ = _run_applescript(
        'tell application "System Events" to key code 145'
    )
    if ok:
        return "Brightness down."
    if shutil.which("brightness"):
        r = subprocess.run(["brightness", "-a", "-0.1"], capture_output=True, timeout=5)
        return "Brightness down." if r.returncode == 0 else "Couldn't change brightness."
    return "Couldn't change brightness. Run: brew install brightness"


def _brightness_set(level: int) -> str:
    level = max(0, min(100, level))
    frac = round(level / 100.0, 2)
    if shutil.which("brightness"):
        r = subprocess.run(["brightness", str(frac)], capture_output=True, timeout=5)
        return f"Brightness set to {level}%." if r.returncode == 0 else "Couldn't set brightness."
    return "Couldn't set brightness to an exact value. Run: brew install brightness"


# ---------------------------------------------------------------------------
# Focus / Do Not Disturb
# ---------------------------------------------------------------------------

# Map spoken names → canonical macOS Focus mode names
_FOCUS_NAME_MAP: dict[str, str] = {
    "do not disturb": "Do Not Disturb",
    "dnd":            "Do Not Disturb",
    "work":           "Work",
    "personal":       "Personal",
    "sleep":          "Sleep",
    "reading":        "Reading",
    "fitness":        "Fitness",
    "gaming":         "Gaming",
    "driving":        "Driving",
    "mindfulness":    "Mindfulness",
    "focus":          "Do Not Disturb",  # generic → DND
    "":               "Do Not Disturb",  # no mode specified → DND
}


def _focus_on(mode: str) -> str:
    canonical = _FOCUS_NAME_MAP.get(mode.lower().strip(), mode.strip() or "Do Not Disturb")

    # Method 1 — shortcuts CLI (user can create shortcuts named e.g. "Do Not Disturb On")
    shortcut_name = f"{canonical} On"
    if shutil.which("shortcuts"):
        r = subprocess.run(
            ["shortcuts", "run", shortcut_name],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return f"{canonical} Focus is on."

    # Method 2 — old AppleScript API (works on macOS 12 and below)
    ok, _ = _run_applescript(
        'tell application "System Events" to tell the current user to set doNotDisturb to true'
    )
    if ok:
        return "Do Not Disturb is on."

    # Method 3 — graceful fallback
    return (
        f"I can't toggle {canonical} Focus directly on this macOS version. "
        f"To enable voice control, create a Shortcut named '{shortcut_name}' in the Shortcuts app."
    )


def _focus_off(mode: str) -> str:
    canonical = _FOCUS_NAME_MAP.get(mode.lower().strip(), mode.strip() or "Do Not Disturb")

    shortcut_name = f"{canonical} Off"
    if shutil.which("shortcuts"):
        r = subprocess.run(
            ["shortcuts", "run", shortcut_name],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return f"{canonical} Focus is off."

    # Try "Turn Off Focus" as a generic shortcut
    r2 = subprocess.run(
        ["shortcuts", "run", "Turn Off Focus"],
        capture_output=True, text=True, timeout=15,
    ) if shutil.which("shortcuts") else None
    if r2 and r2.returncode == 0:
        return "Focus is off."

    ok, _ = _run_applescript(
        'tell application "System Events" to tell the current user to set doNotDisturb to false'
    )
    if ok:
        return "Do Not Disturb is off."

    return (
        f"I can't turn off {canonical} Focus directly on this macOS version. "
        f"To enable voice control, create a Shortcut named '{shortcut_name}' in the Shortcuts app."
    )


# ---------------------------------------------------------------------------
# Wi-Fi
# ---------------------------------------------------------------------------

def _wifi_on() -> str:
    for iface in ("Wi-Fi", "en0", "en1"):
        r = subprocess.run(
            ["networksetup", "-setairportpower", iface, "on"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return "Wi-Fi turned on."
    return "Couldn't turn on Wi-Fi."


def _wifi_off() -> str:
    for iface in ("Wi-Fi", "en0", "en1"):
        r = subprocess.run(
            ["networksetup", "-setairportpower", iface, "off"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return "Wi-Fi turned off."
    return "Couldn't turn off Wi-Fi."


# ---------------------------------------------------------------------------
# Bluetooth
# ---------------------------------------------------------------------------

def _bluetooth_on() -> str:
    if shutil.which("blueutil"):
        r = subprocess.run(["blueutil", "--power", "1"], capture_output=True, timeout=10)
        return "Bluetooth on." if r.returncode == 0 else "Couldn't turn on Bluetooth."
    ok, _ = _run_applescript('''
tell application "System Preferences"
    activate
    set current pane to pane id "com.apple.preferences.Bluetooth"
end tell''')
    return "Bluetooth settings are open. Install blueutil for direct control: brew install blueutil"


def _bluetooth_off() -> str:
    if shutil.which("blueutil"):
        r = subprocess.run(["blueutil", "--power", "0"], capture_output=True, timeout=10)
        return "Bluetooth off." if r.returncode == 0 else "Couldn't turn off Bluetooth."
    ok, _ = _run_applescript('''
tell application "System Preferences"
    activate
    set current pane to pane id "com.apple.preferences.Bluetooth"
end tell''')
    return "Bluetooth settings are open. Install blueutil for direct control: brew install blueutil"


# ---------------------------------------------------------------------------
# Dark / Light mode
# ---------------------------------------------------------------------------

def _dark_mode_on() -> str:
    ok, _ = _run_applescript('''
tell application "System Events"
    tell appearance preferences
        set dark mode to true
    end tell
end tell''')
    return "Dark mode on." if ok else "Couldn't enable dark mode."


def _dark_mode_off() -> str:
    ok, _ = _run_applescript('''
tell application "System Events"
    tell appearance preferences
        set dark mode to false
    end tell
end tell''')
    return "Light mode on." if ok else "Couldn't disable dark mode."


def _dark_mode_toggle() -> str:
    ok, _ = _run_applescript('''
tell application "System Events"
    tell appearance preferences
        set dark mode to not dark mode
    end tell
end tell''')
    return "Appearance toggled." if ok else "Couldn't toggle dark mode."


# ---------------------------------------------------------------------------
# Low Power Mode
# ---------------------------------------------------------------------------

def _low_power_on() -> str:
    r = subprocess.run(
        ["sudo", "-n", "pmset", "-a", "lowpowermode", "1"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode == 0:
        return "Low power mode on."
    # Try without sudo in case terminal has it
    r2 = subprocess.run(
        ["pmset", "-a", "lowpowermode", "1"],
        capture_output=True, text=True, timeout=10,
    )
    if r2.returncode == 0:
        return "Low power mode on."
    ok, _ = _run_applescript('''
tell application "System Preferences"
    activate
    set current pane to pane id "com.apple.preference.battery"
end tell''')
    return "Low power mode needs admin access. Battery settings are open."


def _low_power_off() -> str:
    r = subprocess.run(
        ["sudo", "-n", "pmset", "-a", "lowpowermode", "0"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode == 0:
        return "Low power mode off."
    r2 = subprocess.run(
        ["pmset", "-a", "lowpowermode", "0"],
        capture_output=True, text=True, timeout=10,
    )
    if r2.returncode == 0:
        return "Low power mode off."
    ok, _ = _run_applescript('''
tell application "System Preferences"
    activate
    set current pane to pane id "com.apple.preference.battery"
end tell''')
    return "Low power mode needs admin access. Battery settings are open."


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def _screenshot() -> str:
    filename = f"Screenshot {_time.strftime('%Y-%m-%d at %H.%M.%S')}.png"
    desktop = os.path.expanduser(f"~/Desktop/{filename}")
    r = subprocess.run(
        ["screencapture", "-x", desktop],
        capture_output=True, timeout=10,
    )
    return "Screenshot saved to Desktop." if r.returncode == 0 else "Couldn't take a screenshot."


_FINDER_PATHS: dict[str, str] = {
    "desktop":   "path to desktop",
    "downloads": "path to downloads folder",
    "documents": "path to documents folder",
    "home":      "path to home folder",
    "pictures":  "path to pictures folder",
    "music":     "path to music folder",
    "movies":    "path to movies folder",
}


def _finder_open(folder: str) -> str:
    """
    Open a named folder in Finder.

    Parses "X in [parent]" patterns so "Bills in Documents" searches for Bills
    inside ~/Documents rather than treating the whole phrase as a folder name.

    Priority:
      1. Known system alias (desktop, downloads, documents, etc.)
      2. Scan parent dirs (location-hint first, then common dirs) with fuzzy
         name matching — handles underscores/hyphens and Whisper mishearings
      3. Spotlight mdfind fallback across the whole home tree
    """
    import difflib
    import os
    import re as _re

    folder = folder.strip()

    # Parse "X in [parent]" — extract location hint and strip it from the name
    # e.g. "Bells in Documents" → folder_name="Bells", location_hint="documents"
    location_hint: str | None = None
    m = _re.search(r'\s+in\s+(\w+)\s*$', folder, _re.IGNORECASE)
    if m:
        location_hint = m.group(1).lower().rstrip("s")  # "Documents"→"document"
        folder = folder[: m.start()].strip()

    key = folder.lower()

    # 1. Known system alias (only when no location hint, or hint matches)
    alias = _FINDER_PATHS.get(key)
    if alias and location_hint is None:
        script = f'''tell application "Finder"
    open ({alias} from user domain)
    activate
end tell'''
        ok, _ = _run_applescript(script)
        if ok:
            return f"Opened {folder} in Finder."

    home = os.path.expanduser("~")

    # Map spoken location hints to real paths
    _HINT_MAP = {
        "document": os.path.join(home, "Documents"),
        "desktop":  os.path.join(home, "Desktop"),
        "download": os.path.join(home, "Downloads"),
        "picture":  os.path.join(home, "Pictures"),
        "movie":    os.path.join(home, "Movies"),
        "music":    os.path.join(home, "Music"),
        "home":     home,
    }

    # Build search order: hint dir first (if given), then all common dirs
    search_parents: list[str] = []
    if location_hint:
        hint_path = _HINT_MAP.get(location_hint)
        if hint_path:
            search_parents.append(hint_path)
        else:
            # Unknown hint — try it as a dir name under home
            search_parents.append(os.path.join(home, location_hint.capitalize()))
    search_parents += [p for p in [
        home,
        os.path.join(home, "Documents"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Downloads"),
        os.path.join(home, "Pictures"),
        os.path.join(home, "Movies"),
        os.path.join(home, "Music"),
    ] if p not in search_parents]

    def _normalize(name: str) -> str:
        return name.lower().replace("_", " ").replace("-", " ").replace(".", " ")

    for parent in search_parents:
        if not os.path.isdir(parent):
            continue
        try:
            entries = [e for e in os.listdir(parent) if os.path.isdir(os.path.join(parent, e))]
        except PermissionError:
            continue

        # Exact normalized match first
        for entry in entries:
            if _normalize(entry) == _normalize(key):
                full_path = os.path.join(parent, entry)
                subprocess.run(["open", full_path], timeout=5)
                return f"Opened {entry} in Finder."

        # Fuzzy match — handles Whisper mishearings (tray→trae, bells→bills, trail→trae)
        normalized_entries = {_normalize(e): e for e in entries}
        close = difflib.get_close_matches(_normalize(key), normalized_entries.keys(), n=1, cutoff=0.72)
        if close:
            real_name = normalized_entries[close[0]]
            full_path = os.path.join(parent, real_name)
            subprocess.run(["open", full_path], timeout=5)
            return f"Opened {real_name} in Finder."

    # 3. Spotlight fallback — searches entire home tree
    try:
        result = subprocess.run(
            [
                "mdfind",
                "-onlyin", home,
                f"kMDItemFSName == '{folder}'cd && kMDItemContentType == 'public.folder'",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            path = result.stdout.strip().splitlines()[0]
            subprocess.run(["open", path], timeout=5)
            folder_name = os.path.basename(path)
            return f"Opened {folder_name} in Finder."
    except Exception as exc:
        print(f"[mac_controller] mdfind error: {exc}")

    return f"I couldn't find a folder named '{folder}'."


def _calendar_read(day: str) -> str:
    """Read today's or tomorrow's calendar events."""
    if "tomorrow" in day.lower():
        script = '''set tomorrowStart to (current date) + 1 * days
set (time of tomorrowStart) to 0
set tomorrowEnd to tomorrowStart + 1 * days - 1
set eventList to ""
tell application "Calendar"
    repeat with cal in calendars
        repeat with ev in (events of cal whose start date >= tomorrowStart and start date <= tomorrowEnd)
            set eventList to eventList & summary of ev & ", "
        end repeat
    end repeat
end tell
if eventList is "" then
    return "No events tomorrow."
else
    return "Tomorrow: " & eventList
end if'''
    else:
        script = '''set todayStart to current date
set (time of todayStart) to 0
set todayEnd to todayStart + 1 * days - 1
set eventList to ""
tell application "Calendar"
    repeat with cal in calendars
        repeat with ev in (events of cal whose start date >= todayStart and start date <= todayEnd)
            set eventList to eventList & summary of ev & ", "
        end repeat
    end repeat
end tell
if eventList is "" then
    return "No events today."
else
    return "Today: " & eventList
end if'''
    ok, out = _run_applescript(script)
    if ok:
        return out or "No events found."
    return "I couldn't read your calendar. Make sure Calendar has permission."


def _calendar_add(title: str, when: str) -> str:
    """Add a calendar event."""
    safe_title = title.replace('"', '\\"')
    safe_when = when.replace('"', '\\"')
    script = f'''tell application "Calendar"
    tell calendar "Home"
        make new event with properties {{summary:"{safe_title}", start date:date "{safe_when}", end date:date "{safe_when}" + 1 * hours}}
    end tell
end tell'''
    ok, _ = _run_applescript(script)
    if ok:
        return f"Added event: {title} at {when}."
    return "I couldn't add the event. Calendar may need permission or the time format wasn't recognized."


def _mail_unread_count() -> str:
    script = '''tell application "Mail"
    set unreadCount to unread count of inbox
    return unreadCount as string
end tell'''
    ok, out = _run_applescript(script)
    if ok:
        n = out.strip()
        return f"You have {n} unread email{'s' if n != '1' else ''} in your inbox."
    return "I couldn't check your mail. Make sure Mail has permission."


def _mail_read_latest() -> str:
    script = '''tell application "Mail"
    set msg to message 1 of inbox
    set sub to subject of msg
    set sndr to sender of msg
    return "From " & sndr & ": " & sub
end tell'''
    ok, out = _run_applescript(script)
    if ok and out:
        return out
    return "I couldn't read your latest email."


def _reminders_add(task: str, when: str) -> str:
    safe_task = task.replace('"', '\\"')
    if when:
        safe_when = when.replace('"', '\\"')
        script = f'''tell application "Reminders"
    tell list "Reminders"
        make new reminder with properties {{name:"{safe_task}", due date:date "{safe_when}"}}
    end tell
end tell'''
    else:
        script = f'''tell application "Reminders"
    tell list "Reminders"
        make new reminder with properties {{name:"{safe_task}"}}
    end tell
end tell'''
    ok, _ = _run_applescript(script)
    if ok:
        suffix = f" at {when}" if when else ""
        return f"Reminder set: {task}{suffix}."
    return "I couldn't add the reminder. Reminders may need permission."


def _app_open(app_name: str) -> str:
    import os

    if not app_name.strip():
        return "What would you like me to open?"

    # If target matches a known folder alias, route straight to Finder.
    if app_name.lower().strip() in _FINDER_PATHS:
        return _finder_open(app_name)

    safe = app_name.replace('"', '\\"')
    ok, err = _run_applescript(f'tell application "{safe}" to activate')
    if ok:
        return f"Opened {app_name}."

    # -1728 = "Can't get object" — app doesn't exist, not a permissions issue
    if "-1728" in err or "Can't get application" in err:
        # Try open -a as a second attempt
        try:
            result = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return f"Opened {app_name}."
        except Exception:
            pass

        # Maybe the user named a folder, not an app — try Finder search
        finder_result = _finder_open(app_name)
        if "couldn't find" not in finder_result:
            return finder_result

        return f"I couldn't find '{app_name}' as an app or folder on your Mac."

    return _applescript_fail(app_name)


def _app_quit(app_name: str) -> str:
    safe = app_name.replace('"', '\\"')
    ok, err = _run_applescript(f'tell application "{safe}" to quit')
    if ok:
        return f"Quit {app_name}."
    if "-1728" in err or "Can't get application" in err:
        return f"I couldn't find {app_name} on your Mac."
    return _applescript_fail(app_name)


def _app_hide(app_name: str) -> str:
    safe = app_name.replace('"', '\\"')
    ok, err = _run_applescript(f'tell application "{safe}" to set visible to false')
    if ok:
        return f"Hid {app_name}."
    if "-1728" in err or "Can't get application" in err:
        return f"I couldn't find {app_name} on your Mac."
    return _applescript_fail(app_name)


# ---------------------------------------------------------------------------
# Layer 2 — pyobjc Accessibility API fallback
# ---------------------------------------------------------------------------

def click_element(app_name: str, element_label: str) -> bool:
    """
    Find a running app by name, locate an AX element by label, and press it.
    Returns True on success, False on any failure.
    """
    try:
        import AppKit
        import ApplicationServices as AS

        workspace = AppKit.NSWorkspace.sharedWorkspace()
        running = workspace.runningApplications()
        target_app = None
        for app in running:
            if app_name.lower() in (app.localizedName() or "").lower():
                target_app = app
                break

        if target_app is None:
            print(f"[mac_controller] Layer 2: app not found: {app_name!r}")
            return False

        pid = target_app.processIdentifier()
        ax_app = AS.AXUIElementCreateApplication(pid)

        def _find_element(element, label, depth=0):
            if depth > 8:
                return None
            try:
                err, children = AS.AXUIElementCopyAttributeValue(element, "AXChildren", None)
                if err or not children:
                    return None
                for child in children:
                    err2, title = AS.AXUIElementCopyAttributeValue(child, "AXTitle", None)
                    if not err2 and title and label.lower() in str(title).lower():
                        return child
                    err3, desc = AS.AXUIElementCopyAttributeValue(child, "AXDescription", None)
                    if not err3 and desc and label.lower() in str(desc).lower():
                        return child
                    found = _find_element(child, label, depth + 1)
                    if found is not None:
                        return found
            except Exception:
                pass
            return None

        el = _find_element(ax_app, element_label)
        if el is None:
            print(f"[mac_controller] Layer 2: element {element_label!r} not found in {app_name}")
            return False

        AS.AXUIElementPerformAction(el, "AXPress")
        return True

    except ImportError:
        print("[mac_controller] Layer 2: pyobjc not installed — skipping Accessibility fallback")
        return False
    except Exception as exc:
        print(f"[mac_controller] Layer 2 error: {exc}")
        return False


def _layer2_fallback(app_name: str, action_description: str) -> str | None:
    """
    Try to click a UI element matching action_description.
    Returns a spoken response if successful, None if failed.
    """
    success = click_element(app_name, action_description)
    if success:
        return f"Done via accessibility: {action_description} in {app_name}."
    return None


# ---------------------------------------------------------------------------
# Layer 3 — screencapture + Groq vision
# ---------------------------------------------------------------------------

def read_screen_region(app_name: str) -> str:
    """
    Capture a screenshot and ask Groq vision to describe visible text/UI.
    Returns extracted text or an error string.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="aria_screen_")
        tmp_path = tmp.name
        tmp.close()

        result = subprocess.run(
            ["screencapture", "-x", tmp_path],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(f"[mac_controller] screencapture failed: {result.stderr}")
            return "I couldn't capture the screen."

        with open(tmp_path, "rb") as f:
            import base64
            img_b64 = base64.b64encode(f.read()).decode()

        os.unlink(tmp_path)

        client = _get_client()
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe only the main content visible on screen in 1-2 short sentences. "
                                "Do NOT start with 'The image shows' or describe the UI — just state what the content says."
                                if app_name in ("screen", "")
                                else f"What does {app_name} currently show? Read the key text content in 1-2 short sentences. Skip UI chrome descriptions."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        print(f"[mac_controller] Layer 3 vision error: {exc}")
        return "I couldn't read the screen."


def _layer3_screen_read(app_name: str) -> str:
    """Layer 3 wrapper — returns spoken response."""
    return read_screen_region(app_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _applescript_fail(app_name: str) -> str:
    return (
        f"I wasn't able to control {app_name}. "
        "You may need to grant accessibility permissions in System Settings "
        "→ Privacy & Security → Accessibility → add Terminal."
    )


# ---------------------------------------------------------------------------
# Groq sub-classifier
# ---------------------------------------------------------------------------

_SUB_CLASSIFY_SYSTEM = """You are Aria — Bhanu's personal AI, built to be fast, sharp, and occasionally hilarious. You are parsing a Mac system control command. Return ONLY a JSON object — no markdown, no explanation, no personality in this response.

Schema:
{
  "action": "open" | "quit" | "hide" |
            "volume_set" | "volume_up" | "volume_down" | "mute" | "unmute" |
            "brightness_set" | "brightness_up" | "brightness_down" |
            "focus_on" | "focus_off" |
            "wifi_on" | "wifi_off" |
            "bluetooth_on" | "bluetooth_off" |
            "dark_mode_on" | "dark_mode_off" | "dark_mode_toggle" |
            "low_power_on" | "low_power_off" |
            "screenshot" |
            "calendar_read" | "calendar_add" | "mail_read" | "mail_latest" | "reminder" | "finder" | "screen_read",
  "target": "<app name, focus mode name, folder name, or empty string>",
  "params": {
    "volume_level": <integer 0-100 or null>,
    "brightness_level": <integer 0-100 or null>,
    "focus_mode": "<do not disturb | work | personal | sleep | gaming | reading | fitness | driving | empty>",
    "event_title": "<string or null>",
    "event_time": "<string or null>",
    "day": "<today | tomorrow | empty>",
    "reminder_task": "<string or null>",
    "reminder_time": "<string or null>"
  }
}

PRIORITY RULE (check this first before all others):
- Any command containing "on the screen", "on screen", "on my screen", "read screen", "read my screen", "read the screen", "what's on the screen", "what does the screen say", "what does it say", "copy what's on screen", "copy all visible text", "copy the text on screen" → action=screen_read, target=app name if mentioned else ""
  Examples: "read the mail on the screen", "copy all visible text on the screen", "what's on my screen" → ALL map to screen_read

Rules:
- "set volume to 40" / "volume 40" → action=volume_set, params.volume_level=40
- "volume up" / "turn it up" → action=volume_up
- "volume down" / "turn it down" → action=volume_down
- "mute" → action=mute
- "unmute" / "unmute sound" → action=unmute
- "increase brightness" / "brightness up" / "brighter" → action=brightness_up
- "decrease brightness" / "brightness down" / "dimmer" → action=brightness_down
- "set brightness to 70" → action=brightness_set, params.brightness_level=70
- "turn on do not disturb" / "enable do not disturb" / "dnd on" → action=focus_on, params.focus_mode="do not disturb"
- "turn off do not disturb" / "disable do not disturb" / "dnd off" → action=focus_off, params.focus_mode="do not disturb"
- "enable work focus" / "turn on work mode" / "work focus on" → action=focus_on, params.focus_mode="work"
- "enable sleep focus" / "turn on sleep mode" → action=focus_on, params.focus_mode="sleep"
- "enable personal focus" → action=focus_on, params.focus_mode="personal"
- "turn off focus" / "disable focus" / "focus off" / "focus mode off" → action=focus_off, params.focus_mode=""
- "turn on wifi" / "enable wifi" / "wifi on" → action=wifi_on
- "turn off wifi" / "disable wifi" / "wifi off" → action=wifi_off
- "turn on bluetooth" / "enable bluetooth" / "bluetooth on" → action=bluetooth_on
- "turn off bluetooth" / "disable bluetooth" / "bluetooth off" → action=bluetooth_off
- "dark mode on" / "enable dark mode" / "switch to dark mode" → action=dark_mode_on
- "light mode on" / "enable light mode" / "switch to light mode" / "dark mode off" → action=dark_mode_off
- "toggle dark mode" / "toggle appearance" → action=dark_mode_toggle
- "low power mode on" / "enable low power" / "battery saver on" → action=low_power_on
- "low power mode off" / "disable low power" → action=low_power_off
- "take a screenshot" / "screenshot" / "capture screen" / "screencap" → action=screenshot
- "what's on my calendar today" → action=calendar_read, params.day="today"
- "what's on my calendar tomorrow" → action=calendar_read, params.day="tomorrow"
- "add event [title] at [time]" → action=calendar_add, params.event_title, params.event_time
- "how many unread emails" / "check emails" / "check my emails" / "what are my emails" / "latest emails" / "read my emails" → action=mail_read (ONLY when no "screen" or "on screen" phrase present)
- "read latest email" / "read my latest email" → action=mail_latest (ONLY when no "screen" phrase present)
- "remind me to [task] at [time]" → action=reminder, params.reminder_task, params.reminder_time
- "open downloads" / "show downloads" → action=finder, target="downloads"
- "open documents" / "show desktop" → action=finder, target matching folder name
- "open [AppName]" → action=open, target=AppName
- "quit [AppName]" → action=quit, target=AppName
- "hide [AppName]" → action=hide, target=AppName
"""


def _sub_classify(transcript: str) -> dict:
    """Use Groq to extract structured action/target/params from transcript."""
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": _SUB_CLASSIFY_SYSTEM},
            {"role": "user", "content": transcript.strip()},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw.strip())
    parsed = json.loads(raw)
    if "action" not in parsed:
        raise ValueError(f"No action in sub-classifier response: {parsed}")
    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def handle_app_command(transcript: str) -> str:
    """
    Sub-classify transcript via Groq, route to the appropriate AppleScript handler.
    Falls back through Layer 2 → Layer 3 on failure.
    Returns a spoken-word response string. Never raises.
    """
    try:
        cmd = _sub_classify(transcript)
    except Exception as exc:
        print(f"[mac_controller] Sub-classify failed: {exc}")
        return "I didn't understand that app command, please try again."

    action = cmd.get("action", "")
    target = cmd.get("target", "") or ""
    params = cmd.get("params") or {}

    print(f"[mac_controller] action={action!r} target={target!r} params={params}")

    try:
        if action == "volume_set":
            level = params.get("volume_level")
            if level is None:
                try:
                    level = int(re.search(r"\d+", target).group())
                except Exception:
                    level = 50
            return _volume_set(int(level))
        elif action == "volume_up":
            return _volume_up()
        elif action == "volume_down":
            return _volume_down()
        elif action == "mute":
            return _volume_mute()
        elif action == "unmute":
            return _volume_unmute()
        elif action == "brightness_set":
            level = params.get("brightness_level")
            if level is None:
                try:
                    level = int(re.search(r"\d+", target).group())
                except Exception:
                    level = 50
            return _brightness_set(int(level))
        elif action == "brightness_up":
            return _brightness_up()
        elif action == "brightness_down":
            return _brightness_down()
        elif action == "focus_on":
            mode = params.get("focus_mode") or target or ""
            return _focus_on(mode)
        elif action == "focus_off":
            mode = params.get("focus_mode") or target or ""
            return _focus_off(mode)
        elif action == "wifi_on":
            return _wifi_on()
        elif action == "wifi_off":
            return _wifi_off()
        elif action == "bluetooth_on":
            return _bluetooth_on()
        elif action == "bluetooth_off":
            return _bluetooth_off()
        elif action == "dark_mode_on":
            return _dark_mode_on()
        elif action == "dark_mode_off":
            return _dark_mode_off()
        elif action == "dark_mode_toggle":
            return _dark_mode_toggle()
        elif action == "low_power_on":
            return _low_power_on()
        elif action == "low_power_off":
            return _low_power_off()
        elif action == "screenshot":
            return _screenshot()
        elif action == "calendar_read":
            day = params.get("day") or "today"
            return _calendar_read(day)
        elif action == "calendar_add":
            title = params.get("event_title") or target
            when = params.get("event_time") or ""
            return _calendar_add(title, when)
        elif action == "mail_read":
            return _mail_unread_count()
        elif action == "mail_latest":
            return _mail_read_latest()
        elif action == "reminder":
            task = params.get("reminder_task") or target
            when = params.get("reminder_time") or ""
            return _reminders_add(task, when)
        elif action == "finder":
            return _finder_open(target or "home")
        elif action == "open":
            return _app_open(target)
        elif action == "quit":
            return _app_quit(target)
        elif action == "hide":
            return _app_hide(target)
        elif action == "screen_read":
            return _layer3_screen_read(target or "screen")
        else:
            return f"I don't know how to do {action!r} yet."

    except Exception as exc:
        print(f"[mac_controller] Handler error for action={action!r}: {exc}")
        app = target or "that app"
        return _applescript_fail(app)
