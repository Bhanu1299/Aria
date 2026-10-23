"""dispatch/parse.py — deterministic text → value parsing for direct routes.

normalize(text)            transcript → canonical lowercase command text
unglue(text)               "AriaHeadSubmit" → "Head Submit" (what the agent is given)
parse_duration(text)       "an hour and a half" → 5400 (seconds)
parse_when(text, now)      "tomorrow at 5pm" → datetime
split_task_when(text, now) "call mom at 5pm" → ("call mom", datetime)
humanize_seconds(n)        5400 → "1 hour and 30 minutes"
spoken_when(dt, now)       datetime → "tomorrow at 5 PM"

Every parser raises SlotError when it can't read the input with certainty —
the dispatcher then hands the command to the agent instead of guessing.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta


class SlotError(ValueError):
    """The text doesn't parse with certainty; escalate to the agent."""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WAKE_PREFIX = re.compile(r"^(?:(?:hey|yo|ok|okay|hi)\s+)?aria\b[\s,.:!-]*")
_POLITE_PREFIX = re.compile(
    r"^(?:please|can you|could you|would you|will you|i want you to|i need you to|"
    r"go ahead and|just)\s+"
)
_POLITE_SUFFIX = re.compile(r"\s+(?:please|for me|now|thanks|thank you)$")
# Whisper glues the wake word on: "AriaHeadSubmit" → "Head Submit".
_GLUED_WAKE = re.compile(r"^\s*(?:[Hh]ey\s*)?[Aa]ria(?=[A-Z][a-z])")
# Spoken lead-ins that carry no command ("yeah, I submitted it", "so, open the dashboard"); only
# stripped when words follow, so a bare "yes" / "okay" still reaches the yes/no handlers.
_FILLER_PREFIX = re.compile(r"^(?:yeah|yes|yep|yup|okay|ok|so|alright|all right|um+|uh+|hmm+|oh|well then)\s+(?=\S)")


def unglue(text: str) -> str:
    """Undo Whisper gluing the wake word onto the command ("AriaHeadSubmit" → "Head Submit");
    any other text comes back unchanged. The agent gets this, not the glued word."""
    s = (text or "").strip()
    if _GLUED_WAKE.match(s):
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", _GLUED_WAKE.sub("", s))
    return text


def normalize(text: str) -> str:
    s = unglue(text).strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", s)        # whisper writes "p.m."
    s = re.sub(r"[!?;]+|(?<!\d),|,(?!\d)", " ", s)        # keep 1,200
    s = re.sub(r"\.(?=\s|$)", " ", s)                     # sentence periods only: keep 2.5, google.com
    s = re.sub(r"\s+", " ", s).strip()
    s = _WAKE_PREFIX.sub("", s).strip()
    for _ in range(3):                                    # "yeah so can you please just ..."
        new = _FILLER_PREFIX.sub("", s)
        new = _POLITE_PREFIX.sub("", new)
        new = _POLITE_SUFFIX.sub("", new).strip()
        if new == s:
            break
        s = new
    return s


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def words_to_digits(text: str) -> str:
    """'twenty five minutes' → '25 minutes'; leaves everything else alone."""
    out: list[str] = []
    words = text.replace("-", " ").split()
    i = 0
    while i < len(words):
        w = words[i]
        if w in _TENS:
            n = _TENS[w]
            if i + 1 < len(words) and words[i + 1] in _UNITS and _UNITS[words[i + 1]] < 10:
                n += _UNITS[words[i + 1]]
                i += 1
            out.append(str(n))
        elif w in _UNITS:
            out.append(str(_UNITS[w]))
        else:
            out.append(w)
        i += 1
    return " ".join(out)


# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------

_UNIT_SECONDS = (
    (re.compile(r"^(?:hours?|hrs?|h)$"), 3600),
    (re.compile(r"^(?:minutes?|mins?|m)$"), 60),
    (re.compile(r"^(?:seconds?|secs?|s)$"), 1),
)
_MAX_DURATION = 24 * 3600


def _unit_seconds(word: str) -> int | None:
    for rx, secs in _UNIT_SECONDS:
        if rx.match(word):
            return secs
    return None


def parse_duration(text: str) -> int:
    s = words_to_digits(normalize(text))
    # "half an hour", "an hour and a half", "2 and a half minutes"
    s = re.sub(r"\bhalf (?:an? )?(hour|minute)\b", r"0.5 \1s", s)
    s = re.sub(r"\ban? (hour|minute|second)\b", r"1 \1", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?) (hours?|minutes?) and a half\b",
               lambda m: f"{float(m.group(1)) + 0.5} {m.group(2)}", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?) and a half (hours?|minutes?)\b",
               lambda m: f"{float(m.group(1)) + 0.5} {m.group(2)}", s)
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)              # "5min" → "5 min"

    tokens = [t for t in s.split() if t != "and"]
    if not tokens or len(tokens) % 2:
        raise SlotError(f"not a duration: {text!r}")
    total = 0.0
    for num, unit in zip(tokens[::2], tokens[1::2]):
        secs = _unit_seconds(unit)
        if secs is None or not re.fullmatch(r"\d+(?:\.\d+)?", num):
            raise SlotError(f"not a duration: {text!r}")
        total += float(num) * secs
    total_int = int(round(total))
    if not 0 < total_int <= _MAX_DURATION:
        raise SlotError(f"duration out of range: {text!r}")
    return total_int


def humanize_seconds(seconds: int) -> str:
    seconds = int(seconds)
    parts = []
    for name, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        n, seconds = divmod(seconds, size)
        if n:
            parts.append(f"{n} {name}{'s' if n != 1 else ''}")
    if not parts:
        return "0 seconds"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# ---------------------------------------------------------------------------
# Points in time
# ---------------------------------------------------------------------------

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PARTS_OF_DAY = {"morning": 9, "afternoon": 15, "evening": 18, "night": 20}
_CLOCK = re.compile(
    r"^(?:at|by|around)?\s*"
    r"(?:(?P<noon>noon|midday)|(?P<midnight>midnight)|"
    r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>am|pm)?(?:\s*o'?clock)?)$"
)
_DAY = re.compile(
    r"^(?:(?P<today>today)|(?P<tonight>tonight)|(?P<tomorrow>tomorrow)"
    r"(?:\s+(?P<tpart>morning|afternoon|evening|night))?|"
    r"this\s+(?P<thispart>morning|afternoon|evening)|"
    r"(?:on\s+|next\s+)?(?P<weekday>" + "|".join(_WEEKDAYS) + r")"
    r"(?:\s+(?P<wpart>morning|afternoon|evening|night))?)$"
)


def _parse_clock(text: str) -> tuple[int, int, bool] | None:
    """→ (hour, minute, ambiguous_ampm) or None."""
    m = _CLOCK.match(text.strip())
    if not m:
        return None
    if m.group("noon"):
        return 12, 0, False
    if m.group("midnight"):
        return 0, 0, False
    h, mins, ampm = int(m.group("h")), int(m.group("m") or 0), m.group("ampm")
    if mins > 59:
        return None
    if ampm:
        if not 1 <= h <= 12:
            return None
        return (h % 12) + (12 if ampm == "pm" else 0), mins, False
    if h > 23:
        return None
    return h, mins, 1 <= h <= 12


def _parse_day(text: str, now: datetime) -> tuple[datetime, int | None] | None:
    """→ (date at midnight, default hour if no clock given) or None."""
    m = _DAY.match(text.strip())
    if not m:
        return None
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if m.group("today"):
        return today, None
    if m.group("tonight"):
        return today, 20
    if m.group("tomorrow"):
        part = m.group("tpart")
        return today + timedelta(days=1), _PARTS_OF_DAY[part] if part else 9
    if m.group("thispart"):
        return today, _PARTS_OF_DAY[m.group("thispart")]
    target = _WEEKDAYS.index(m.group("weekday"))
    ahead = (target - today.weekday()) % 7 or 7
    part = m.group("wpart")
    return today + timedelta(days=ahead), _PARTS_OF_DAY[part] if part else 9


def parse_when(text: str, now: datetime | None = None) -> datetime:
    now = now or datetime.now()
    s = words_to_digits(normalize(text))

    in_m = re.match(r"^in (.+)$", s)
    if in_m:
        return now + timedelta(seconds=parse_duration(in_m.group(1)))

    # Split into a day part and a clock part, in either order.
    day, clock = None, None
    words = s.split()
    for i in range(len(words) + 1):
        left, right = " ".join(words[:i]), " ".join(words[i:])
        for d_txt, c_txt in ((left, right), (right, left)):
            d = _parse_day(d_txt, now) if d_txt else None
            c = _parse_clock(c_txt) if c_txt else None
            if (d_txt and d is None) or (c_txt and c is None):
                continue
            if d or c:
                day, clock = d, c
                break
        if day or clock:
            break
    if not (day or clock):
        raise SlotError(f"not a time: {text!r}")

    if clock is None:
        base, default_hour = day
        if default_hour is None:
            raise SlotError(f"no time given: {text!r}")
        return base.replace(hour=default_hour)

    hour, minute, ambiguous = clock
    if day is None:
        # Bare clock time: the next time it comes round ("at 5" → 5 AM or 5 PM, whichever is next).
        today = now.replace(second=0, microsecond=0)
        options = [hour, hour + 12] if ambiguous and hour < 12 else [hour]
        candidates = []
        for offset in (0, 1):
            for h in options:
                t = today.replace(hour=h % 24, minute=minute) + timedelta(days=offset)
                if t > now:
                    candidates.append(t)
        return min(candidates)

    base, default_hour = day
    if ambiguous and hour < 12:
        # "tonight at 8" / "tomorrow evening at 7" → PM; "friday at 3" → PM (nobody means
        # 3 AM); "tomorrow at 7" → AM; "today at 9" said at 10 AM → 9 PM.
        if default_hour is not None and default_hour >= 15:
            hour += 12
        elif hour <= 6:
            hour += 12
        elif base.replace(hour=hour, minute=minute) <= now:
            hour += 12
    result = base.replace(hour=hour, minute=minute)
    if result <= now:
        raise SlotError(f"time is in the past: {text!r}")
    return result


_TIME_KEYWORDS = {"at", "by", "in", "on", "tomorrow", "tonight", "today", "this", "next", *_WEEKDAYS}
_TIMEISH = re.compile(
    r"\d|\b(?:am|pm|noon|midnight|tonight|tomorrow|today|morning|afternoon|evening|"
    r"minutes?|hours?|o'?clock|" + "|".join(_WEEKDAYS) + r")\b"
)


def split_task_when(text: str, now: datetime | None = None) -> tuple[str, datetime | None]:
    """'call mom at 5pm' → ('call mom', <5pm>); 'buy milk' → ('buy milk', None).

    Tries each tail that starts with a time keyword, longest first. A tail that
    looks like a time but doesn't parse raises SlotError rather than guessing.
    """
    words = normalize(text).split()
    for i in range(1, len(words)):
        if words[i] not in _TIME_KEYWORDS:
            continue
        try:
            when = parse_when(" ".join(words[i:]), now)
        except SlotError:
            continue
        return " ".join(words[:i]), when
    if any(_TIMEISH.search(words_to_digits(" ".join(words[i:])))
           for i in range(1, len(words)) if words[i] in _TIME_KEYWORDS):
        raise SlotError(f"unreadable time in: {text!r}")
    if not words:
        raise SlotError("empty task")
    return " ".join(words), None


def spoken_when(dt: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now()
    clock = dt.strftime("%-I:%M %p").replace(":00 ", " ")
    delta = dt - now
    if timedelta(0) < delta < timedelta(hours=1):
        return "in " + humanize_seconds(int(round(delta.total_seconds() / 60.0)) * 60)
    days = (dt.date() - now.date()).days
    if days == 0:
        return f"at {clock}"
    if days == 1:
        return f"tomorrow at {clock}"
    if days < 7:
        return f"on {dt.strftime('%A')} at {clock}"
    return f"on {dt.strftime('%A, %B')} {dt.day} at {clock}"
