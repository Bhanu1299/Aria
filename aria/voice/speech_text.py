"""voice/speech_text.py — text the way it should sound, right before `say` reads it.

The agent is told its words are spoken (agent.py "# Voice"), but tool output and canned lines
still reach the speaker with things a voice reads badly: job IDs ("J0220" → "J zero two two
zero"), markdown, URLs, bullets, "e.g.", em dashes (2026-10-25 log: "The closest match is
J0220 — Full St…"). spoken(text) fixes those in code — no LLM, never changes the meaning.

best_voice() picks the most natural installed voice: Premium, then Enhanced, then Samantha.
ARIA_VOICE=<name> overrides; ARIA_VOICE_RATE=<words per minute> sets the speed.
"""
from __future__ import annotations

import functools
import os
import re
import subprocess
from typing import Optional

_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)?[^)]+\)")
_URL = re.compile(r"\bhttps?://(?:www\.)?([^/\s]+)\S*", re.I)
_JOB_ID = re.compile(r"\bJ0*(\d{1,6})\b")
_BULLET = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+", re.M)
_ABBREV = [(re.compile(r"\be\.g\.,?", re.I), "for example,"), (re.compile(r"\bi\.e\.,?", re.I), "that is,"),
           (re.compile(r"\betc\.", re.I), "and so on."), (re.compile(r"\bvs\.?(?=\s)", re.I), "versus"),
           (re.compile(r"\bw/(?=\s)"), "with"), (re.compile(r"\s&\s"), " and ")]


def spoken(text: str) -> str:
    """Markdown, URLs, IDs and symbols turned into what a person would say. Never raises."""
    try:
        t = text or ""
        t = re.sub(r"```.*?```", " ", t, flags=re.S)
        t = _LINK.sub(r"\1", t)
        t = _URL.sub(lambda m: m.group(1).split(":")[0], t)          # https://jobs.lever.co/x → jobs.lever.co
        t = re.sub(r"(\*\*|__|`|^#+\s*)", "", t, flags=re.M)
        t = re.sub(r"(?<!\w)\*(\S[^*]*?)\*(?!\w)", r"\1", t)
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if len(lines) > 1:                                              # a list read as sentences
            lines = [_BULLET.sub("", ln) for ln in lines]
            t = " ".join(ln if re.search(r"[.!?:;]$", ln) else ln + "." for ln in lines)
        else:
            t = _BULLET.sub("", t)
        t = re.sub(r"\bID\s+J0*(\d{1,6})\b", lambda m: f"ID {int(m.group(1))}", t)      # "ID J0200" → "ID 200"
        t = _JOB_ID.sub(lambda m: f"job {int(m.group(1))}", t)
        for pat, rep in _ABBREV:
            t = pat.sub(rep, t)
        t = re.sub(r"\s+[—–]\s+|—", ", ", t)                             # a dash is a pause
        t = re.sub(r"\s*\|\s*", ", ", t)
        t = re.sub(r"\s+([,.!?;:])", r"\1", t)
        t = re.sub(r",\s*,", ",", t)
        return re.sub(r"\s+", " ", t).strip()
    except Exception:
        return text or ""


def _pick(listing: str) -> Optional[str]:
    voices = []
    for line in listing.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if m and m.group(2) in ("en_US", "en_GB", "en_AU", "en_IN", "en_IE"):
            voices.append((m.group(1).strip(), m.group(2)))
    for tier in ("(Premium)", "(Enhanced)"):
        hits = [v for v in voices if tier in v[0]]
        hits.sort(key=lambda v: v[1] != "en_US")
        if hits:
            return hits[0][0]
    for name in ("Samantha", "Ava", "Allison", "Evan", "Tom", "Alex"):
        for v, _loc in voices:
            if v == name or v.startswith(name + " ("):
                return v
    return None


@functools.lru_cache(maxsize=1)
def best_voice() -> Optional[str]:
    """The voice `say -v` should use, or None for the system default."""
    forced = os.getenv("ARIA_VOICE", "").strip()
    if forced:
        return forced
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    return _pick(out)


def say_args(text: str) -> list:
    """argv for `say`: the best voice, the rate if set, the spoken text."""
    args = ["say"]
    voice = best_voice()
    if voice:
        args += ["-v", voice]
    rate = os.getenv("ARIA_VOICE_RATE", "").strip()
    if rate.isdigit():
        args += ["-r", rate]
    return args + [spoken(text)]
