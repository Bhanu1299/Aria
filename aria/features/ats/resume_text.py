"""features/ats/resume_text.py — the resume's plain text, the only facts open-text answers may use.

text(path) → str   pdftotext (poppler) on the resume PDF, cached in ~/.aria/resume.txt
                   and rebuilt when the PDF changes. .txt / .md resumes are read as-is.
                   "" when there is no resume or it can't be read. Never raises.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE = Path.home() / ".aria" / "resume.txt"
MAX_CHARS = 12000


def _cache_meta() -> Path:
    return CACHE.with_suffix(".json")


def _cached(src: Path) -> Optional[str]:
    try:
        meta = json.loads(_cache_meta().read_text())
        st = src.stat()
        if meta.get("path") == str(src) and meta.get("mtime") == st.st_mtime and meta.get("size") == st.st_size:
            return CACHE.read_text()
    except (OSError, ValueError):
        pass
    return None


def _extract_pdf(src: Path) -> str:
    exe = shutil.which("pdftotext") or ("/opt/homebrew/bin/pdftotext" if Path("/opt/homebrew/bin/pdftotext").exists() else None)
    if not exe:
        logger.warning("resume_text: pdftotext not installed (brew install poppler)")
        return ""
    out = subprocess.run([exe, "-layout", "-enc", "UTF-8", str(src), "-"], capture_output=True, timeout=30)
    if out.returncode != 0:
        logger.warning("resume_text: pdftotext failed on %s: %s", src, out.stderr.decode(errors="replace")[:200])
        return ""
    return out.stdout.decode("utf-8", errors="replace")


def _tidy(raw: str) -> str:
    lines = [" ".join(ln.split()) for ln in raw.splitlines()]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()[:MAX_CHARS]


def text(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        src = Path(path).expanduser()
        if not src.is_file():
            return ""
        if src.suffix.lower() in (".txt", ".md"):
            return _tidy(src.read_text(errors="replace"))
        hit = _cached(src)
        if hit is not None:
            return hit
        body = _tidy(_extract_pdf(src))
        if body:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(body)
            st = src.stat()
            _cache_meta().write_text(json.dumps({"path": str(src), "mtime": st.st_mtime, "size": st.st_size}))
        return body
    except Exception as exc:
        logger.error("resume_text.text(%s) failed: %s", path, exc)
        return ""
