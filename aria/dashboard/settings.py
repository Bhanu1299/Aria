"""dashboard/settings.py — LLM provider keys + tier models. The ONLY place the dashboard writes.

Writes go to exactly two files, both in GateLLM's directory (config.GATEWAY_DIR):
  .env         provider API keys (created if missing, always chmod 600)
  config.yaml  the `tiers:` lines only, edited in place so comments survive
then GateLLM is told to reload (POST /admin/reload). Keys are never logged, never
returned in full (mask() only), and never leave this machine except to GateLLM's
own process via its .env file.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx

import aria.core.config as config

logger = logging.getLogger(__name__)

PROVIDER_LABELS = {"openai": "OpenAI", "anthropic": "Anthropic", "groq": "Groq", "deepseek": "DeepSeek",
                   "moonshot": "Moonshot (Kimi)"}
_KEY_OK = re.compile(r"^[A-Za-z0-9_\-.:/+=]{8,512}$")
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_TIER_LINE = re.compile(r"^(\s+)([A-Za-z0-9_\-]+)(\s*:\s*)\[(.*)\](\s*(?:#.*)?)$")


class SettingsError(Exception):
    """Bad input or an unusable GateLLM directory. str(exc) is safe to show (never a key)."""


def gateway_dir() -> Path:
    return Path(config.GATEWAY_DIR)


def env_path() -> Path:
    return gateway_dir() / ".env"


def config_path() -> Path:
    return gateway_dir() / "config.yaml"


def mask(key: str) -> str:
    """sk-...a1b2 — never more than the first 3 and last 4 characters."""
    key = (key or "").strip()
    if not key:
        return ""
    return f"{key[:3]}...{key[-4:]}" if len(key) >= 12 else "set (short key)"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _yaml() -> dict:
    import yaml
    path = config_path()
    if not path.exists():
        raise SettingsError(f"GateLLM config not found at {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _env_values() -> dict:
    out = {}
    path = env_path()
    if path.exists():
        for line in path.read_text().splitlines():
            m = _ENV_LINE.match(line)
            if m and not line.lstrip().startswith("#"):
                out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


def providers() -> list:
    """[{name, label, env, masked, has_key, models}] — masked values only."""
    cfg = _yaml()
    env = _env_values()
    models_by = {}
    for m in cfg.get("models", []):
        models_by.setdefault(m.get("provider"), []).append(m)
    out = []
    for name, pc in (cfg.get("providers") or {}).items():
        var = pc.get("api_key_env", "")
        value = env.get(var, "")
        out.append({"name": name, "label": PROVIDER_LABELS.get(name, name.title()), "env": var,
                    "masked": mask(value), "has_key": bool(value),
                    "models": [m["id"] for m in models_by.get(name, [])]})
    return out


def tiers() -> dict:
    """{"tiers": {tier: [model ids]}, "models": [{id, provider, input, output}]}"""
    cfg = _yaml()
    models = [{"id": m["id"], "provider": m.get("provider", ""), "input": m.get("input_per_mtok"),
               "output": m.get("output_per_mtok")} for m in cfg.get("models", [])]
    return {"tiers": {k: list(v) for k, v in (cfg.get("tiers") or {}).items()}, "models": models}


# ---------------------------------------------------------------------------
# writing — .env and config.yaml only
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, text: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    os.chmod(path, mode)


def _provider_env(provider: str) -> str:
    for p in providers():
        if p["name"] == provider:
            return p["env"]
    raise SettingsError(f"unknown provider {provider!r}")


def set_key(provider: str, key: Optional[str]) -> dict:
    """Add/replace (key) or remove (key None/'') a provider's key in GateLLM's .env.
    Returns the provider's new masked status. Raises SettingsError on bad input."""
    var = _provider_env(provider)
    key = (key or "").strip()
    if key and not _KEY_OK.match(key):
        raise SettingsError("That doesn't look like an API key (8–512 characters, no spaces or quotes).")
    path = env_path()
    lines = path.read_text().splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        m = _ENV_LINE.match(line)
        if m and m.group(1) == var and not line.lstrip().startswith("#"):
            if key and not done:
                out.append(f"{var}={key}")
                done = True
            continue  # replaced, removed, or a duplicate
        out.append(line)
    if key and not done:
        out.append(f"{var}={key}")
    _atomic_write(path, "\n".join(out) + "\n", 0o600)
    logger.info("dashboard: %s key %s", provider, "saved" if key else "removed")  # never the key
    return {"provider": provider, "masked": mask(key), "has_key": bool(key)}


def set_tier(tier: str, models: list) -> list:
    """Replace one tier's model chain in config.yaml (that line only). Validated first."""
    info = tiers()
    known = {m["id"] for m in info["models"]}
    if tier not in info["tiers"]:
        raise SettingsError(f"unknown tier {tier!r}")
    chain = [str(m).strip() for m in models if str(m).strip()]
    if not chain:
        raise SettingsError("A tier needs at least one model.")
    unknown = [m for m in chain if m not in known]
    if unknown:
        raise SettingsError(f"Unknown model(s): {', '.join(unknown)}")
    if len(set(chain)) != len(chain):
        raise SettingsError("A model is listed twice.")
    path = config_path()
    lines = path.read_text().split("\n")
    in_tiers, changed = False, False
    for i, line in enumerate(lines):
        if re.match(r"^tiers\s*:", line):
            in_tiers = True
            continue
        if in_tiers and line and not line[0].isspace() and not line.lstrip().startswith("#"):
            break
        m = _TIER_LINE.match(line) if in_tiers else None
        if m and m.group(2) == tier:
            lines[i] = f"{m.group(1)}{tier}{m.group(3)}[{', '.join(chain)}]{m.group(5)}"
            changed = True
            break
    if not changed:
        raise SettingsError(f"Couldn't find a one-line `{tier}: [...]` entry under tiers: in config.yaml.")
    _atomic_write(path, "\n".join(lines), os.stat(path).st_mode & 0o777)
    return chain


# ---------------------------------------------------------------------------
# GateLLM: reload + test call
# ---------------------------------------------------------------------------

def reload_gateway() -> dict:
    """Ask a running GateLLM to re-read .env + config.yaml. Never raises."""
    try:
        r = httpx.post(config.GATEWAY_URL.rstrip("/") + "/admin/reload", timeout=10)
    except httpx.HTTPError:
        return {"ok": True, "message": "GateLLM isn't running; it reads the new settings when it next starts."}
    if r.status_code == 200:
        return {"ok": True, "message": "GateLLM reloaded."}
    if r.status_code == 404:
        return {"ok": False, "message": "This GateLLM has no reload endpoint yet; restart it to apply the change."}
    try:
        msg = r.json()["error"]["message"]
    except Exception:
        msg = f"HTTP {r.status_code}"
    return {"ok": False, "message": f"GateLLM didn't reload: {msg[:300]}"}


def _scrub(text: str) -> str:
    """Drop anything key-shaped from an error message before it reaches the browser."""
    for value in _env_values().values():
        if value and len(value) >= 8:
            text = text.replace(value, mask(value))
    return re.sub(r"\b(sk|gsk|xai|key)[-_][A-Za-z0-9_\-]{12,}", lambda m: mask(m.group(0)), text)


def test_provider(provider: str) -> dict:
    """One tiny call through GateLLM on the provider's cheapest model, no fallback."""
    from aria.llm import autostart
    cfg = _yaml()
    models = [m for m in cfg.get("models", []) if m.get("provider") == provider]
    if not models:
        raise SettingsError(f"GateLLM has no models for {provider!r}")
    model = min(models, key=lambda m: (m.get("input_per_mtok") or 0) + (m.get("output_per_mtok") or 0))["id"]
    url = config.GATEWAY_URL.rstrip("/")
    autostart.ensure_running(url)
    start = time.monotonic()
    try:
        r = httpx.post(url + "/v1/chat/completions", timeout=60,
                       headers={"X-Gateway-Fallback": "off", "X-Gateway-Task": "settings_test"},
                       json={"model": model, "max_tokens": 16,
                             "messages": [{"role": "user", "content": "Reply with the word OK."}]})
    except httpx.HTTPError as exc:
        return {"ok": False, "model": model, "message": f"GateLLM not reachable: {type(exc).__name__}"}
    ms = int((time.monotonic() - start) * 1000)
    if r.status_code == 200:
        cost = r.headers.get("x-gateway-cost-usd", "")
        return {"ok": True, "model": model, "message": f"ok · {ms} ms" + (f" · ${float(cost):.6f}" if cost else "")}
    try:
        msg = r.json()["error"]["message"]
    except Exception:
        msg = r.text[:200]
    return {"ok": False, "model": model, "message": _scrub(f"HTTP {r.status_code}: {msg}")[:300]}
