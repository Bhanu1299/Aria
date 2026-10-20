"""integrations/google_auth.py — one Google OAuth token for Gmail, Calendar and Sheets.

Setup (once, and again whenever SCOPES grows):
    python -m aria.setup google [--account you@gmail.com]

Files (~/.aria/credentials/):
    google_client_secret.json   OAuth client ("Desktop app") from Google Cloud Console
    google.json                 the user token, written by run_setup()

The token file's "scopes" field records what the user actually GRANTED. google-auth
can't be trusted for this: Credentials.from_authorized_user_file(path, scopes)
overwrites the stored scopes with whatever the caller asks for, so a token missing
a newly added scope would load fine and then 403 at call time. load_credentials()
compares the recorded grant against what the caller needs and raises
GoogleAuthError("…run python -m aria.setup google") instead.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

CREDENTIALS_DIR = Path.home() / ".aria" / "credentials"
TOKEN_PATH = CREDENTIALS_DIR / "google.json"
CLIENT_SECRETS_PATH = CREDENTIALS_DIR / "google_client_secret.json"

GMAIL_MODIFY = "https://www.googleapis.com/auth/gmail.modify"
GMAIL_LABELS = "https://www.googleapis.com/auth/gmail.labels"
GMAIL_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
CALENDAR = "https://www.googleapis.com/auth/calendar"
SPREADSHEETS = "https://www.googleapis.com/auth/spreadsheets"

# Everything Aria asks for in one consent screen.
SCOPES = [GMAIL_MODIFY, GMAIL_LABELS, GMAIL_COMPOSE, CALENDAR, SPREADSHEETS]

SETUP_CMD = "python -m aria.setup google"


class GoogleAuthError(Exception):
    """Google can't be used right now. str(exc) is safe to speak to the user."""


def is_configured() -> bool:
    return TOKEN_PATH.exists()


def granted_scopes(token_path: Path = TOKEN_PATH) -> list[str]:
    """Scopes recorded in the token file at setup time ([] if unreadable)."""
    try:
        return list(json.loads(token_path.read_text()).get("scopes") or [])
    except (OSError, ValueError):
        return []


def missing_scopes(required: Sequence[str], token_path: Path = TOKEN_PATH) -> list[str]:
    granted = set(granted_scopes(token_path))
    return [s for s in required if s not in granted]


def load_credentials(required: Sequence[str], token_path: Path = TOKEN_PATH):
    """Valid google.oauth2 Credentials covering `required`, refreshed if expired.

    Raises GoogleAuthError when not connected, missing a scope, or the refresh fails.
    """
    if not token_path.exists():
        raise GoogleAuthError(f"Google isn't connected yet. Run: {SETUP_CMD}")
    missing = missing_scopes(required, token_path)
    if missing:
        short = ", ".join(s.rsplit("/", 1)[-1] for s in missing)
        raise GoogleAuthError(
            f"Google needs new permissions ({short}). Run: {SETUP_CMD}"
        )
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        recorded = granted_scopes(token_path)
        creds = Credentials.from_authorized_user_file(str(token_path), recorded)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _write_token(creds, recorded, token_path)
        if not creds.valid:
            raise GoogleAuthError(f"Google auth expired. Run: {SETUP_CMD}")
        return creds
    except GoogleAuthError:
        raise
    except Exception as exc:
        logger.warning("google_auth: load/refresh failed: %s", exc)
        raise GoogleAuthError(f"Google auth expired. Run: {SETUP_CMD}") from exc


def build_service(api: str, version: str, required: Sequence[str]):
    """googleapiclient service for api/version. Raises GoogleAuthError."""
    creds = load_credentials(required)
    from googleapiclient.discovery import build
    return build(api, version, credentials=creds, cache_discovery=False)


def _write_token(creds, scopes: Sequence[str], token_path: Path = TOKEN_PATH) -> None:
    data = json.loads(creds.to_json())
    data["scopes"] = sorted(scopes)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(data, indent=2))
    token_path.chmod(0o600)


def run_setup(account: Optional[str] = None) -> int:
    """Interactive browser consent for SCOPES. Returns a process exit code."""
    if not CLIENT_SECRETS_PATH.exists():
        print(
            "OAuth client not found.\n"
            "Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID\n"
            "(type: Desktop app), enable the Gmail, Calendar and Sheets APIs, then save the\n"
            f"downloaded JSON as:\n  {CLIENT_SECRETS_PATH}"
        )
        return 1
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_PATH), SCOPES)
        extra = {"login_hint": account} if account else {}
        # prompt=consent guarantees a refresh token even if this app was approved before.
        creds = flow.run_local_server(port=0, prompt="consent", **extra)
        granted = list(getattr(creds, "granted_scopes", None) or creds.scopes or SCOPES)
        _write_token(creds, granted)
    except Exception as exc:
        print(f"Google setup failed: {exc}")
        return 1

    missing = [s for s in SCOPES if s not in granted]
    print(f"Google connected. Token saved to {TOKEN_PATH}")
    if missing:
        print("Some permissions were NOT granted (unchecked on the consent screen):")
        for s in missing:
            print(f"  - {s}")
        print(f"Features that need them will ask you to run `{SETUP_CMD}` again.")
    return 0
