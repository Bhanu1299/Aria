"""integrations/google_auth.py — Google OAuth tokens, one per Aria account.

Aria uses two Google accounts, each with its own token. Every caller names its
account explicitly; there is no default and no fallback between them, so a
missing jobs token is an error, never a silent switch to the personal one.

  personal  google.json       Gmail, Calendar, context index — the assistant
  jobs      google_jobs.json  Muse sheet, outreach drafts — the job pipeline

Setup (once per account, and again whenever its scopes grow):
    python -m aria.setup google                    # personal
    python -m aria.setup google --account jobs     # jobs
    python -m aria.setup google --status           # which account each tool uses

Files (~/.aria/credentials/):
    google_client_secret.json   OAuth client ("Desktop app"), shared by both accounts
    google.json / google_jobs.json   user tokens, written by run_setup()

Each token records the signed-in address ("email") and what the user actually
GRANTED ("scopes"). google-auth can't be trusted for the latter:
Credentials.from_authorized_user_file(path, scopes) overwrites the stored scopes
with whatever the caller asks for, so a token missing a newly added scope would
load fine and then 403 at call time. load_credentials() compares the recorded
grant against what the caller needs and raises GoogleAuthError instead.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import aria.core.config as config

logger = logging.getLogger(__name__)

CREDENTIALS_DIR = Path.home() / ".aria" / "credentials"
CLIENT_SECRETS_PATH = CREDENTIALS_DIR / "google_client_secret.json"

GMAIL_MODIFY = "https://www.googleapis.com/auth/gmail.modify"
GMAIL_LABELS = "https://www.googleapis.com/auth/gmail.labels"
GMAIL_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
CALENDAR = "https://www.googleapis.com/auth/calendar"
SPREADSHEETS = "https://www.googleapis.com/auth/spreadsheets"
# Identity only — lets setup read the signed-in address from the ID token
# without depending on any API being enabled in the Cloud project.
IDENTITY_SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email"]

PERSONAL = "personal"
JOBS = "jobs"
ACCOUNTS = (PERSONAL, JOBS)

_TOKEN_FILES = {PERSONAL: "google.json", JOBS: "google_jobs.json"}

# What each account asks for on its consent screen.
ACCOUNT_SCOPES = {
    PERSONAL: [GMAIL_MODIFY, GMAIL_LABELS, GMAIL_COMPOSE, CALENDAR, SPREADSHEETS],
    JOBS: [GMAIL_MODIFY, GMAIL_LABELS, GMAIL_COMPOSE, SPREADSHEETS],
}
SCOPES = ACCOUNT_SCOPES[PERSONAL]

# Who uses which account — printed by `python -m aria.setup google --status`.
ACCOUNT_USES = {
    PERSONAL: [
        "gmail_list, gmail_read, gmail_create_draft, gmail_send, gmail_reply",
        "calendar_list, calendar_create, calendar_delete",
        "context_sync / context_for (Gmail + Calendar index)",
    ],
    JOBS: [
        "job_from_sheet, apply_job, submit/skip follow-ups (Muse sheet read/write)",
        "job_outreach_draft (Gmail drafts for outreach)",
    ],
}

TOKEN_PATH = CREDENTIALS_DIR / _TOKEN_FILES[PERSONAL]
SETUP_CMD = "python -m aria.setup google"


class GoogleAuthError(Exception):
    """Google can't be used right now. str(exc) is safe to speak to the user."""


def _check(account: str) -> str:
    if account not in ACCOUNTS:
        raise ValueError(f"unknown Google account {account!r} (expected one of {ACCOUNTS})")
    return account


def token_path(account: str) -> Path:
    return CREDENTIALS_DIR / _TOKEN_FILES[_check(account)]


def setup_cmd(account: str) -> str:
    return SETUP_CMD if _check(account) == PERSONAL else f"{SETUP_CMD} --account {account}"


def label(account: str) -> str:
    """'Google' for personal, 'your jobs Google account' for jobs — for spoken errors."""
    return "Google" if _check(account) == PERSONAL else "Your jobs Google account"


def expected_email(account: str) -> str:
    """Address the account must sign in as (from .env), '' if unset."""
    return (config.GOOGLE_ACCOUNT if _check(account) == PERSONAL else config.GOOGLE_JOBS_ACCOUNT).lower()


def is_configured(account: str) -> bool:
    return token_path(account).exists()


def account_email(account: str) -> str:
    """Signed-in address recorded at setup ('' if unknown)."""
    try:
        return str(json.loads(token_path(account).read_text()).get("email") or "")
    except (OSError, ValueError):
        return ""


def granted_scopes(path: Path) -> list[str]:
    """Scopes recorded in the token file at setup time ([] if unreadable)."""
    try:
        return list(json.loads(path.read_text()).get("scopes") or [])
    except (OSError, ValueError):
        return []


def missing_scopes(required: Sequence[str], path: Path) -> list[str]:
    granted = set(granted_scopes(path))
    return [s for s in required if s not in granted]


def load_credentials(required: Sequence[str], account: str, path: Optional[Path] = None):
    """Valid google.oauth2 Credentials for `account` covering `required`, refreshed
    if expired. Raises GoogleAuthError when not connected, missing a scope, or the
    refresh fails. `path` overrides the account's token file (tests)."""
    path = path or token_path(account)
    cmd, who = setup_cmd(account), label(account)
    if not path.exists():
        raise GoogleAuthError(f"{who} isn't connected yet. Run: {cmd}")
    missing = missing_scopes(required, path)
    if missing:
        short = ", ".join(s.rsplit("/", 1)[-1] for s in missing)
        raise GoogleAuthError(f"{who} needs new permissions ({short}). Run: {cmd}")
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        recorded = granted_scopes(path)
        creds = Credentials.from_authorized_user_file(str(path), recorded)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _write_token(creds, recorded, path)
        if not creds.valid:
            raise GoogleAuthError(f"{who} auth expired. Run: {cmd}")
        return creds
    except GoogleAuthError:
        raise
    except Exception as exc:
        logger.warning("google_auth[%s]: load/refresh failed: %s", account, exc)
        raise GoogleAuthError(f"{who} auth expired. Run: {cmd}") from exc


def build_service(api: str, version: str, required: Sequence[str], *, account: str):
    """googleapiclient service for api/version as `account`. Raises GoogleAuthError."""
    creds = load_credentials(required, account)
    from googleapiclient.discovery import build
    return build(api, version, credentials=creds, cache_discovery=False)


def _write_token(creds, scopes: Sequence[str], path: Path, email: str = "") -> None:
    data = json.loads(creds.to_json())
    data["scopes"] = sorted(scopes)
    if not email:
        try:
            email = json.loads(path.read_text()).get("email") or ""
        except (OSError, ValueError):
            email = ""
    if email:
        data["email"] = email
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    path.chmod(0o600)


def _signed_in_email(creds) -> str:
    """Address from the OAuth ID token (setup asks for IDENTITY_SCOPES). '' if absent.

    The token comes straight from Google's token endpoint over TLS, so the payload
    is read without signature verification."""
    import base64
    token = getattr(creds, "id_token", None) or ""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return str(claims.get("email") or "").lower()
    except (IndexError, ValueError):
        return ""


def run_setup(account: str = PERSONAL, email: Optional[str] = None) -> int:
    """Interactive browser consent for `account`. Returns a process exit code.

    `email` (default: GOOGLE_ACCOUNT / GOOGLE_JOBS_ACCOUNT) pre-selects the
    account and must match whoever signs in — a wrong-account token is never saved.
    """
    _check(account)
    if not CLIENT_SECRETS_PATH.exists():
        print(
            "OAuth client not found.\n"
            "Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID\n"
            "(type: Desktop app), enable the Gmail, Calendar and Sheets APIs, then save the\n"
            f"downloaded JSON as:\n  {CLIENT_SECRETS_PATH}"
        )
        return 1
    want = (email or expected_email(account)).lower()
    if account == JOBS and not want:
        print("Set GOOGLE_JOBS_ACCOUNT=<jobs address> in .env (or pass --email) so Aria can\n"
              "check the jobs token belongs to the jobs account.")
        return 1
    scopes = ACCOUNT_SCOPES[account]
    path = token_path(account)
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow

        print(f"Connecting the {account} account" + (f" ({want})" if want else "") + " — sign in in the browser.")
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_PATH), scopes + IDENTITY_SCOPES)
        extra = {"login_hint": want} if want else {}
        # prompt=consent guarantees a refresh token even if this app was approved before.
        creds = flow.run_local_server(port=0, prompt="consent", **extra)
        granted = list(getattr(creds, "granted_scopes", None) or creds.scopes or scopes)
        signed_in = _signed_in_email(creds)
    except Exception as exc:
        print(f"Google setup failed: {exc}")
        return 1

    if want and signed_in != want:
        print(f"Signed in as {signed_in or 'an unknown account'}, but the {account} account is {want}.\n"
              f"Nothing was saved. Run `{setup_cmd(account)}` again and pick {want}.")
        return 1
    other = [a for a in ACCOUNTS if a != account and account_email(a).lower() == signed_in]
    if other:
        print(f"{signed_in} is already connected as the {other[0]} account. Nothing was saved —\n"
              f"the {account} account must be a different Google account.")
        return 1

    _write_token(creds, granted, path, signed_in)
    missing = [s for s in scopes if s not in granted]
    print(f"Google {account} account connected as {signed_in}. Token saved to {path}")
    if missing:
        print("Some permissions were NOT granted (unchecked on the consent screen):")
        for s in missing:
            print(f"  - {s}")
        print(f"Features that need them will ask you to run `{setup_cmd(account)}` again.")
    return 0


def print_status() -> int:
    """Which Google account each Aria tool uses, and whether it's connected."""
    for account in ACCOUNTS:
        if is_configured(account):
            state = f"connected as {account_email(account) or 'unknown address (re-run setup to record it)'}"
        else:
            state = f"NOT connected — run: {setup_cmd(account)}"
        print(f"{account}: {state}")
        print(f"  token: {token_path(account)}")
        for use in ACCOUNT_USES[account]:
            print(f"  - {use}")
    return 0
