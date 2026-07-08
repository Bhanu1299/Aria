"""plugins/productivity/setup.py — OAuth CLI helper for Google services."""
from __future__ import annotations

from pathlib import Path

_CREDS_PATH = Path.home() / ".aria" / "credentials" / "google.json"
_CLIENT_SECRETS = Path.home() / ".aria" / "credentials" / "google_client_secret.json"
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/calendar",
]


def run_gmail_setup() -> None:
    """Interactive OAuth flow. Saves tokens to ~/.aria/credentials/google.json."""
    if not _CLIENT_SECRETS.exists():
        print(
            "Client secrets not found. Download OAuth2 credentials from Google Cloud Console\n"
            f"and save to: {_CLIENT_SECRETS}"
        )
        return
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        _CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        flow = InstalledAppFlow.from_client_secrets_file(str(_CLIENT_SECRETS), _SCOPES)
        creds = flow.run_local_server(port=0)
        _CREDS_PATH.write_text(creds.to_json())
        _CREDS_PATH.chmod(0o600)
        print(f"Google connected. Credentials saved to {_CREDS_PATH}")
    except Exception as exc:
        print(f"Setup failed: {exc}")


if __name__ == "__main__":
    run_gmail_setup()
