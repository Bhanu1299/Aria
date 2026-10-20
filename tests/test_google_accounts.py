"""Two Google accounts: personal (assistant) and jobs (job pipeline), never mixed."""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import aria.core.config as config
from aria.integrations import google_auth, sheets
from aria.plugins.productivity.gmail import GmailClient

ARIA = Path(__file__).parent.parent / "aria"
JOB_PIPELINE = [ARIA / "features" / "applicator.py", *sorted((ARIA / "features" / "ats").glob("*.py")),
                *sorted((ARIA / "plugins" / "apply").glob("*.py")), ARIA / "integrations" / "sheets.py"]


@pytest.fixture
def creds_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(google_auth, "CREDENTIALS_DIR", tmp_path)
    monkeypatch.setattr(google_auth, "CLIENT_SECRETS_PATH", tmp_path / "google_client_secret.json")
    return tmp_path


def _token(path: Path, scopes, email: str = "") -> None:
    path.write_text(json.dumps({"scopes": scopes, "token": "t", **({"email": email} if email else {})}))


def test_each_account_has_its_own_token_file(creds_dir):
    assert google_auth.token_path("personal") == creds_dir / "google.json"
    assert google_auth.token_path("jobs") == creds_dir / "google_jobs.json"
    assert google_auth.setup_cmd("jobs") == "python -m aria.setup google --account jobs"
    with pytest.raises(ValueError):
        google_auth.token_path("work")


def test_jobs_never_falls_back_to_personal(creds_dir):
    _token(creds_dir / "google.json", google_auth.ACCOUNT_SCOPES["personal"], "me@personal.test")
    assert google_auth.is_configured("personal") and not google_auth.is_configured("jobs")
    with pytest.raises(google_auth.GoogleAuthError, match="--account jobs"):
        google_auth.load_credentials([google_auth.SPREADSHEETS], "jobs")
    out = GmailClient("jobs").create_draft("r@co.test", "Hi", "Body")
    assert "jobs Google account isn't connected" in out and "--account jobs" in out


def test_sheet_uses_jobs_account(monkeypatch):
    seen = []

    def fake_build(api, version, required, *, account):
        seen.append(account)
        raise google_auth.GoogleAuthError("x")

    monkeypatch.setattr(google_auth, "build_service", fake_build)
    with pytest.raises(sheets.SheetError):
        sheets.JobSheet("id", "Sheet1").get_next_job()
    assert seen == ["jobs"]


def test_outreach_tool_drafts_in_jobs_account(monkeypatch):
    from aria.core.tool import ToolRegistry
    from aria.plugins.apply import ApplyPlugin
    seen = []
    monkeypatch.setattr(GmailClient, "create_draft",
                        lambda self, *a, **k: seen.append(self.account) or "Draft saved")
    reg = ToolRegistry()
    ApplyPlugin().register(reg)
    assert reg.get("job_outreach_draft").execute({"to": "r@co.test", "body": "Hi"}) == "Draft saved"
    assert seen == ["jobs"]


def test_personal_tools_use_personal_account(monkeypatch):
    from aria.core.tool import ToolRegistry
    from aria.plugins.productivity import ProductivityPlugin
    import aria.plugins.productivity.cron as cron
    monkeypatch.setattr(cron.CronScheduler, "start", lambda self: None)
    built = []
    monkeypatch.setattr(google_auth, "build_service",
                        lambda api, version, required, *, account: built.append((api, account)) or MagicMock())
    monkeypatch.setattr(google_auth, "is_configured", lambda account: True)
    reg = ToolRegistry()
    ProductivityPlugin().register(reg)
    reg.get("gmail_list").execute({})
    reg.get("calendar_list").execute({})
    assert built == [("gmail", "personal"), ("calendar", "personal")]


def test_every_google_call_names_its_account():
    """build_service / GmailClient must name an account at every call site."""
    offenders = []
    for path in ARIA.rglob("*.py"):
        if path.name == "google_auth.py":
            continue
        text = path.read_text()
        for m in re.finditer(r"build_service\((?:[^()]|\([^()]*\))*\)", text):
            if "account=" not in m.group(0):
                offenders.append(f"{path.relative_to(ARIA)}: {m.group(0)[:60]}")
        for m in re.finditer(r"GmailClient\(\)", text):
            offenders.append(f"{path.relative_to(ARIA)}: {m.group(0)}")
    assert not offenders, offenders


def test_job_pipeline_never_names_personal_account():
    pattern = re.compile(r"google_auth\.PERSONAL|[\"']personal[\"']")
    offenders = [f"{p.relative_to(ARIA)}: {m.group(0)}" for p in JOB_PIPELINE for m in pattern.finditer(p.read_text())]
    assert not offenders, offenders


# ── setup ───────────────────────────────────────────────────────────────────

class _Flow:
    def __init__(self, email):
        self.email, self.kwargs = email, {}

    def run_local_server(self, **kwargs):
        self.kwargs = kwargs
        creds = MagicMock()
        creds.granted_scopes = google_auth.ACCOUNT_SCOPES["jobs"]
        creds.to_json.return_value = json.dumps({"token": "t", "refresh_token": "r"})
        return creds


@pytest.fixture
def fake_oauth(creds_dir, monkeypatch):
    (creds_dir / "google_client_secret.json").write_text("{}")
    state = {"email": "jobs@jobs.test", "flow": None}

    def from_secrets(path, scopes):
        state["scopes"] = scopes
        state["flow"] = _Flow(state["email"])
        return state["flow"]

    import google_auth_oauthlib.flow as flow_mod
    monkeypatch.setattr(flow_mod.InstalledAppFlow, "from_client_secrets_file", staticmethod(from_secrets))
    monkeypatch.setattr(google_auth, "_signed_in_email", lambda creds: state["email"])
    monkeypatch.setattr(config, "GOOGLE_JOBS_ACCOUNT", "jobs@jobs.test")
    return state


def test_setup_jobs_writes_jobs_token_with_email(fake_oauth, creds_dir):
    assert google_auth.run_setup("jobs") == 0
    data = json.loads((creds_dir / "google_jobs.json").read_text())
    assert data["email"] == "jobs@jobs.test"
    assert not (creds_dir / "google.json").exists()
    assert fake_oauth["flow"].kwargs["login_hint"] == "jobs@jobs.test"
    assert google_auth.CALENDAR not in fake_oauth["scopes"]
    assert set(google_auth.IDENTITY_SCOPES) <= set(fake_oauth["scopes"])


def test_signed_in_email_reads_id_token():
    import base64
    body = base64.urlsafe_b64encode(json.dumps({"email": "Jobs@Jobs.test"}).encode()).decode().rstrip("=")
    assert google_auth._signed_in_email(MagicMock(id_token=f"h.{body}.s")) == "jobs@jobs.test"
    assert google_auth._signed_in_email(MagicMock(id_token=None)) == ""


def test_setup_refuses_wrong_account(fake_oauth, creds_dir):
    fake_oauth["email"] = "me@personal.test"
    assert google_auth.run_setup("jobs") == 1
    assert not (creds_dir / "google_jobs.json").exists()


def test_setup_refuses_same_address_as_other_account(fake_oauth, creds_dir, monkeypatch):
    _token(creds_dir / "google.json", google_auth.ACCOUNT_SCOPES["personal"], "jobs@jobs.test")
    assert google_auth.run_setup("jobs") == 1
    assert not (creds_dir / "google_jobs.json").exists()


def test_setup_jobs_requires_configured_address(fake_oauth, monkeypatch, capsys):
    monkeypatch.setattr(config, "GOOGLE_JOBS_ACCOUNT", "")
    assert google_auth.run_setup("jobs") == 1
    assert "GOOGLE_JOBS_ACCOUNT" in capsys.readouterr().out


def test_status_lists_accounts_and_tools(creds_dir, capsys):
    _token(creds_dir / "google.json", google_auth.ACCOUNT_SCOPES["personal"], "me@personal.test")
    google_auth.print_status()
    out = capsys.readouterr().out
    assert "personal: connected as me@personal.test" in out
    assert "jobs: NOT connected — run: python -m aria.setup google --account jobs" in out
    assert "job_outreach_draft" in out and "gmail_create_draft" in out
