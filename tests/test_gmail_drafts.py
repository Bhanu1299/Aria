"""tests/test_gmail_drafts.py — drafts never send; send/reply need a spoken yes; the job pipeline can't send."""
from __future__ import annotations

import base64
import email
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aria.integrations import google_auth
from aria.plugins.productivity import ProductivityPlugin, is_send_confirmation
from aria.plugins.productivity.gmail import _SCOPES, GmailClient

ARIA = Path(__file__).parent.parent / "aria"


def _client(svc):
    client = GmailClient("personal")
    client._service = svc
    return client


def _raw(call) -> email.message.Message:
    raw = call.kwargs["body"]["message"]["raw"]
    return email.message_from_bytes(base64.urlsafe_b64decode(raw))


def test_create_draft_saves_and_never_sends():
    svc = MagicMock()
    client = _client(svc)
    with patch.object(client, "is_configured", return_value=True):
        out = client.create_draft("recruiter@example.com", "New-grad FDE", "Hi Sam,\n\nAlex here.")
    create = svc.users().drafts().create
    assert create.called and not svc.users().messages().send.called
    msg = _raw(create.call_args)
    assert (msg["to"], msg["subject"]) == ("recruiter@example.com", "New-grad FDE")
    assert "Alex here." in msg.get_payload()
    assert "nothing was sent" in out


def test_reply_draft_is_threaded():
    svc = MagicMock()
    svc.users().messages().get().execute.return_value = {
        "threadId": "t-1",
        "payload": {"headers": [{"name": "From", "value": "rec@acme.com"}, {"name": "Subject", "value": "Interview"},
                                {"name": "Message-ID", "value": "<m1@acme>"}]},
    }
    client = _client(svc)
    with patch.object(client, "is_configured", return_value=True):
        client.create_draft("", "", "Thanks!", reply_to_id="abc")
    call = svc.users().drafts().create.call_args
    assert call.kwargs["body"]["message"]["threadId"] == "t-1"
    msg = _raw(call)
    assert (msg["to"], msg["subject"], msg["In-Reply-To"]) == ("rec@acme.com", "Re: Interview", "<m1@acme>")


def test_draft_needs_compose_scope():
    assert google_auth.GMAIL_COMPOSE in _SCOPES


@pytest.mark.parametrize("reply,ok", [
    ("Yes", True), ("yes, send it", True), ("Send it.", True), ("yeah go ahead", True), ("Confirm", True),
    ("No", False), ("no, wait", False), ("yes but don't send", False), ("not yet", False),
    ("", False), ("uh what", False), ("save it as a draft", False), ("yes, actually no", False),
])
def test_is_send_confirmation(reply, ok):
    assert is_send_confirmation(reply) is ok


def _plugin(heard=None):
    if heard is None:
        return ProductivityPlugin()                       # no voice stack
    speaker, vc, tr = MagicMock(), MagicMock(), MagicMock()
    vc.record_once.return_value = "/tmp/x.wav"
    tr.transcribe.return_value = heard
    return ProductivityPlugin(speaker=speaker, voice_capture=vc, transcriber=tr)


PARAMS = {"to": "a@b.com", "subject": "Hello", "body": "Hi"}


@pytest.mark.parametrize("heard", [None, "no", "wait, not yet"])
def test_send_without_spoken_yes_saves_a_draft(heard):
    client = MagicMock()
    client.create_draft.return_value = "Draft saved for a@b.com: Hello. It's in your Gmail drafts; nothing was sent."
    out = _plugin(heard)._gmail_send(client).execute(dict(PARAMS))
    assert out.startswith("Not sent. Draft saved") and not client.send_message.called


def test_send_with_spoken_yes_sends():
    client = MagicMock()
    client.send_message.return_value = "Sent."
    plugin = _plugin("yes, send it")
    assert plugin._gmail_send(client).execute(dict(PARAMS)) == "Sent."
    spoken = plugin._speaker.say.call_args[0][0]
    assert "a@b.com" in spoken and "Hello" in spoken


def test_reply_without_spoken_yes_drafts_in_thread():
    client = MagicMock()
    client.create_draft.return_value = "Draft saved"
    out = _plugin("no")._gmail_reply(client).execute({"email_id": "abc", "body": "Thanks"})
    assert out.startswith("Not sent.") and not client.reply_message.called
    assert client.create_draft.call_args.kwargs["reply_to_id"] == "abc"


def test_job_pipeline_cannot_send_email():
    pipeline = [ARIA / "features" / "applicator.py", *sorted((ARIA / "features" / "ats").glob("*.py")),
                *sorted((ARIA / "plugins" / "apply").glob("*.py")), ARIA / "integrations" / "sheets.py"]
    pattern = re.compile(r"send_message|reply_message|gmail_send|gmail_reply|messages\(\)\.send")
    offenders = [f"{p.relative_to(ARIA)}: {m.group(0)}" for p in pipeline for m in pattern.finditer(p.read_text())]
    assert not offenders, offenders
    # The only Gmail the pipeline may touch is the jobs account's, and only for drafts.
    clients = [m.group(0) for p in pipeline for m in re.finditer(r"GmailClient\([^)]*\)", p.read_text())]
    assert clients and all(c == "GmailClient(google_auth.JOBS)" for c in clients), clients


def test_save_draft_returns_a_link_to_the_draft_in_that_account():
    svc = MagicMock()
    svc.users().drafts().create().execute.return_value = {"id": "r-1", "message": {"id": "18f2a"}}
    client = _client(svc)
    with patch.object(client, "is_configured", return_value=True), \
            patch.object(google_auth, "account_email", return_value="candidate@example.com"):
        out = client.save_draft("pat@acme.com", "Founding Engineer — Bhanu", "Hi Pat,")
    assert out["ok"] and out["draft_id"] == "r-1" and out["message_id"] == "18f2a"
    assert out["link"] == "https://mail.google.com/mail/u/candidate@example.com/#drafts?compose=18f2a"
    assert not svc.users().messages().send.called


def test_save_draft_attaches_files_as_they_are(tmp_path):
    pdf = tmp_path / "Resume.pdf"
    pdf.write_bytes(b"%PDF-1.4 resume bytes")
    svc = MagicMock()
    client = _client(svc)
    with patch.object(client, "is_configured", return_value=True):
        out = client.save_draft("pat@acme.com", "triaging 911 surges", "Hi Pat,\n\nBody.", attachments=[str(pdf)])
    assert out["ok"] and not svc.users().messages().send.called
    msg = _raw(svc.users().drafts().create.call_args)
    assert (msg["to"], msg["subject"]) == ("pat@acme.com", "triaging 911 surges") and msg.is_multipart()
    text, att = msg.get_payload()
    assert "Body." in text.get_payload()
    assert att.get_filename() == "Resume.pdf" and att.get_payload(decode=True) == b"%PDF-1.4 resume bytes"
