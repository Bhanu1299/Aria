"""tests/test_keychain_and_email_code.py — Keychain logins (fake backend, never the real
Keychain) and the verification-email reader (fake Gmail, never the real inbox)."""
from __future__ import annotations

import logging

import pytest

from aria.features.ats import email_code
from aria.system import keychain


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, pw):
        self.store[(service, user)] = pw

    def delete_password(self, service, user):
        del self.store[(service, user)]


@pytest.fixture
def ring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(keychain, "_backend", lambda: fake)
    return fake


# ── keychain ────────────────────────────────────────────────────────────────

def test_save_and_get_per_tenant_host(ring):
    assert keychain.save("Acme.wd5.myworkdayjobs.com", "me@example.com", "S3cret!pass")
    cred = keychain.get("acme.wd5.myworkdayjobs.com", "me@example.com")
    assert (cred.host, cred.username, cred.password) == ("acme.wd5.myworkdayjobs.com", "me@example.com", "S3cret!pass")
    assert ("aria-apply:acme.wd5.myworkdayjobs.com", "me@example.com") in ring.store
    assert keychain.get("other.wd1.myworkdayjobs.com", "me@example.com") is None


def test_credential_repr_never_shows_the_password(ring):
    keychain.save("acme.wd5.myworkdayjobs.com", "me@example.com", "S3cret!pass")
    cred = keychain.get("acme.wd5.myworkdayjobs.com", "me@example.com")
    assert "S3cret" not in repr(cred) and "S3cret" not in str(cred) and "<hidden>" in repr(cred)


def test_keychain_errors_never_raise_or_log_the_password(monkeypatch, caplog):
    class Locked:
        def get_password(self, *a):
            raise RuntimeError("locked")

        def set_password(self, service, user, pw):
            raise RuntimeError(f"denied {pw}")
    monkeypatch.setattr(keychain, "_backend", lambda: Locked())
    caplog.set_level(logging.DEBUG)
    assert keychain.get("h", "u") is None
    assert keychain.save("h", "u", "Topsecret#99") is False
    assert "Topsecret" not in caplog.text


def test_save_refuses_empty_values(ring):
    assert not keychain.save("", "u", "p") and not keychain.save("h", "u", "") and not ring.store


def test_generated_passwords_meet_portal_rules():
    for _ in range(50):
        pw = keychain.generate()
        assert len(pw) == 20
        assert any(c.islower() for c in pw) and any(c.isupper() for c in pw)
        assert any(c.isdigit() for c in pw) and any(c in keychain._SYMBOLS for c in pw)
        assert not set(pw) & set("O0Il1")
    assert len(keychain.generate(4)) == 12
    assert keychain.generate() != keychain.generate()


# ── email_code.extract (pure) ───────────────────────────────────────────────

@pytest.mark.parametrize("subject,text,html,want", [
    ("Verify your account", "Your verification code is 483920. It expires in 10 minutes.", "", ("code", "483920")),
    ("Your code", "Code: 48 39 20", "", ("code", "483920")),
    ("483920 is your Acme sign-in code", "", "", ("code", "483920")),
    ("Security check", "Use passcode - A7K29Q to continue", "", ("code", "A7K29Q")),
    ("Verify", "", "<p>Your one-time code</p><p><b>551203</b></p>", ("code", "551203")),
    ("Verify your email", "", '<a href="https://acme.myworkday.com/acme/verify?token=abc&amp;x=1">Verify</a>'
                              '<a href="https://acme.myworkday.com/unsubscribe">Unsubscribe</a>',
     ("link", "https://acme.myworkday.com/acme/verify?token=abc&x=1")),
    ("Welcome", "Thanks for creating an account. The code is below in the app 4839 and more", "", None),
    ("Your application", "Requisition 482913 received. Zip 14214.", "", None),
    ("Welcome", "Thanks for applying!", "", None),
])
def test_extract(subject, text, html, want):
    assert email_code.extract(subject, text, html) == want


def test_query_names_only_the_senders():
    assert email_code.query(["myworkday.com", "acme"]) == "newer_than:1h from:(myworkday.com OR acme)"


# ── client_for / wait_for ───────────────────────────────────────────────────

def test_client_is_the_jobs_account_only_when_it_is_the_application_email(monkeypatch):
    from aria.integrations import google_auth
    from aria.plugins.productivity import gmail
    monkeypatch.setattr(google_auth, "account_email", lambda a: {"personal": "Me@Personal.com", "jobs": "Jobs@X.com"}[a])
    monkeypatch.setattr(gmail.GmailClient, "is_configured", lambda self: True)
    assert email_code.client_for("jobs@x.com").account == "jobs"
    assert email_code.client_for("me@personal.com") is None          # never the personal inbox
    assert email_code.client_for("someone-else@x.com") is None
    assert email_code.client_for("") is None
    monkeypatch.setattr(gmail.GmailClient, "is_configured", lambda self: False)
    assert email_code.client_for("jobs@x.com") is None


class FakeGmail:
    def __init__(self, inbox, appear_after=0):
        self.inbox, self.appear_after, self.calls = inbox, appear_after, 0
        self.queries = []

    def list_messages(self, query, limit=5):
        self.queries.append(query)
        self.calls += 1
        return [{"id": k} for k in self.inbox] if self.calls > self.appear_after else []

    def message_bodies(self, mid):
        return self.inbox[mid]


def test_wait_for_polls_until_the_code_arrives():
    gm = FakeGmail({"m1": {"from": "acme@myworkday.com", "subject": "Verify", "text": "Your code is 112233",
                           "html": "", "ts": 2000.0}}, appear_after=2)
    slept = []
    got = email_code.wait_for("me@x.com", ["myworkday.com"], since=1990.0, polls=4, poll_secs=15, client=gm,
                              sleep=slept.append)
    assert got == ("code", "112233") and slept == [15, 15] and gm.queries[0].startswith("newer_than:1h from:(")


def test_wait_for_ignores_mail_older_than_the_request_and_gives_up():
    gm = FakeGmail({"old": {"from": "acme@myworkday.com", "subject": "Verify", "text": "Your code is 999999",
                            "html": "", "ts": 100.0}})
    assert email_code.wait_for("me@x.com", ["myworkday.com"], since=2000.0, polls=3, client=gm,
                               sleep=lambda s: None) is None
    assert gm.calls == 3


def test_wait_for_hands_off_without_a_matching_inbox(monkeypatch):
    monkeypatch.setattr(email_code, "client_for", lambda email: None)
    assert email_code.wait_for("me@x.com", ["myworkday.com"], since=0, sleep=lambda s: None) is None


def test_wait_for_stops_when_gmail_is_unavailable():
    class Down:
        def list_messages(self, q, limit=5):
            return "Gmail is rate-limited. Try again in a minute."
    assert email_code.wait_for("me@x.com", ["x"], since=0, client=Down(), sleep=lambda s: None) is None
