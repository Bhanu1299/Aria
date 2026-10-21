"""tests/test_outreach_channel.py — step 2: Coldstart emails + recommend_channel (Coldstart mocked)."""
from __future__ import annotations

import sys
import types

import pytest

from aria.features.outreach import channel
from aria.features.outreach.contacts import Person


def _p(name="Pat Lee", **lookup):
    p = Person(name, "Recruiter", "search")
    p.lookup = lookup
    return p


@pytest.mark.parametrize("lookup,want", [
    ({"email": "pat@acme.com", "status": "valid", "confidence": 0.97}, "email"),
    ({"email": "pat@acme.com", "status": "catch_all", "confidence": 0.85}, "email"),
    ({"email": "pat@acme.com", "status": "unknown", "confidence": 0.4, "source": "site"}, "email"),
    ({"email": "pat@acme.com", "status": "catch_all", "confidence": 0.5}, "linkedin"),
    ({"email": "pat@acme.com", "status": "invalid", "confidence": 0.0}, "linkedin"),
    ({"email": "pat@acme.com", "status": "blocked", "confidence": 0.9}, "linkedin"),
    ({"email": "jobs@acme.com", "status": "role_inbox", "confidence": 0.9}, "linkedin"),
    ({"email": None, "status": "unknown", "confidence": 0.0}, "linkedin"),
    ({}, "linkedin"),
])
def test_recommend_channel(lookup, want):
    assert channel.recommend_channel(_p(**lookup))[0] == want


def test_published_posting_address_counts_when_coldstart_has_none():
    p = _p()
    p.email = "pat.lee@acme.com"
    assert channel.recommend_channel(p) == ("email", "address published in the posting")
    assert channel.email_of(p) == "pat.lee@acme.com"


def test_choose_prefers_an_emailable_person_over_rank():
    top = _p("Pat Lee", email="pat@acme.com", status="invalid")
    second = _p("Ana Diaz", email="ana@acme.com", status="valid", confidence=0.97)
    person, ch, _why = channel.choose([top, second])
    assert (person.name, ch) == ("Ana Diaz", "email")
    person, ch, _why = channel.choose([top])
    assert (person.name, ch) == ("Pat Lee", "linkedin")
    assert channel.choose([]) is None


def _fake_coldstart(monkeypatch, fn):
    api = types.ModuleType("coldstart.api")
    api.find_emails = fn
    monkeypatch.setitem(sys.modules, "coldstart.api", api)
    import coldstart
    monkeypatch.setattr(coldstart, "api", api, raising=False)


def test_find_emails_calls_coldstart_with_names_and_titles(monkeypatch):
    calls = []

    def find(domain, contacts, company="", verify=True):
        calls.append((domain, contacts, company))
        return [{"name": c["name"], "email": "pat@acme.com", "status": "valid", "confidence": 0.97} for c in contacts]

    _fake_coldstart(monkeypatch, find)
    people = channel.find_emails("acme.com", [Person("Pat Lee", "Recruiter")], company="Acme")
    assert calls == [("acme.com", [{"name": "Pat Lee", "title": "Recruiter"}], "Acme")]
    assert people[0].lookup["email"] == "pat@acme.com"


def test_find_emails_fails_soft(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("smtp down")

    _fake_coldstart(monkeypatch, boom)
    people = channel.find_emails("acme.com", [Person("Pat Lee", "Recruiter")])
    assert people[0].lookup == {}
    assert channel.find_emails("", [Person("Pat Lee", "Recruiter")])[0].lookup == {}
