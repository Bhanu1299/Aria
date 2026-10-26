"""tests/test_company_fact.py — what a company solves: which pages are read, and the LLM's
pick is kept only when it's grounded in those pages."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aria.features.outreach import company_fact as cf

HOME = ("The AI Platform for ECCs and PSAPs. Aurelian is built to support PSAPs before, during, and "
        "after every call, so telecommunicators can deliver fast, consistent support.")
STUDY = ("MACC 911 Case Study. Refocusing on priorities by automating non-emergency calls. MACC 911, the "
         "unified dispatch center for Grant County, WA, uses AVA to answer non-emergency calls.")
PAGES = {"https://aurelian.com/studies/macc-911": STUDY, "https://aurelian.io": HOME}


def _llm(monkeypatch, reply):
    calls = []

    def complete_checked(task, messages, validate, system=None, max_tokens=1024):
        calls.append({"task": task, "content": messages[0]["content"]})
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(text=text, stop_reason="end_turn"), validate(text)
    monkeypatch.setattr(cf.llm_client, "complete_checked", complete_checked)
    return calls


def _find(**kw):
    return cf.find("Aurelian", "aurelian.io", source="Published on aurelian.com/studies/macc-911",
                   fetch=lambda url: PAGES.get(url, ""), **kw)


def test_pages_in_order_source_on_the_company_site_then_home_then_posting():
    urls = cf.candidate_urls("aurelian.io", "Published on aurelian.com/studies/macc-911; github.com/x/y",
                             "https://jobs.ashbyhq.com/aurelian/1")
    assert urls == ["https://aurelian.com/studies/macc-911", "https://aurelian.io",
                    "https://jobs.ashbyhq.com/aurelian/1"]
    assert cf.candidate_urls("aurelian.io", "Coldstart (guess)") == ["https://aurelian.io"]


def test_a_grounded_fact_is_kept(monkeypatch):
    calls = _llm(monkeypatch, {
        "problem": "Aurelian helps 911 dispatch centers (PSAPs) by automating non-emergency calls.",
        "detail": "MACC 911, the unified dispatch center for Grant County, uses AVA to answer non-emergency calls.",
        "url": "https://aurelian.com/studies/macc-911"})
    fact = _find()
    assert fact and fact.url == "https://aurelian.com/studies/macc-911" and "Grant County" in fact.detail
    assert calls[0]["task"] == "outreach_company_fact"
    assert "Page: https://aurelian.com/studies/macc-911" in calls[0]["content"]


@pytest.mark.parametrize("reply", [
    {"none": True},
    "not json at all",
    {"problem": "Aurelian automates non-emergency calls for dispatch centers.",
     "detail": "Aurelian raised a $40M Series B led by Sequoia.",                         # not on the page
     "url": "https://aurelian.com/studies/macc-911"},
    {"problem": "Aurelian automates non-emergency calls for dispatch centers.",
     "detail": "MACC 911 uses AVA to answer non-emergency calls.", "url": "https://aurelian.com/about"},  # page not read
    {"problem": "Great team.", "detail": "MACC 911 uses AVA.", "url": "https://aurelian.io"},  # too thin
])
def test_ungrounded_or_empty_picks_are_dropped(monkeypatch, reply):
    _llm(monkeypatch, reply)
    assert _find() is None


def test_no_page_loads_means_no_llm_call(monkeypatch):
    calls = _llm(monkeypatch, {"none": True})
    assert cf.find("Aurelian", "aurelian.io", fetch=lambda url: "") is None and calls == []


def test_grounded_ignores_the_company_name():
    assert not cf.grounded("Aurelian Aurelian rocks", "nothing here", company="Aurelian")
    assert cf.grounded("automating non-emergency calls", STUDY)
