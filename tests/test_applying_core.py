from __future__ import annotations

import threading
import time

import pytest

from aria.features.applying import control, tabs, walls


@pytest.fixture(autouse=True)
def registry(monkeypatch, tmp_path):
    monkeypatch.setattr(tabs, "REGISTRY_PATH", tmp_path / "apply_tabs.json")
    return tmp_path / "apply_tabs.json"


def _job(row=15, company="Scale AI", role="Software Engineer - New Grad",
         url="https://job-boards.greenhouse.io/scaleai/jobs/4730836005"):
    return {"row": row, "company": company, "role": role, "url": url, "location": "SF"}


def test_add_update_roundtrip_persists(registry):
    e = tabs.add(_job(), ats="greenhouse")
    assert e["state"] == tabs.FILLING and e["id"].startswith("apply-15-")
    tabs.update(e["id"], state=tabs.FILLED, needs_me=["Pronouns"])
    assert registry.exists()
    again = tabs.get(e["id"])
    assert again["state"] == tabs.FILLED and again["needs_me"] == ["Pronouns"]
    assert again["company"] == "Scale AI" and again["ats"] == "greenhouse"


def test_update_unknown_id_returns_none():
    assert tabs.update("apply-nope", state=tabs.FILLED) is None


def test_live_hides_closed():
    a = tabs.add(_job())
    b = tabs.add(_job(16, "SingleStore", "SWE", "https://job-boards.greenhouse.io/singlestore/jobs/8205427"))
    tabs.mark_closed(a["id"])
    assert [e["id"] for e in tabs.live()] == [b["id"]]


def test_by_company_matches_loosely():
    a = tabs.add(_job())
    tabs.add(_job(16, "SingleStore", "SWE", "https://x.example/1"))
    assert [e["id"] for e in tabs.by_company("scale")] == [a["id"]]
    assert [e["id"] for e in tabs.by_company("Scale A.I.")] == [a["id"]]
    assert tabs.by_company("ookla") == []


def test_is_open_for_row_or_url():
    tabs.add(_job())
    assert tabs.is_open_for({"row": 15, "url": ""})
    assert tabs.is_open_for({"row": None, "url": "https://job-boards.greenhouse.io/scaleai/jobs/4730836005"})
    assert not tabs.is_open_for({"row": 16, "url": "https://x.example/2"})


def test_corrupt_registry_reads_empty(registry):
    registry.write_text("{not json")
    assert tabs.live() == []


def test_walls_ignore_invisible_recaptcha():
    frames = [{"src": "https://www.google.com/recaptcha/api2/anchor?k=x", "visible": False},
              {"src": "https://www.google.com/recaptcha/api2/bframe?k=x", "visible": False}]
    assert walls.detect("Apply for this job First Name", "https://job-boards.greenhouse.io/x", frames) is None


def test_walls_visible_hcaptcha():
    frames = [{"src": "https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html#frame=challenge",
               "visible": True}]
    assert walls.detect("", "https://jobs.jobvite.com/ookla", frames) == "captcha"
    assert walls.detect("Please verify you are human", "https://x.example", []) == "captcha"


def test_walls_login_and_email_code():
    assert walls.detect("Sign in to continue\n[password-field]", "https://x.example/login", []) == "login"
    assert walls.detect("Sign in to apply", "https://x.example/job", []) is None   # no password box
    assert walls.detect("Enter the verification code we sent to your email", "https://x.example", []) == "email_code"


def test_checkpoint_blocks_until_resume():
    ctl = control.RunControl()
    ctl.pause()
    done = []
    t = threading.Thread(target=lambda: (ctl.checkpoint(), done.append(1)))
    t.start()
    time.sleep(0.15)
    assert done == [] and ctl.paused
    ctl.resume()
    t.join(1)
    assert done == [1]


def test_checkpoint_raises_after_stop():
    ctl = control.RunControl()
    ctl.pause()
    errs = []

    def wait():
        try:
            ctl.checkpoint()
        except control.Stopped:
            errs.append("stopped")
    t = threading.Thread(target=wait)
    t.start()
    ctl.stop()
    t.join(1)
    assert errs == ["stopped"]
    with pytest.raises(control.Stopped):
        ctl.checkpoint()


def test_privacy_notice_with_accept_and_decline_is_a_wall():
    assert walls.detect("Privacy notice\n\nI AcceptI Decline", "", []) == "agreement"     # buttons' innerText runs together
    assert walls.detect("I agree to the privacy policy", "", []) is None          # a form checkbox isn't a wall
