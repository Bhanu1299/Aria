"""adapters/workday.py — Workday: every company is its own tenant (acme.wd5.myworkdayjobs.com)
with its own account, so each tenant host has its own Keychain entry.

login — a state machine over Workday's data-automation-id markers (classify(), pure):
  posting         → click Apply (adventureButton)
  start           → "Autofill with Resume" when your resume file is set (user, 2026-10-24: Workday
                    fills the experience / education sections from it; the bank's answers still
                    overwrite every field it knows), else "Use My Last Application", else "Apply Manually"
  upload          → the autofill step: your resume attached, its upload waited for, then Continue
  sign_in         → the tenant's Keychain login typed straight into the page; no saved login →
                    Create Account; a rejected login → handed to you (logged)
  create_account  → your application email + a generated password, saved to the Keychain
                    BEFORE the form is sent; the terms box is yours to tick (Aria waits, once
                    per tenant); then Create Account
  verify_email    → the code / link from the Gmail account that IS the application email
                    (email_code); none after a few polls → handed to you
  signed_in       → the wizard
fill — the form engine page by page with Workday's own field ids mapped to bank keys (FIELD_KEYS),
  "Save and Continue" between pages, never Submit. A page counts as left only when its step /
  questions changed (Workday re-draws a page it refuses to leave). A page Workday won't leave
  (its validation errors) stops the run there, and every error it shows comes back as a field
  for you — nothing is guessed to get past it.

Passwords: read from the Keychain inside the browser step that types them; never in a prompt,
a log line, a progress message, a review row or a correction snapshot.
"""
from __future__ import annotations

import logging
import re
import time
from urllib.parse import urlparse

from aria.features.ats import apply_target, corrections, email_code
from aria.features.ats.adapters import Adapter, ApplyContext, FillReport
from aria.features.ats.apply_target import Tab
from aria.system import keychain

logger = logging.getLogger(__name__)

MAX_STEPS = 12
AFTER_CLICK_SECS = 2.5
# Workday draws a step's content after its progress bar (a spinner in between): a page that
# isn't recognizable yet is polled this many times before it counts as unrecognized.
LOAD_WAITS = 8
LOAD_POLL_SECS = 1.5
UPLOAD_WAIT_SECS = 45

STATE_JS = r"""() => {
  const vis = e => { if (!e) return false; const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const aid = a => vis(document.querySelector('[data-automation-id="' + a + '"]'));
  const box = document.querySelector('[data-automation-id="createAccountCheckbox"]');
  const code = document.querySelector('[data-automation-id="verificationCode"], input[name*="verification" i], input[autocomplete="one-time-code"]');
  const step = document.querySelector('[data-automation-id="progressBarActiveStep"]');   // the bar itself lists every step
  const fileIn = document.querySelector('[data-automation-id="file-upload-input-ref"], input[type=file]');
  const done = document.querySelector('[data-automation-id="file-upload-successful"], [data-automation-id="delete-file"]');
  return {
    posting: aid('adventureButton'),
    start: aid('applyManually') || aid('useMyLastApplication') || aid('autofillWithResume'),
    last_app: aid('useMyLastApplication'),
    autofill: aid('autofillWithResume'),
    file_input: !!fileIn,
    uploaded: vis(done),
    sign_in: aid('signInSubmitButton') || aid('signInContent') || (aid('password') && !aid('verifyPassword')),
    create: aid('createAccountSubmitButton') || aid('verifyPassword'),
    create_link: aid('createAccountLink'),
    email_button: aid('SignInWithEmailButton'),
    password_box: aid('password'),
    step: step ? (step.innerText || '').trim().slice(0, 80) : '',
    terms: vis(box), terms_checked: !!(box && (box.checked || box.getAttribute('aria-checked') === 'true')),
    code_box: vis(code),
    wizard: aid('progressBar') || aid('applyFlowPage') || aid('bottom-navigation-next-button'),
    errors: Array.from(document.querySelectorAll('[data-automation-id="errorMessage"], [data-automation-id="inputAlert"], ' +
      '[data-automation-id="errorBanner"], [role=alert]')).filter(vis).map(e => (e.innerText || '').trim()).filter(Boolean).slice(0, 8),
    text: (document.body ? document.body.innerText : '').slice(0, 3000),
  };
}"""

# The wizard's progress bar shows on the sign-in step too ("current step 1 of 7 Create
# Account/Sign In"): on that step the progress bar is not "signed in".
_LOGIN_STEP = re.compile(r"sign ?in|create account|log ?in", re.I)
_REVIEW_STEP = re.compile(r"\breview\b", re.I)
_AUTOFILL_STEP = re.compile(r"autofill|upload (?:your )?(?:resume|cv)", re.I)
_VERIFY_TEXT = re.compile(r"verify your (?:email|account)|verification (?:email|link|code)|"
                          r"check your (?:email|inbox)|we(?:'ve| have)? sent (?:you )?an? (?:email|code)", re.I)


def classify(s: dict) -> str:
    """posting | start | upload | sign_in | create_account | verify_email | signed_in | loading | unknown.
    loading = the wizard is on its sign-in step but hasn't drawn the sign-in box yet.
    upload = the "Autofill with Resume" step (its file box; My Experience's resume box isn't it)."""
    login_step = bool(_LOGIN_STEP.search(s.get("step") or ""))
    on_autofill = _AUTOFILL_STEP.search(s.get("step") or "") or (
        not s.get("step") and re.search(r"autofill with resume", (s.get("text") or "")[:600], re.I))
    if s.get("file_input") and on_autofill and not (s.get("sign_in") or s.get("create") or s.get("code_box")):
        return "upload"
    if s.get("wizard") and not (s.get("sign_in") or s.get("create") or s.get("code_box") or login_step):
        return "signed_in"
    if s.get("create"):
        return "create_account"
    if s.get("code_box") or _VERIFY_TEXT.search(s.get("text") or ""):
        return "verify_email"
    if s.get("sign_in"):
        return "sign_in"
    if s.get("start"):
        return "start"
    if s.get("posting"):
        return "posting"
    return "loading" if s.get("wizard") else "unknown"


def tenant(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower()


def senders(host: str) -> list:
    """Who a tenant's verification mail comes from: Workday itself, and the company's name."""
    name = host.split(".")[0] if host else ""
    return ["myworkday.com", "myworkdayjobs.com"] + ([name] if name and not name.startswith("wd") else [])


def _click_aid(page, aid: str) -> bool:
    loc = page.locator(f'[data-automation-id="{aid}"]').first
    if loc.count() == 0:
        return False
    try:
        loc.click(timeout=3000)
    except Exception:          # Workday lays a click_filter overlay over its buttons
        loc.click(force=True, timeout=3000)
    return True


def _next_button(page) -> bool:
    return _click_aid(page, "bottom-navigation-next-button") or _click_aid(page, "pageFooterNextButton")


def _type_aid(page, aid: str, value: str) -> bool:
    loc = page.locator(f'input[data-automation-id="{aid}"], [data-automation-id="{aid}"] input').first
    if loc.count() == 0:
        return False
    loc.fill(value, timeout=5000)
    loc.evaluate("e => e.dispatchEvent(new Event('blur', {bubbles: true}))")
    return True


class WorkdayAdapter(Adapter):
    name = "workday"
    VISION_FALLBACK = False
    FIELD_KEYS = (
        (r"legalNameSection_firstName", "first_name"), (r"legalNameSection_lastName", "last_name"),
        (r"preferredNameSection_firstName", "preferred_name"),
        (r"addressSection_addressLine1", "address_line1"), (r"addressSection_city", "city"),
        (r"addressSection_postalCode", "zip"), (r"addressSection_countryRegion", "state"),
        (r"countryDropdown|addressSection_country", "country"),
        (r"phone-number|phoneNumber", "phone"), (r"email", "email"),
        (r"linkedinQuestion|linkedIn(?:Url)?", "linkedin"),
    )

    def field_key(self, raw: dict) -> str:
        key = super().field_key(raw)
        if key == "phone" and re.search(r"extension|\bext\b|country ?phone ?code|device type",
                                        f"{raw.get('label') or ''} {raw.get('aid') or ''} {raw.get('name') or ''}", re.I):
            return ""                 # the extension / dial code inside the phone section is not the number
        return key

    # -- helpers ------------------------------------------------------------
    def _state(self, tab: Tab) -> dict:
        return tab.run(lambda page: page.evaluate(STATE_JS))

    def _now(self, tab: Tab) -> str:
        return classify(self._state(tab))

    def _after_click(self, tab: Tab) -> None:
        time.sleep(AFTER_CLICK_SECS)
        apply_target.settle(tab)

    def _hand_off(self, tab: Tab, ctx: ApplyContext, kind: str, why: str) -> bool:
        """The user clears it in the tab; True once the wizard shows."""
        corrections.record(ctx.job, "login_failed", "login", f"Workday {tenant(ctx.target.url)}: {why}",
                           ats=self.name, run=tab.run)
        if ctx.wait_fn is not None and ctx.wait_fn(kind, cleared=lambda: self._now(tab) == "signed_in"):
            return True
        return False

    def _upload_resume(self, tab: Tab, ctx: ApplyContext) -> bool:
        """The autofill step: attach the resume, wait for Workday to take it, Continue."""
        from aria.features.ats import planner
        resume = planner.resume_path()
        if not resume:
            return False
        ctx.progress_fn("Uploading your resume so Workday can autofill from it.")
        tab.run(lambda page: page.locator('[data-automation-id="file-upload-input-ref"], input[type=file]')
                .first.set_input_files(resume, timeout=10000))
        name = resume.rsplit("/", 1)[-1].lower()
        deadline = time.time() + UPLOAD_WAIT_SECS
        while time.time() < deadline:
            time.sleep(1.5)
            st = self._state(tab)
            if st.get("uploaded") or name in (st.get("text") or "").lower():
                break
        tab.run(_next_button)
        self._after_click(tab)
        return True

    @staticmethod
    def _email() -> str:
        import aria.features.answer_bank as answer_bank
        found = answer_bank.lookup_key("email")
        return found.value if found else ""

    # -- the state machine --------------------------------------------------
    def login(self, tab: Tab, ctx: ApplyContext) -> bool:
        host = tenant(ctx.target.url or ctx.job.get("url", ""))
        email = self._email()
        say = ctx.progress_fn
        tried_sign_in = created = verified = email_form = uploaded = False
        started = time.time()
        waits = 0
        for _ in range(MAX_STEPS + LOAD_WAITS):
            s = self._state(tab)
            state = classify(s)
            if state in ("loading", "unknown") and waits < LOAD_WAITS:
                waits += 1
                time.sleep(LOAD_POLL_SECS)
                continue
            if state == "signed_in":
                ctx.target.url = tab.run(lambda page: page.url)
                return True
            if state == "posting":
                tab.run(lambda page: _click_aid(page, "adventureButton"))
                self._after_click(tab)
            elif state == "start":
                from aria.features.ats import planner
                if s.get("autofill") and planner.resume_path():
                    choice, line = "autofillWithResume", "Choosing Autofill with Resume."
                elif s.get("last_app"):
                    choice, line = "useMyLastApplication", "Using your last application on this Workday."
                else:
                    choice, line = "applyManually", "Choosing Apply Manually."
                say(line)
                tab.run(lambda page, c=choice: _click_aid(page, c))
                self._after_click(tab)
            elif state == "upload":
                if uploaded or not self._upload_resume(tab, ctx):
                    return self._hand_off(tab, ctx, "login", "the resume upload for autofill didn't go through")
                uploaded = True
            elif state == "sign_in":
                if s.get("email_button") and not s.get("password_box"):
                    # Newer tenants: Google / LinkedIn / "Sign in with email" first; the email
                    # and password boxes (and Create Account) appear after the email choice.
                    if email_form:
                        return self._hand_off(tab, ctx, "login", "the email sign-in form never appeared")
                    say("Choosing Sign in with email.")
                    tab.run(lambda page: _click_aid(page, "SignInWithEmailButton"))
                    email_form = True
                    self._after_click(tab)
                    continue
                cred = keychain.get(host, email)
                if cred is None:
                    if s.get("create_link") and not created:
                        say(f"No saved login for {host}. Creating an account with your application email.")
                        tab.run(lambda page: _click_aid(page, "createAccountLink"))
                        self._after_click(tab)
                        continue
                    return self._hand_off(tab, ctx, "login", "no saved login and no Create Account link")
                if tried_sign_in:
                    why = "; ".join(s.get("errors") or []) or "still on the sign-in page"
                    return self._hand_off(tab, ctx, "login", f"sign-in rejected ({why[:160]})")
                say(f"Signing in to {host} with your saved login.")

                def sign_in(page, cred=cred) -> None:
                    _type_aid(page, "email", cred.username)
                    _type_aid(page, "password", cred.password)
                    _click_aid(page, "signInSubmitButton")
                tab.run(sign_in)
                tried_sign_in = True
                self._after_click(tab)
            elif state == "create_account":
                if created:
                    why = "; ".join(s.get("errors") or []) or "the account wasn't created"
                    return self._hand_off(tab, ctx, "login", f"account creation failed ({why[:160]})")
                if not email:
                    return self._hand_off(tab, ctx, "login", "no email in your answer bank to create an account")
                cred = keychain.get(host, email)
                password = cred.password if cred else keychain.generate()
                if not keychain.save(host, email, password):
                    return self._hand_off(tab, ctx, "login", "couldn't save a password to the Keychain, so no account was made")
                def account_form(page, password=password) -> None:
                    _type_aid(page, "email", email)
                    _type_aid(page, "password", password)
                    _type_aid(page, "verifyPassword", password)
                tab.run(account_form)
                if s.get("terms") and not s.get("terms_checked"):
                    say("Workday's account terms are yours to accept.")
                    ticked = ctx.wait_fn is not None and ctx.wait_fn(
                        "terms", cleared=lambda: bool(self._state(tab).get("terms_checked")))
                    if not ticked:
                        corrections.record(ctx.job, "login_failed", "login", f"Workday {host}: terms box not ticked",
                                           ats=self.name, run=tab.run)
                        return False
                tab.run(lambda page: _click_aid(page, "createAccountSubmitButton"))
                created = True
                self._after_click(tab)
            elif state == "verify_email":
                if verified:
                    return self._hand_off(tab, ctx, "email_code", "the verification didn't go through")
                verified = True
                say("Looking for Workday's verification email.")
                found = email_code.wait_for(email, senders(host), since=started - 60)
                if found is None:
                    return self._hand_off(tab, ctx, "email_code", "no verification email found in the application inbox")
                if found[0] == "code":
                    tab.run(lambda page, c=found[1]: page.locator(
                        '[data-automation-id="verificationCode"] input, input[data-automation-id="verificationCode"], '
                        'input[name*="verification" i], input[autocomplete="one-time-code"]').first.fill(c, timeout=5000))
                    tab.run(lambda page: _click_aid(page, "verifyButton") or _click_aid(page, "submitButton"))
                else:
                    import aria.web.agent_browser as agent_browser
                    agent_browser.open_in_tab(found[1], settle_secs=3.0)       # a background tab; the form tab stays
                    tab.navigate(ctx.job.get("url") or ctx.target.url, settle_secs=2.0)
                self._after_click(tab)
            else:
                break
        return self._hand_off(tab, ctx, "login", f"stuck on an unrecognized page ({tab.run(lambda page: page.url)[:120]})")

    def fill(self, tab: Tab, ctx: ApplyContext) -> FillReport:
        s = self._state(tab)
        if classify(s) == "upload" and self._upload_resume(tab, ctx):     # picked up on the autofill step
            s = self._state(tab)
        if classify(s) != "signed_in":
            report = FillReport(pages=0, stopped_at=f"Workday {s.get('step') or 'sign-in'} step",
                                form_url=tab.run(lambda page: page.url),
                                error="Workday is still on its sign-in step, so nothing was filled.")
            ctx.report = report
            return report
        report = super().fill(tab, ctx)
        if report.stopped_at == "no fields" and not report.rows and _REVIEW_STEP.search(s.get("step") or ""):
            report.stopped_at = "Workday review page (filled from your last application)"
        if report.stopped_at.startswith("next did nothing"):
            errors = self._state(tab).get("errors") or []
            for n, err in enumerate(errors):
                report.rows.append({"name": f"workday_error_{n}", "label": err[:200], "kind": "text", "category": "fact",
                                    "value": "", "source": "needs_me", "required": True, "page": report.pages,
                                    "note": "Workday won't continue until this is answered", "status": "unknown"})
            report.stopped_at = f"Workday page {report.pages}: {len(errors) or 'unlisted'} error(s) — yours to answer"
        ctx.report = report
        return report
