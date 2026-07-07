# Aria — Public Release Checklist

Status: **ON HOLD** (2026-07-07). Feature work continues; revisit when core is daily-reliable.
This is the complete list of what must happen before strangers can install Aria.

---

## 1. Privacy & Trust (highest stakes — one bad thread kills the product)

- [ ] **Consent screen on first run**: plain-language explanation that voice audio is
      transcribed locally, but screenshots, highlighted text, and queries are sent to
      Groq/Anthropic servers. Require explicit opt-in per capability.
- [ ] **Visible capture indicator**: menu bar icon change + optional sound whenever
      `screencapture` fires (screen_qa/screen_explain). Never capture silently for a public build.
- [ ] **Privacy policy** (hosted URL): what is sent where, retention, third parties
      (Groq, Anthropic), how to delete local data (identity.json, ChromaDB, SQLite).
- [ ] **Data-wipe command**: "Aria, forget everything" → wipes identity.json learned facts,
      ChromaDB collection, SQLite memory, session notes.
- [ ] **Redaction pass (stretch)**: blur password fields / detect password managers before
      sending screenshots.
- [ ] **iMessage/WhatsApp/Gmail safety**: confirm-before-send is mandatory (voice confirm),
      never auto-send on a public build.

## 2. Onboarding & Permissions

- [ ] **First-run wizard**: guided flow for Microphone → Accessibility → Screen Recording
      permissions, with live checks (detect missing permission, deep-link to the right
      System Settings pane: `x-apple.systempreferences:com.apple.preference.security?Privacy_*`).
- [ ] **Graceful denial paths**: every feature must degrade with a spoken explanation when
      its permission is missing (screen features already do this — audit the rest).
- [ ] **API key setup**: BYO-key flow (paste Groq + optional Anthropic key into a settings UI),
      OR a hosted proxy backend with metered accounts. BYO-key is the realistic v1.
- [ ] **Google OAuth**: plugins/productivity/setup.py flow needs to be click-through simple,
      with its own consent screen (Gmail read scope is sensitive — Google verification required
      for >100 users; budget 4-6 weeks for their review).

## 3. Packaging & Distribution

- [ ] **Signed, notarized .app** via py2app (or PyInstaller .app bundle): Apple Developer
      account ($99/yr), codesign + notarytool in CI.
- [ ] **Bundle weight**: faster-whisper model (~140MB) — download on first run with progress,
      don't ship in the bundle. Same for aria.onnx (ship, it's tiny) and
      sentence-transformers/torch (~2GB — consider swapping embedder to a smaller ONNX model
      or making semantic memory optional).
- [ ] **Python runtime**: bundle a private venv inside the .app; zero "pip install" for users.
- [ ] **Auto-update**: Sparkle framework or a simple version-check + download prompt.
- [ ] **Crash reporting (opt-in)**: local log file + "send report" button; no silent telemetry.
- [ ] **Login item**: "Start Aria at login" toggle (SMAppService).

## 4. Reliability bar (must hold before any of the above matters)

- [ ] Fix pre-existing test failures: tests/test_computer_use_* (drifted after 5F llm layer),
      test_jobs.py (3), test_llm_groq.py (1), tests/training/* (need optional deps
      audiomentations + edge-tts — either add to requirements-dev or mark skipif).
- [ ] Script-style tests (test_voice_capture, test_hotkey, test_browser, test_speaker,
      test_transcriber) → convert to pytest-with-skip-markers so `pytest tests/` is green.
- [ ] **Delete dead code**: main.py `_handle_intent` + router.py classifier path are no longer
      called (agent tool-loop is the only path). ~700 lines of drift risk. Verify nothing
      imports router.route, then remove.
- [ ] **End-to-end soak test**: 7 days of daily-driver use, log every misrouted/failed command
      to a file, fix the top offenders (tests/checklist.md is the manual pass).
- [ ] **Rate-limit behavior**: Groq free tier throttles — verify the tier failover chain
      degrades to a spoken "I'm rate limited, give me a minute" instead of silence.
- [ ] **Wake word false-positive rate**: measure over a workday; tune aria.onnx threshold
      (0.6–0.8) or retrain with more negative samples.

## 5. Cost model

- [ ] Estimate per-user daily token spend (agent smart-tier hits Sonnet first — measure
      real usage, consider cheap-tier default with smart-tier escalation).
- [ ] Decide: BYO API keys (v1, free for you) vs hosted proxy (needs billing, abuse controls).

## 6. Legal / misc

- [ ] License decision (closed source vs MIT for the client + hosted brain).
- [ ] Terms of service if any hosted component exists.
- [ ] App name/trademark check ("Aria" is used by Opera's AI — a rename may be needed
      for public distribution).
- [ ] Website + demo video (screen recording of hotkey → voice → result, 60s).

---

*Maintained by Claude sessions. When release work resumes, work top to bottom;
sections 1–2 are the moat, section 3 is the grind, section 4 is the gate.*
