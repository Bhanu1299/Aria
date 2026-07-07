# 02 — Debugging

Use this when anything is broken, failing, or surprising. The core
discipline: understand the bug BEFORE changing code. A fix you can't
explain is a bug you've hidden.

## The loop

### 1. Reproduce it

Find the smallest command that shows the failure, and run it yourself.
- If you can't reproduce it, you can't verify a fix — getting a
  reproduction IS the task right now.
- Save the exact command; you'll re-run it to prove the fix.

### 2. Read the real error

Read the full error output, bottom to top. The last line names the
error; the traceback names the exact file and line. Go read that line
and the ten lines around it.
- Do NOT skim and pattern-match ("probably an import issue"). What the
  error actually says beats what it usually means.

### 3. One hypothesis, stated out loud

Write: "I think ___ because ___. If I'm right, then ___ should be true."
- ONE hypothesis at a time. Not three maybes.
- It must be checkable: a print/log, a smaller test, reading a value,
  running one function in isolation.

### 4. Check it — cheapest test first

Run the check. Two outcomes:
- **Confirmed** → you now know the root cause. Go to 5.
- **Wrong** → good, you've eliminated it. Say so, form the next
  hypothesis from what you just learned. Back to 3.

### 5. Fix the root cause

Make the smallest change that removes the cause — not one that hides
the symptom (no blanket try/except, no sleep(), no "just restart it"
unless the root cause genuinely is state).

### 6. Prove it

Re-run the exact reproduction from step 1 — it must pass now. Then run
the surrounding tests to check you broke nothing. If the project has a
test suite, add a test that would have caught this bug.

## Hard rules

- **Never fix blind.** No code changes before you have a confirmed (or
  at minimum, stated and testable) hypothesis.
- **Three failed fixes = stop.** You're guessing. Go back to step 1,
  re-read everything, and question your assumptions ("is the code I'm
  editing even the code that runs?").
- **Changed it and the error is different?** That's progress, not
  failure. Start the loop again on the new error.
- **Two bugs can wear one mask.** If the fix works "sometimes," suspect
  a second cause.
