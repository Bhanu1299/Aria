# 03 — Writing Code

Rules that apply the moment you write or edit code, in any task.

## Before the first line

1. Read the file you're editing — at least the whole function/class
   you're touching and its callers.
2. Find one existing piece of similar code in the project and match its
   style: naming, imports, error handling, how it's tested. The
   codebase's conventions beat your preferences.
3. Check for an existing helper before writing a new one. Duplicating a
   utility that already exists is a bug you're planting.

## While writing

4. **Smallest diff that works.** Change what the task needs, nothing
   else. No drive-by refactors, no reformatting untouched lines, no
   "while I'm here" improvements — they hide your real change and add
   risk.
5. **Handle the failure paths.** For every call that can fail (file I/O,
   network, subprocess, parsing user data), decide what happens when it
   does. A graceful fallback or a clear error message — never an
   unhandled crash, and never a bare `except: pass` that hides it.
6. **No placeholders, ever.** No TODO, no stub, no hardcoded fake value
   "for now". Code you commit works now or doesn't exist.
7. **Names carry meaning.** A reader should understand a function from
   its name and signature without opening it. If you can't name it
   simply, it's doing too much — split it.
8. **Comments explain constraints, not code.** Write a comment only for
   something the code can't say (why a weird workaround exists, what
   invariant must hold). Never narrate ("increment the counter").
9. **Simple and working beats clever and fragile.** If there's a boring
   obvious way, use it.

## Test-first (when the project has a test suite)

10. Write the test for the new behavior FIRST. Run it — it must FAIL
    (if it passes before your change, the test is broken or the feature
    already exists).
11. Write the minimal code to make it pass. Run it — it must PASS.
12. Run the wider test suite for the area you touched.

Skipping the "watch it fail" step is how you end up with tests that
test nothing.

## After writing

13. Run the code — actually execute the path you changed, not just the
    tests (see `04-verify-done.md`).
14. Re-read your diff top to bottom as a reviewer would: leftover debug
    prints? unused imports? a case you forgot?
