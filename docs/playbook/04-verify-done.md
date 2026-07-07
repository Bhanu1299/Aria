# 04 — Verify Before Saying "Done"

Read this every time you are about to tell the user something works,
is fixed, or is complete. The rule is one sentence:

**Evidence before claims. If you didn't run it in this session and read
the output, you may not say it works.**

## The gate

Before the words "done", "fixed", "works", or "passing" leave your
mouth, all of these must be true:

- [ ] I ran the code path the task was about (not just a related test)
      and saw the expected behavior with my own eyes.
- [ ] I ran the tests for the area I touched, and I read the summary
      line — it says what I'm about to claim it says.
- [ ] The original request is fully satisfied — re-read the user's
      actual words; check each part of what they asked, not just the
      part I found interesting.
- [ ] Nothing I wrote contains a placeholder, stub, or "temporary" hack
      I forgot about.
- [ ] Any file I created or modified is listed in my summary with its
      full path.

If any box is unchecked, you are not done — go check it. If you cannot
check one (e.g., it needs a device or credential you don't have), say
exactly that: "Implemented, but I could not verify ___ because ___."
That sentence is honest; "it should work" is not.

## What counts as evidence

| Claim | Required evidence |
|---|---|
| "The tests pass" | You ran them now; the output shows 0 failures. |
| "The bug is fixed" | The original reproduction command now succeeds. |
| "The feature works" | You exercised the feature end-to-end and saw the result. |
| "It compiles / imports" | You ran the build or imported the module, no errors. |

"I wrote code that should do X" is not evidence of anything.

## Verify the RIGHT thing

Tests passing is necessary, not sufficient. A wrong config path, a
handler that's never registered, a function nobody calls — all pass
tests. So also run the real entry point the user will use and watch
the actual behavior once.

## Reporting

State the result plainly with the evidence attached:

> Done. `pytest tests/test_foo.py` → 12 passed. Ran
> `python main.py --demo`, the new command responds correctly.
> Files changed: /full/path/a.py, /full/path/b.py.

If something failed or was skipped, that goes FIRST, not buried at the
bottom.
