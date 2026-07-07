# 01 — Breaking Down a Task

Use this when asked to build or change something. The goal: never start
coding with a vague plan, and never carry more than one step in your
head at a time.

## Phase 1 — Understand (no code yet)

1. Write down the goal in one sentence and the success test:
   "Done means: ___ happens when ___."
2. List what you don't know yet (which file handles X? does a helper
   already exist? what does the data look like?). Each unknown is a
   question you answer by LOOKING, not by assuming.

## Phase 2 — Explore (still no code)

3. Answer each unknown by reading the project:
   - Search for related names: `grep -rn "keyword" --include="*.py"`
   - Read the files you'll change, and one similar existing feature —
     copy its patterns (imports, error handling, naming, tests).
4. Write one sentence on how the change fits the existing design.
   If your plan fights the codebase's patterns, change your plan.

## Phase 3 — Plan

5. Break the work into steps where each step:
   - changes ONE thing,
   - can be verified on its own (a test passes, a command prints the
     right output, the app does the new behavior),
   - leaves the project working (never plan a step that breaks things
     for a later step to fix).
6. Order the steps so the riskiest or most uncertain one comes FIRST.
   If the hard part is impossible, find out before writing the easy 80%.
7. Show the plan as a numbered list. For anything non-trivial, let the
   user see it before you execute.

## Phase 4 — Execute

8. Do step 1. Verify it (run the test / command). State the evidence.
9. Only then do step 2. Repeat.
10. If a step fails or reveals the plan was wrong: STOP. Say what you
    learned, revise the remaining steps, then continue. Do not push
    through a broken plan.

## Red flags — stop and go back a phase

- "I'll just start writing and figure it out" → back to Phase 1.
- You're editing a file you never read → back to Phase 2.
- A step is "wire everything together" → too big, split it (Phase 3).
- You did three steps without verifying any → back to step 8.
