# Aria — Daily Voice Test Questions

<!-- GENERATED from tests/daily_questions.json — edit the JSON, then run:
     venv/bin/python daily_check.py list --md > tests/daily_questions.md -->

Read these aloud to Aria (hotkey or wake word), then run `venv/bin/python daily_check.py report` to see what failed.
**Bold** questions are the ~3-minute daily core set; the rest are the weekly sweep.

## Knowledge (direct LLM answer)

- **Who wrote The Great Gatsby?**
  - Expected: Says F. Scott Fitzgerald in one sentence, no tools needed.
- What's the capital of Australia?
  - Expected: Says Canberra.

## Web search (current info)

- **What's the weather right now?**
  - Expected: Uses web search, speaks current conditions for Buffalo.
- Search the web for the latest Python version
  - Expected: Speaks the current release number, not a stale one from memory.

## Conversation mode (follow-up, no hotkey)

- **(right after the weather answer, into the open mic) What about tomorrow?**
  - Expected: Mic pop plays after the previous answer; follow-up is understood WITHOUT hotkey or wake word and answers for tomorrow.
- (into the open mic) Thanks
  - Expected: Aria says 'Anytime.' and the conversation window closes.

## App launching

- **Open Notes**
  - Expected: Notes opens WITHOUT stealing focus from the current app; Aria confirms.

## Browser navigation

- Open github.com
  - Expected: GitHub opens in the default browser; Aria says 'Opening GitHub'.

## Mac system control

- Turn the brightness up
  - Expected: Screen brightness visibly increases; Aria confirms.

## Media playback

- **Play some jazz**
  - Expected: Music starts (Apple Music/Spotify/YouTube per router); Aria names what it's playing.
- **Pause**
  - Expected: Playback pauses. Single-word command must be accepted.
- What song is playing?
  - Expected: Names the current track, or says nothing is playing.

## Screen intelligence

- **What's on my screen right now?**
  - Expected: Describes the actual visible window contents — never guesses. Needs Screen Recording permission.
- Show me where the Apple menu is
  - Expected: Draws a box near the top-left Apple menu and says where it is. Known weak link: ~80% coordinate accuracy.

## Messaging (read + confirm-cancel)

- **Read my latest text messages**
  - Expected: Summarizes recent iMessages, does not dump raw data.
- Text mom saying I'll call tonight — then answer NO at the confirmation
  - Expected: Aria states the message and asks to confirm BEFORE sending; saying no must abort with nothing sent.

## Memory (store + recall)

- **Remember that my favorite coffee is cold brew**
  - Expected: Acknowledges saving it.
- **What's my favorite coffee?**
  - Expected: Says cold brew — recalled from vector memory, works even next session.

## Email / calendar / reminders

- **Check my email**
  - Expected: Summarizes recent inbox (needs Gmail setup); graceful message if not configured.
- What's on my calendar today?
  - Expected: Reads today's events or says the day is clear.
- Remind me to stretch in two hours
  - Expected: Confirms the reminder; it should actually fire later via cron.

## Job search

- Search for machine learning engineer jobs
  - Expected: Speaks top results with company names; 'tell me more about the second job' should then work.

## Morning briefing

- Give me my briefing
  - Expected: Weather + calendar + email + news in one spoken summary.

## Multi-step browser research

- Compare the price of AirPods Pro on Amazon and Best Buy
  - Expected: Runs the background browser loop, speaks progress, returns both prices. Browser must not steal focus.

## Code execution

- Write a Python script that prints the first ten prime numbers and run it
  - Expected: Writes, runs, and speaks the result.

## Self-report

- **How have you been performing this week?**
  - Expected: Reads the flight-recorder stats: command count, failure count, worst tool.

## Robustness / graceful failure

- (hold the hotkey and say nothing)
  - Expected: Says 'I didn't catch that' — never crashes. (Not log-matched; verify by ear.)
- Flurble the wombat sideways
  - Expected: Asks for clarification or admits it can't — never pretends it did something.
