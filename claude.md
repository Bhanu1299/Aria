# Aria — Claude Code Operating Instructions

## Who I am
Bhanu Teja Veeramachaneni. MS CS graduate, active job seeker.
Full stack + GenAI background. Python fluent, Swift beginner.
Building Aria — a background Mac voice agent.

## The product
Aria runs silently on Mac. Hotkey → voice → background browser
execution → spoken result. Screen never moves. User stays in flow.
Throwaway prototype proving the core loop works.

## Rules — always follow
- Full file paths on every file created or modified, no exceptions
- Code must be copy-pasteable and run immediately
- No placeholders, no TODOs, no stubs
- Add error handling wherever something can fail
- Simple and working beats clever and fragile
- Read .claude/project-state.md before starting any session
- Write .claude/session-log.md before ending any session

## Stack
Python 3.11+, Playwright, faster-whisper (base model),
pynput, sounddevice, soundfile, rumps, macOS say command

## Non-negotiables
- Browser never steals focus
- Whisper preloaded at startup in main.py
- All processing local except Claude.ai interaction
- No Swift, no API calls, no alternative tools
