# Domain Context

## Claude.ai DOM — what to look for
The response text lives inside a streaming div.
Poll until streaming stops before reading full text.
If DOM structure changes, this is the first thing to debug.

## Whisper base model
~140MB download on first run.
Stored in ~/.cache/huggingface/ after first run.
Subsequent runs load from cache instantly.

## pynput on macOS
Requires Accessibility permission to capture global hotkeys.
Without it, hotkey silently does nothing.
Detect this at startup and print a clear error.

## rumps + pynput threading
rumps runs its own main loop.
pynput listener runs in a background thread.
Do not run both on the same thread. Test this early.
