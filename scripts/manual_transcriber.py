"""Manual check: record 3s from the real mic and transcribe it with Whisper."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aria.voice.transcriber import Transcriber
from aria.voice.voice_capture import VoiceCapture

print("[TEST] transcriber.py")
print("First run may download ~140MB Whisper model...")

transcriber = Transcriber()  # loads the Whisper model, blocks until ready

capture = VoiceCapture()
capture.start_recording()
print("Speak now... (recording for 3 seconds)")
time.sleep(3)
wav_path = capture.stop_recording()

result = transcriber.transcribe(wav_path)

if isinstance(result, str) and len(result) > 0:
    print(f'PASS — transcribed: "{result}"')
else:
    print("FAIL — result was empty or not a string")
    sys.exit(1)
