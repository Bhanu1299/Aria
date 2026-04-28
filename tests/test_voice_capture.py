import os
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voice_capture import VoiceCapture

RECORD_SECONDS = 3

print("[TEST] voice_capture.py")
print(f"Speak now... (recording for {RECORD_SECONDS} seconds)")

vc = VoiceCapture()

try:
    vc.start_recording()
except Exception as e:
    print(f"FAIL — could not start recording: {e}")
    sys.exit(1)

time.sleep(RECORD_SECONDS)

try:
    wav_path = vc.stop_recording()
except Exception as e:
    print(f"FAIL — could not stop recording: {e}")
    sys.exit(1)

# --- Verification ---

# 1. File exists
if not os.path.exists(wav_path):
    print(f"FAIL — file does not exist: {wav_path}")
    sys.exit(1)

# 2. File size > 0
size = os.path.getsize(wav_path)
if size == 0:
    print(f"FAIL — file is empty: {wav_path}")
    sys.exit(1)

# 3. Valid WAV via soundfile
try:
    info = sf.info(wav_path)
except Exception as e:
    print(f"FAIL — soundfile could not read WAV: {e}")
    sys.exit(1)

duration = round(info.duration, 1)
samplerate = info.samplerate

print(f"PASS — recorded {duration}s of audio at {samplerate}Hz")
