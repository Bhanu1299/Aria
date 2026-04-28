import sys
import time

sys.path.insert(0, "/Users/bhanuteja/Documents/trae_projects/Aria")

from transcriber import Transcriber, load_model
from voice_capture import VoiceCapture

print("[TEST] transcriber.py")
print("First run may download ~140MB Whisper model...")

model = load_model()
transcriber = Transcriber(model)

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
