import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speaker import Speaker, speak

print("[TEST] speaker.py")

try:
    speak("Aria is working correctly.")
    print("PASS — say command completed successfully")
except FileNotFoundError:
    print("FAIL — macOS say command not found (this only works on macOS)")
except Exception as e:
    print(f"FAIL — unexpected error: {e}")
