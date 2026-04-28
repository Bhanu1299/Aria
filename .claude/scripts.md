# Aria — Scripts & Commands Reference

## Setup (run once)
cd aria
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium

## Run Aria
source venv/bin/activate
python main.py

## Run individual module tests
python tests/test_hotkey.py
python tests/test_voice_capture.py
python tests/test_transcriber.py
python tests/test_browser.py
python tests/test_speaker.py

## Run all tests in sequence
for f in tests/test_*.py; do python "$f"; done

## Check permissions (macOS)
# Microphone: System Settings → Privacy & Security → Microphone
# Accessibility: System Settings → Privacy & Security → Accessibility

## Deactivate venv
deactivate
