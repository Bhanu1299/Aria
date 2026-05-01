from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildPromptStaticTerms(unittest.TestCase):
    def test_build_prompt_includes_static_terms(self) -> None:
        import voice_keyterms

        prompt = voice_keyterms.build_prompt()
        self.assertIn("LinkedIn", prompt)
        self.assertIn("Python", prompt)
        self.assertIn("Playwright", prompt)


class TestBuildPromptIdentitySkills(unittest.TestCase):
    def test_build_prompt_includes_identity_skills(self) -> None:
        import voice_keyterms

        fake_identity = {"skills": ["FastAPI", "Redis"]}
        with patch.object(voice_keyterms, "_load_identity", return_value=fake_identity):
            prompt = voice_keyterms.build_prompt()

        self.assertIn("FastAPI", prompt)
        self.assertIn("Redis", prompt)


class TestBuildPromptMissingIdentity(unittest.TestCase):
    def test_build_prompt_handles_missing_identity(self) -> None:
        import voice_keyterms

        # Simulate a missing / corrupt identity.json by raising on open
        with patch("builtins.open", side_effect=FileNotFoundError("no file")):
            # _load_identity catches exceptions and returns {}
            prompt = voice_keyterms.build_prompt()

        self.assertTrue(len(prompt) > 0)
        # Static terms should still be present
        self.assertIn("LinkedIn", prompt)


class TestTranscriberPassesInitialPrompt(unittest.TestCase):
    def test_transcriber_passes_initial_prompt_to_model(self) -> None:
        """Verify that Transcriber.transcribe() forwards initial_prompt to the model."""
        # Stub faster_whisper so Transcriber can be imported without the real dep
        fake_fw = types.ModuleType("faster_whisper")

        mock_model_instance = MagicMock()
        # transcribe() returns (segments_iterable, info)
        mock_model_instance.transcribe.return_value = (iter([]), MagicMock())

        fake_fw_class = MagicMock(return_value=mock_model_instance)
        fake_fw.WhisperModel = fake_fw_class

        with patch.dict(sys.modules, {"faster_whisper": fake_fw}):
            # Re-import to pick up the stubbed module
            if "transcriber" in sys.modules:
                del sys.modules["transcriber"]
            from transcriber import Transcriber

            t = Transcriber()
            t.transcribe("/tmp/fake.wav", initial_prompt="LinkedIn Python Playwright")
            t.stop()

        # The model's transcribe() must have been called with initial_prompt kwarg
        call_kwargs = mock_model_instance.transcribe.call_args
        self.assertIsNotNone(call_kwargs, "model.transcribe was never called")
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        self.assertIn(
            "initial_prompt",
            kwargs,
            "initial_prompt not forwarded to model.transcribe()",
        )
        self.assertEqual(kwargs["initial_prompt"], "LinkedIn Python Playwright")


if __name__ == "__main__":
    unittest.main()
