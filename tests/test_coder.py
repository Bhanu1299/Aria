"""Tests for coder.py — agentic code execution engine."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call


def _tool_use_block(name: str, input_: dict, tool_use_id: str = "tu_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_
    block.id = tool_use_id
    return block


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _end_turn_response(text: str = "Done."):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [_text_block(text)]
    return resp


def _tool_use_response(name: str, input_: dict, then_end_turn: bool = True):
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [_tool_use_block(name, input_)]
    return resp


class TestCoderTools(unittest.TestCase):
    """Unit tests for the individual tool functions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tool_file_write_creates_file(self):
        import aria.features.coder as coder
        path = os.path.join(self.tmpdir, "hello.py")
        coder._tool_file_write(path, "print('hello')")
        with open(path) as f:
            self.assertEqual(f.read(), "print('hello')")

    def test_tool_file_read_returns_content(self):
        import aria.features.coder as coder
        path = os.path.join(self.tmpdir, "read_me.txt")
        with open(path, "w") as f:
            f.write("content here")
        result = coder._tool_file_read(path)
        self.assertEqual(result, "content here")

    def test_tool_file_read_missing_returns_error(self):
        import aria.features.coder as coder
        result = coder._tool_file_read("/nonexistent/file.py")
        self.assertIn("Error", result)

    def test_tool_file_edit_replaces_string(self):
        import aria.features.coder as coder
        path = os.path.join(self.tmpdir, "edit_me.py")
        with open(path, "w") as f:
            f.write("def foo():\n    pass\n")
        result = coder._tool_file_edit(path, "pass", "return 42")
        with open(path) as f:
            content = f.read()
        self.assertIn("return 42", content)
        self.assertNotIn("pass", content)

    def test_tool_file_edit_returns_error_if_old_str_missing(self):
        import aria.features.coder as coder
        path = os.path.join(self.tmpdir, "no_match.py")
        with open(path, "w") as f:
            f.write("hello world")
        result = coder._tool_file_edit(path, "not_here", "replacement")
        self.assertIn("Error", result)

    def test_tool_glob_finds_files(self):
        import aria.features.coder as coder
        open(os.path.join(self.tmpdir, "a.py"), "w").close()
        open(os.path.join(self.tmpdir, "b.py"), "w").close()
        result = coder._tool_glob(os.path.join(self.tmpdir, "*.py"))
        self.assertIn("a.py", result)
        self.assertIn("b.py", result)

    def test_tool_bash_runs_command(self):
        import aria.features.coder as coder
        result = coder._tool_bash("echo hello", cwd=self.tmpdir)
        self.assertIn("hello", result)

    def test_tool_bash_captures_stderr(self):
        import aria.features.coder as coder
        result = coder._tool_bash("cat /nonexistent_file_xyz", cwd=self.tmpdir)
        self.assertGreater(len(result), 0)

    def test_tool_bash_timeout_returns_error(self):
        import aria.features.coder as coder
        with patch("aria.features.coder._BASH_TIMEOUT", 1):
            result = coder._tool_bash("sleep 5", cwd=self.tmpdir)
        self.assertIn("timeout", result.lower())


class TestCoderProjectManagement(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.identity_path = os.path.join(self.tmpdir, "identity.json")
        with open(self.identity_path, "w") as f:
            json.dump({"active_project": "", "name": "Test"}, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_project_creates_directory(self):
        import aria.features.coder as coder
        with patch("aria.features.coder._IDENTITY_PATH", self.identity_path), \
             patch("aria.features.coder.config.PROJECTS_HOME", self.tmpdir):
            coder.new_project("my-app")
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "my-app")))

    def test_new_project_updates_active_project(self):
        import aria.features.coder as coder
        with patch("aria.features.coder._IDENTITY_PATH", self.identity_path), \
             patch("aria.features.coder.config.PROJECTS_HOME", self.tmpdir):
            coder.new_project("my-app")
        with open(self.identity_path) as f:
            data = json.load(f)
        self.assertIn("my-app", data["active_project"])

    def test_get_active_project_returns_projects_home_when_empty(self):
        import aria.features.coder as coder
        with patch("aria.features.coder._IDENTITY_PATH", self.identity_path), \
             patch("aria.features.coder.config.PROJECTS_HOME", self.tmpdir):
            result = coder.get_active_project()
        self.assertEqual(result, self.tmpdir)

    def test_switch_project_updates_identity(self):
        import aria.features.coder as coder
        os.makedirs(os.path.join(self.tmpdir, "other-app"))
        with patch("aria.features.coder._IDENTITY_PATH", self.identity_path), \
             patch("aria.features.coder.config.PROJECTS_HOME", self.tmpdir):
            coder.switch_project("other-app")
        with open(self.identity_path) as f:
            data = json.load(f)
        self.assertIn("other-app", data["active_project"])


if __name__ == "__main__":
    unittest.main()
