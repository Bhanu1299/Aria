"""tests/test_app_bundle.py — Aria.app builder, launcher behavior, single-instance lock."""
from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_app  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS bundle")


@pytest.fixture(scope="module")
def fake_repo(tmp_path_factory):
    """A repo whose 'python' just records its args/env and exits with $FAKE_EXIT."""
    repo = tmp_path_factory.mktemp("repo")
    py = repo / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text('#!/bin/bash\necho "ran $* in $(pwd) bundle=$ARIA_APP_BUNDLE"\n'
                  'exit "${FAKE_EXIT:-0}"\n')
    py.chmod(0o755)
    (repo / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    return repo


@pytest.fixture(scope="module")
def app(tmp_path_factory, fake_repo):
    return build_app.build(tmp_path_factory.mktemp("dist"), repo=fake_repo)


def _launch(app, home, exit_code):
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "ARIA_LAUNCHER_QUIET": "1",
           "FAKE_EXIT": str(exit_code)}
    result = subprocess.run([str(app / "Contents" / "MacOS" / "Aria")], env=env,
                            capture_output=True, text=True, timeout=30)
    log = (home / ".aria" / "logs" / "aria.log").read_text()
    return result.returncode, log


def test_bundle_has_menubar_plist_with_permission_strings(app):
    with open(app / "Contents" / "Info.plist", "rb") as fh:
        info = plistlib.load(fh)
    assert info["CFBundleIdentifier"] == build_app.BUNDLE_ID
    assert info["LSUIElement"] is True
    assert info["CFBundleExecutable"] == "Aria"
    # must be a native binary, not a script — TCC grants never match a script executable
    exe = (app / "Contents" / "MacOS" / "Aria").read_bytes()
    assert not exe.startswith(b"#!") and exe[:4] in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")
    assert "NSMicrophoneUsageDescription" in info
    assert (app / "Contents" / "Resources" / "Aria.icns").stat().st_size > 10_000


def test_bundle_is_signed(app):
    subprocess.run(["codesign", "--verify", str(app)], check=True)


def test_launcher_runs_main_from_repo_and_logs(app, fake_repo, tmp_path):
    code, log = _launch(app, tmp_path, 0)
    assert code == 0
    assert f"ran main.py in {fake_repo}" in log and "bundle=1" in log
    assert "Starting up" in log and "FAILED" not in log


def test_launcher_reports_already_running(app, tmp_path):
    _, log = _launch(app, tmp_path, 3)
    assert "already running" in log and "FAILED" not in log


def test_launcher_surfaces_crashes_with_log_tail(app, tmp_path):
    code, log = _launch(app, tmp_path, 1)
    assert code == 0
    assert "FAILED: Exit code 1" in log


def test_launcher_explains_missing_venv(tmp_path):
    app = build_app.build(tmp_path / "dist", repo=tmp_path / "nowhere")
    code, log = _launch(app, tmp_path, 0)
    assert code == 1 and "No Python venv" in log


def test_launcher_explains_protected_folder_instead_of_crashing(tmp_path, fake_repo):
    repo = tmp_path / "locked"
    (repo / "venv" / "bin").mkdir(parents=True)
    shutil.copy(fake_repo / "venv" / "bin" / "python", repo / "venv" / "bin" / "python")
    cfg = repo / "venv" / "pyvenv.cfg"
    cfg.write_text("home = /usr/bin\n")
    cfg.chmod(0)  # same symptom as a TCC-protected folder: stat ok, read denied
    try:
        app = build_app.build(tmp_path / "dist", repo=repo)
        code, log = _launch(app, tmp_path, 0)
    finally:
        cfg.chmod(0o644)
    assert code == 1 and "protected folder" in log and "ran main.py" not in log


def test_install_refuses_to_replace_a_foreign_app(app, tmp_path):
    foreign = tmp_path / "Aria.app" / "Contents"
    foreign.mkdir(parents=True)
    with open(foreign / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleIdentifier": "com.opera.aria"}, fh)
    with pytest.raises(SystemExit):
        build_app.install(app, tmp_path)
    target = build_app.install(app, tmp_path / "apps")
    assert build_app.install(app, tmp_path / "apps") == target  # re-install over our own is fine


def test_single_instance_lock(tmp_path):
    from aria.system import single_instance
    lock = tmp_path / "aria.lock"
    assert single_instance.acquire(lock) is True
    assert single_instance.acquire(lock) is False
    held = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); from aria.system import single_instance as s; "
         "sys.exit(0 if s.acquire(__import__('pathlib').Path(%r)) else 3)" % (str(ROOT), str(lock))])
    assert held.returncode == 3
