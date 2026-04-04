import sys, os, importlib, types, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_match_skill_returns_none_for_unknown():
    """An unrecognised command returns None."""
    from skills import skill_loader
    importlib.reload(skill_loader)
    skill_loader.load_skills()
    assert skill_loader.match_skill("open YouTube") is None


def test_load_skills_handles_broken_skill(tmp_path, monkeypatch):
    """A broken skill module must not prevent other skills from loading."""
    from skills import skill_loader
    importlib.reload(skill_loader)

    # Patch skills dir to a tmp dir with one bad skill and one good skill
    bad_dir = tmp_path / "broken_skill"
    bad_dir.mkdir()
    (bad_dir / "__init__.py").write_text("raise ImportError('intentional')\n")

    good_dir = tmp_path / "good_skill"
    good_dir.mkdir()
    (good_dir / "__init__.py").write_text(
        "TRIGGERS = ['good trigger']\ndef handle(c): return 'ok'\n"
    )

    monkeypatch.setattr(skill_loader, "_SKILLS_DIR", tmp_path)
    skill_loader.load_skills()

    assert skill_loader.match_skill("good trigger") is not None
    assert skill_loader.match_skill("broken") is None


def test_load_skills_skips_no_handle(tmp_path, monkeypatch):
    """A skill with TRIGGERS but no handle() is skipped."""
    from skills import skill_loader
    importlib.reload(skill_loader)

    no_handle_dir = tmp_path / "no_handle"
    no_handle_dir.mkdir()
    (no_handle_dir / "__init__.py").write_text("TRIGGERS = ['test trigger']\n")

    monkeypatch.setattr(skill_loader, "_SKILLS_DIR", tmp_path)
    skill_loader.load_skills()

    assert skill_loader.match_skill("test trigger") is None


def test_load_skills_skips_no_triggers(tmp_path, monkeypatch):
    """A skill with handle() but no TRIGGERS is skipped."""
    from skills import skill_loader
    importlib.reload(skill_loader)

    no_trigger_dir = tmp_path / "no_trigger"
    no_trigger_dir.mkdir()
    (no_trigger_dir / "__init__.py").write_text("def handle(c): return 'ok'\n")

    monkeypatch.setattr(skill_loader, "_SKILLS_DIR", tmp_path)
    skill_loader.load_skills()

    # Nothing was registered — should return None for anything
    assert skill_loader.match_skill("anything") is None


def test_match_skill_case_insensitive(tmp_path, monkeypatch):
    """Trigger matching is case-insensitive."""
    from skills import skill_loader
    importlib.reload(skill_loader)

    skill_dir = tmp_path / "test_skill"
    skill_dir.mkdir()
    (skill_dir / "__init__.py").write_text(
        "TRIGGERS = ['my trigger']\ndef handle(c): return 'hit'\n"
    )

    monkeypatch.setattr(skill_loader, "_SKILLS_DIR", tmp_path)
    skill_loader.load_skills()

    assert skill_loader.match_skill("MY TRIGGER command") is not None
    assert skill_loader.match_skill("say MY TRIGGER please") is not None
