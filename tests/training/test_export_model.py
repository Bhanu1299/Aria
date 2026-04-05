# tests/training/test_export_model.py
import sys
import shutil
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


def test_export_copies_model(tmp_path):
    """export_model() copies aria.onnx to the specified destination."""
    from export_model import export_model

    src = tmp_path / "models" / "aria.onnx"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"fake onnx content")

    dest_dir = tmp_path / "aria_dir"
    out = export_model(src_path=str(src), dest_dir=str(dest_dir))

    assert Path(out).exists()
    assert Path(out).read_bytes() == b"fake onnx content"


def test_export_creates_dest_dir(tmp_path):
    """export_model() creates the destination directory if it doesn't exist."""
    from export_model import export_model

    src = tmp_path / "aria.onnx"
    src.write_bytes(b"model")
    dest_dir = tmp_path / "new_dir" / "nested"

    out = export_model(src_path=str(src), dest_dir=str(dest_dir))
    assert Path(out).exists()


def test_export_raises_if_src_missing(tmp_path):
    """export_model() raises FileNotFoundError if source model is missing."""
    from export_model import export_model

    with pytest.raises(FileNotFoundError):
        export_model(src_path=str(tmp_path / "missing.onnx"), dest_dir=str(tmp_path))
