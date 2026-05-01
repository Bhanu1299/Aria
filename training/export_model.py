"""
Export trained model to ~/.aria/aria.onnx for use by wake_word.py.

Usage:
    python export_model.py
"""
import shutil
from pathlib import Path

from common import MODELS_DIR, ONNX_MODEL_NAME

ARIA_DIR = Path.home() / ".aria"


def export_model(
    src_path: str = str(MODELS_DIR / ONNX_MODEL_NAME),
    dest_dir: str = str(ARIA_DIR),
) -> str:
    """Copy src_path to dest_dir/aria.onnx. Returns destination path."""
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Model not found at {src_path}. Run train_model.py first."
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / ONNX_MODEL_NAME
    shutil.copy2(str(src), str(out_path))
    return str(out_path)


if __name__ == "__main__":
    out = export_model()
    print(f"[Done] Model exported to {out}")
    print()
    print("Next: update wake_word.py with these two lines:")
    print(f'  _OWW_MODEL = "{out}"')
    print('  _OWW_THRESHOLD = 0.7  # tune after real-world testing (range 0.6–0.8)')
