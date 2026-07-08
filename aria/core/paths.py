"""aria.core.paths — single source of truth for on-disk locations.

Repo-root data files used to be located with os.path.dirname(__file__)
from flat root modules; after the package restructure every module goes
through here instead.
"""

from __future__ import annotations

from pathlib import Path

# aria/core/paths.py -> aria/core -> aria -> repo root
ROOT: Path = Path(__file__).resolve().parent.parent.parent

IDENTITY_JSON: Path = ROOT / "identity.json"
SCENES_JSON: Path = ROOT / "scenes.json"
ENV_FILE: Path = ROOT / ".env"
