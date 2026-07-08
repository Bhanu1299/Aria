"""plugins — Aria's drop-in capability packs.

Each subpackage defines one PluginBase subclass in its __init__.py.
discover() imports every subpackage and returns those classes — drop a
new folder in here and Aria loads it at the next startup, no core
edits. A plugin that fails to import is skipped with an error line;
it can never take down startup.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def discover() -> list:
    """Return every PluginBase subclass defined by a plugins.* subpackage.

    Sorted phase-1 first (requires_agent=False), then by class name so
    load order is deterministic. Never raises.
    """
    from aria.core.plugin import PluginBase

    found: list = []
    try:
        modules = list(pkgutil.iter_modules(__path__))
    except Exception as exc:
        logger.error("Plugin scan failed: %s", exc)
        return found

    for modinfo in modules:
        name = f"{__name__}.{modinfo.name}"
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            logger.error("Plugin %r failed to import — skipped: %s", name, exc)
            print(f"[Aria] Plugin {modinfo.name!r} failed to import — skipped: {exc}")
            continue
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, PluginBase)
                and obj is not PluginBase
                and obj.__module__ == mod.__name__
            ):
                found.append(obj)

    found.sort(key=lambda c: (bool(c.requires_agent), c.__name__))
    return found
