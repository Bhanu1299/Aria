#!/usr/bin/env python3
"""Aria entry point — the application lives in the aria/ package.

Kept at the repo root so `python main.py` (and run.sh) keep working.
Also supports: python main.py --login <gmail|google|linkedin>
"""

from __future__ import annotations

from aria.app import cli

if __name__ == "__main__":
    cli()
