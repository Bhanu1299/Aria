"""python -m aria start — run Aria from a terminal.

Starts Aria only: no dashboard, no job applications. Those start when you ask by voice
("let's apply to some jobs" / "start applications"), which brings up GateLLM,
Coldstart, the dashboard and Chrome first.
"""
from __future__ import annotations

import sys

USAGE = "usage: python -m aria start   (runs Aria; say \"let's apply to some jobs\" to start applications)"


def _run_aria() -> int:
    from aria.app import main as aria_main
    from aria.system import single_instance
    try:
        aria_main()
    except SystemExit as exc:
        if exc.code == single_instance.ALREADY_RUNNING_EXIT:
            print("Aria is already running. Say \"start applications\" to it instead.")
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def main(argv: list) -> int:
    if argv[:1] != ["start"]:
        print(USAGE)
        return 2
    return _run_aria()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
