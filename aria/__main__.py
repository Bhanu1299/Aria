"""python -m aria start — the one start command, from a terminal.

Brings up GateLLM, Coldstart, the dashboard and Aria's Chrome (health-checked; the
first failure is printed and nothing else starts), opens the dashboard on Live run,
then runs Aria with ARIA_START_APPLICATIONS=1 so it starts filling new sheet jobs as
soon as it's ready — the same as saying "Aria, start applications".
"""
from __future__ import annotations

import os
import sys

USAGE = "usage: python -m aria start   (GateLLM + Coldstart + dashboard + Aria, then start applications)"


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
    from aria.features.applying import startup
    print("Starting GateLLM, Coldstart, the dashboard and Chrome...")
    ok, message = startup.bring_up()
    print(message)
    if not ok:
        return 1
    os.environ["ARIA_START_APPLICATIONS"] = "1"
    return _run_aria()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
