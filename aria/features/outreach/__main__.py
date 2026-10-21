"""python -m aria.features.outreach [--dry-run] — run the outreach loop from a shell."""
from __future__ import annotations

import argparse
import logging

from aria.features.outreach import pipeline


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m aria.features.outreach", description=pipeline.__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="write nothing (drafts, sheet, Coldstart); save a report")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(pipeline.run_outreach(progress_fn=print, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
