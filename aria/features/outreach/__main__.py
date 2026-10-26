"""python -m aria.features.outreach [--dry-run] — run the outreach loop from a shell.
python -m aria.features.outreach contacts [--one] [--no-draft] — the Contacts tab (Coldstart → drafts)."""
from __future__ import annotations

import argparse
import logging

from aria.features.outreach import pipeline


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m aria.features.outreach", description=pipeline.__doc__.split("\n")[0])
    ap.add_argument("mode", nargs="?", default="sheet", choices=("sheet", "contacts"),
                    help="sheet: the job-sheet outreach loop; contacts: the Contacts tab via Coldstart")
    ap.add_argument("--dry-run", action="store_true", help="sheet mode: write nothing (drafts, sheet, Coldstart); save a report")
    ap.add_argument("--one", action="store_true", help="contacts mode: one pending contact, end to end")
    ap.add_argument("--no-draft", action="store_true", help="contacts mode: look up emails only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.mode == "contacts":
        from aria.features.outreach import contacts_tab
        from aria.web import websearch
        print(contacts_tab.run(limit=1 if args.one else None, draft=not args.no_draft, step_fn=print,
                               search_fn=lambda q: websearch.search(q, max_results=8)))
        return
    print(pipeline.run_outreach(progress_fn=print, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
