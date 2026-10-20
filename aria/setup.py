"""aria.setup — one-time account setup.

    python -m aria.setup google                    personal account: Gmail + Calendar + Sheets
    python -m aria.setup google --account jobs     jobs account: Muse sheet + outreach drafts
    python -m aria.setup google --status           which account each tool uses
    python -m aria.setup gmail | sheets            aliases for google

The address each account must sign in as comes from GOOGLE_ACCOUNT /
GOOGLE_JOBS_ACCOUNT in .env; --email overrides it.
"""
from __future__ import annotations

import sys

_GOOGLE_ALIASES = {"google", "gmail", "sheets", "calendar"}


def _flag(argv: list[str], name: str) -> str:
    if name not in argv:
        return ""
    i = argv.index(name)
    return argv[i + 1] if i + 1 < len(argv) else ""


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0 if argv else 1
    service = argv[0].lower()
    if service in _GOOGLE_ALIASES:
        from aria.integrations import google_auth

        if "--status" in argv:
            return google_auth.print_status()
        account = (_flag(argv, "--account") or google_auth.PERSONAL).lower()
        if account not in google_auth.ACCOUNTS:
            hint = " (to pick an address, use --email)" if "@" in account else ""
            print(f"--account must be one of: {', '.join(google_auth.ACCOUNTS)}{hint}")
            return 1
        return google_auth.run_setup(account, _flag(argv, "--email") or None)
    print(f"No setup available for {service!r}. Try: python -m aria.setup google")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
