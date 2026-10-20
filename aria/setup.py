"""aria.setup — one-time account setup.

    python -m aria.setup google [--account you@gmail.com]   Gmail + Calendar + Sheets
    python -m aria.setup gmail | sheets                      aliases for google

--account (or GOOGLE_ACCOUNT in .env) pre-selects which Google account to sign in with.
"""
from __future__ import annotations

import sys

_GOOGLE_ALIASES = {"google", "gmail", "sheets", "calendar"}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0 if argv else 1
    service = argv[0].lower()
    if service in _GOOGLE_ALIASES:
        import aria.core.config as config
        from aria.integrations import google_auth

        account = config.GOOGLE_ACCOUNT
        if "--account" in argv:
            i = argv.index("--account")
            account = argv[i + 1] if i + 1 < len(argv) else ""
        return google_auth.run_setup(account or None)
    print(f"No setup available for {service!r}. Try: python -m aria.setup google")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
