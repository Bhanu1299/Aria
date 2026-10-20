"""python -m aria.dashboard [--port 8787] [--no-browser] — serve the pipeline page on 127.0.0.1."""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"


def render_page() -> str:
    from aria.dashboard import model, render, sources
    try:
        srcs = sources.load_all()
        return render.page(model.build(srcs))
    except Exception as exc:  # last resort: never a blank 500
        logger.exception("dashboard: page failed")
        return ("<!doctype html><meta charset=utf-8><title>Job Pipeline</title>"
                f"<p>The dashboard failed to build: {render.e(exc)}</p>")


def make_handler(build: Callable[[], str]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/health":
                return self._send(200, "ok", "text/plain")
            if self.path.split("?")[0] not in ("/", "/index.html"):
                return self._send(404, "not found", "text/plain")
            self._send(200, build(), "text/html; charset=utf-8")

        def _send(self, code: int, body: str, ctype: str) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args) -> None:
            logger.debug("dashboard: " + fmt, *args)

    return Handler


def serve(port: int, build: Callable[[], str] = render_page) -> ThreadingHTTPServer:
    """Bind 127.0.0.1:port (0 = any free port). Caller runs serve_forever()."""
    return ThreadingHTTPServer((HOST, port), make_handler(build))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m aria.dashboard", description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        server = serve(args.port)
    except OSError:
        server = serve(0)  # port taken: any free one
    url = f"http://{HOST}:{server.server_address[1]}/"
    print(f"Job pipeline dashboard: {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
