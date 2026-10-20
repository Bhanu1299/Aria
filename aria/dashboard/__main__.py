"""python -m aria.dashboard [--port 8787] [--no-browser] — serve the pipeline page on 127.0.0.1.

Endpoints:
  GET  /                    the page (all tabs)
  GET  /api/runs            last 10 pipeline runs (Live run tab polls this every 2 s)
  POST /api/settings/key    {provider, key|null}    add / replace / remove a GateLLM key
  POST /api/settings/test   {provider}              one tiny call through GateLLM
  POST /api/settings/tier   {tier, models: [...]}   edit a tier's model chain
  GET  /health

Every request must carry a Host of 127.0.0.1/localhost on our port (DNS rebinding);
POSTs also need this process's random token (in the page) and, if the browser sends an
Origin, our own origin (CSRF). Request bodies are never logged.
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
MAX_BODY = 16 * 1024


def render_page(token: str = "") -> str:
    from aria.dashboard import costs, model, render, sources, tabs
    from aria.dashboard import settings as st
    from aria.observability import run_log
    try:
        srcs = sources.load_all()
        mdl = model.build(srcs)
        built = {}

        def guard(key: str, fn: Callable[[], str]) -> None:
            try:
                built[key] = fn()
            except Exception as exc:
                logger.exception("dashboard: %s tab failed", key)
                built[key] = render._no_data(f"{type(exc).__name__}: {exc}")

        guard("live", lambda: tabs.live_tab(run_log.read_runs(10)))

        def costs_html() -> str:
            calls = costs.fetch_calls()
            summary = costs.summarize(calls["calls"], costs.fetch_flight(), mdl["now"],
                                      applied_jobs=model.cost(mdl)["applied"])
            return tabs.costs_tab(summary, calls["via"])
        guard("costs", costs_html)
        guard("settings", lambda: tabs.settings_tab(st.providers(), st.tiers()))
        return render.page(mdl, built, token)
    except Exception as exc:  # last resort: never a blank 500
        logger.exception("dashboard: page failed")
        return ("<!doctype html><meta charset=utf-8><title>Job Pipeline</title>"
                f"<p>The dashboard failed to build: {render.e(exc)}</p>")


def runs_json() -> dict:
    from aria.observability import run_log
    return {"runs": run_log.read_runs(10)}


def settings_action(path: str, body: dict) -> tuple:
    """(status, json) for a settings POST. The only route that writes."""
    from aria.dashboard import settings as st
    try:
        if path == "/api/settings/key":
            provider = str(body.get("provider") or "")
            key = body.get("key")
            out = st.set_key(provider, key if isinstance(key, str) else None)
            out["reload"] = st.reload_gateway()["message"]
            return 200, out
        if path == "/api/settings/test":
            return 200, st.test_provider(str(body.get("provider") or ""))
        if path == "/api/settings/tier":
            models = body.get("models")
            if not isinstance(models, list):
                raise st.SettingsError("models must be a list")
            chain = st.set_tier(str(body.get("tier") or ""), models)
            return 200, {"tier": body.get("tier"), "models": chain, "reload": st.reload_gateway()["message"]}
    except st.SettingsError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        logger.error("dashboard: settings action failed: %s", type(exc).__name__)  # no body, no key
        return 500, {"error": f"failed: {type(exc).__name__}"}
    return 404, {"error": "not found"}


def make_handler(build: Callable[[], str], token: str, port_ref: list,
                 runs: Callable[[], dict] = runs_json, act: Callable = settings_action):
    class Handler(BaseHTTPRequestHandler):
        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            return host in {f"{HOST}:{port_ref[0]}", f"localhost:{port_ref[0]}"}

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            return origin is None or origin in {f"http://{HOST}:{port_ref[0]}", f"http://localhost:{port_ref[0]}"}

        def do_GET(self) -> None:
            if not self._host_ok():
                return self._send(403, "forbidden", "text/plain")
            path = self.path.split("?")[0]
            if path == "/health":
                return self._send(200, "ok", "text/plain")
            if path == "/api/runs":
                return self._send(200, json.dumps(runs()), "application/json")
            if path not in ("/", "/index.html"):
                return self._send(404, "not found", "text/plain")
            self._send(200, build(), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            if not (self._host_ok() and self._origin_ok()) or \
                    not secrets.compare_digest(self.headers.get("X-Dashboard-Token") or "", token):
                return self._send(403, json.dumps({"error": "forbidden"}), "application/json")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY or "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._send(400, json.dumps({"error": "bad request"}), "application/json")
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._send(400, json.dumps({"error": "bad json"}), "application/json")
            status, out = act(path, body)
            self._send(status, json.dumps(out), "application/json")

        def _send(self, code: int, body: str, ctype: str) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args) -> None:
            logger.debug("dashboard: %s %s", self.command, self.path.split("?")[0])

    return Handler


def serve(port: int, build: Optional[Callable[[], str]] = None, token: Optional[str] = None,
          **handler_kw) -> ThreadingHTTPServer:
    """Bind 127.0.0.1:port (0 = any free port). Caller runs serve_forever().
    The server's token is on server.token."""
    token = token or secrets.token_urlsafe(24)
    port_ref = [port]
    server = ThreadingHTTPServer((HOST, port), make_handler(build or (lambda: render_page(token)), token,
                                                            port_ref, **handler_kw))
    port_ref[0] = server.server_address[1]
    server.token = token
    return server


def main(argv: list) -> int:
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
