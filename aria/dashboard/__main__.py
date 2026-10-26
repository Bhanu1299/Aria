"""python -m aria.dashboard [--port 8787] [--no-browser] — serve the pipeline page on 127.0.0.1.

Endpoints:
  GET  /                    the page (all tabs)
  GET  /api/runs            last 10 pipeline runs + open application tabs (Live run polls every 2 s)
  POST /api/settings/key    {provider, key|null}    add / replace / remove a GateLLM key
  POST /api/settings/test   {provider}              one tiny call through GateLLM
  POST /api/settings/tier   {tier, models: [...]}   edit a tier's model chain
  POST /api/answers         {op, ...}               answer-bank edits (My answers tab)
  GET  /api/sheet?tab=jobs|applied                  the Muse sheet tab, read fresh (Sheet tab polls it)
  POST /api/sheet/move_applied {}                   every Jobs row marked Applied → the Applied tab
  POST /api/sheet/cell      {tab, row, column, value, expect, company, role}   one sheet cell
  POST /api/jobprep         {op, ...}               follow-up drafts, reply Approve/Dismiss (the sheet's
                                                    Status), do-not-apply list (dashboard/jobprep.py)
  GET  /review-img?f=x.png  a review screenshot / field crop from ~/.aria/reviews
  GET  /health

The only writes: GateLLM keys/tiers (settings.py), the answer bank (answers.py →
aria.features.answer_bank), single Muse sheet cells (sheet_tab.py → Sheets API; an approved
reply's Status via jobprep.py) and job prep state in aria.db (follow-ups, replies, do-not-apply)
plus jobs-Gmail drafts. Everything
else is read-only.

Every request must carry a Host of 127.0.0.1/localhost on our port (DNS rebinding);
POSTs also need this process's random token (in the page) and, if the browser sends an
Origin, our own origin (CSRF). Request bodies are never logged.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
MAX_BODY = 64 * 1024        # the base resume JSON (My answers) is the largest body


def render_page(token: str = "") -> str:
    from aria.dashboard import render, sources
    try:
        return page_from(sources.load_all(), token=token)
    except Exception as exc:  # last resort: never a blank 500
        logger.exception("dashboard: page failed")
        return ("<!doctype html><meta charset=utf-8><title>Job Pipeline</title>"
                f"<p>The dashboard failed to build: {render.e(exc)}</p>")


def page_from(srcs: dict, now=None, token: str = "", calls: Optional[dict] = None, flight: Optional[list] = None,
              runs: Optional[list] = None) -> str:
    """Every tab from already-loaded sources (tests pass fixtures here). Each tab is guarded."""
    from aria.dashboard import answers, applications, costs, health, model, outreach, render, sheet_tab, tabs, today
    from aria.dashboard import settings as st
    from aria.observability import run_log
    mdl = model.build(srcs, now)
    built = {}

    def guard(key: str, fn: Callable[[], str]) -> None:
        try:
            built[key] = fn()
        except Exception as exc:
            logger.exception("dashboard: %s tab failed", key)
            built[key] = render._no_data(f"{type(exc).__name__}: {exc}")

    questions = answers.open_count(srcs.get("reviews"))
    guard("today", lambda: today.render(mdl, questions) + _jobprep(mdl) + render.today_extras(mdl))
    guard("outreach", lambda: outreach.render(mdl))
    guard("answers", lambda: answers.render(srcs.get("reviews")) + _base_resume())
    guard("sheet", sheet_tab.render)
    guard("applications", lambda: applications.render(mdl) + render.funnel_details(mdl) + render.all_jobs_table(mdl))
    guard("health", lambda: health.render(mdl))
    guard("live", lambda: tabs.live_tab(run_log.read_runs(10) if runs is None else runs, apps=open_apps()))

    def costs_html() -> str:
        got = calls if calls is not None else costs.fetch_calls()
        summary = costs.summarize(got["calls"], costs.fetch_flight() if flight is None else flight, mdl["now"],
                                  applied_jobs=model.cost(mdl)["applied"])
        return tabs.costs_tab(summary, got["via"]) + render.pipeline_cost(mdl)
    guard("costs", costs_html)
    guard("settings", lambda: tabs.settings_tab(st.providers(), st.tiers()))
    return render.page(mdl, built, token, badges(mdl, questions))


def _jobprep(mdl: dict) -> str:
    """Follow-ups, employer replies, reposts and the do-not-apply list — never breaks Today."""
    from aria.dashboard import jobprep
    try:
        return jobprep.render(mdl)
    except Exception:
        logger.exception("dashboard: job prep cards failed")
        return ""


def _base_resume() -> str:
    from aria.dashboard import jobprep
    try:
        return jobprep.base_resume_card()
    except Exception:
        logger.exception("dashboard: base resume card failed")
        return ""


def badges(mdl: dict, questions: Optional[int]) -> dict:
    """Sidebar counts: Today = what Today lists, Outreach = LinkedIn notes + drafts to review,
    My answers = open questions."""
    from aria.dashboard import outreach, today
    try:
        d = today.build(mdl)
    except Exception:
        logger.exception("dashboard: badge counts failed")
        return {"answers": questions}
    return {"today": len(d["finish"]) + len(d["interviews"]) + len(d["deadlines"]) + (questions or 0),
            "outreach": len(d["linkedin"]) + outreach.review_count(mdl), "answers": questions}


def open_apps() -> list:
    """Open application tabs for the Live run tab: no filled values, just what to look at."""
    from aria.features.applying import tabs as apply_tabs
    return [{"company": e.get("company") or "", "role": e.get("role") or "", "row": e.get("row"),
             "state": e.get("state", ""), "needs_me": list(e.get("needs_me") or []), "note": e.get("note") or ""}
            for e in apply_tabs.live()]


def runs_json() -> dict:
    from aria.observability import run_log
    return {"runs": run_log.read_runs(10), "apps": open_apps()}


def settings_action(path: str, body: dict) -> tuple:
    """(status, json) for a POST: GateLLM settings, answer-bank edits or a sheet cell — the only writes."""
    from aria.dashboard import answers, sheet_tab
    if path == "/api/jobprep":
        from aria.dashboard import jobprep
        try:
            return 200, jobprep.act(body)
        except jobprep.JobPrepError as exc:
            return 400, {"error": str(exc)}
        except Exception as exc:
            logger.error("dashboard: job prep action failed: %s", type(exc).__name__)
            return 500, {"error": f"failed: {type(exc).__name__}: {str(exc)[:120]}"}
    if path == "/api/sheet/move_applied":
        return 200, sheet_tab.move_applied()
    if path == "/api/sheet/cell":
        from aria.integrations import sheets
        try:
            return 200, sheet_tab.edit(body)
        except (sheet_tab.SheetEditError, sheets.SheetError) as exc:
            return 400, {"error": str(exc)}
        except Exception as exc:
            logger.error("dashboard: sheet edit failed: %s", type(exc).__name__)   # never the value
            return 500, {"error": f"failed: {type(exc).__name__}"}
    from aria.dashboard import settings as st
    if path == "/api/answers":
        try:
            return 200, answers.act(body)
        except answers.AnswerError as exc:
            return 400, {"error": str(exc)}
        except Exception as exc:
            logger.error("dashboard: answer edit failed: %s", type(exc).__name__)
            return 500, {"error": f"failed: {type(exc).__name__}"}
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


def sheet_json(tab: str) -> dict:
    from aria.dashboard import sheet_tab
    return sheet_tab.data(tab)


def make_handler(build: Callable[[], str], token: str, port_ref: list,
                 runs: Callable[[], dict] = runs_json, act: Callable = settings_action,
                 sheet: Callable[[str], dict] = sheet_json):
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
            if path == "/api/sheet":
                from urllib.parse import parse_qs, urlparse
                tab = (parse_qs(urlparse(self.path).query).get("tab") or ["jobs"])[0]
                return self._send(200, json.dumps(sheet(tab)), "application/json")
            if path == "/review-img":
                return self._review_img()
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

        def _review_img(self) -> None:
            from urllib.parse import parse_qs, urlparse

            from aria.features.applicator import REVIEWS_DIR
            name = (parse_qs(urlparse(self.path).query).get("f") or [""])[0]
            path = REVIEWS_DIR / name
            if not re.fullmatch(r"[\w.-]+\.png", name) or not path.is_file():
                return self._send(404, "not found", "text/plain")
            self._send_bytes(200, path.read_bytes(), "image/png")

        def _send(self, code: int, body: str, ctype: str) -> None:
            self._send_bytes(code, body.encode("utf-8"), ctype)

        def _send_bytes(self, code: int, data: bytes, ctype: str) -> None:
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
