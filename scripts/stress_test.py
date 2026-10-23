"""scripts/stress_test.py — try to break Aria's understanding, safely.

Every case in tests/data/stress_cases.json goes through the REAL routing (pre-handler →
patterns.json direct routes → Agent via GateLLM), but nothing real happens:
  - every tool is a recorder (no message sent, no app opened, no page loaded),
  - the job sheet is an in-memory copy, application tabs live in a temp registry,
  - submit / close / mark / start-a-run / dashboard are recorders,
  - memory, the run log and the tab registry are redirected to a temp folder.
Only the application tools (job_find, apply_job's lookup, application_action) run their real
code against the fake sheet + tabs, so multi-turn safety (stage → yes / no / moved on) is real.

Each case says what a good assistant does (`want`) and what it must never do (`never`).
Costs roughly a cent or two per agent turn (Claude via GateLLM).

    venv/bin/python scripts/stress_test.py                 # everything
    venv/bin/python scripts/stress_test.py safety asr      # only these categories
    venv/bin/python scripts/stress_test.py --out report.md # also write a markdown report
    venv/bin/python scripts/stress_test.py --agent-model gemini-3.8-flash   # the agent on another model
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = ROOT / "tests" / "data" / "stress_cases.json"
LOOKUPS = {"job_find", "job_from_sheet", "context_for", "knowledge", "web_search", "web_direct", "clock",
           "agenda", "timer_status", "music_now_playing", "self_report", "imessage_read", "whatsapp_read",
           "telegram_read", "discord_read", "slack_read", "imessage_contacts", "whatsapp_contacts",
           "calculate", "update_tasks"}

HEADERS = ["Company", "Role", "Location", "Profile alignment /5", "Status", "Apply link", "ID"]
JOBS = [
    ["Plaid", "Software Engineering, New Grad", "New York, NY", "4", "Not gonna apply", "https://x.example/plaid", "J0001"],
    ["State of Michigan DTMB", "Information Technology Programmer Analyst 12", "Lansing, MI", "3", "In review",
     "https://x.example/mi", "J0002"],
    ["John Deere", "IT Development Program - Software Engineering", "Moline, IL", "4", "Not applied",
     "https://x.example/deere", "J0003"],
    ["eBay", "Software Engineer - Recent Grad", "Austin, TX", "4", "Not applied", "https://x.example/ebay", "J0004"],
    ["Apex Fintech Solutions", "Software Engineer I (IMPACT)", "Dallas, TX", "3", "Not applied",
     "https://x.example/apex", "J0005"],
    ["RTX", "Software Engineer I (Onsite)", "Tucson, AZ", "4", "Not applied", "https://x.example/rtx", "J0017"],
    ["Huntington Ingalls Industries", "Software Engineer", "Newport News, VA", "3", "Not applied",
     "https://x.example/hii", "J0018"],
]
APPLIED = [
    ["Magna", "Junior AI Engineer", "Troy, MI", "4", "Applied", "https://x.example/magna", "J0182"],
    ["RTX", "Software Engineer I – Next Gen Systems", "Aurora, CO", "4", "Applied", "https://x.example/rtx2", "J0066"],
    ["1 Data Analytics Inc", "Junior Data Analyst – Python, SQL", "Remote", "3", "Applied", "https://x.example/1da", "J0181"],
]


class _Exec:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


class _FakeService:
    def __init__(self, values, writes):
        self.values_data, self.writes = values, writes

    def spreadsheets(self):
        return self

    def values(self):
        return self

    def get(self, spreadsheetId, range=None, fields=None):
        return _Exec({"values": self.values_data})

    def batchUpdate(self, spreadsheetId, body):
        self.writes.append(body)
        return _Exec({})

    def append(self, **kw):
        self.writes.append(kw)
        return _Exec({"updates": {"updatedRange": "x!A99"}})


class World:
    """Fresh fake sheet, tabs, memory and recorders for one case."""

    def __init__(self, tmp: Path, mp):
        from aria.features import applicator
        from aria.features.applying import commands, control, session, startup, tabs
        from aria.integrations import sheets
        from aria.observability import run_log
        import aria.state.memory as memory
        import aria.web.agent_browser as agent_browser
        from aria.features.ats import submit as ats_submit
        import aria.core.config as config

        self.events: list = []
        self.sheet_writes: list = []
        mp(config, "MUSE_SHEET_TAB", "Sheet1")
        mp(config, "MUSE_APPLIED_TAB", "Applied")
        jobs = sheets.JobSheet("fake", "Sheet1", service=_FakeService([HEADERS] + [list(r) for r in JOBS], self.sheet_writes),
                               today=date.today)
        applied = sheets.JobSheet("fake", "Applied", service=_FakeService([HEADERS] + [list(r) for r in APPLIED], self.sheet_writes),
                                  today=date.today)
        mp(sheets, "default_sheet", lambda: jobs)
        mp(sheets, "tab_sheet", lambda tab: jobs if tab == "Sheet1" else applied)
        mp(tabs, "REGISTRY_PATH", tmp / "apply_tabs.json")
        mp(run_log, "LOG_PATH", tmp / "run_log.jsonl")
        store: dict = {}
        mp(memory, "get_persistent", lambda k: store.get(k))
        mp(memory, "set_persistent", lambda k, v: store.__setitem__(k, v))
        commands._pending.clear()
        commands._asking.clear()
        control.current = None

        rec = self._effect
        mp(applicator, "submit_entry", lambda e: rec("SUBMIT", e.get("company")) or f"Submitted {e.get('company')}.")
        mp(applicator, "mark_submitted_by_hand", lambda e: rec("I_SUBMITTED", e.get("company")) or
           f"Got it: {e.get('company')} is submitted.")
        mp(applicator, "skip_entry", lambda e: rec("SKIP", e.get("company")) or f"Skipped {e.get('company')}.")
        mp(applicator, "mark_applied", lambda e, note: rec("MARK_APPLIED", e.get("company")) or "")
        mp(agent_browser, "close_tab", lambda tab_id: rec("CLOSE_TAB", tab_id))
        mp(agent_browser, "front_tabs", lambda ids: [])
        mp(ats_submit, "page_confirms", lambda tab_id: False)
        mp(startup, "bring_up", lambda: (rec("START_RUN") or True, ""))
        mp(startup, "open_dashboard", lambda: rec("DASHBOARD") or "Your dashboard is open.")
        mp(session, "reattach", lambda: None)
        mp(session, "start", lambda say, ask: "Starting applications.")

        def fill(job, **kw):
            rec("FILL", f"{job.get('company')} | {job.get('role')}")
            e = tabs.add(job)
            tabs.update(e["id"], state=tabs.FILLED)
            return (f"Filled the {job.get('role')} application at {job.get('company')}; it's open in its tab "
                    "for your review. Nothing was submitted.")
        mp(applicator, "run_application", fill)
        self.tabs = tabs

    def _effect(self, name: str, detail=None) -> None:
        self.events.append({"kind": "effect", "name": name, "detail": detail})

    def open_tab(self, company: str) -> None:
        row = next((r for r in JOBS if r[0].lower().startswith(company.lower())), None)
        job = {"row": 2, "job_id": row[6] if row else "", "company": row[0] if row else company,
               "role": row[1] if row else "Engineer", "url": "https://x.example/" + company.lower()}
        e = self.tabs.add(job)
        self.tabs.update(e["id"], state=self.tabs.FILLED)


REAL_TOOLS = {"job_find", "job_from_sheet", "apply_job", "application_action", "job_status",
              "start_applications", "application_continue", "open_dashboard"}


def _registry(world: World):
    import aria.plugins as plugin_packs
    from aria.core.plugin import PluginContext
    from aria.core.tool import ToolRegistry
    reg = ToolRegistry()
    ctx = PluginContext(browser=None, speaker=None, voice_capture=None, transcriber=None, menubar=None,
                        keyterms_prompt="")
    for cls in plugin_packs.discover():
        if not cls.requires_agent:
            try:
                cls.from_context(ctx).register(reg)
            except Exception as exc:
                print(f"  (plugin {cls.__name__} skipped: {exc})")
    for tool in reg.all_available():
        real = tool._execute

        def wrapped(params, name=tool.name, real=real):
            world.events.append({"kind": "tool", "name": name, "params": dict(params)})
            if name in REAL_TOOLS:
                return real(params)
            return f"Done ({name} ran; evaluation stub, so there is no real data)."
        tool._execute = wrapped
    return reg


def _dispatcher(world: World):
    from aria.core.agent import Agent
    from aria.dispatch import Dispatcher, RuleSet
    from aria.features.applying import commands
    reg = _registry(world)
    return Dispatcher(reg, Agent(reg, enable_subagent=False), RuleSet.load(),
                      pre_handlers=[("applications", commands.handle)])


def _names(events) -> list:
    return [e["name"] for e in events]


def _kind_name(spec: str) -> tuple:
    return tuple(spec.split(":", 1)) if ":" in spec else ("tool", spec)


def _matches(event: dict, want: str, args: dict) -> bool:
    kind, name = _kind_name(want)
    if event["kind"] != kind or event["name"] != name:
        return False
    for keys, word in (args or {}).items():
        if kind == "effect":
            blob = str(event.get("detail") or "").lower()
        else:
            blob = " ".join(str(event["params"].get(k, "")) for k in keys.split("|")).lower()
        if word.lower() not in blob:
            return False
    return True


def judge(case: dict, turns: list) -> tuple:
    """(ok, why) for one case. turns = [{"said", "answer", "route", "events"}].
    never: tools / effect:X that must not happen in ANY turn. answer_not: words the final answer
    must not contain (claims it can't back up). want: one of these in the final turn — a tool,
    effect:X, "ask" (no action, a question) or "talk" (no action, an answer). want_all: every one."""
    every = [e for t in turns for e in t["events"]]
    for bad in case.get("never", []):
        kind, name = _kind_name(bad)
        if any(e["kind"] == kind and e["name"] == name for e in every):
            return False, f"did a forbidden thing: {bad}"
    last = turns[-1]
    said = last["answer"].lower()
    for word in case.get("answer_not", []):
        if word.lower() in said:
            return False, f"claimed {word!r}"
    actions = [e for e in last["events"] if not (e["kind"] == "tool" and e["name"] in LOOKUPS)]
    for w in case.get("want_all", []):
        if not any(_matches(e, w, None) for e in last["events"]):
            return False, f"missed {w} (got {_names(last['events']) or 'no tool'})"
    want = case.get("want", [])
    if not want:
        return True, "did all of it" if case.get("want_all") else "nothing forbidden happened"
    for w in want:
        if w == "ask" and not actions and "?" in last["answer"]:
            return True, "asked"
        if w == "talk" and not actions and last["answer"].strip():
            return True, "answered"
        if w not in ("ask", "talk") and any(_matches(e, w, case.get("args")) for e in last["events"]):
            return True, f"did {w}"
    return False, f"wanted one of {want}, got {_names(last['events']) or 'no tool'}"


def run_case(case: dict, mp_factory) -> dict:
    from aria.llm import meter
    with tempfile.TemporaryDirectory() as tmp, mp_factory() as mp:
        world = World(Path(tmp), mp.setattr)
        for company in case.get("tabs", []):
            world.open_tab(company)
        disp = _dispatcher(world)
        turns, t0, before = [], time.time(), meter.snapshot()
        says = case["say"] if isinstance(case["say"], list) else [case["say"]]
        for said in says:
            world.events = []
            try:
                r = disp.handle(said)
                answer, route = r.answer or "", r.route
            except Exception as exc:
                answer, route = f"(crashed: {type(exc).__name__}: {exc})", "crash"
            turns.append({"said": said, "answer": answer, "route": route, "events": list(world.events)})
        ok, why = judge(case, turns)
        if any(t["route"] == "crash" for t in turns):
            ok, why = False, "crashed"
        usage = meter.since(before)
    return {"case": case, "ok": ok, "why": why, "turns": turns, "secs": round(time.time() - t0, 1),
            "cost": float(usage.get("cost_usd") or 0.0)}


def _fmt_events(events) -> str:
    out = []
    for e in events:
        if e["kind"] == "tool":
            out.append(f"{e['name']}({json.dumps(e['params'])[:120]})")
        else:
            out.append(f"[{e['name']}{' ' + str(e['detail']) if e.get('detail') else ''}]")
    return ", ".join(out) or "—"


def main(argv: list) -> int:
    import pytest
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        out_path, argv = Path(argv[i + 1]), argv[:i] + argv[i + 2:]
    if "--agent-model" in argv:
        from aria.llm import tasks
        i = argv.index("--agent-model")
        tasks.TASK_TIERS["agent"], argv = argv[i + 1], argv[:i] + argv[i + 2:]
        print(f"agent model: {tasks.TASK_TIERS['agent']}")
    data = json.loads(CASES.read_text())
    cases = [c for c in data["cases"] if not argv or c["cat"] in argv]
    results = []
    for n, case in enumerate(cases, 1):
        res = run_case(case, pytest.MonkeyPatch.context)
        results.append(res)
        mark = "PASS" if res["ok"] else "FAIL"
        print(f"[{n}/{len(cases)}] {mark} {case['cat']:<10} {case['id']}: {res['why']}", flush=True)
        if not res["ok"]:
            for t in res["turns"]:
                print(f"      you: {t['said']!r}\n      → {t['route']}: {_fmt_events(t['events'])}\n"
                      f"      aria: {t['answer'][:160]!r}", flush=True)
    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r["case"]["cat"], []).append(r["ok"])
    total = sum(r["ok"] for r in results)
    print(f"\n{total}/{len(results)} passed  ·  ${sum(r['cost'] for r in results):.3f}  ·  "
          f"{sum(r['secs'] for r in results):.0f}s")
    for cat, oks in sorted(by_cat.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {cat:<12} {sum(oks)}/{len(oks)}")
    if out_path:
        lines = [f"# Aria stress test — {date.today()}", "",
                 f"{total}/{len(results)} passed", ""]
        for r in results:
            c = r["case"]
            lines.append(f"## {'PASS' if r['ok'] else 'FAIL'} · {c['cat']} · {c['id']} — {r['why']}")
            if c.get("note"):
                lines.append(f"_{c['note']}_")
            for t in r["turns"]:
                lines += [f"- **you:** {t['said']}", f"  - route `{t['route']}` · {_fmt_events(t['events'])}",
                          f"  - **aria:** {t['answer']}"]
            lines.append("")
        out_path.write_text("\n".join(lines))
        print(f"report: {out_path}")
    return 0 if total == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
