"""scripts/manual_intent_eval.py — does the agent understand what the user means?

Runs every layer=agent case in tests/data/intent_cases.json through the real Agent (GateLLM must
be up; costs a few cents) with EVERY tool replaced by a recorder: nothing is filled, written,
opened or sent. A case passes when the first tool the agent calls is one of the expected tools
and its arguments contain the expected words. Prints a score and the misses.

    venv/bin/python scripts/manual_intent_eval.py            # all agent cases
    venv/bin/python scripts/manual_intent_eval.py magna rtx  # only cases containing these words
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = ROOT / "tests" / "data" / "intent_cases.json"


def _registry(calls: list):
    import aria.plugins as plugin_packs
    from aria.core.plugin import PluginContext
    from aria.core.tool import ToolRegistry
    reg = ToolRegistry()
    ctx = PluginContext(browser=None, speaker=None, voice_capture=None, transcriber=None,
                        menubar=None, keyterms_prompt="")
    for cls in plugin_packs.discover():
        if cls.requires_agent:
            continue
        try:
            cls.from_context(ctx).register(reg)
        except Exception as exc:
            print(f"  (plugin {cls.__name__} skipped: {exc})")
    for tool in reg.all_available():
        tool._execute = (lambda name: lambda params: calls.append((name, dict(params))) or
                         "Done (evaluation stub: nothing actually happened).")(tool.name)
    return reg


def _arg_ok(params: dict, want: dict) -> bool:
    for keys, word in want.items():
        values = " ".join(str(params.get(k, "")) for k in keys.split("|")).lower()
        if word.lower() not in values:
            return False
    return True


def main(argv: list) -> int:
    from aria.core.agent import Agent
    from aria.dispatch.parse import unglue
    cases = [c for c in json.loads(CASES.read_text())["cases"] if c["layer"] == "agent"]
    if argv:
        cases = [c for c in cases if any(w.lower() in c["said"].lower() for w in argv)]
    passed, misses = 0, []
    for case in cases:
        calls: list = []
        agent = Agent(_registry(calls), enable_subagent=False)
        try:
            answer = agent.run(unglue(case["said"]))          # what the dispatcher hands the agent
        except Exception as exc:
            answer = f"(agent raised {exc})"
        first = next(((n, p) for n, p in calls if n != "update_tasks"), None)
        ok = bool(first) and first[0] in case["tools"] and _arg_ok(first[1], case.get("args", {}))
        passed += ok
        mark = "PASS" if ok else "MISS"
        got = f"{first[0]}({json.dumps(first[1])})" if first else f"no tool — said: {answer[:90]!r}"
        print(f"{mark}  {case['said']!r}\n      → {got}")
        if not ok:
            misses.append(case["said"])
    total = len(cases)
    print(f"\n{passed}/{total} understood ({100 * passed // max(total, 1)}%)")
    for m in misses:
        print(f"  missed: {m!r}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
