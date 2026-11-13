"""Diagnostic only: exact P2 and separately hashed budget-only boss baselines.

Run from a P2 repository checkout. No submitted/frozen source is modified.
The catalog extension is process-local, including spawned harness workers.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from public_levels import catalog
from tools.benchmark_p1 import load_agent, run_case

catalog.PUBLIC_GAMEPLAY_LEVEL_BUILDERS.update(catalog.PUBLIC_BOSS_LEVEL_BUILDERS)

EXPECTED = {
    "p2a": "e478c1ccb78c21ea4e97fa3c54ab08a28a401c9f4e4ca2846dfcb0590bb7c2cb",
    "p2b": "551ee00377ffe42a343d18c887b1c7f36ef4baf4fa0fff6a22b0053c9845edd0",
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def audit_trace(module, state, actions):
    from grid_adventure.grid import step
    from grid_adventure.step import Action

    reasons = Counter()
    frames = 0
    first_unsupported = None
    for index in range(len(actions) + 1):
        try:
            module.encode(state)
            reasons["supported"] += 1
        except module.UnsupportedState as error:
            reason = str(error)
            reasons[reason] += 1
            if first_unsupported is None:
                first_unsupported = {"turn": state.turn, "reason": reason}
        frames += 1
        if index < len(actions):
            state = step(state, Action(actions[index]))
    return {
        "frames": frames,
        "support_counts": dict(reasons),
        "first_unsupported": first_unsupported,
        "replayed_win": state.win,
        "replayed_score": state.score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    for variant, expected in EXPECTED.items():
        source = Path(f"artifacts/p2/{variant}-ready/task1_agent.py")
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        frozen = out / f"{variant}-frozen.py"
        frozen.write_bytes(raw)
        sources.append((variant + "-frozen", frozen, expected, 3.0, 3.5))
        text = raw.decode()
        for prefix in ("", "reference_"):
            for name, old, new in (
                ("INITIAL_SEARCH_SECONDS", "3.0", "11.0"),
                ("SEARCH_CUTOFF_SECONDS", "3.5", "12.0"),
            ):
                before = f"\n{prefix}{name} = {old}\n"
                assert text.count(before) == 1
                text = text.replace(before, f"\n{prefix}{name} = {new}\n")
        adjusted = out / f"{variant}-budget15.py"
        adjusted.write_text(text)
        sources.append((variant + "-budget15", adjusted, expected, 11.0, 12.0))
    manifest = [
        {
            "variant": name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parent_sha256": parent,
            "initial_search_seconds": initial,
            "cutoff_seconds": cutoff,
            "episode_limit_seconds": 15.0,
            "change": "none" if name.endswith("frozen") else "four deadline constants only",
        }
        for name, path, parent, initial, cutoff in sources
    ]
    save(out / "manifest.json", manifest)
    rows = json.loads((out / "boss.json").read_text())["results"] if args.audit_only else []
    # Round-robin repeated, sequential isolated workers; no concurrent runners.
    for repetition in range(0 if args.audit_only else 3):
        for name, source, *_ in sources:
            for label in catalog.PUBLIC_BOSS_LEVEL_LABELS:
                row = run_case(str(source), label, 42 + repetition, evaluation_limit=15.0)
                row.update(variant=name, repetition=repetition)
                rows.append(row)
                save(out / "boss.json", {"suite": "structured-boss", "results": rows})
                print(name, repetition, label, row["status"], row["reward"], row["evaluation_seconds"], flush=True)

    # Coverage/replay occurs outside timed acceptance workers.
    agent = load_agent(str(sources[0][1]))
    module = sys.modules[agent.__module__]
    coverage = []
    for label, builder in catalog.PUBLIC_LEVEL_BUILDERS.items():
        state = builder()
        state.turn_limit = 150
        coverage.append({"suite": "initial-public24", "level": label, **audit_trace(module, state, [])})
    for report_name, suite in (("public-repeat3", "public"), ("heldout-v2", "p2-v2")):
        report = json.loads(Path(f"artifacts/p2/p2a-ready/{report_name}.json").read_text())
        for row in report["results"]:
            if row["repetition"] != 0:
                continue
            if suite == "public":
                state = catalog.PUBLIC_LEVEL_BUILDERS[row["level"]]()
            else:
                from tests.p2_fixtures import heldout_case

                state = heldout_case(int(row["level"].split("-")[-1]))[0]
            state.turn_limit = 150
            state.seed = row["seed"]
            coverage.append({"suite": suite, "level": row["level"], **audit_trace(module, state, row["trace_actions"])})
    for row in rows:
        if row.get("trace_actions") is None:
            continue
        state = catalog.PUBLIC_BOSS_LEVEL_BUILDERS[row["level"]]()
        state.turn_limit = 150
        state.seed = row["seed"]
        coverage.append({"suite": "boss", "variant": row["variant"], "repetition": row["repetition"], "level": row["level"], **audit_trace(module, state, row["trace_actions"])})
    save(out / "coverage.json", {"source_sha256": EXPECTED["p2a"], "results": coverage})
    print("coverage complete", len(coverage), flush=True)


if __name__ == "__main__":
    main()
