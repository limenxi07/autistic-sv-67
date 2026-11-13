"""Fresh-worker structured benchmarks; never patch acceptance-source policy.

Example: taskset -c 0 python -m tools.benchmark_boss --source SOURCE.py
  --suite boss --repeat 3 --output NEW.json
Use --diagnostic-copy NEW.py separately to observe independently replayed
reference incumbents. Its timings are diagnostic, never acceptance timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from tools import benchmark_planner_p1 as harness

BASE_WORKER = harness._worker


def worker(connection, spec, label, seed, split, evaluation_limit, profile):
    from public_levels import catalog

    catalog.PUBLIC_GAMEPLAY_LEVEL_BUILDERS.update(catalog.PUBLIC_BOSS_LEVEL_BUILDERS)
    original = harness.load_agent

    def load(source):
        cls = original(source)
        module = sys.modules[cls.__module__]
        cls = getattr(module, os.environ.get("BOSS_ENTRYPOINT", "Agent"))

        class Captured(cls):
            def info(self):
                stats = dict(super().info())
                backend = getattr(self, "_reference", None)
                stats["reference"] = backend.info() if backend else None
                events = getattr(module, "BOSS_DIAGNOSTIC_EVENTS", None)
                if events is not None:
                    stats["verified_incumbents"] = [
                        {**e, "seconds": e["at"] - self._constructed} for e in events
                    ]
                return stats

        return Captured

    harness.load_agent = load
    if split in ("heldout-v2", "stress"):
        from functools import partial

        from tests import planner_p1_fixtures
        from tests.compact_p2_fixtures import heldout_case

        if split == "stress":
            from tests.boss_p2_next_fixtures import boss_like

            build = lambda i: boss_like(i, split="heldout")[0]
        else:
            build = lambda i: heldout_case(i)[0]
        planner_p1_fixtures.fixture_cases = lambda _: {
            f"case-{i:03d}": (partial(build, i), ()) for i in range(200)
        }
    BASE_WORKER(connection, spec, label, seed, split, evaluation_limit, profile)


def replay(row):
    from grid_adventure.grid import step
    from grid_adventure.step import Action

    from public_levels.catalog import PUBLIC_LEVEL_BUILDERS

    if row["split"] == "heldout-v2":
        from tests.compact_p2_fixtures import heldout_case

        state = heldout_case(int(row["level"].split("-")[-1]))[0]
    elif row["split"] == "stress":
        from tests.boss_p2_next_fixtures import boss_like

        state = boss_like(int(row["level"].split("-")[-1]), split="heldout")[0]
    else:
        state = PUBLIC_LEVEL_BUILDERS[row["level"]]()
    state.seed, state.turn_limit = row["seed"], 150
    for action in row.get("trace_actions", ()):
        state = step(state, Action(action))
    return {
        "win": state.win,
        "lose": state.lose,
        "reward": state.score,
        "matches": (state.win, state.lose, state.score)
        == (row["win"], row["lose"], row["reward"]),
    }


def diagnostic_copy(source, target):
    """Instrument our planner source only, never any engine source."""
    text = source.read_text()
    needle = "                            incumbent = child\n"
    assert text.count(needle) == 1
    text = text.replace(
        needle,
        needle
        + "                            _boss_verify(start, nodes, child, deadline)\n",
    )
    text += """

BOSS_DIAGNOSTIC_EVENTS = []

def _boss_verify(start, nodes, child, deadline):
    actions, keys = [], []
    index = child
    while index != -1:
        parent, action, key, _, _ = nodes[index]
        keys.append(key)
        if parent != -1:
            actions.append(action)
        index = parent
    actions.reverse()
    keys.reverse()
    current = start
    before = perf_counter()
    for i, action in enumerate(actions):
        if perf_counter() >= deadline:
            BOSS_DIAGNOSTIC_EVENTS.append(dict(at=perf_counter(), status="deadline"))
            return
        if reference_state_key(current) != keys[i]:
            raise AssertionError("reference prefix mismatch")
        current = grid_step(current, action)
    assert reference_state_key(current) == keys[-1]
    assert current.win and not current.lose
    BOSS_DIAGNOSTIC_EVENTS.append(dict(at=perf_counter(), status="verified",
        reward=current.score, verification_seconds=perf_counter()-before,
        actions=[a.value for a in actions]))
"""
    target.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--entrypoint", default="Agent")
    parser.add_argument(
        "--suite", choices=("boss", "public", "heldout-v2", "stress"), default="boss"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--diagnostic-copy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh output path; preserve evidence"
    from public_levels.catalog import (
        PUBLIC_BOSS_LEVEL_BUILDERS,
        PUBLIC_GAMEPLAY_LEVEL_BUILDERS,
    )

    original_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    source = args.source
    if args.diagnostic_copy:
        assert not args.diagnostic_copy.exists()
        diagnostic_copy(source, args.diagnostic_copy)
        source = args.diagnostic_copy
    split = args.suite if args.suite in ("heldout-v2", "stress") else None
    labels = (
        [f"case-{i:03d}" for i in range(args.count)]
        if split
        else list(
            PUBLIC_BOSS_LEVEL_BUILDERS
            if args.suite == "boss"
            else PUBLIC_GAMEPLAY_LEVEL_BUILDERS
        )
    )
    limit = args.limit or (15.0 if args.suite in ("boss", "stress") else 5.0)
    os.environ["BOSS_ENTRYPOINT"] = args.entrypoint
    harness._worker = worker
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        suite=args.suite,
        original_sha256=original_hash,
        instrumented=bool(args.diagnostic_copy),
        entrypoint=args.entrypoint,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=rows,
    )
    for repetition in range(args.repeat):
        for label in labels:
            row = harness.run_case(
                str(source), label, 42 + repetition, split=split, evaluation_limit=limit
            )
            row["repetition"] = repetition
            row["conservative_cold_seconds"] = row["worker_wall_seconds"]
            if row.get("trace_actions") is not None:
                row["replay"] = replay(row)
            rows.append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                label,
                repetition,
                row["status"],
                row["reward"],
                row["evaluation_seconds"],
                flush=True,
            )


if __name__ == "__main__":
    main()
