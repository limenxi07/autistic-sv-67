"""P2 diagnostic probes; never mutate source bytes used for acceptance runs."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.benchmark_planner_p1 import load_agent, provenance


def operation_probe(source, label, *, transitions=None, seconds=None, profile=False):
    from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

    agent = load_agent(source)
    module: Any = sys.modules[agent.__module__]
    start = PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label]()
    start.turn_limit = 150
    calls = 0
    first_win = first_win_seconds = None
    compact = hasattr(module, "FastState")
    field = "fast_step" if compact else "grid_step"
    original = getattr(module, field)
    began = perf_counter()

    def transition(*args):
        nonlocal calls, first_win, first_win_seconds
        result = original(*args)
        calls += 1
        terminal = result[0] if compact else result
        if terminal.win and first_win is None:
            first_win, first_win_seconds = calls, perf_counter() - began
        return result

    setattr(module, field, transition)
    clock = (lambda: calls) if transitions is not None else perf_counter
    deadline = (
        transitions
        if transitions is not None
        else began + (3.0 if seconds is None else seconds)
    )
    profiler = cProfile.Profile() if profile else None
    try:
        if profiler:
            profiler.enable()
        if compact:
            board, state, value = module.encode(start)
            result = module.search(
                board,
                state,
                value,
                mode=module.PRODUCTION_MODE,
                deadline=deadline,
                clock=clock,
                verify_candidate=lambda actions, keys: module.verify_plan(
                    board, start, actions, keys, deadline=deadline, clock=clock
                ),
            )
        else:
            result = module.search(start, mode="guided", deadline=deadline, clock=clock)
    finally:
        if profiler:
            profiler.disable()
        setattr(module, field, original)
    functions: list[dict[str, Any]] = []
    if profiler:
        for entry in profiler.getstats():
            code = entry.code
            functions.append(
                {
                    "function": code if isinstance(code, str) else code.co_name,
                    "calls": entry.callcount,
                    "total": entry.totaltime,
                    "own": entry.inlinetime,
                }
            )
        functions.sort(key=lambda e: -e["own"])
    return {
        "level": label,
        "requested_transitions": transitions,
        "requested_seconds": seconds,
        "transitions": calls,
        "first_win_transitions": first_win,
        "first_win_seconds": first_win_seconds,
        "elapsed": perf_counter() - began,
        "winning": result.winning,
        "reward": result.final_score,
        "stop_reason": result.stop_reason,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "profiled": profile,
        "functions": functions,
        "provenance": provenance(),
    }


def _generated_worker(connection, spec, label, seed, split, evaluation_limit, profile):
    from functools import partial

    from tests import planner_p1_fixtures
    from tools.benchmark_planner_p1 import _worker

    original = planner_p1_fixtures.fixture_cases

    def fixtures(name):
        if name in ("p2-heldout", "p2-development", "p2-heldout-v2"):
            chosen = name.removeprefix("p2-")
            return {
                f"p2-{i:03d}": (partial(_build_generated, i, chosen), ())
                for i in range(200)
            }
        return original(name)

    cast(Any, planner_p1_fixtures).fixture_cases = fixtures
    _worker(connection, spec, label, seed, split, evaluation_limit, profile)


def _build_generated(index, split):
    from tests.compact_p2_fixtures import generated, heldout_case

    return (
        heldout_case(index)[0]
        if split == "heldout-v2"
        else generated(index, split=split)[0]
    )


def benchmark(source, suite, repeat, output):
    from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
    from tools import benchmark_planner_p1 as harness

    if suite == "public":
        cases = [(label, None) for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS]
    elif suite == "p1-regression":
        from tests.planner_p1_fixtures import fixture_cases

        cases = [
            (label, split)
            for split in ("tuning", "heldout")
            for label in fixture_cases(split)
        ]
    else:
        cases = [(f"p2-{i:03d}", "p2-" + suite) for i in range(200)]
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    original = harness._worker
    try:
        if suite in ("development", "heldout", "heldout-v2"):
            cast(Any, harness)._worker = _generated_worker
        for repetition in range(repeat):
            for label, split in cases:
                result = harness.run_case(source, label, 42 + repetition, split=split)
                result["repetition"] = repetition
                rows.append(result)
                temporary = output.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"suite": suite, "results": rows}, indent=2) + "\n"
                )
                temporary.replace(output)
                print(
                    label,
                    result["status"],
                    result["reward"],
                    result["evaluation_seconds"],
                    flush=True,
                )
    finally:
        harness._worker = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source")
    p.add_argument("--agent")
    p.add_argument(
        "--suite",
        choices=("public", "p1-regression", "development", "heldout", "heldout-v2"),
    )
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.suite:
        benchmark(args.agent or args.source, args.suite, args.repeat, args.output)
        return
    if not args.source:
        p.error("--source required for diagnostics")
    from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

    labels = [
        s
        for s in PUBLIC_GAMEPLAY_LEVEL_BUILDERS
        if s.startswith(("A1 ", "A2 ", "A7 ", "B5 ", "B6 "))
    ]
    rows = []
    for label in labels:
        for budget in (300, 600, 1500, 5000):
            rows.append(operation_probe(args.source, label, transitions=budget))
        for seconds in (0.75, 3.0):
            rows.append(operation_probe(args.source, label, seconds=seconds))
        rows.append(operation_probe(args.source, label, seconds=3.0, profile=True))
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(label, "done", flush=True)


if __name__ == "__main__":
    main()
