"""Deterministic virtual-CPU dispatch regression; never acceptance wall timing."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tests import test_hierarchical_p3_dispatch
from tools.benchmark_planner_p1 import load_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--costs", type=float, nargs="+", default=[0.004, 0.0055, 0.0065, 0.0072]
    )
    args = p.parse_args()
    assert not args.output.exists()
    test_hierarchical_p3_dispatch.hierarchical_p3 = sys.modules[
        load_agent(str(args.source)).__module__
    ]
    rows = []
    for cost in args.costs:
        state, elapsed, searches, trace = (
            test_hierarchical_p3_dispatch.slow_episode(cost)
        )
        row = {
            "virtual_public_transition_seconds": cost,
            "win": state.win,
            "reward": state.score,
            "virtual_episode_seconds": elapsed,
            "searches": searches,
            "actions": len(trace),
            "wait_actions": sum(a.value == "wait" for a in trace),
        }
        rows.append(row)
        print(row, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "diagnostic": True,
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "model": "deterministic virtual CPU cost, not acceptance wall time",
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
