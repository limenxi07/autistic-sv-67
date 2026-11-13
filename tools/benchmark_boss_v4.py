"""Check public boss levels in fresh workers, with optional simulated slowdown."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

WORKER = r"""
import json, sys, time
real_clock = time.perf_counter
origin = real_clock()
scale = float(sys.argv[4])
def clock():
    return origin + scale * (real_clock() - origin)
time.perf_counter = clock
from tools.benchmark_planner_p1 import load_agent
from tools.benchmark_vision_v4 import semantic_labels
from grid_adventure.env import GridAdventureEnv
from grid_adventure.grid import to_state, from_state
from public_levels.catalog import PUBLIC_BOSS_LEVEL_BUILDERS, PUBLIC_LEVEL_MAX_REWARDS
source, label, mode = sys.argv[1:4]
seed = int(sys.argv[5])
state = PUBLIC_BOSS_LEVEL_BUILDERS[label]()
state.seed, state.turn_limit = seed, 150
env = GridAdventureEnv(initial_state_fn=lambda *a, **k: to_state(state),
    width=state.width, height=state.height, observation_type=mode)
cls = load_agent(source)
start = clock()
agent = cls()
obs, _ = env.reset()
win = lose = False
reward = 0
actions = []
first = None
exact = None
budget = tuple(agent._budget)
while not (win or lose) and len(actions) < 150 and clock() - start < 15:
    action = agent.step(obs)
    actions.append(action.name)
    if first is None:
        planner = agent._planner if mode == 'image' else agent._structured
        first = {'seconds': clock() - start,
                 'planner': planner.info() if planner else None}
        if mode == 'image':
            history = agent._history
            exact = bool(history and history.hypotheses and
                semantic_labels(history.hypotheses[0].state) ==
                semantic_labels(from_state(env.state)))
    obs, r, win, lose, _ = env.step(action)
    reward += r
elapsed = clock() - start
optimal = PUBLIC_LEVEL_MAX_REWARDS[label]
print(json.dumps(dict(label=label, observation=mode, seed=seed, scale=scale,
    win=bool(win), lose=bool(lose), timeout=elapsed >= 15, reward=reward,
    performance=max(0, (reward - 1.5 * optimal) / (-.5 * optimal)) if win else 0,
    seconds=elapsed, first=first, initial_board_exact=exact, actions=actions,
    budget_seconds=[value-budget[0] for value in budget[1:]],
    budget_reset=tuple(agent._budget) != budget, info=agent.info())))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--label")
    parser.add_argument("--mode", choices=("gridstate", "image"))
    args = parser.parse_args()
    from public_levels.catalog import PUBLIC_BOSS_LEVEL_BUILDERS

    rows = []
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    for label in [args.label] if args.label else PUBLIC_BOSS_LEVEL_BUILDERS:
        for mode in [args.mode] if args.mode else ("gridstate", "image"):
            for seed in args.seeds:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        WORKER,
                        args.source,
                        label,
                        mode,
                        str(args.scale),
                        str(seed),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                    check=True,
                )
                row = json.loads(result.stdout)
                rows.append(row)
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("actions", "info")}
                    ),
                    flush=True,
                )
    report = {
        "source": args.source,
        "sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        "synthetic_clock_stress": args.scale != 1,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
