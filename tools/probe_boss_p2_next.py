"""Inspect boss-search timing and resource use through the public API."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

from tools.benchmark_planner_p1 import load_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--seconds", type=float, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from public_levels.catalog import PUBLIC_BOSS_LEVEL_BUILDERS

    module = sys.modules[load_agent(args.source).__module__]
    start = list(PUBLIC_BOSS_LEVEL_BUILDERS.values())[1]()
    board, current, score = module.encode(start)
    counts = Counter()
    best = dict(gem_distance=10000)
    original = module.fast_step

    def transition(board, state, action):
        successor, delta = original(board, state, action)
        kinds = {board.slots[i].kind for i, _, _ in successor.effects}
        counts["all"] += 1
        counts["lose"] += successor.lose
        counts["ghost"] += "ghost" in kinds
        counts["boots"] += "boots" in kinds
        counts["key_carried"] += bool(successor.inventory & board.keys)
        counts["door_opened"] += successor.doors != current.doors
        counts["gem_picked"] += not successor.items & board.gems
        counts["inside_room"] += (
            4 <= successor.position % board.width <= 10
            and 4 <= successor.position // board.width <= 10
        )
        gem = next(s.point for s in board.slots if s.kind == "gem")
        dist = abs(gem % board.width - successor.position % board.width) + abs(
            gem // board.width - successor.position // board.width
        )
        if dist < best["gem_distance"] and not successor.lose:
            best.update(
                gem_distance=dist,
                turn=successor.turn,
                effects=successor.effects,
                position=successor.position,
                inventory=successor.inventory,
            )
        return successor, delta

    module.fast_step = transition
    began = perf_counter()
    deadline = began + args.seconds
    result = module.search(
        board,
        current,
        score,
        deadline=deadline,
        mode="p2a",
        verify_candidate=lambda a, k: module.verify_plan(
            board, start, a, k, deadline=deadline
        ),
    )
    args.output.write_text(
        json.dumps(
            dict(
                instrumented=True,
                result=result._asdict(),
                counts=dict(counts),
                best=best,
                seconds=perf_counter() - began,
            ),
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
