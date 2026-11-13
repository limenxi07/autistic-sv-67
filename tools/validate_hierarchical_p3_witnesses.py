"""Replay known solutions and check endpoint mechanics against the public engine."""

import argparse
import hashlib
import json
from pathlib import Path

from grid_adventure.grid import step
from grid_adventure.step import Action

from agents import hierarchical_p3
from tests.boss_p2_next_fixtures import boss_like, semantic_projection
from tests.hierarchical_p3_fixtures import HOLDOUT_COUNT, holdout_case
from tools.differential_compact_p2 import project, project_fast


def validate():
    rows = []
    comparisons = 0
    for suite, count in [("development", 12), ("holdout", HOLDOUT_COUNT)]:
        for i in range(count):
            state, actions, _ = (
                holdout_case(i) if suite == "holdout" else boss_like(i, split="heldout")
            )
            state.seed = 42
            b, c, value = hierarchical_p3.encode(state)
            frames = 0
            for action in (None,) + actions:
                if action is not None:
                    state = step(state, action)
                    c, delta = hierarchical_p3.fast_step(b, c, action)
                    value += delta
                assert hierarchical_p3.read_state(b, state) == (c, value), (
                    suite,
                    i,
                    "prefix",
                )
                frames += 1
                for alternative in Action:
                    nxt, delta = hierarchical_p3.fast_step(b, c, alternative)
                    assert semantic_projection(
                        project(step(state, alternative)), b.keys.bit_count()
                    ) == semantic_projection(
                        project_fast(b, nxt, value + delta), b.keys.bit_count()
                    ), (suite, i, state.turn, alternative)
                    comparisons += 1
            assert state.win and not state.lose, (suite, i, "invalid witness")
            rows.append(
                {
                    "suite": suite,
                    "index": i,
                    "frames": frames,
                    "reward": state.score,
                    "win": state.win,
                }
            )
    return {
        "passed": True,
        "source_sha256": hashlib.sha256(
            Path("agents/hierarchical_p3.py").read_bytes()
        ).hexdigest(),
        "fixture_sha256": hashlib.sha256(
            Path("tests/hierarchical_p3_fixtures.py").read_bytes()
        ).hexdigest(),
        "comparisons": comparisons,
        "results": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    r = validate()
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print("passed", r["comparisons"])


if __name__ == "__main__":
    main()
