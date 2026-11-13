"""Compare every action with the reference engine and audit mechanic coverage."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

from grid_adventure.grid import step
from grid_adventure.step import Action

from tests.boss_p2_next_fixtures import boss_like, semantic_projection
from tools.benchmark_planner_p1 import load_agent
from tools.differential_compact_p2 import project, project_fast


def legacy_entries():
    from public_levels.catalog import PUBLIC_LEVEL_BUILDERS
    from tests.compact_p2_fixtures import heldout_case

    paths = [
        ("public", Path("artifacts/p2/p2a-ready/public-repeat3.json")),
        ("p2-v2", Path("artifacts/p2/p2a-ready/heldout-v2.json")),
        ("boss", Path("artifacts/p2-next/baseline-frozen.json")),
        ("candidate-boss", Path("artifacts/p2-next/candidate-task3-boss-v2.json")),
    ]
    for suite, path in paths:
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["results"]:
            if row["repetition"] != 0:
                continue
            state = (
                heldout_case(int(row["level"].split("-")[-1]))[0]
                if suite == "p2-v2"
                else PUBLIC_LEVEL_BUILDERS[row["level"]]()
            )
            state.turn_limit, state.seed = 150, row["seed"]
            yield (
                suite + ":" + row["level"],
                state,
                tuple(Action(a) for a in row["trace_actions"]),
                row,
            )


def audit(module, source):
    from public_levels.catalog import PUBLIC_LEVEL_BUILDERS

    sites = []
    for node in ast.walk(ast.parse(Path(source).read_text())):
        if (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "UnsupportedState"
        ):
            sites.append(dict(line=node.lineno, reason=ast.unparse(node.exc.args[0])))
    rows = []
    entries = [
        ("initial:" + name, build(), (), None)
        for name, build in PUBLIC_LEVEL_BUILDERS.items()
    ]
    entries += list(legacy_entries())
    for label, state, actions, expected in entries:
        counts = Counter()
        first = None
        for index in range(len(actions) + 1):
            try:
                module.encode(state)
                counts["supported"] += 1
            except module.UnsupportedState as error:
                counts[str(error)] += 1
                if first is None:
                    first = dict(turn=state.turn, reason=str(error))
            if index < len(actions):
                state = step(state, actions[index])
        rows.append(
            dict(
                label=label,
                counts=dict(counts),
                first_unsupported=first,
                win=state.win,
                reward=state.score,
                replay_matches=None
                if expected is None
                else (state.win, state.lose, state.score)
                == (expected["win"], expected["lose"], expected["reward"]),
            )
        )
    return dict(rejection_sites=sorted(sites, key=lambda s: s["line"]), traces=rows)


def run(source, layouts=1000):
    module = sys.modules[load_agent(str(source)).__module__]
    rng = random.Random(20261109)
    from tests.compact_p2_fixtures import identity_case
    from tests.planner_p1_fixtures import fixture_cases
    from tests.test_compact_p2_step import interactions

    entries = [
        (name, build(), tuple(actions), None)
        for name, (build, actions) in fixture_cases("mechanics").items()
    ]
    entries += [
        (f"interaction-{i}", s, tuple(a), None)
        for i, (s, a) in enumerate(interactions())
    ]
    entries += [(f"identity-{i}", *identity_case(i), None) for i in range(100)]
    entries += list(legacy_entries())
    entries += [(f"many-{i}", *boss_like(i)[:2], None) for i in range(layouts)]
    count = frames = layouts_supported = many_initial_supported = many_compared = 0
    unsupported = []
    coverage = Counter()
    started = perf_counter()
    for label, start, actions, _ in entries:
        try:
            board, _, _ = module.encode(start)
            if label.startswith("many-"):
                many_initial_supported += 1
        except module.UnsupportedState as error:
            unsupported.append(dict(label=label, reason=str(error)))
            # Preserve conservative exclusion; test later supported frames.
            actions = tuple(actions)
            while actions:
                start, actions = step(start, actions[0]), actions[1:]
                try:
                    board, _, _ = module.encode(start)
                    break
                except module.UnsupportedState:
                    pass
            else:
                continue
        layouts_supported += 1
        states = [start]
        state = start
        for action in actions:
            state = step(state, action)
            states.append(state)
            if state.win or state.lose:
                break
        if label.startswith("many-"):
            # Stratified prefix/suffix plus random witness and off-witness frames.
            states = states[:5] + states[-4:] + rng.sample(states, min(10, len(states)))
        state = start
        for _ in range(5):
            state = step(state, rng.choice(tuple(Action)))
            states.append(state)
        for state in states:
            current, score = module.read_state(board, state)
            frames += 1
            key_count = board.keys.bit_count()
            want = semantic_projection(project(state), key_count)
            got = semantic_projection(project_fast(board, current, score), key_count)
            if want != got:
                return dict(
                    passed=False,
                    label=label,
                    turn=state.turn,
                    action="read_state",
                    expected=repr(want),
                    actual=repr(got),
                    comparisons=count,
                )
            kinds = {board.slots[i].kind for i, _, _ in current.effects}
            coverage["many_items_one_key"] += (
                key_count == 1 and (board.coins | board.gems).bit_count() >= 4
            )
            coverage["mixed_inventory"] += current.inventory.bit_count() > 4
            coverage["key_consumed"] += bool(
                board.keys and not board.keys & (current.items | current.inventory)
            )
            coverage["stacked_effects"] += len(current.effects) > 1
            coverage["expiry"] += any(t == 1 for _, t, _ in current.effects)
            coverage["boxes"] += bool(current.boxes)
            coverage["overlaps"] += any(len(c) > 1 for col in state.grid for c in col)
            coverage["boots"] += "boots" in kinds
            coverage["phasing"] += "ghost" in kinds
            coverage["shield"] += "shield" in kinds
            for action in Action:
                successor, delta = module.fast_step(board, current, action)
                actual = step(state, action)
                want = semantic_projection(project(actual), key_count)
                got = semantic_projection(
                    project_fast(board, successor, score + delta), key_count
                )
                count += 1
                if want != got:
                    return dict(
                        passed=False,
                        label=label,
                        turn=state.turn,
                        action=action.value,
                        comparisons=count,
                        expected=repr(want),
                        actual=repr(got),
                    )
                # Actual read-back must agree too, not only projection.
                if module.read_state(board, actual) != (successor, score + delta):
                    return dict(
                        passed=False,
                        label=label,
                        turn=state.turn,
                        action=action.value,
                        comparisons=count,
                        reason="read-back mismatch",
                    )
        if label.startswith("many-"):
            many_compared += 1
            if int(label[5:]) % 100 == 0:
                print(label, count, flush=True)
    return dict(
        passed=count >= 100000
        and many_initial_supported >= 1000
        and many_compared >= 1000
        and all(coverage.values()),
        comparisons=count,
        frames=frames,
        layouts=layouts,
        supported_layouts=layouts_supported,
        many_initial_supported=many_initial_supported,
        many_compared=many_compared,
        coverage=dict(coverage),
        unsupported=unsupported,
        elapsed_seconds=perf_counter() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--layouts", type=int, default=1000)
    args = parser.parse_args()
    assert not args.output.exists()
    module = sys.modules[load_agent(str(args.source)).__module__]
    result = (
        audit(module, args.source)
        if args.audit_only
        else run(args.source, args.layouts)
    )
    result["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k
            not in ("traces", "unsupported", "rejection_sites", "expected", "actual")
        },
        flush=True,
    )
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
