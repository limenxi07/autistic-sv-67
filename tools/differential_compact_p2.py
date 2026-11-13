"""Independent public semantic projection and transition differential checks."""

from __future__ import annotations


def project(observation):
    """Inspect every public gameplay component, independently of the encoder."""

    key_exists = any(
        getattr(e, "key", None) is not None
        for col in observation.grid
        for cell in col
        for actor in cell
        for e in (actor, *getattr(actor, "inventory_list", ()))
    )

    def components(entity):
        values = tuple(
            sorted(
                (
                    name,
                    tuple(
                        sorted(
                            (k, tuple(sorted(v)) if isinstance(v, set) else v)
                            for k, v in vars(value).items()
                        )
                    ),
                )
                for name, value in vars(entity).items()
                if value is not None
                and name
                not in ("entity_id", "appearance", "inventory_list", "status_list")
            )
        )

        if getattr(entity, "agent", None) is not None:
            order = tuple(e.entity_id for e in entity.inventory_list)
            if not key_exists:
                order = tuple(sorted(order))
            values += (("_inventory_order", (("ids", order),)),)
        return tuple(sorted(values))

    entities = []
    for x, column in enumerate(observation.grid):
        for y, cell in enumerate(column):
            for entity in cell:
                entities.append(
                    (
                        entity.entity_id,
                        x + y * observation.width,
                        "grid",
                        components(entity),
                    )
                )
                for location, field in [
                    ("inventory", "inventory_list"),
                    ("effect", "status_list"),
                ]:
                    for item in getattr(entity, field, ()):
                        entities.append(
                            (item.entity_id, -1, location, components(item))
                        )
    return (
        observation.turn,
        observation.score,
        observation.win,
        observation.lose,
        tuple(sorted(entities)),
    )


def project_fast(board, state, score):
    entities = []
    effects = {i: (t, u) for i, t, u in state.effects}
    boxes = dict(zip(board.box_slots, state.boxes))
    for i, slot in enumerate(board.slots):
        components = dict(slot.components)
        point, location = slot.point, "grid"
        if slot.kind == "agent":
            point = state.position
            components["_inventory_order"] = (
                ("ids", tuple(board.slots[i].identity for i in state.inventory_order)),
            )
            components["health"] = (
                ("current_health", state.health),
                ("max_health", state.max_health),
            )
            components["inventory"] = (("item_ids", ()),)
            components["status"] = (("effect_ids", ()),)
        elif i in boxes:
            point = boxes[i]
        elif slot.kind == "door" and not state.doors & (1 << i):
            components = {}
        elif i in effects:
            point, location = -1, "effect"
            components.pop("collectible", None)
            t, u = effects[i]
            if t != -1:
                components["time_limit"] = (("amount", t),)
            if u != -1:
                components["usage_limit"] = (("amount", u),)
        elif state.inventory & (1 << i):
            point, location = -1, "inventory"
            if slot.kind != "key":
                components.pop("collectible", None)
        elif slot.kind in (
            "coin",
            "gem",
            "key",
            "boots",
            "ghost",
            "shield",
        ) and not state.items & (1 << i):
            continue
        entities.append(
            (slot.identity, point, location, tuple(sorted(components.items())))
        )
    return state.turn, score, state.win, state.lose, tuple(sorted(entities))


def run(
    *,
    layouts=1000,
    comparisons=100000,
    seed=20261108,
    output,
    source="agents/compact_p2.py",
):
    import hashlib
    import json
    import random
    import sys
    from pathlib import Path
    from time import perf_counter
    from typing import Any

    from grid_adventure.grid import step
    from grid_adventure.step import Action

    from tools.benchmark_planner_p1 import load_agent

    module: Any = sys.modules[load_agent(str(source)).__module__]
    encode, fast_step, predicted_key, read_state = (
        module.encode,
        module.fast_step,
        module.predicted_key,
        module.read_state,
    )
    from tests.compact_p2_fixtures import generated
    from tests.planner_p1_fixtures import fixture_cases
    from tests.test_compact_p2_step import interactions

    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path("agents/compact_p2.py"),
            Path(__file__),
            Path("tests/compact_p2_fixtures.py"),
        )
    }
    hashes["agents/compact_p2.py"] = hashlib.sha256(
        Path(source).read_bytes()
    ).hexdigest()
    rng = random.Random(seed)
    mandatory = (
        "movement",
        "pickup",
        "key-door",
        "box",
        "overlap",
        "damage",
        "shield",
        "stacked",
        "boots",
        "phasing",
        "wait",
        "terminal",
        "final-turn",
    )
    coverage = dict.fromkeys(mandatory, 0)
    count = 0
    fast_seconds = reference_seconds = 0.0
    entries = [
        (builder(), actions, "mechanic-" + label)
        for label, (builder, actions) in fixture_cases("mechanics").items()
    ]
    entries += [
        (s, a, "interaction-" + str(i)) for i, (s, a) in enumerate(interactions())
    ]
    from tests.planner_p1_fixtures import corridor

    s = corridor()
    s.turn_limit = 2
    entries.append((s, (Action.RIGHT, Action.RIGHT), "final-turn"))
    for i in range(layouts):
        s, a, _, _ = generated(i)
        entries.append((s, a, "development-" + str(i)))
    from tests.compact_p2_fixtures import identity_case

    entries += [
        (s, a, "identity-" + str(i)) for i in range(100) for s, a in (identity_case(i),)
    ]
    baseline_path = Path("artifacts/p2/baseline/public-repeat3.json")
    public_cases = 0
    if baseline_path.exists():
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        baseline = json.loads(baseline_path.read_text())
        for row in baseline["results"]:
            if row["repetition"] != 0:
                continue
            start = PUBLIC_GAMEPLAY_LEVEL_BUILDERS[row["level"]]()
            start.turn_limit = 150
            actions = tuple(Action(a) for a in row["trace_actions"])
            entries.append((start, actions, "public-" + row["level"]))
            public_cases += 1
    frames_per = max(1, (comparisons + 7 * len(entries) - 1) // (7 * len(entries)))
    traces = []
    unsupported = []
    for start, actions, label in entries:
        actions = tuple(actions)
        try:
            board, _, _ = encode(start)
        except module.UnsupportedState as error:
            unsupported.append({"layout": label, "reason": str(error)})
            # A public-only prefix can establish a supported repair state.
            while actions:
                start = step(start, actions[0])
                actions = actions[1:]
                try:
                    board, _, _ = encode(start)
                    break
                except module.UnsupportedState:
                    continue
            else:
                continue
        frames = [(start, ())]
        s = start
        for action in actions:
            if s.win or s.lose:
                break
            s = step(s, action)
            frames.append((s, frames[-1][1] + (action,)))
        # Add a few off-witness reachable frames, preserving their action trace.
        s = start
        prefix = ()
        for _ in range(10):
            action = rng.choice(tuple(Action))
            prefix += (action,)
            s = step(s, action)
            frames.append((s, prefix))
            if s.win or s.lose:
                break
        sampled = (
            frames
            if not label.startswith("development")
            else rng.sample(frames, min(frames_per, len(frames)))
        )
        for s, prefix in sampled:
            # Use the original board slots for the actual comparison.
            compact, score = read_state(board, s)
            kinds = {board.slots[i].kind for i, _, _ in compact.effects}
            for action in Action:
                before = perf_counter()
                actual = step(s, action)
                reference_seconds += perf_counter() - before
                before = perf_counter()
                successor, delta = fast_step(board, compact, action)
                predicted_key(board, successor, score + delta)
                fast_seconds += perf_counter() - before
                want, got = (
                    project(actual),
                    project_fast(board, successor, score + delta),
                )
                count += 1
                if want != got:
                    # Greedy action deletion gives a reproducible shorter witness.
                    minimal = list(prefix)

                    def mismatches(trace, start=start, board=board, action=action):
                        test = start
                        for a in trace:
                            test = step(test, a)
                        c, v = read_state(board, test)
                        n, d = fast_step(board, c, action)
                        return project(step(test, action)) != project_fast(
                            board, n, v + d
                        )

                    for j in range(len(minimal) - 1, -1, -1):
                        trial = minimal[:j] + minimal[j + 1 :]
                        if mismatches(trial):
                            minimal = trial
                    report = {
                        "passed": False,
                        "seed": seed,
                        "source_hashes": hashes,
                        "first_differing_field": next(
                            (
                                name
                                for name, a, b in zip(
                                    ("turn", "score", "win", "lose", "entities"),
                                    want,
                                    got,
                                )
                                if a != b
                            ),
                            "unknown",
                        ),
                        "count": count,
                        "label": label,
                        "prefix": [a.value for a in minimal],
                        "action": action.value,
                        "expected": repr(want),
                        "actual": repr(got),
                        "coverage": coverage,
                    }
                    Path(output).write_text(json.dumps(report, indent=2) + "\n")
                    return report
                coverage["movement"] += action in (
                    Action.UP,
                    Action.DOWN,
                    Action.LEFT,
                    Action.RIGHT,
                )
                coverage["pickup"] += action == Action.PICK_UP
                coverage["key-door"] += bool(board.doors and compact.inventory)
                coverage["box"] += bool(compact.boxes)
                coverage["overlap"] += any(
                    len(cell) > 1 for col in s.grid for cell in col
                )
                coverage["damage"] += any(slot.kind == "lava" for slot in board.slots)
                coverage["shield"] += "shield" in kinds
                coverage["stacked"] += len(compact.effects) > 1
                coverage["boots"] += "boots" in kinds
                coverage["phasing"] += "ghost" in kinds
                coverage["wait"] += action == Action.WAIT
                coverage["terminal"] += actual.win or actual.lose
                coverage["final-turn"] += compact.turn == board.turn_limit - 1
            traces.append({"layout": label, "actions": [a.value for a in prefix]})
    report = {
        "passed": count >= comparisons and layouts >= 1000 and all(coverage.values()),
        "seed": seed,
        "comparisons": count,
        "layouts": layouts,
        "coverage": coverage,
        "source_hashes": hashes,
        "public_trace_cases": public_cases,
        "unsupported_reference_prefixes": unsupported,
        "fast_transition_key_seconds": fast_seconds,
        "reference_step_seconds": reference_seconds,
        "transition_key_speedup": reference_seconds / fast_seconds,
        "corpus": traces,
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "corpus"}


def main():
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=Path("agents/compact_p2.py")
    )
    p.add_argument("--layouts", type=int, default=1000)
    p.add_argument("--comparisons", type=int, default=100000)
    p.add_argument("--seed", type=int, default=20261108)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(**vars(args))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
