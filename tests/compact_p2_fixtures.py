"""Deterministic disjoint P2 layout families; no planner-derived labels."""

from __future__ import annotations

import itertools
import random

from grid_adventure.entities import (
    AgentEntity,
    BoxEntity,
    CoinEntity,
    ExitEntity,
    GemEntity,
    KeyEntity,
    LavaEntity,
    LockedDoorEntity,
    PhasingPowerUpEntity,
    ShieldPowerUpEntity,
    SpeedPowerUpEntity,
    WallEntity,
)
from grid_adventure.grid import GridState
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
from grid_adventure.step import Action

FAMILIES = (
    "maze",
    "multi-gem",
    "coins",
    "keys-doors",
    "boxes",
    "lava-shields",
    "boots",
    "phasing",
    "mixed-effects",
    "combined",
)


def generated(index, *, split="development"):
    family = FAMILIES[index % 10]
    seed = (923817 if split == "heldout" else 472991) + index * 7919
    rng = random.Random(seed)
    w = h = 15 if index % 3 == 0 else 7 + index % 8
    s = GridState(
        width=w,
        height=h,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        step_cost=3,
        turn_limit=150,
    )
    agent = AgentEntity()
    agent.set_health(1 if index % 4 == 0 else 5)
    s.add((0, 1), agent)
    s.add((w - 1, h - 2), ExitEntity())
    # Witness turns through a horizontal corridor and a vertical arm. Vary
    # dead ends, junctions, hazards, and objectives off that path independently.
    for x in range(w - 1):
        for y in range(h):
            if y != 1 and not (x == w - 2 and y == 2) and rng.random() < 0.38:
                s.add((x, y), WallEntity())
    witness = []
    for x in range(1, w):
        if x == 3 and family in ("keys-doors", "combined"):
            s.add((x, 1), LockedDoorEntity())
            witness.append(Action.USE_KEY)
        witness.extend(
            (Action.DOWN, Action.RIGHT)
            if family == "boxes" and x == w - 1
            else (Action.RIGHT,)
        )
        if x == 1:
            if family in ("coins", "combined"):
                s.add((x, 1), CoinEntity())
            if family in ("keys-doors", "combined"):
                s.add((x, 1), KeyEntity())
            if family in ("lava-shields", "mixed-effects", "combined"):
                s.add((x, 1), ShieldPowerUpEntity())
            if family in ("boots", "mixed-effects"):
                s.add((x, 1), SpeedPowerUpEntity())
            if family in ("phasing", "mixed-effects"):
                s.add((x, 1), PhasingPowerUpEntity())
            if family not in ("maze", "multi-gem", "boxes"):
                witness.append(Action.PICK_UP)
                # Fix witness speed at one without relying on inferred timers.
                witness.extend((Action.WAIT,) * 5)
        if x in (w // 2, w - 2) and family in ("multi-gem", "combined", "coins"):
            s.add((x, 1), GemEntity() if family != "coins" else CoinEntity())
            witness.append(Action.PICK_UP)
        if x == 2 and family == "phasing":
            # Ghost expired above: characterize phasing in the differential
            # witnesses; this qualification map still contains an optional ghost.
            pass
        if x in (3, 4) and family in ("lava-shields", "mixed-effects"):
            s.add((x, 1), LavaEntity())
        if x == 2 and family == "boxes":
            s.add((x, 1), BoxEntity())
    witness.extend((Action.DOWN,) * (h - 4 if family == "boxes" else h - 3))
    return s, tuple(witness), family, seed


def heldout_case(index):
    """Qualification v2: DFS spanning-tree mazes, farthest exits and branches.

    This generator never calls the development elbow/corridor builder. Seeds,
    topology, start/exit distribution and objective routes are independent.
    Witness selection uses geometry alone and public replay, never a planner.
    """
    from collections import deque

    family = FAMILIES[index % 10]
    seed = 66181937 + index * 104729
    w = 7 + 2 * (index % 5)
    h = 7 + 2 * ((index // 5) % 5)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def neighbor(p, d):
        return p[0] + d[0], p[1] + d[1]

    def inside(p):
        return 0 < p[0] < w - 1 and 0 < p[1] < h - 1

    # Require a winding route; deterministic rejection depends only on geometry.
    for attempt in range(1000):
        rng = random.Random(seed + attempt * 100000007)
        origin = (1, 1) if index % 2 == 0 else (w - 2, h - 2)
        free = {origin}
        stack = [origin]
        while stack:
            p = stack[-1]
            options = [
                d
                for d in directions
                if inside(neighbor(neighbor(p, d), d))
                and neighbor(neighbor(p, d), d) not in free
            ]
            if not options:
                stack.pop()
                continue
            d = rng.choice(options)
            q = neighbor(p, d)
            r = neighbor(q, d)
            free.update((q, r))
            stack.append(r)
        parents = {origin: None}
        queue = deque([origin])
        last = origin
        while queue:
            last = queue.popleft()
            for d in directions:
                q = neighbor(last, d)
                if q in free and q not in parents:
                    parents[q] = last
                    queue.append(q)
        route = []
        p = last
        while p is not None:
            route.append(p)
            p = parents[p]
        route.reverse()
        route_dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(route)]
        turns = sum(a != b for a, b in itertools.pairwise(route_dirs))
        branch = None
        if family in ("boxes", "multi-gem"):
            for n, p in enumerate(route[2:-2], 2):
                for d in directions:
                    q = neighbor(p, d)
                    r = neighbor(q, d)
                    if (
                        inside(q)
                        and q not in route
                        and (family != "boxes" or inside(r) and r not in route)
                    ):
                        branch = (n, d, q, r)
                        break
                if branch:
                    break
        if (
            turns >= 4
            and len(route) < 110
            and (family not in ("boxes", "multi-gem") or branch)
        ):
            break
    else:
        raise RuntimeError("No bounded maze witness")
    state = GridState(
        width=w,
        height=h,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        step_cost=3,
        turn_limit=150,
    )
    agent = AgentEntity()
    agent.set_health(1 if index % 3 == 0 else 5)
    state.add(origin, agent)
    state.add(route[-1], ExitEntity())
    pickups = set()

    def item(n, cls):
        state.add(route[n], cls())
        pickups.add(n)

    if family in ("coins", "combined"):
        for n in (1, len(route) - 3):
            item(n, CoinEntity)
    if family in ("multi-gem", "combined"):
        for n in sorted({len(route) // 3, 2 * len(route) // 3}):
            item(n, GemEntity)
    if family in ("keys-doors", "combined"):
        item(1, KeyEntity)
        state.add(route[4], LockedDoorEntity())
    if family in ("lava-shields", "mixed-effects", "combined"):
        item(1, ShieldPowerUpEntity)
        for n in (6, 7, 8):
            state.add(route[n], LavaEntity())
    if family in ("phasing", "mixed-effects"):
        item(1, PhasingPowerUpEntity)
        state.add(route[2], WallEntity())
    if family == "boots":
        item(1, SpeedPowerUpEntity)
    if branch:
        n, d, q, r = branch
        free.add(q)
        if family == "boxes":
            free.add(r)
            state.add(q, BoxEntity())
        state.add(q, GemEntity())
    for x in range(w):
        for y in range(h):
            if (x, y) not in free:
                state.add((x, y), WallEntity())
    to_action = {
        (1, 0): Action.RIGHT,
        (-1, 0): Action.LEFT,
        (0, 1): Action.DOWN,
        (0, -1): Action.UP,
    }
    actions = []
    for n, d in enumerate(route_dirs, 1):
        if n == 4 and family in ("keys-doors", "combined"):
            actions.append(Action.USE_KEY)
        actions.append(to_action[d])
        if n in pickups:
            actions.append(Action.PICK_UP)
        if n == 1 and family == "boots":
            actions.extend((Action.WAIT,) * 5)
        if branch and n == branch[0]:
            direction = branch[1]
            actions.extend(
                (
                    to_action[direction],
                    Action.PICK_UP,
                    to_action[(-direction[0], -direction[1])],
                )
            )
    return state, tuple(actions), family, seed


def identity_case(index):
    """Single-pickup identity stress, with sparse IDs and stacked active effects."""
    from tests.planner_p1_fixtures import layout

    rng = random.Random(987611 + index)
    state = layout(width=15)
    ids = rng.sample(range(10**12, 10**12 + 10000), 12)
    kinds = (
        KeyEntity,
        KeyEntity,
        CoinEntity,
        GemEntity,
        ShieldPowerUpEntity,
        ShieldPowerUpEntity,
        SpeedPowerUpEntity,
        PhasingPowerUpEntity,
    )
    for j, cls in enumerate(kinds):
        entity = cls()
        entity.entity_id = ids[j]
        state.add((j, 1), entity)
    for j, cls in enumerate((LockedDoorEntity, LavaEntity), 9):
        entity = cls()
        entity.entity_id = ids[j]
        state.add((j, 1), entity)
    return state, (Action.RIGHT, Action.PICK_UP, Action.PICK_UP) * 10
