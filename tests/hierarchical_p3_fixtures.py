"""Frozen P3 holdout: vertical switchbacks, cross corridors, and comb rooms.

Distinct seeds and rotated topology construction from P2's horizontal snakes.
Only public constructors; the witness is generated without consulting a planner.
"""

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

HOLDOUT_COUNT = 18


def holdout_case(index):
    seed = 991731001 + index * 15485863
    rng = random.Random(seed)
    width, height = 9 + 2 * (index % 3), 9 + 2 * ((index // 3) % 4)
    route = []
    for col, x in enumerate(range(1, width - 1, 2)):
        ys = list(range(1, height - 1))
        if col % 2:
            ys.reverse()
        if route:
            route.append((x - 1, ys[0]))
        route.extend((x, y) for y in ys)
    if index % 2:
        route = [(width - 1 - x, y) for x, y in route]
    free = set(route)
    # Cross corridors and comb rooms, distinct from the original horizontal bays.
    if index % 3 == 1:
        free.update((x, y) for x in range(1, width - 1) for y in (height // 2,))
    if index % 3 == 2:
        free.update((x + 1, y) for x, y in route if x + 1 < width - 1 and y % 3 == 1)
    state = GridState(
        width=width,
        height=height,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        step_cost=3,
        turn_limit=150,
    )
    player = AgentEntity()
    player.set_health(3 if index % 2 else 5)
    state.add(route[0], player)
    state.add(route[-1], ExitEntity())
    pickups = set()
    used = set()

    def put(n, cls, pick=True):
        state.add(route[n], cls())
        used.add(n)
        if pick:
            pickups.add(n)

    put(1, KeyEntity)
    put(4, LockedDoorEntity, False)
    put(2, ShieldPowerUpEntity)
    put(3, ShieldPowerUpEntity)
    put(6, SpeedPowerUpEntity)
    put(8, PhasingPowerUpEntity)
    put(10, LavaEntity, False)
    put(11, LavaEntity, False)
    choices = [n for n in range(12, len(route) - 1) if n not in used]
    rng.shuffle(choices)
    for j, n in enumerate(choices[:12]):
        put(n, GemEntity if j < 4 else CoinEntity)
    extra = sorted(free - set(route))
    if extra:
        state.add(rng.choice(extra), BoxEntity())
    for x in range(width):
        for y in range(height):
            if (x, y) not in free:
                state.add((x, y), WallEntity())
    for e, identity in zip(
        [e for c in state.grid for cell in c for e in cell],
        rng.sample(range(10**13, 10**13 + 1000000), width * height),
    ):
        e.entity_id = identity
    directions = {
        (1, 0): Action.RIGHT,
        (-1, 0): Action.LEFT,
        (0, 1): Action.DOWN,
        (0, -1): Action.UP,
    }
    actions = []
    for n in range(1, len(route)):
        if n == 4:
            actions.append(Action.USE_KEY)
        x, y = route[n - 1]
        a, b = route[n]
        actions.append(directions[a - x, b - y])
        if n in pickups:
            actions.append(Action.PICK_UP)
            if n in (6, 8):
                actions.extend([Action.WAIT] * 6)
    return state, tuple(actions), seed
