"""Independent many-item snake/chamber families, with public replay witnesses."""

import random
from dataclasses import replace

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


def boss_like(index, *, split="development"):
    seed = (731993 if split == "development" else 88277191) + index * 104729
    rng = random.Random(seed)
    width = 9 + 2 * (index % 4)
    height = 7 + 2 * ((index // 4) % 3)
    route = []
    for row, y in enumerate(range(1, height - 1, 2)):
        xs = list(range(1, width - 1))
        if row % 2:
            xs.reverse()
        if route:
            route.append((xs[0], y - 1))
        route.extend((x, y) for x in xs)
    if index % 2:
        route = [(width - 1 - x, height - 1 - y) for x, y in route]
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
    player.set_health(5)
    state.add(route[0], player)
    state.add(route[-1], ExitEntity())
    free = set(route)
    # Three topology families: narrow snake, alternating chambers, side bays.
    if index % 3 == 1:
        for x in range(1, width - 1):
            for y in range(2, height - 2, 2):
                if rng.random() < 0.3:
                    free.add((x, y))
    if index % 3 == 2:
        free.update((x, y + 1) for x, y in route if y + 1 < height - 1 and x % 3 == 0)
    occupied = set()
    pickups = set()

    def put(n, cls):
        state.add(route[n], cls())
        occupied.add(n)
        if cls not in (LavaEntity, LockedDoorEntity):
            pickups.add(n)

    put(1, KeyEntity)
    door_index = 4
    put(door_index, LockedDoorEntity)
    # Stacked active shields, speed expiry and phasing expiry get exercised
    # without ambiguous simultaneous pickups. Extra WAITs fix witness speed.
    for n, cls in (
        (2, ShieldPowerUpEntity),
        (3, ShieldPowerUpEntity),
        (6, SpeedPowerUpEntity),
        (8, PhasingPowerUpEntity),
    ):
        put(n, cls)
    for n in (10, 11):
        put(n, LavaEntity)
    candidates = [n for n in range(5, len(route) - 1) if n not in occupied]
    rng.shuffle(candidates)
    for j, n in enumerate(candidates[: min(12, len(candidates))]):
        put(n, GemEntity if j < 3 else CoinEntity)
    # Movable box on an optional side bay: actions still test contact/push.
    bays = sorted(free - set(route))
    if bays:
        state.add(rng.choice(bays), BoxEntity())
    for x in range(width):
        for y in range(height):
            if (x, y) not in free:
                state.add((x, y), WallEntity())
    # Re-identify ALL entities before transitions, including colliding residues.
    entities = [e for col in state.grid for cell in col for e in cell]
    ids = rng.sample(range(10**12, 10**12 + 100000), len(entities))
    for e, identity in zip(entities, ids):
        e.entity_id = identity
    key_name = "key-" + str(index % 7)
    for e in entities:
        if getattr(e, "key", None):
            e.key = replace(e.key, key_id=key_name)
        if getattr(e, "locked", None):
            e.locked = replace(e.locked, key_id=key_name)
    directions = {
        (1, 0): Action.RIGHT,
        (-1, 0): Action.LEFT,
        (0, 1): Action.DOWN,
        (0, -1): Action.UP,
    }
    actions = []
    for n in range(1, len(route)):
        if n == door_index:
            actions.append(Action.USE_KEY)
        x, y = route[n - 1]
        a, b = route[n]
        actions.append(directions[a - x, b - y])
        if n in pickups:
            actions.append(Action.PICK_UP)
            if n in (6, 8):
                actions.extend((Action.WAIT,) * 6)
    return state, tuple(actions), seed


def semantic_projection(projection, key_count):
    """Only quotient proven irrelevant inventory order for <= one key."""
    if key_count > 1:
        return projection
    turn, score, win, lose, entities = projection
    result = []
    for identity, point, location, components in entities:
        fields = dict(components)
        if "_inventory_order" in fields:
            ids = dict(fields["_inventory_order"])["ids"]
            fields["_inventory_order"] = (("ids", tuple(sorted(ids))),)
        result.append((identity, point, location, tuple(sorted(fields.items()))))
    return turn, score, win, lose, tuple(result)
