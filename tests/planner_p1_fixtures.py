"""Public-constructor fixtures; witnesses are replayed with the reference stepper."""

from grid_adventure.constants import STEP_COST
from grid_adventure.entities import AgentEntity, ExitEntity
from grid_adventure.grid import GridState
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
from grid_adventure.step import Action


class WaitAgent:
    def step(self, state):
        return Action.WAIT


class HangAgent:
    def step(self, state):
        while True:
            pass


class ErrorAgent:
    def step(self, state):
        raise RuntimeError("intentional harness fixture")


def corridor():
    state = GridState(
        width=3,
        height=1,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        step_cost=STEP_COST,
        turn_limit=150,
        seed=42,
    )
    state.add((0, 0), AgentEntity())
    state.add((2, 0), ExitEntity())
    return state


def layout(width=8, height=3, *, health=5, turn_limit=150):
    state = GridState(
        width=width,
        height=height,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        step_cost=STEP_COST,
        turn_limit=turn_limit,
        seed=42,
    )
    agent = AgentEntity()
    agent.set_health(health)
    state.add((0, 1), agent)
    state.add((width - 1, 1), ExitEntity())
    return state


def mechanic(kind):
    from grid_adventure.entities import (
        BoxEntity,
        CoinEntity,
        GemEntity,
        KeyEntity,
        LavaEntity,
        LockedDoorEntity,
        PhasingPowerUpEntity,
        ShieldPowerUpEntity,
        SpeedPowerUpEntity,
    )

    state = layout(width=10)
    if kind == "key_door":
        state.add((1, 1), KeyEntity())
        state.add((3, 1), LockedDoorEntity())
        actions = (Action.RIGHT, Action.PICK_UP, Action.RIGHT, Action.USE_KEY) + (
            Action.RIGHT,
        ) * 7
    elif kind in ("coin", "gem"):
        state.add((1, 1), CoinEntity() if kind == "coin" else GemEntity())
        actions = (Action.RIGHT, Action.PICK_UP) + (Action.RIGHT,) * 8
    elif kind == "box_coin":
        state.add((2, 1), BoxEntity())
        state.add((3, 1), CoinEntity())
        actions = (
            (Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.PICK_UP, Action.UP)
            + (Action.RIGHT,) * 6
            + (Action.DOWN,)
        )
    elif kind == "damage":
        state.add((1, 1), LavaEntity())
        actions = (Action.RIGHT,) * 9
    elif kind in ("boots", "ghost"):
        state.add(
            (1, 1), SpeedPowerUpEntity() if kind == "boots" else PhasingPowerUpEntity()
        )
        actions = (
            (Action.RIGHT, Action.PICK_UP) + (Action.WAIT,) * 5 + (Action.RIGHT,) * 8
        )
    elif kind == "shield":
        state.add((1, 1), ShieldPowerUpEntity())
        for x in range(2, 7):
            state.add((x, 1), LavaEntity())
        actions = (Action.RIGHT, Action.PICK_UP) + (Action.RIGHT,) * 8
    else:
        raise ValueError(kind)
    return state, actions


def generated_case(index):
    """Structural variants: even indices tune, odd indices remain held out."""
    import random

    from grid_adventure.entities import (
        BoxEntity,
        CoinEntity,
        GemEntity,
        KeyEntity,
        LavaEntity,
        LockedDoorEntity,
        PhasingPowerUpEntity,
        ShieldPowerUpEntity,
        SpeedPowerUpEntity,
        WallEntity,
    )

    rng = random.Random(20261108 + index)
    width = 7 + index % 7
    height = 3 + (index // 7) % 5
    row = 1 + (index // 3) % (height - 1)
    state = GridState(
        width=width,
        height=height,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        step_cost=STEP_COST,
        turn_limit=150,
        seed=42,
    )
    agent = AgentEntity()
    agent.set_health(3 + index % 3)
    state.add((0, row), agent)
    state.add((width - 1, row), ExitEntity())
    style = (index // 2) % 5
    actions = []
    # Keep the witness route clear; vary off-route obstacles.
    for x in range(width):
        for y in range(height):
            if y not in (row, row - 1) and rng.random() < 0.30:
                state.add((x, y), WallEntity())
    if style == 0:
        gem_x = 2 + rng.randrange(width - 3)
        state.add((1, row), CoinEntity())
        state.add((gem_x, row), GemEntity())
        for x in range(1, width):
            actions.append(Action.RIGHT)
            if x in (1, gem_x):
                actions.append(Action.PICK_UP)
    elif style == 1:
        key_x = 1 + rng.randrange(width - 4)
        door_x = key_x + 2
        state.add((key_x, row), KeyEntity())
        state.add((door_x, row), LockedDoorEntity())
        for x in range(1, width):
            if x == door_x:
                actions.append(Action.USE_KEY)
            actions.append(Action.RIGHT)
            if x == key_x:
                actions.append(Action.PICK_UP)
    elif style == 2:
        state.add((2, row), BoxEntity())
        state.add((3, row), CoinEntity())
        state.add((width - 2, row - 1), GemEntity())
        actions.extend((Action.RIGHT,) * 3 + (Action.PICK_UP, Action.UP))
        for x in range(4, width):
            actions.append(Action.RIGHT)
            if x == width - 2:
                actions.append(Action.PICK_UP)
        actions.append(Action.DOWN)
    elif style == 3:
        effect = ShieldPowerUpEntity() if index % 3 else PhasingPowerUpEntity()
        state.add((1, row), effect)
        if index % 3:
            for x in range(2, min(width - 1, 5)):
                state.add((x, row), LavaEntity())
        else:
            state.add((2, row), WallEntity())
        actions.extend((Action.RIGHT, Action.PICK_UP) + (Action.RIGHT,) * (width - 2))
    else:
        state.add((1, row), SpeedPowerUpEntity())
        state.add((width - 2, row), GemEntity())
        actions.extend((Action.RIGHT, Action.PICK_UP) + (Action.WAIT,) * 5)
        for x in range(2, width):
            actions.append(Action.RIGHT)
            if x == width - 2:
                actions.append(Action.PICK_UP)
    return state, tuple(actions)


def fixture_cases(split):
    from functools import partial

    if split == "mechanics":
        cases = {"corridor": (corridor, (Action.RIGHT, Action.RIGHT))}
        for kind in (
            "key_door",
            "coin",
            "gem",
            "box_coin",
            "damage",
            "boots",
            "ghost",
            "shield",
        ):
            _, witness = mechanic(kind)
            cases[kind] = (partial(_mechanic_state, kind), witness)
        return cases
    if split not in ("tuning", "heldout"):
        raise ValueError(split)
    parity = 0 if split == "tuning" else 1
    result = {}
    for i in range(parity, 100, 2):
        _, witness = generated_case(i)
        result[f"generated-{i:03d}"] = (partial(_generated_state, i), witness)
    return result


def _mechanic_state(kind):
    return mechanic(kind)[0]


def _generated_state(index):
    return generated_case(index)[0]
