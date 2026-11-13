"""Public image/metadata contract, without access to an evaluation environment.

Grid-only IDs occupy negative local slots. IDs explicitly exposed in public
resource metadata are retained after pickup: changing their integer values can
change Python-set key consumption order. No unobserved engine IDs are read.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any

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
    UnlockedDoorEntity,
    WallEntity,
)
from grid_adventure.grid import GridState
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES

from agents.v4_types import BoardHypothesis

CLASSES = (
    "boots",
    "box",
    "coin",
    "exit",
    "floor",
    "gem",
    "ghost",
    "human",
    "key",
    "lava",
    "locked",
    "opened",
    "shield",
    "wall",
)
ENTITY_TYPES: dict[str, Any] = dict(
    zip(
        CLASSES,
        (
            SpeedPowerUpEntity,
            BoxEntity,
            CoinEntity,
            ExitEntity,
            None,
            GemEntity,
            PhasingPowerUpEntity,
            AgentEntity,
            KeyEntity,
            LavaEntity,
            LockedDoorEntity,
            UnlockedDoorEntity,
            ShieldPowerUpEntity,
            WallEntity,
        ),
    )
)
EFFECT_TYPES = {"SPEED": "boots", "PHASING": "ghost", "IMMUNITY": "shield"}


def split_cells(observation):
    image = observation["image"]
    config = observation["info"]["config"]
    width, height = config["width"], config["height"]
    ih, iw = image.shape[:2]
    if width <= 0 or height <= 0 or iw < width or ih < height:
        raise ValueError("Invalid image/grid geometry")
    return [
        [
            image[
                y * ih // height : (y + 1) * ih // height,
                x * iw // width : (x + 1) * iw // width,
                :,
            ]
            for x in range(width)
        ]
        for y in range(height)
    ]


def decode_metadata(info):
    config, status, agent = info["config"], info["status"], info["agent"]
    objective = config["objective"]
    if objective == "collect_exit":
        objective = "collect_gems_and_exit"
    if objective not in OBJECTIVES or config["movement"] not in MOVEMENTS:
        raise ValueError("Unsupported image configuration")
    return {
        **{key: config[key] for key in ("width", "height", "movement")},
        "objective": objective,
        "seed": None if config["seed"] == -1 else config["seed"],
        "turn_limit": None if config["turn_limit"] == -1 else config["turn_limit"],
        **{key: status[key] for key in ("turn", "score", "phase")},
        "health": deepcopy(agent["health"]),
        "inventory": deepcopy(agent["inventory"]),
        "effects": deepcopy(agent["effects"]),
    }


def _resource(entry, *, effect=False):
    kind = EFFECT_TYPES[entry["type"]] if effect else entry["appearance_name"]
    constructor = ENTITY_TYPES[kind]
    if constructor is None or kind not in (
        "boots",
        "ghost",
        "shield",
        "coin",
        "gem",
        "key",
    ):
        raise ValueError("Unsupported resource type")
    entity = constructor(entity_id=entry["id"])
    if not isinstance(entry["id"], int) or entry["id"] < 0:
        raise ValueError("Unsupported public resource ID")
    if kind == "key":
        entity.key = replace(entity.key, key_id=entry["key_id"])
    else:
        entity.collectible = None
    if effect:
        field = {"TIME": "time_limit", "USAGE": "usage_limit"}[entry["limit_type"]]
        setattr(
            entity, field, replace(getattr(entity, field), amount=entry["limit_amount"])
        )
        if kind == "boots":
            entity.speed = replace(entity.speed, multiplier=entry["multiplier"])
    return entity


def reconstruct(labels, info, previous=None):
    """Build a conditional hypothesis; tuple labels assert observed full layers.

    A single actor/box label on a later frame does not establish the covered
    layers. A same-turn predicted previous state supplies corroborated history;
    without it the ambiguity is returned explicitly.
    """
    meta = decode_metadata(info)
    width, height = meta["width"], meta["height"]
    if len(labels) != height or any(len(row) != width for row in labels):
        raise ValueError("Labels do not match metadata geometry")
    if previous is not None and (
        (previous.width, previous.height) != (width, height)
        or previous.turn != meta["turn"]
    ):
        raise ValueError("History must be the same-turn prediction")
    state = GridState(
        width=width,
        height=height,
        movement=MOVEMENTS[meta["movement"]],
        objective=OBJECTIVES[meta["objective"]],
        seed=meta["seed"],
        turn_limit=meta["turn_limit"],
        step_cost=3,
        turn=meta["turn"],
        score=meta["score"],
        win=meta["phase"] == "win",
        lose=meta["phase"] == "lose",
        message=info.get("message") or None,
    )
    unresolved = []
    old_ids = (
        [e.entity_id for col in previous.grid for cell in col for e in cell]
        if previous
        else []
    )
    next_id = min([-1] + old_ids) - 1
    agents: list[AgentEntity] = []
    for y, row in enumerate(labels):
        for x, label in enumerate(row):
            explicit_layers = isinstance(label, (tuple, list))
            kinds = tuple(label) if explicit_layers else (label,)
            old = list(previous.grid[x][y]) if previous else []
            old = deepcopy(old)
            for kind in kinds:
                if kind not in ENTITY_TYPES:
                    unresolved.append((x, y, "unknown_class", kind))
                    continue
                if kind == "floor":
                    continue
                constructor = ENTITY_TYPES[kind]
                match = next((e for e in old if isinstance(e, constructor)), None)
                if match is None and kind == "human" and previous is not None:
                    # There is one actor. A visually established unexpected
                    # position must preserve its local identity during repair.
                    match = next(
                        (
                            deepcopy(e)
                            for col in previous.grid
                            for cell in col
                            for e in cell
                            if getattr(e, "agent", None) is not None
                        ),
                        None,
                    )
                if match is not None:
                    if match in old:
                        old.remove(match)
                    entity = match
                else:
                    entity = constructor(entity_id=next_id)
                    next_id -= 1
                state.add((x, y), entity)
                if kind == "human":
                    agents.append(entity)
            if not explicit_layers and any(k in ("human", "box") for k in kinds):
                if previous is None and meta["turn"] > 0:
                    unresolved.append((x, y, "covered_layers_without_history"))
                elif previous is not None:
                    for entity in old:
                        state.add((x, y), entity)
    if len(agents) != 1:
        unresolved.append(("agent_count", len(agents)))
    for agent in agents:
        agent.health = replace(agent.health, **meta["health"])
        agent.inventory_list = [_resource(entry) for entry in meta["inventory"]]
        agent.status_list = [_resource(entry, effect=True) for entry in meta["effects"]]
    return BoardHypothesis(state, 1.0 if not unresolved else 0.0, tuple(unresolved))
