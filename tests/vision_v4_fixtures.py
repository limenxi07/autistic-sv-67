"""Independent public-API transitions for temporal V4 data and recovery tests."""

from grid_adventure.entities import AgentEntity, ExitEntity, GemEntity
from grid_adventure.grid import GridState, step
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
from grid_adventure.step import Action

from agents.vision_v4_bridge import ENTITY_TYPES


def composite_sequence(kind, *, box=False):
    state = GridState(
        width=8,
        height=3,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=20261109,
        turn_limit=150,
        step_cost=3,
    )
    state.add((0, 1), AgentEntity())
    state.add((7, 1), ExitEntity())
    # A required gem off the path allows standing on an exit without winning.
    state.add((7, 0), GemEntity())
    if box:
        state.add((1, 1), ENTITY_TYPES["box"]())
        state.add((2, 1), ENTITY_TYPES[kind]())
        actions = (Action.RIGHT, Action.RIGHT, Action.RIGHT)
    elif kind in ("wall", "locked"):
        state.add((1, 1), ENTITY_TYPES["ghost"]())
        state.add((2, 1), ENTITY_TYPES[kind]())
        actions = (Action.RIGHT, Action.PICK_UP, Action.RIGHT, Action.WAIT)
    elif kind == "lava":
        state.add((1, 1), ENTITY_TYPES["shield"]())
        state.add((2, 1), ENTITY_TYPES[kind]())
        actions = (Action.RIGHT, Action.PICK_UP, Action.RIGHT, Action.WAIT)
    else:
        state.add((1, 1), ENTITY_TYPES[kind]())
        actions = (Action.RIGHT, Action.WAIT, Action.PICK_UP, Action.RIGHT)
    frames = []
    current = state
    for action in actions:
        current = step(current, action)
        if current.lose or current.win:
            break
        frames.append(current)
    return state, tuple(frames), actions


def configure_vision():
    from agents.vision_v4_model import POLICY, TEMPERATURE, get_model
    from agents.vision_v4_perception import configure

    configure(get_model, TEMPERATURE, policy=POLICY)
