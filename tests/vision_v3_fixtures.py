"""Evaluator-only V3 fixtures. Never included in submitted source."""

from copy import deepcopy

from grid_adventure.env import GridAdventureEnv
from grid_adventure.grid import to_state


def configure_vision():
    from pathlib import Path

    from agents.vision_v3_perception import configure

    root = Path(__file__).resolve().parents[1] / "artifacts/v3/models/selected-export"
    namespace = {}
    exec((root / "model_loader.py").read_text(), namespace)  # noqa: S102
    configure(namespace["get_model"], 0.25)


def vision_observation(state):
    """Normal renderer, validation-only artwork; no oracle labels in the image."""
    import inspect
    from pathlib import Path

    from agents.vision_v3_bridge import CLASSES

    original = (
        inspect.signature(GridAdventureEnv).parameters["render_image_map"].default
    )
    mapping = type(original)(
        {key: value for key, value in original.items() if value in CLASSES}
    )
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts/v3/data/new-layout-unseen-art-v2/asset-root"
    )
    env = GridAdventureEnv(
        initial_state_fn=lambda *args, **kwargs: to_state(state),
        width=state.width,
        height=state.height,
        observation_type="image",
        render_asset_root=str(root),
        render_image_map=mapping,
    )
    return env.reset()[0]


def image_env(state, seed=None):
    state = deepcopy(state)
    if seed is not None:
        state.seed = seed
    return GridAdventureEnv(
        initial_state_fn=lambda *args, **kwargs: to_state(state),
        width=state.width,
        height=state.height,
        observation_type="image",
    )


def observable(state):
    """Public observations with resource IDs erased, preserving entry order.

    Numeric IDs are not a semantic comparison across re-identification. Resource
    components, list ordering and all board layers remain in the comparison.
    """
    observation, _ = image_env(state).reset()
    info = deepcopy(observation["info"])
    for field in ("inventory", "effects"):
        for entry in info["agent"][field]:
            entry.pop("id")
    cells = []
    for y in range(state.height):
        for x in range(state.width):
            cell = []
            for entity in state.grid[x][y]:
                components = []
                for name, value in vars(entity).items():
                    if value is None or name in (
                        "entity_id",
                        "appearance",
                        "inventory_list",
                        "status_list",
                        "inventory",
                        "status",
                    ):
                        continue
                    components.append((name, tuple(sorted(vars(value).items()))))
                cell.append(tuple(sorted(components)))
            cells.append(tuple(sorted(cell)))
    return info, tuple(cells)
