from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sys

# Needed when this is loaded with grid-play --plugin-file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grid_universe.renderer.image import ImageMap
from grid_universe.state import State

from grid_adventure.env import GridAdventureEnv
from grid_adventure.rendering import IMAGE_MAP
from grid_play.config.sources.base import register_level_source
from grid_play.config.sources.level_selection import (
    Builder,
    make_level_selection_source,
)
from public_levels.catalog import PUBLIC_LEVEL_BUILDERS, PUBLIC_TURN_LIMIT


SOURCE_NAME = "Public Tests"
BUILDERS: dict[str, Builder] = dict(PUBLIC_LEVEL_BUILDERS)


def _env_factory(
    initial_state_fn: Callable[..., State], _image_map: ImageMap
) -> GridAdventureEnv:
    def initial_state_with_public_turn_limit(**kwargs: object) -> State:
        return replace(initial_state_fn(**kwargs), turn_limit=PUBLIC_TURN_LIMIT)

    sample_state = initial_state_with_public_turn_limit()
    return GridAdventureEnv(
        render_mode="rgb_array",
        initial_state_fn=initial_state_with_public_turn_limit,
        width=sample_state.width,
        height=sample_state.height,
        render_image_map=IMAGE_MAP,
    )


source = make_level_selection_source(
    name=SOURCE_NAME,
    builders=BUILDERS,
    builder_returns_gridstate=True,
    env_factory=_env_factory,
    image_maps=[IMAGE_MAP],
)

register_level_source(source)
