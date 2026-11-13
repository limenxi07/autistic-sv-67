from grid_adventure.grid import GridState
from grid_adventure.env import ImageObservation
from grid_adventure.constants import STEP_COST
from grid_adventure.entities import AgentEntity, BoxEntity, CoinEntity, ExitEntity, GemEntity, KeyEntity, LavaEntity, LockedDoorEntity, PhasingPowerUpEntity, ShieldPowerUpEntity, SpeedPowerUpEntity, UnlockedDoorEntity, WallEntity
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
from grid_adventure.rendering import DEFAULT_ASSET_ROOT
from grid_universe.grid.entity import BaseEntity
from grid_adventure.step import Action
from grid_adventure.grid import step as grid_step
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image

class Agent:
    """Grid Adventure: Variant 1 agent template.

        This class is the single public interface that Coursemology will import and
        interact with when evaluating your submission. You should extend the
        internals (add helper classes / functions in other files if you wish) but
        MUST preserve:

    # State steppers
        1. The class name: Agent
        2. The public method: step(self, state: GridState | ImageObservation) -> Action

    # Utility helpers
        High-level lifecycle per environment tick:
            state  --->  step(...)  --->  Action

        The "state" object type depends on the task:
        - Task 1: A fully structured GridState instance.
        - Task 2: An ImageObservation dictionary whose primary observation is an RGBA image
          plus limited structured metadata in the 'info' sub-dict. In this case you
          typically perform perception to build (or approximate) an internal
          structured representation before planning.
        - Task 3: Input state could be either a GridState instance 
          or an ImageObservation dictionary

        Constraints:
        - Keep per-step latency small (single CPU, 1.5 GB RAM). Avoid O(W*H) scans of
          the full grid every step.
        - Determinism helps reproducibility; seed your own RNG if you add any
          random components.

        You may add __init__ parameters (with defaults) if needed for your own
        development, but the grader will instantiate Agent() with no arguments.
        """

    def __init__(self):
        """Initialize your agent.

            Put all one-time setup here (e.g., hardcoded ML model weights, 
            precomputing heuristic tables). Keep it fast and memory-light 
            to respect platform limits.
            """
        pass

    def step(self, state: GridState | ImageObservation) -> Action:
        """Return the next action given the current environment state.

            Parameters
            ----------
            state : GridState | ImageObservation
                - If a GridState instance (Tasks 1 and 3): you have direct, structured
                  access to grid, entities, objective message, score, etc.
                - If an ImageObservation dictionary (Tasks 2 and 3): contains 'image' (HxWx4 RGBA uint8)
                  plus 'info' sub-dictionary (agent stats, partial config, message).
                  You likely need to parse the image into an internal representation.

            Returns
            -------
            Action  # Placeholder for any future initialization logic
                A valid action from the Action enum. Must always return a member;
                never return None.
            """
        if not isinstance(state, GridState):
            state = self.parse(state)
        frontier = deque([(state, [])])
        visited = {str(state.grid)}
        while frontier:
            current_state, path = frontier.popleft()
            for action in Action:
                next_state = grid_step(current_state, action)
                if next_state.lose:
                    continue
                next_path = path + [action]
                if next_state.win:
                    return next_path[0]
                key = str(next_state.grid)
                if key in visited:
                    continue
                visited.add(key)
                frontier.append((next_state, next_path))
        return Action.WAIT

    def parse(self, observation: ImageObservation) -> GridState:
        """Convert an image observation into a GridState.

            Grid Play also calls this method to display the agent's reconstructed view.
            """
        info = observation['info']
        config = info['config']
        status = info['status']
        width = config['width']
        height = config['height']
        gridstate = GridState(width=width, height=height, movement=MOVEMENTS['cardinal'], objective=OBJECTIVES['collect_gems_and_exit'], step_cost=STEP_COST, seed=None if config['seed'] == -1 else config['seed'], turn=status['turn'], score=status['score'], win=status['phase'] == 'win', lose=status['phase'] == 'lose', message=info['message'] or None, turn_limit=None if config['turn_limit'] == -1 else config['turn_limit'])
        agent_count = 0
        for y, row in enumerate(self.split_cells(observation)):
            for x, cell_image in enumerate(row):
                entity = self.predict(cell_image)
                if entity is not None:
                    if isinstance(entity, AgentEntity):
                        agent_count += 1
                        entity.set_health(info['agent']['health']['current_health'])
                    gridstate.add((x, y), entity)
        if agent_count != 1:
            raise ValueError(f'predict() must identify exactly one AgentEntity, but found {agent_count}.')
        return gridstate

    def split_cells(self, observation: ImageObservation) -> list[list[np.ndarray]]:
        """Split an RGBA observation into rows of cell images."""
        _image = observation['image']
        config = observation['info']['config']
        width = config['width']
        height = config['height']
        image_height, image_width = _image.shape[:2]
        cells = []
        for y in range(height):
            y_start = y * image_height // height
            y_end = (y + 1) * image_height // height
            row = []
            for x in range(width):
                x_start = x * image_width // width
                x_end = (x + 1) * image_width // width
                row.append(_image[y_start:y_end, x_start:x_end, :])
            cells.append(row)
        return cells

    def predict(self, cell_image: np.ndarray) -> BaseEntity | None:
        """Return the entity most similar to one RGBA cell image, or None for floor."""
        entity_types = {'boots': SpeedPowerUpEntity, 'box': BoxEntity, 'coin': CoinEntity, 'exit': ExitEntity, 'gem': GemEntity, 'ghost': PhasingPowerUpEntity, 'human': AgentEntity, 'key': KeyEntity, 'lava': LavaEntity, 'locked': LockedDoorEntity, 'opened': UnlockedDoorEntity, 'shield': ShieldPowerUpEntity, 'wall': WallEntity}
        image_size = (cell_image.shape[1], cell_image.shape[0])
        asset_root = Path(DEFAULT_ASSET_ROOT)
        with Image.open(asset_root / 'floor' / 'floor_1.png') as _image:
            floor = _image.convert('RGBA').resize(image_size)
        observed = cell_image.astype(np.int16)
        best_entity = None
        best_error = np.abs(observed - np.asarray(floor)).mean()
        for asset_name, entity_type in entity_types.items():
            asset_path = asset_root / asset_name / f'{asset_name}_1.png'
            with Image.open(asset_path) as _image:
                asset = _image.convert('RGBA').resize(image_size)
            candidate = np.asarray(Image.alpha_composite(floor, asset))
            error = np.abs(observed - candidate).mean()
            if error < best_error:
                best_error = error
                best_entity = entity_type
        return None if best_entity is None else best_entity()

    def info(self) -> dict[str, str]:
        """Return info about the agent.

            NOTE: This method is optional and intended for debugging in Grid Play only. 
            You do not need to implement it for Coursemology submission, but it can be 
            helpful for visualizing your agent's internal state during development. 
            Implementing this method will not affect grading.
            """
        return {'name': 'BFS Agent'}