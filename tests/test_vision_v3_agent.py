import unittest
from unittest.mock import patch

from grid_adventure.entities import WallEntity
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents.vision_v3 import Agent
from tests.vision_v3_fixtures import vision_observation
from tests.planner_p1_fixtures import corridor, layout


class AgentTests(unittest.TestCase):
    def test_observed_actor_template_recognizes_human_covering_boots(self):
        import inspect
        from pathlib import Path

        from grid_adventure.env import GridAdventureEnv
        from grid_adventure.grid import to_state

        from agents.vision_v3_bridge import CLASSES
        from tools.vision_v3_data import new_layout

        state = new_layout(38, split="validation")[0]
        state.seed = 2030090603
        mapping = (
            inspect.signature(GridAdventureEnv).parameters["render_image_map"].default
        )
        env = GridAdventureEnv(
            initial_state_fn=lambda *a, **k: to_state(state),
            width=state.width,
            height=state.height,
            observation_type="image",
            render_asset_root=str(
                Path("artifacts/v3/data/train-boards-600-v2/asset-root")
            ),
            render_image_map=type(mapping)(
                {k: v for k, v in mapping.items() if v in CLASSES}
            ),
        )
        observation, _ = env.reset()
        agent = Agent()
        first = agent.step(observation)
        self.assertEqual(first, Action.LEFT)
        observation, *_ = env.step(first)
        self.assertEqual(agent.step(observation), Action.PICK_UP)

    def test_parse_later_frame_uses_history_without_mutating_execution(self):
        from grid_adventure.entities import GemEntity

        state = corridor()
        state.add((1, 0), GemEntity())
        agent = Agent()
        action = agent.step(vision_observation(state))
        state = step(state, action)
        before = agent.info()
        parsed = agent.parse(vision_observation(state))
        self.assertTrue(any(isinstance(e, GemEntity) for e in parsed.grid[1][0]))
        self.assertEqual(parsed.turn, 1)
        self.assertEqual(agent.info(), before)

    def test_unsupported_metadata_returns_legal_action_and_uncertainty(self):
        from agents.vision_v3 import PerceptionUncertainty

        observation = vision_observation(corridor())
        observation["info"]["config"]["movement"] = "unsupported"
        agent = Agent()
        self.assertEqual(agent.step(observation), Action.WAIT)
        self.assertTrue(agent.info()["reason"].startswith("unsupported_observation"))
        with self.assertRaises(PerceptionUncertainty):
            agent.parse(observation)

    def test_unconfirmed_pickup_keeps_hidden_gem_and_retries(self):
        from grid_adventure.entities import GemEntity

        state = corridor()
        state.add((1, 0), GemEntity())
        agent = Agent()
        self.assertEqual(agent.step(vision_observation(state)), Action.RIGHT)
        state = step(state, Action.RIGHT)
        self.assertEqual(agent.step(vision_observation(state)), Action.PICK_UP)
        # A missed action has the same cost/turn as picking up this gem. The
        # independent inventory observation proves the pickup did not register.
        state = step(state, Action.WAIT)
        retry = agent.step(vision_observation(state))
        self.assertEqual(retry, Action.PICK_UP)
        self.assertEqual(agent.info()["recoveries"], "0")
        state = step(state, retry)
        finish = agent.step(vision_observation(state))
        self.assertEqual(agent.info()["recoveries"], "1")
        state = step(state, finish)
        self.assertTrue(state.win)

    def test_counter_agreement_cannot_dispatch_stale_right_action(self):
        state = layout(width=3)
        state.add((1, 0), WallEntity())
        state.add((2, 0), WallEntity())
        agent = Agent()
        first = agent.step(vision_observation(state))
        self.assertEqual(first, Action.RIGHT)
        # Unexpected real motion: same score, turn and health as planned RIGHT.
        actual = step(state, Action.UP)
        second = agent.step(vision_observation(actual))
        self.assertEqual(second, Action.DOWN)
        self.assertGreaterEqual(int(agent.info()["visual_discrepancies"]), 1)

    def test_parse_does_not_advance_execution_tracker(self):
        observation = vision_observation(corridor())
        agent = Agent()
        before = agent.info().copy()
        parsed = agent.parse(observation)
        self.assertEqual((parsed.turn, parsed.score), (0, 0))
        self.assertEqual(agent.info(), before)
        self.assertEqual(agent.step(observation), Action.RIGHT)

    def test_structured_dispatch_uses_p2_next_without_perception(self):
        agent = Agent()
        with patch(
            "agents.vision_v3.classify_initial",
            side_effect=AssertionError("vision on structured input"),
        ):
            self.assertEqual(agent.step(corridor()), Action.RIGHT)
        self.assertEqual(agent.info()["input_mode"], "gridstate")
