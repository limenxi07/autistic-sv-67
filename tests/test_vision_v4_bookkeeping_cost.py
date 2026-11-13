"""Cost and pixel invariants for emission bookkeeping (no model shortcuts)."""

import unittest
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from grid_adventure.entities import AgentEntity, BoxEntity
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents import vision_v4
from agents import vision_v4_tracking as tracking
from agents.vision_v4 import Agent
from agents.vision_v4_bridge import split_cells
from agents.vision_v4_types import BoardHypothesis, EpisodeBudget
from tests.planner_p1_fixtures import corridor, layout
from tests.vision_v3_fixtures import vision_observation


class BookkeepingTests(unittest.TestCase):
    def emit(self, state, image, old):
        agent = Agent()
        agent._background_image = old
        agent._emission_observation = {
            "image": image,
            "info": {"config": {"width": state.width, "height": state.height}},
        }
        agent._emit_action(
            Action.WAIT,
            turn=state.turn,
            hypotheses=(BoardHypothesis(state, 1.0, ()),),
            reason="conditional_plan",
            verified_successors=(state,),
        )
        return agent

    def test_sparse_background_refresh_does_not_build_all_tile_views(self):
        state = layout(width=15, height=15)
        state.add((5, 4), BoxEntity())
        image = np.arange(47 * 49 * 4, dtype=np.uint8).reshape(47, 49, 4)
        old = np.full_like(image, 213)
        expected = image.copy()
        # Known occupied cells; floor/exit pixels must come from this frame.
        for x, y in ((0, 1), (5, 4)):
            expected[
                y * 47 // 15 : (y + 1) * 47 // 15, x * 49 // 15 : (x + 1) * 49 // 15
            ] = 213
        with patch.object(vision_v4, "split_cells", wraps=split_cells) as split:
            agent = self.emit(state, image, old)
        self.assertTrue(agent._last_emission.prediction_valid)
        np.testing.assert_array_equal(agent._background_image, expected)
        np.testing.assert_array_equal(agent._previous_image, image)
        self.assertFalse(np.shares_memory(agent._background_image, image))
        self.assertFalse(np.shares_memory(agent._background_image, old))
        self.assertEqual(split.call_count, 0)

    def test_malformed_cell_still_drops_emission_prediction(self):
        state = corridor()
        state.grid[1][0] = None
        agent = self.emit(
            state, np.zeros((4, 12, 4), np.uint8), np.ones((4, 12, 4), np.uint8)
        )
        self.assertFalse(agent._last_emission.prediction_valid)
        self.assertEqual(agent._last_emission.reason, "emission_failure:TypeError")

    def test_different_background_dtype_retains_assignment_conversion(self):
        state = corridor()
        image = np.full((3, 11, 4), 300, dtype=np.int16)
        old = np.full((3, 11, 4), 27, dtype=np.uint8)
        agent = self.emit(state, image, old)
        expected = np.full_like(old, 44)
        expected[:, :3] = 27
        np.testing.assert_array_equal(agent._background_image, expected)
        self.assertEqual(agent._background_image.dtype, old.dtype)

    def test_invalid_geometry_still_drops_emission_prediction(self):
        state = corridor()
        agent = self.emit(
            state, np.zeros((1, 2, 4), np.uint8), np.ones((1, 2, 4), np.uint8)
        )
        self.assertFalse(agent._last_emission.prediction_valid)
        self.assertEqual(agent._last_emission.reason, "emission_failure:ValueError")

    def test_long_public_episode_background_matches_original_pixel_updates(self):
        from agents.vision_v4_planner import PlannerAdapter
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
        from tests.vision_v3_fixtures import image_env

        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 10, now + 10))
        planner.step(state)
        actions = (planner._reference or planner)._actions
        self.assertEqual(len(actions), 130)
        env = image_env(state)
        observation, _ = env.reset()
        agent = Agent()
        agent._budget = EpisodeBudget(now, now + 30, now + 30)
        expected = None
        for action in actions:
            # Execute the old full-grid update independently as the pixel oracle.
            if expected is None:
                expected = observation["image"].copy()
            else:
                rows = split_cells(observation)
                tiles = split_cells({**observation, "image": expected})
                for x in range(state.width):
                    for y in range(state.height):
                        names = [e.appearance.name for e in state.grid[x][y]]
                        if "human" not in names and "box" not in names:
                            tiles[y][x][:] = rows[y][x]
            successor = step(state, action)
            agent._emission_observation = observation
            agent._emit_action(
                action,
                turn=state.turn,
                hypotheses=(BoardHypothesis(state, 1.0, ()),),
                reason="conditional_plan",
                verified_successors=(successor,),
            )
            self.assertTrue(agent._last_emission.prediction_valid)
            np.testing.assert_array_equal(agent._background_image, expected)
            observation, _, win, lose, _ = env.step(action)
            self.assertFalse(lose)
            state = successor
        self.assertTrue(win)
        self.assertTrue(state.win)
        self.assertEqual(state.score, -387)

    def test_sparse_agent_scan_avoids_per_empty_cell_any(self):
        state = layout(width=15, height=15)
        with patch.object(tracking, "any", wraps=any, create=True) as scan:
            self.assertEqual(tracking.agent_position(state), (0, 1))
        self.assertLessEqual(scan.call_count, 2)

    def test_multiple_agents_in_one_cell_are_one_position(self):
        state = corridor()
        state.add((0, 0), AgentEntity())
        self.assertEqual(tracking.agent_position(state), (0, 0))
        state.add((1, 0), AgentEntity())
        self.assertIsNone(tracking.agent_position(state))
        self.assertIsNone(tracking.agent_position(SimpleNamespace(grid=[[[]]])))

    def test_visible_priority_and_unknown_entities(self):
        def e(name):
            return SimpleNamespace(appearance=SimpleNamespace(name=name))

        for cell, expected in (
            ([], "floor"),
            ([e("unknown")], "floor"),
            ([e("exit")], "exit"),
            ([e("box"), e("human")], "human"),
            ([e("unknown"), e("gem")], "gem"),
            ([e("door")], "opened"),
        ):
            self.assertEqual(tracking.visible_kind(cell), expected)
        with self.assertRaises(AttributeError):
            tracking.visible_kind([e("human"), object()])

    def test_budget_crossed_during_check_reports_cutoff(self):
        state = corridor()
        agent = Agent()
        state = step(state, agent.step(vision_observation(state)))
        now = agent._budget.started_at
        original = vision_v4.check_observation

        def slow_check(*args, **kwargs):
            nonlocal now
            result = original(*args, **kwargs)
            now = agent._budget.cutoff
            return result

        with (
            patch.object(vision_v4, "perf_counter", side_effect=lambda: now),
            patch.object(tracking, "perf_counter", side_effect=lambda: now),
            patch.object(vision_v4, "check_observation", side_effect=slow_check),
        ):
            self.assertEqual(agent.step(vision_observation(state)), Action.WAIT)
        self.assertEqual(agent.info()["reason"], "cutoff")
        self.assertFalse(agent._last_emission.prediction_valid)


class ActorTemplateCostTests(unittest.TestCase):
    def test_duplicate_floor_pixels_are_converted_once_with_exact_mask(self):
        from PIL import Image

        from agents import vision_v4_perception as perception

        # The border is identical; a central bright square is the entire actor.
        image = np.zeros((48, 48 * 20, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        image[10:38, 10:38, :3] = 255
        # Distinct floor tile must still be evaluated, even after duplicate tiles.
        image[:, 48 * 19 :, :3] = 20
        observation = {"image": image, "info": {"config": {"width": 20, "height": 1}}}
        with patch.object(Image, "fromarray", wraps=Image.fromarray) as convert:
            actor, mask = perception.make_actor_template(
                observation, (0, 0), [(x, 0) for x in range(1, 20)]
            )
        expected = np.zeros((48, 48), dtype=bool)
        expected[11:37, 11:37] = True
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(actor, image[:, :48, :3].astype(np.int16))
        self.assertEqual(convert.call_count, 3)


class SelectedTileCostTests(unittest.TestCase):
    def test_partial_classification_reads_only_selected_tiles_with_exact_scores(self):
        from agents import vision_v4_perception as perception
        from tests.vision_v4_fixtures import configure_vision

        configure_vision()
        rng = np.random.default_rng(67)
        image = rng.integers(0, 256, (47, 49, 4), dtype=np.uint8)
        observation = {"image": image, "info": {"config": {"width": 15, "height": 15}}}
        from PIL import Image

        selected = ((0, 0), (14, 14), (7, 8))
        rows = split_cells(observation)
        crops = np.stack(
            [
                np.asarray(
                    Image.fromarray(rows[y][x])
                    .convert("RGB")
                    .resize((48, 48), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ).transpose(2, 0, 1)
                / 255
                for x, y in selected
            ]
        )
        logits = (
            perception._get_model()(crops) / perception._perception_runtime.temperature
        )
        expected = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected /= expected.sum(axis=1, keepdims=True)
        with patch.object(perception, "split_cells", wraps=split_cells) as split:
            actual = perception.classify(observation, selected)
        for i, (x, y) in enumerate(selected):
            np.testing.assert_array_equal(actual.probabilities[y, x], expected[i])
        self.assertEqual(np.isfinite(actual.probabilities).sum(), len(selected) * 14)
        self.assertEqual(split.call_count, 0)

    def test_cell_access_preserves_geometry_and_index_validation(self):
        from agents.vision_v4_perception import _cell_access

        for shape, width, height in (
            ((7, 11, 4), 3, 2),
            ((7, 11, 4), 0, 2),
            ((7, 11, 4), 12, 2),
            ((7, 11), 3, 2),
            ((7, 11, 4), 3.0, 2),
            ((7, 11, 4), True, 2),
            ((7, 11, 4), np.int64(3), 2),
        ):
            observation = {
                "image": np.zeros(shape, np.uint8),
                "info": {"config": {"width": width, "height": height}},
            }
            try:
                rows = split_cells(observation)
            except (ValueError, TypeError, IndexError) as error:
                with self.assertRaises(type(error)):
                    _cell_access(observation)
                continue
            actual_width, actual_height, tile = _cell_access(observation)
            self.assertEqual((actual_width, actual_height), (len(rows[0]), len(rows)))
            for x, y in ((0, 0), (-1, -1), (np.int64(0), 0), (99, 0), (0.5, 0)):
                try:
                    expected = rows[y][x]
                except (IndexError, TypeError) as error:
                    with self.assertRaises(type(error)):
                        tile(x, y)
                else:
                    np.testing.assert_array_equal(tile(x, y), expected)

    def test_all_visual_views_match_legacy_splitter_scores(self):
        from agents import vision_v4_perception as perception
        from tests.vision_v4_fixtures import configure_vision

        rng = np.random.default_rng(68)
        image = rng.integers(0, 256, (53, 71, 4), dtype=np.uint8)
        observation = {"image": image, "info": {"config": {"width": 5, "height": 3}}}
        before = image.copy()
        before[20:25, 25:35, :3] = 30
        selected = ((1, 1), (4, 2))

        def legacy_access(frame):
            rows = split_cells(frame)
            return len(rows[0]), len(rows), lambda x, y: rows[y][x]

        for classify in (
            lambda: perception.classify(observation),
            lambda: perception.classify_subcrops(observation, selected),
            lambda: perception.classify_foregrounds(observation, before, selected),
        ):
            configure_vision()
            with patch.object(perception, "_cell_access", side_effect=legacy_access):
                expected = classify()
            configure_vision()
            with patch.object(perception, "split_cells", wraps=split_cells) as split:
                actual = classify()
            np.testing.assert_array_equal(actual.probabilities, expected.probabilities)
            self.assertEqual(split.call_count, 0)


class RemainingScanCostTests(unittest.TestCase):
    def test_observation_check_scans_actor_position_once_and_no_empty_pushable_any(
        self,
    ):
        from tests.vision_v4_fixtures import configure_vision

        configure_vision()
        state = layout(width=15, height=15)
        observation = vision_observation(state)
        with (
            patch.object(
                tracking, "agent_position", wraps=tracking.agent_position
            ) as positions,
            patch.object(tracking, "any", wraps=any, create=True) as scans,
        ):
            check = tracking.check_observation(observation, state, observation["image"])
        self.assertTrue(check.consistent, check.mismatches)
        self.assertEqual(positions.call_count, 1)
        self.assertLess(scans.call_count, 10)

    def test_template_location_reads_selected_tiles_only(self):
        from agents import vision_v4_perception as perception

        image = np.zeros((48, 48 * 3, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        image[10:38, 10:38, :3] = 255
        observation = {"image": image, "info": {"config": {"width": 3, "height": 1}}}
        actor = image[:, :48, :3].astype(np.int16)
        mask = np.zeros((48, 48), bool)
        mask[11:37, 11:37] = True
        with patch.object(perception, "split_cells", wraps=split_cells) as split:
            actual = perception.locate_actor_template(
                observation, ((0, 0), (2, 0)), (actor, mask)
            )
        self.assertEqual(actual, (0, 0))
        self.assertEqual(split.call_count, 0)
