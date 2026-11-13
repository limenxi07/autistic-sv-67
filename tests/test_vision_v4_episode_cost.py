import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from agents import vision_v4_backend as backend
from agents import vision_v4_perception as perception
from agents.vision_v4_planner import PlannerAdapter
from agents.vision_v4_types import EpisodeBudget


class RawPixelCacheTests(unittest.TestCase):
    def setUp(self):
        class Model:
            numpy_inference = True

            def __init__(self):
                self.samples = 0

            def eval(self):
                return self

            def __call__(self, crops):
                self.samples += len(crops)
                logits = np.zeros((len(crops), 14), dtype=np.float32)
                logits[:, 0] = crops.mean(axis=(1, 2, 3))
                return logits

        self.model = Model()
        perception.configure(lambda: self.model)
        self.obs: dict[str, Any] = {
            "image": np.zeros((24, 48, 4), dtype=np.uint8),
            "info": {"config": {"width": 2, "height": 1}},
        }
        self.obs["image"][:, :, 3] = 255

    def test_identical_pixels_reuse_inference_before_preprocessing(self):
        from PIL import Image

        first = perception.classify(self.obs)
        calls = self.model.samples
        with patch.object(Image, "fromarray", wraps=Image.fromarray) as convert:
            second = perception.classify(self.obs)
        np.testing.assert_array_equal(first.probabilities, second.probabilities)
        self.assertEqual(self.model.samples, calls)
        self.assertEqual(convert.call_count, 0)

    def test_changed_pixels_invalidate_and_returned_probabilities_are_isolated(self):
        first = perception.classify(self.obs)
        first.probabilities[:] = 99
        self.obs["image"][:, :24, :3] = 255
        second = perception.classify(self.obs)
        self.assertGreater(second.probabilities[0, 0, 0], second.probabilities[0, 1, 0])
        np.testing.assert_allclose(
            second.probabilities[0, 1], np.full(14, 1 / 14), rtol=1e-6
        )

    def test_configure_clears_cache_and_partial_selection_stays_nan(self):
        perception.classify(self.obs)
        calls = self.model.samples
        perception.configure(lambda: self.model, temperature=2.0)
        partial = perception.classify(self.obs, ((1, 0),))
        self.assertGreater(self.model.samples, calls)
        self.assertTrue(np.isnan(partial.probabilities[0, 0]).all())
        self.assertTrue(np.isfinite(partial.probabilities[0, 1]).all())

    def test_pixel_cache_has_byte_limit_for_large_cells(self):
        self.obs["info"]["config"] = {"width": 1, "height": 1}
        for shade in range(12):
            self.obs["image"] = np.full((512, 512, 4), shade, dtype=np.uint8)
            perception.classify(self.obs)
        retained = sum(
            len(key[2]) for key in perception._perception_runtime.pixel_cache
        )
        self.assertLessEqual(retained, 8 * 1024 * 1024)


class ReplayConversionCostTests(unittest.TestCase):
    def test_disagreeing_key_proposal_falls_back_without_dispatching_it(self):
        from time import perf_counter

        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 3, now + 3.5))
        # The compact inventory model is speculative. Reject its replay while
        # keeping the actual reference search and its separate replay intact.
        with patch.object(
            planner,
            "_verify_cached",
            return_value=backend.Verification("invalid", None, ()),
        ):
            action = planner.step(state)
        self.assertNotEqual(action, backend.Action.WAIT)
        self.assertFalse(planner._verified_edges)
        self.assertIsNotNone(planner._reference)
        for action in planner._reference._actions:
            state = backend.grid_step(state, action)
        self.assertTrue(state.win)
        self.assertEqual(state.score, -387)

    def test_long_key_route_needs_only_one_public_winning_replay(self):
        from time import perf_counter

        from agents import vision_v4_planner
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 3, now + 3.5))
        with (
            patch.object(
                vision_v4_planner,
                "v4_reference_transition",
                wraps=vision_v4_planner.v4_reference_transition,
            ) as replay,
            patch(
                "agents.vision_v4_search.v4_reference_transition",
                wraps=vision_v4_planner.v4_reference_transition,
            ) as search,
        ):
            action = planner.step(state)
        self.assertNotEqual(action, backend.Action.WAIT)
        active = planner._reference or planner
        self.assertEqual(len(active._actions), 130)
        for action in active._actions:
            state = backend.grid_step(state, action)
        self.assertTrue(state.win)
        self.assertEqual(state.score, -387)
        self.assertLessEqual(replay.call_count + search.call_count, 150)

    def test_static_state_keys_avoid_recursive_serialization_without_losing_fields(
        self,
    ):
        from types import SimpleNamespace

        from grid_adventure.entities import WallEntity

        from agents import vision_v4_search
        from tests.planner_p1_fixtures import layout

        state = layout(width=4)
        wall: Any = WallEntity()
        state.add((1, 0), wall)
        key_of = getattr(vision_v4_search, "v4_state_key", backend.reference_state_key)
        with (
            patch.object(
                backend, "reference__entity_key", wraps=backend.reference__entity_key
            ) as recursive,
            patch.object(vision_v4_search, "reference__entity_key", recursive),
        ):
            actual = key_of(state)
        self.assertEqual(actual, backend.reference_state_key(state))
        self.assertLess(recursive.call_count, 3)
        wall.blocking = SimpleNamespace(extra=42)
        self.assertEqual(key_of(state), backend.reference_state_key(state))
        wall.unexpected = [1, {"x": 2}]
        self.assertEqual(key_of(state), backend.reference_state_key(state))

    def test_mutated_shared_wall_rejects_cached_transition(self):
        from time import perf_counter

        from grid_adventure.entities import WallEntity

        from agents import vision_v4_planner
        from tests.planner_p1_fixtures import layout

        state = layout(width=4)
        wall: Any = WallEntity()
        state.add((1, 0), wall)
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 3, now + 3.5))
        action = planner.step(state)
        cached = planner.predict(state, action)
        self.assertIs(cached.grid[1][0][0], wall)
        wall.blocking = None
        with patch.object(
            vision_v4_planner, "grid_step", wraps=backend.grid_step
        ) as simulate:
            actual = planner.predict(state, action)
        self.assertEqual(simulate.call_count, 1)
        self.assertIsNot(actual, cached)
        self.assertEqual(
            backend.reference_state_key(actual),
            backend.reference_state_key(backend.grid_step(state, action)),
        )

    def test_cached_dispatch_reads_current_state_once(self):
        from time import perf_counter

        from agents import vision_v4_planner
        from tests.planner_p1_fixtures import corridor

        state = corridor()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 3, now + 3.5))
        action = planner.step(state)
        state = planner.predict(state, action)
        with patch.object(
            vision_v4_planner, "v4_observe_key", wraps=vision_v4_planner.v4_observe_key
        ) as observe:
            action = planner.step(state)
        self.assertEqual(observe.call_count, 1)
        self.assertTrue(backend.grid_step(state, action).win)

    def test_winning_replay_avoids_reconstructing_unchanged_scenery(self):
        from time import perf_counter

        from grid_universe.grid import convert

        from agents import vision_v4_search
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B3 So Close Yet So Far"]()
        before = backend.reference_state_key(state)
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 10, now + 10))
        original = convert._entity_object_from_state
        calls = []

        def count(native, eid):
            calls.append(eid)
            return original(native, eid)

        with (
            patch.object(convert, "_entity_object_from_state", count),
            patch.object(vision_v4_search, "v4_entity_from_state", count),
        ):
            planner.step(state)
        self.assertTrue(planner._actions)
        self.assertLess(len(calls), len(planner._actions) * 15)
        self.assertEqual(backend.reference_state_key(state), before)
        current = state
        for action in planner._actions:
            current = backend.grid_step(current, action)
        self.assertTrue(current.win)
        self.assertEqual(current.score, -315)
