import unittest
from time import perf_counter

from grid_adventure.entities import GemEntity
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents.vision_v3_bridge import reconstruct
from agents.vision_v3_planner import PlannerAdapter
from agents.vision_v3_types import EpisodeBudget
from tests.vision_v3_fixtures import image_env
from tests.planner_p1_fixtures import corridor
from tools.benchmark_vision_v3 import evaluate_episode, oracle_labels


class HarnessTests(unittest.TestCase):
    def test_corpus_renderer_checks_decoded_pixels_across_png_encoders(self):
        import hashlib
        import json
        from pathlib import Path

        import numpy as np
        from PIL import Image

        from tools.benchmark_vision_v3 import recorded_appearance
        from tools.vision_v3_data import new_layout

        options = json.loads(
            Path("artifacts/v3/validation/recorded-jobs.json").read_text()
        )[0]
        record = json.loads(
            Path("artifacts/v3/data/known-layout-unseen-art-v2/boards.jsonl")
            .read_text()
            .splitlines()[0]
        )
        with Image.open(
            Path("artifacts/v3/data/known-layout-unseen-art-v2") / record["image"]
        ) as image:
            options["expected_pixels_sha256"] = hashlib.sha256(
                np.asarray(image).tobytes()
            ).hexdigest()
        options["expected_image_sha256"] = "different-PNG-compression"
        state = new_layout(options["generated"], split=options["layout_split"])[0]
        observation, _ = recorded_appearance(image_env(state), options).reset()
        self.assertEqual(
            hashlib.sha256(observation["image"].tobytes()).hexdigest(),
            options["expected_pixels_sha256"],
        )

    def test_selected_p2_next_backend_supports_one_key_many_items(self):
        from grid_adventure.entities import CoinEntity, KeyEntity, LockedDoorEntity

        from tests.planner_p1_fixtures import layout

        state = layout(width=10)
        state.add((1, 1), KeyEntity())
        for x in (2, 3, 4, 5):
            state.add((x, 1), CoinEntity())
        state.add((6, 1), GemEntity())
        state.add((7, 1), LockedDoorEntity())
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 1, now + 2))
        self.assertIsInstance(planner.step(state), Action)
        self.assertIsNone(planner._reference)
        self.assertEqual(planner.info()["search_calls"], "1")

    def test_reference_suffix_survives_public_resource_reidentification(self):
        from grid_adventure.entities import KeyEntity, LockedDoorEntity
        from grid_adventure.grid import from_state

        from tests.planner_p1_fixtures import layout

        state = layout(width=8)
        state.add((0, 1), KeyEntity(entity_id=1005))
        for x in range(1, 5):
            state.add((x, 0), KeyEntity(entity_id=1005 + x * 37))
        state.add((1, 1), LockedDoorEntity())
        from grid_adventure.entities import WallEntity

        for x in range(8):
            state.add((x, 0), WallEntity())
            state.add((x, 2), WallEntity())
        env = image_env(state)
        obs, _ = env.reset()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 1, now + 2))
        predicted = None
        for index in range(3):
            hyp = reconstruct(
                oracle_labels(from_state(env.state)), obs["info"], predicted
            )
            action = planner.step(hyp.state)
            if index == 0:
                self.assertEqual(action, Action.PICK_UP)
            elif index == 1:
                self.assertEqual(action, Action.USE_KEY)
            predicted = step(hyp.state, action)
            obs, _, _, _, _ = env.step(action)
        assert planner._reference is not None
        self.assertEqual(planner._reference.info()["search_calls"], "1")
        self.assertGreaterEqual(
            int(planner.info().get("reference_suffix_rebindings", "0")), 1
        )

    def test_public_pickup_reidentification_retains_verified_suffix(self):
        from grid_adventure.grid import from_state

        state = corridor()
        state.add((1, 0), GemEntity(entity_id=1005))
        env = image_env(state)
        obs, _ = env.reset()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 1, now + 2))
        predicted = None
        actions = []
        for _ in range(3):
            hyp = reconstruct(
                oracle_labels(from_state(env.state)), obs["info"], predicted
            )
            action = planner.step(hyp.state)
            actions.append(action)
            predicted = step(hyp.state, action)
            obs, _, win, _, _ = env.step(action)
        self.assertTrue(win)
        self.assertEqual(actions, [Action.RIGHT, Action.PICK_UP, Action.RIGHT])
        # A purely observable identity change must not consume another search.
        self.assertEqual(planner.info().get("suffix_rebindings"), "1")
        self.assertEqual(planner.info()["search_calls"], "1")

    def test_planner_never_resets_original_deadline(self):
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 1, now + 2))
        state = corridor()
        self.assertEqual(planner.step(state), Action.RIGHT)
        expired = PlannerAdapter(EpisodeBudget(now - 10, now - 7, now - 6.5))
        self.assertEqual(expired.step(state), Action.WAIT)
        planner.budget = EpisodeBudget(now - 10, now - 7, now - 6.5)
        # Cached suffix must also respect expiry; no inner restart.
        self.assertEqual(planner.step(step(state, Action.RIGHT)), Action.WAIT)

    def test_bad_parser_is_scored_against_real_environment(self):
        state = corridor()
        state.add((1, 0), GemEntity())

        class BadParser:
            def step(self, observation):
                # Exit-only imagined board wins, true board retains a gem.
                return Action.RIGHT

        imagined = step(step(corridor(), Action.RIGHT), Action.RIGHT)
        self.assertTrue(imagined.win)
        result = evaluate_episode(image_env(state), BadParser(), observation="image")
        self.assertFalse(result["win"])
        self.assertEqual(result["effective_performance"], 0.0)
        self.assertEqual(result["status"], "loss")

    def test_image_mode_receives_only_image_and_metadata(self):
        class ImageOnly:
            def step(self, observation):
                if set(observation) != {"image", "info"}:
                    raise AssertionError("truth leaked into image interface")
                return Action.RIGHT

        result = evaluate_episode(
            image_env(corridor()), ImageOnly(), observation="image"
        )
        self.assertTrue(result["win"])
