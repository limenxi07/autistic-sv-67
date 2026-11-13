import unittest
from copy import deepcopy
from time import perf_counter
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from grid_adventure.entities import CoinEntity, GemEntity, SpeedPowerUpEntity
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents.vision_v3_bridge import CLASSES, reconstruct
from agents.vision_v3_tracking import (
    check_observation,
    common_safe_prefix,
    initial_hypotheses,
    reconcile,
)
from agents.vision_v3_types import (
    BoardHypothesis,
    CellScores,
    EpisodeBudget,
    ObservationCheck,
)
from tests.vision_v3_fixtures import configure_vision, vision_observation
from tests.planner_p1_fixtures import corridor, mechanic
from tests.test_planner_p1_state import player
from tools.benchmark_vision_v3 import oracle_labels


class TrackingTests(unittest.TestCase):
    def test_confident_agreement_prunes_disproved_alternative_and_finishes(self):
        state = corridor()
        observation = vision_observation(state)
        scores = np.zeros((1, 3, 14), np.float32)
        for x, kind in enumerate(("human", "floor", "exit")):
            scores[0, x, CLASSES.index(kind)] = 1
        scores[0, 1, CLASSES.index("floor")] = 0.51
        scores[0, 1, CLASSES.index("gem")] = 0.49
        hypotheses = initial_hypotheses(CellScores(scores, 3, 1), observation["info"])
        self.assertEqual(len(hypotheses), 2)
        check = check_observation(observation, hypotheses[0].state, None)
        self.assertTrue(check.consistent)
        now = perf_counter()
        budget = EpisodeBudget(now, now + 1, now + 2)
        repaired = reconcile(observation, hypotheses, check, budget)
        self.assertEqual(len(repaired), 1)
        self.assertFalse(repaired[0].unresolved)
        prefix = common_safe_prefix(repaired, (Action.RIGHT, Action.RIGHT), budget)
        self.assertEqual(prefix, (Action.RIGHT, Action.RIGHT))
        for action in prefix:
            state = step(state, action)
        self.assertTrue(state.win)

    def test_clean_background_rejects_vacated_lava_silhouette(self):
        from grid_adventure.entities import LavaEntity

        from tests.planner_p1_fixtures import layout

        state = layout(width=5)
        state.add((1, 1), LavaEntity())
        state.add((2, 1), LavaEntity())
        background = vision_observation(state)["image"]
        middle = step(state, Action.RIGHT)
        after = step(middle, Action.RIGHT)
        check = check_observation(
            vision_observation(after),
            after,
            vision_observation(middle)["image"],
            background_image=background,
        )
        self.assertTrue(check.consistent, check)

    def test_reordered_consumed_coins_do_not_invalidate_plan(self):
        state = corridor()
        actor = player(state)
        actor.inventory_list = [CoinEntity(entity_id=(1 << 40) + i) for i in range(10)]
        for item in actor.inventory_list:
            cast(Any, item).collectible = None
        observation = vision_observation(state)
        observation["info"]["agent"]["inventory"] = tuple(
            reversed(observation["info"]["agent"]["inventory"])
        )
        self.assertTrue(check_observation(observation, state, None).consistent)

    def test_pixel_changes_map_to_fractional_cells(self):
        from agents.vision_v3_tracking import changed_cells

        before = np.zeros((11, 17, 4), dtype=np.uint8)
        after = before.copy()
        after[3, 5, 0] = 1
        after[10, 16, 2] = 255
        self.assertEqual(changed_cells(after, before, 3, 3), {(1, 1), (2, 2)})

    def test_lava_and_wall_composites_localize_from_actual_foreground(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        for prefix, actions in (
            ("A5 ", (Action.RIGHT,) * 3),
            (
                "A7 ",
                (Action.UP,) * 3
                + (Action.PICK_UP,)
                + (Action.DOWN,) * 3
                + (Action.RIGHT,) * 3,
            ),
            (
                "A8 ",
                (Action.UP,) * 3 + (Action.PICK_UP, Action.DOWN) + (Action.RIGHT,) * 5,
            ),
        ):
            state = next(
                v()
                for k, v in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items()
                if k.startswith(prefix)
            )
            state.seed = 2
            for action in actions[:-1]:
                state = step(state, action)
            before = vision_observation(state)
            after = step(state, actions[-1])
            check = check_observation(vision_observation(after), after, before["image"])
            self.assertTrue(check.consistent, (prefix, check))

    def test_independent_actor_subcrop_repairs_corrupt_full_cell_prediction(self):
        from agents import vision_v3_tracking

        state = corridor()
        observation = vision_observation(state)
        real = vision_v3_tracking.classify_initial

        def corrupt(observation, cells=None):
            scores = real(observation, cells)
            values = scores.probabilities.copy()
            values[0, 0] = 0
            values[0, 0, CLASSES.index("boots")] = 1
            return CellScores(values, scores.width, scores.height)

        with patch(
            "agents.vision_v3_tracking.classify_initial",
            side_effect=corrupt,
        ):
            check = check_observation(observation, state, None)
        self.assertTrue(check.consistent, check.mismatches)

    def test_actor_subcrops_localize_reachable_stacked_effects(self):
        from grid_adventure.entities import (
            PhasingPowerUpEntity,
            ShieldPowerUpEntity,
            WallEntity,
        )

        from agents.vision_v3_perception import classify_subcrops
        from tests.planner_p1_fixtures import layout

        state = layout(width=10)
        state.add((1, 1), SpeedPowerUpEntity())
        state.add((3, 1), PhasingPowerUpEntity())
        state.add((5, 1), ShieldPowerUpEntity())
        for x in range(10):
            state.add((x, 0), WallEntity())
        for action in (Action.RIGHT, Action.PICK_UP) * 3:
            state = step(state, action)
        self.assertEqual(len(player(state).status_list), 3)
        observation = vision_observation(state)
        cells = tuple((x, y) for y in range(3) for x in range(10))
        scores = classify_subcrops(observation, cells)
        humans = [
            (x, y)
            for x, y in cells
            if CLASSES[int(scores.probabilities[y, x].argmax())] == "human"
        ]
        self.assertEqual(humans, [(5, 1)])

    def test_unconfirmed_pickup_without_history_stays_unresolved(self):
        from agents.vision_v3_tracking import fatal_uncertainty

        state = corridor()
        state.add((1, 0), GemEntity())
        state = step(state, Action.RIGHT)
        before = vision_observation(state)
        actual = step(state, Action.WAIT)
        predicted = step(state, Action.PICK_UP)
        observation = vision_observation(actual)
        check = check_observation(observation, predicted, before["image"])
        repaired = reconcile(
            observation, (BoardHypothesis(predicted, 1, ()),), check, self.budget()
        )
        self.assertTrue(repaired)
        self.assertTrue(all(fatal_uncertainty(h) for h in repaired))

    def test_third_persistent_ambiguity_cannot_drop_unrepresented_boards(self):
        from tests.planner_p1_fixtures import layout

        state = layout(width=5, height=3)
        observation = vision_observation(state)
        values = np.zeros((3, 5, 14), np.float32)
        values[:, :, CLASSES.index("floor")] = 1
        values[1, 0] = 0
        values[1, 0, CLASSES.index("human")] = 1
        values[1, 4] = 0
        values[1, 4, CLASSES.index("exit")] = 1
        for x in (1, 2):
            values[1, x] = 0
            values[1, x, CLASSES.index("gem")] = 0.51
            values[1, x, CLASSES.index("coin")] = 0.49
        hypotheses = initial_hypotheses(CellScores(values, 5, 3), observation["info"])
        self.assertEqual(len(hypotheses), 4)
        third = np.zeros(14)
        third[CLASSES.index("floor")] = 0.51
        third[CLASSES.index("gem")] = 0.49
        check = ObservationCheck(
            False,
            ((3, 1),),
            (("uncertain_cell", (3, 1), "floor", "floor", tuple(third)),),
        )
        repaired = reconcile(observation, hypotheses, check, self.budget())
        self.assertEqual(
            common_safe_prefix(repaired, (Action.RIGHT,), self.budget()), ()
        )

    def setUp(self):
        configure_vision()

    def budget(self):
        now = perf_counter()
        return EpisodeBudget(now, now + 3, now + 3.5)

    def test_wrong_position_same_counters_invalidates_suffix(self):
        predicted = corridor()
        actual = deepcopy(predicted)
        actor = player(actual)
        actual.grid[0][0].remove(actor)
        actual.add((1, 0), actor)
        previous = vision_observation(predicted)
        observation = vision_observation(actual)
        self.assertEqual(previous["info"]["status"], observation["info"]["status"])
        check = check_observation(observation, predicted, previous["image"])
        self.assertFalse(check.consistent)
        self.assertTrue(any(m[0] == "agent_position" for m in check.mismatches))
        self.assertIn((1, 0), check.changed_cells)

    def test_pickup_does_not_erase_occluded_floor_item(self):
        state, _ = mechanic("coin")
        before = vision_observation(state)
        known = reconstruct(oracle_labels(state), before["info"]).state
        actual = step(state, Action.RIGHT)
        predicted = step(known, Action.RIGHT)
        observation = vision_observation(actual)
        check = check_observation(observation, predicted, before["image"])
        repaired = reconcile(
            observation, (BoardHypothesis(predicted, 1, ()),), check, self.budget()
        )
        self.assertTrue(repaired)
        cell = repaired[0].state.grid[1][1]
        self.assertTrue(any(isinstance(e, CoinEntity) for e in cell))
        collected = step(repaired[0].state, Action.PICK_UP)
        self.assertEqual(collected.score - repaired[0].state.score, 2)

    def test_box_item_overlap_recovery(self):
        state, _ = mechanic("box_coin")
        before = vision_observation(state)
        predicted = reconstruct(oracle_labels(state), before["info"]).state
        for action in (Action.RIGHT, Action.RIGHT):
            state, predicted = step(state, action), step(predicted, action)
        observation = vision_observation(state)
        check = check_observation(observation, predicted, before["image"])
        repaired = reconcile(
            observation, (BoardHypothesis(predicted, 1, ()),), check, self.budget()
        )
        self.assertTrue(repaired)
        cell = repaired[0].state.grid[3][1]
        self.assertTrue(any(isinstance(e, CoinEntity) for e in cell))
        self.assertTrue(any(getattr(e, "pushable", None) is not None for e in cell))
        exposed = step(repaired[0].state, Action.RIGHT)
        self.assertEqual(step(exposed, Action.PICK_UP).score - exposed.score, 2)

    def test_effect_expiry_uses_observed_counters(self):
        state, _ = mechanic("boots")
        for action in (Action.RIGHT, Action.PICK_UP) + (Action.WAIT,) * 5:
            state = step(state, action)
        predicted = deepcopy(state)
        effect = SpeedPowerUpEntity(entity_id=1005)
        cast(Any, effect).collectible = None
        player(predicted).status_list.append(effect)
        observation = vision_observation(state)
        check = check_observation(observation, predicted, None)
        self.assertFalse(check.consistent)
        self.assertTrue(any(m[0] == "effects" for m in check.mismatches))
        repaired = reconcile(
            observation, (BoardHypothesis(predicted, 1, ()),), check, self.budget()
        )
        self.assertTrue(repaired)
        self.assertEqual(player(repaired[0].state).status_list, [])

    def test_ambiguous_parse_is_not_certified_by_replay(self):
        state = corridor()
        state.add((1, 0), GemEntity())
        observation = vision_observation(state)
        values = np.zeros((1, 3, 14), np.float32)
        values[0, 0, CLASSES.index("human")] = 1
        values[0, 2, CLASSES.index("exit")] = 1
        values[0, 1, CLASSES.index("gem")] = 0.51
        values[0, 1, CLASSES.index("coin")] = 0.49
        hypotheses = initial_hypotheses(CellScores(values, 3, 1), observation["info"])
        self.assertEqual(len(hypotheses), 2)
        self.assertTrue(all(h.unresolved for h in hypotheses))
        # Skipping the pickup wins only on the imagined coin board.
        prefix = common_safe_prefix(
            hypotheses, (Action.RIGHT, Action.RIGHT), self.budget()
        )
        self.assertEqual(prefix, (Action.RIGHT,))
        self.assertTrue(all(h.unresolved for h in hypotheses))

    def test_repair_at_cutoff_cannot_restart_clock(self):
        state = corridor()
        observation = vision_observation(state)
        check = check_observation(observation, state, None)
        now = perf_counter()
        expired = EpisodeBudget(now - 10, now - 7, now - 6.5)
        with patch(
            "agents.vision_v3_tracking.classify_initial",
            side_effect=AssertionError("inference after cutoff"),
        ):
            self.assertEqual(
                reconcile(
                    observation, (BoardHypothesis(state, 1, ()),), check, expired
                ),
                (),
            )
