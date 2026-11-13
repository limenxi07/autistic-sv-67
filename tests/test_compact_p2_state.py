import unittest
from copy import deepcopy
from dataclasses import replace

from grid_adventure.entities import BoxEntity, CoinEntity, ShieldPowerUpEntity
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents import compact_p2
from tests.planner_p1_fixtures import corridor, fixture_cases, mechanic
from tests.test_planner_p1_state import player


class CompactStateTests(unittest.TestCase):
    def test_mechanic_frames_match_independent_projection(self):
        from tools.differential_compact_p2 import project, project_fast

        for builder, actions in fixture_cases("mechanics").values():
            s = builder()
            board, _, _ = compact_p2.encode(s)
            for action in (None,) + actions:
                if action is not None:
                    s = step(s, action)
                state, score = compact_p2.read_state(board, s)
                self.assertEqual(project(s), project_fast(board, state, score))
                self.assertEqual(
                    compact_p2.observe_key(board, s),
                    compact_p2.predicted_key(board, state, score),
                )
                self.assertEqual(
                    compact_p2.observe_key(board, deepcopy(s)),
                    compact_p2.observe_key(board, s),
                )
                hash(state)

    def test_score_turn_health_and_overlaps_are_distinct(self):
        s = corridor()
        s.add((1, 0), BoxEntity())
        s.add((1, 0), CoinEntity())
        board, state, score = compact_p2.encode(s)
        for field, value in [("score", 8), ("turn", 12), ("win", True), ("lose", True)]:
            other = deepcopy(s)
            setattr(other, field, value)
            self.assertNotEqual(
                compact_p2.observe_key(board, s),
                compact_p2.observe_key(board, other),
            )
        other = deepcopy(s)
        player(other).health = replace(player(other).health, current_health=2)
        self.assertNotEqual(
            compact_p2.observe_key(board, s),
            compact_p2.observe_key(board, other),
        )
        self.assertEqual(score, 0)
        s.score = 17
        self.assertEqual(compact_p2.encode(s)[2], 17)
        self.assertEqual(len(state.boxes), 1)

    def test_stacked_effects_keep_identity_and_counters(self):
        s, _ = mechanic("shield")
        s.add((1, 1), ShieldPowerUpEntity())
        s = step(step(s, Action.RIGHT), Action.PICK_UP)
        board, _, _ = compact_p2.encode(s)
        other = deepcopy(s)
        player(other).status_list[0].usage_limit = replace(
            player(other).status_list[0].usage_limit, amount=2
        )
        self.assertNotEqual(
            compact_p2.observe_key(board, s),
            compact_p2.observe_key(board, other),
        )

    def test_unknown_fields_or_changed_geometry_reject(self):
        for mutate in (
            lambda s: setattr(s, "mystery", 1),
            lambda s: setattr(player(s), "moving", object()),
            lambda s: setattr(s, "step_cost", 4),
        ):
            s = corridor()
            mutate(s)
            with self.assertRaises(compact_p2.UnsupportedState):
                compact_p2.encode(s)
        s = corridor()
        board, _, _ = compact_p2.encode(s)
        s.add((1, 0), CoinEntity())
        with self.assertRaises(compact_p2.UnsupportedState):
            compact_p2.observe_key(board, s)

    def test_changed_lock_identity_is_an_unsupported_live_observation(self):
        s, _ = mechanic("key_door")
        board, _, _ = compact_p2.encode(s)
        door = next(
            e for c in s.grid for cell in c for e in cell if getattr(e, "locked", None)
        )
        door.locked = replace(door.locked, key_id="B")
        with self.assertRaises(compact_p2.UnsupportedState):
            compact_p2.observe_key(board, s)

    def test_nonpositive_or_noninteger_power_counters_reject(self):
        from grid_adventure.entities import PhasingPowerUpEntity, SpeedPowerUpEntity

        for cls, field in (
            (ShieldPowerUpEntity, "usage_limit"),
            (PhasingPowerUpEntity, "time_limit"),
            (SpeedPowerUpEntity, "time_limit"),
        ):
            for amount in (0, -1, 1.5, True):
                s = corridor()
                e = cls()
                setattr(e, field, replace(getattr(e, field), amount=amount))
                s.add((1, 0), e)
                with self.assertRaises(compact_p2.UnsupportedState):
                    compact_p2.encode(s)

    def test_ambiguous_simultaneous_pickups_and_doors_use_reference(self):
        from grid_adventure.entities import GemEntity, LockedDoorEntity

        for cls in (GemEntity, LockedDoorEntity):
            s = corridor()
            s.add((1, 0), cls())
            s.add((1, 0), cls())
            with self.assertRaises(compact_p2.UnsupportedState):
                compact_p2.encode(s)

    def test_coin_inventory_order_is_irrelevant_without_keys(self):
        from grid_adventure.entities import CoinEntity

        s = corridor()
        # Populate through separate public pickups, then reverse wrapper order.
        s.add((0, 0), CoinEntity())
        s = step(s, Action.PICK_UP)
        s.add((1, 0), CoinEntity())
        s = step(step(s, Action.RIGHT), Action.PICK_UP)
        board, _, _ = compact_p2.encode(s)
        other = deepcopy(s)
        player(other).inventory_list.reverse()
        self.assertEqual(
            compact_p2.observe_key(board, s),
            compact_p2.observe_key(board, other),
        )

    def test_large_keyed_inventory_schema_is_conservatively_rejected(self):
        from grid_adventure.entities import KeyEntity

        s = mechanic("key_door")[0]
        for x in (2, 4, 5, 6):
            s.add((x, 1), KeyEntity())
        with self.assertRaises(compact_p2.UnsupportedState):
            compact_p2.encode(s)

    def test_key_reference_order_remains_semantic_but_status_order_does_not(self):
        from grid_adventure.entities import KeyEntity, LockedDoorEntity

        from tools.differential_compact_p2 import project

        s = corridor()
        for identity in (1005, 1037):
            key = KeyEntity()
            key.entity_id = identity
            s.add((0, 0), key)
        door = LockedDoorEntity()
        door.entity_id = 5000
        s.add((1, 0), door)
        s = step(s, Action.PICK_UP)
        board, _, _ = compact_p2.encode(s)
        other = deepcopy(s)
        player(other).inventory_list.reverse()
        self.assertNotEqual(
            compact_p2.observe_key(board, s),
            compact_p2.observe_key(board, other),
        )
        self.assertNotEqual(
            project(step(s, Action.USE_KEY)), project(step(other, Action.USE_KEY))
        )
        s, _ = mechanic("shield")
        s.add((1, 1), ShieldPowerUpEntity())
        s = step(step(s, Action.RIGHT), Action.PICK_UP)
        board, _, _ = compact_p2.encode(s)
        other = deepcopy(s)
        player(other).status_list.reverse()
        self.assertEqual(
            compact_p2.observe_key(board, s),
            compact_p2.observe_key(board, other),
        )
        for action in Action:
            self.assertEqual(project(step(s, action)), project(step(other, action)))
