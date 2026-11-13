import unittest
from copy import deepcopy

from grid_adventure.entities import (
    BoxEntity,
    CoinEntity,
    GemEntity,
    KeyEntity,
    LavaEntity,
    LockedDoorEntity,
    PhasingPowerUpEntity,
    ShieldPowerUpEntity,
    SpeedPowerUpEntity,
    WallEntity,
)
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents import compact_p2
from tests.planner_p1_fixtures import fixture_cases, layout
from tools.differential_compact_p2 import project, project_fast


def interactions():
    for items in [
        (SpeedPowerUpEntity, LavaEntity),
        (ShieldPowerUpEntity, ShieldPowerUpEntity, LavaEntity),
        (SpeedPowerUpEntity, PhasingPowerUpEntity, WallEntity),
        (CoinEntity, GemEntity, KeyEntity),
        (BoxEntity, CoinEntity),
        (BoxEntity, BoxEntity),
        (KeyEntity, LockedDoorEntity, LockedDoorEntity),
    ]:
        s = layout(width=8)
        for i, cls in enumerate(items):
            s.add((1 if i < len(items) - 1 else 2, 1), cls())
        yield (
            s,
            (Action.RIGHT, Action.PICK_UP, Action.USE_KEY)
            + (Action.RIGHT,) * 5
            + (Action.WAIT,) * 6,
        )

    s = layout()
    s.add((1, 1), BoxEntity())
    s.add((1, 1), BoxEntity())
    yield s, (Action.RIGHT, Action.RIGHT, Action.WAIT)


class TransitionTests(unittest.TestCase):
    def compare(self, start, actions):
        try:
            board, current, score = compact_p2.encode(start)
        except compact_p2.UnsupportedState:
            # Unsupported simultaneous pickups are explicitly routed to public
            # search. Characterize their reachable supported repair frames too.
            remaining = tuple(actions)
            while remaining:
                start = step(start, remaining[0])
                remaining = remaining[1:]
                try:
                    compact_p2.encode(start)
                except compact_p2.UnsupportedState:
                    continue
                self.compare(start, remaining)
                return
            return
        for action in (None,) + tuple(actions):
            if action is not None:
                current, delta = compact_p2.fast_step(
                    board, current, action
                )
                score += delta
                start = step(start, action)
            self.assertEqual(project(start), project_fast(board, current, score))
            for alternative in Action:
                old = deepcopy(current)
                actual = step(start, alternative)
                successor, delta = compact_p2.fast_step(
                    board, current, alternative
                )
                self.assertEqual(current, old)
                self.assertEqual(
                    project(actual),
                    project_fast(board, successor, score + delta),
                    str(alternative),
                )

    def test_all_actions_at_every_mechanic_witness_frame(self):
        for builder, actions in fixture_cases("mechanics").values():
            self.compare(builder(), actions)

    def test_stacked_overlapping_and_blocked_interactions(self):
        for s, actions in interactions():
            self.compare(s, actions)

    def test_death_turn_is_free_but_horizon_loss_is_charged(self):
        s = layout(health=1)
        s.add((1, 1), LavaEntity())
        self.compare(s, (Action.RIGHT,))
        s = layout(turn_limit=1)
        self.compare(s, (Action.WAIT,))

    def test_invalid_action_rejected(self):
        s = layout()
        b, c, _ = compact_p2.encode(s)
        with self.assertRaises(ValueError):
            step(s, "invalid")  # ty: ignore[invalid-argument-type]
        with self.assertRaises(ValueError):
            compact_p2.fast_step(b, c, "invalid")

    def test_final_turn_win_is_free_and_terminal_is_absorbing(self):
        from tests.planner_p1_fixtures import corridor

        s = corridor()
        s.turn_limit = 2
        self.compare(s, (Action.RIGHT,) * 3)

    def test_speed_terminal_latching_and_stacked_hazard_consumption(self):
        from grid_adventure.entities import ExitEntity

        s = layout(width=5, health=1)
        s.add((0, 1), SpeedPowerUpEntity())
        s = step(s, Action.PICK_UP)
        s.add((1, 1), ExitEntity())
        s.add((2, 1), LavaEntity())
        self.compare(s, (Action.RIGHT,))
        s = layout()
        for _ in range(2):
            s.add((0, 1), ShieldPowerUpEntity())
        s = step(s, Action.PICK_UP)
        for _ in range(2):
            s.add((1, 1), LavaEntity())
        self.compare(s, (Action.RIGHT, Action.WAIT, Action.WAIT))

    def test_phasing_expiry_inside_wall_and_distinct_key_ids(self):
        from dataclasses import replace

        from tests.test_planner_p1_state import player

        s = layout(width=10)
        s.add((0, 1), PhasingPowerUpEntity())
        s = step(s, Action.PICK_UP)
        effect = player(s).status_list[0]
        effect.time_limit = replace(effect.time_limit, amount=1)
        s.add((1, 1), WallEntity())
        self.compare(s, (Action.RIGHT, Action.RIGHT))
        s = layout()
        for name in ("B", "A"):
            e = KeyEntity()
            e.key = replace(e.key, key_id=name)
            s.add((0, 1), e)
            e = LockedDoorEntity()
            e.locked = replace(e.locked, key_id=name)
            s.add((1, 1), e)
        self.compare(s, (Action.PICK_UP, Action.USE_KEY, Action.RIGHT))

    def test_overlapping_boxes_push_together(self):
        s = layout()
        s.add((1, 1), BoxEntity())
        s.add((1, 1), BoxEntity())
        self.compare(s, (Action.RIGHT, Action.RIGHT))

    def test_duplicate_keys_preserve_the_reference_consumption_identity(self):
        s = layout()
        s.add((0, 1), KeyEntity())
        s.add((0, 1), KeyEntity())
        s.add((1, 1), LockedDoorEntity())
        self.compare(s, (Action.PICK_UP, Action.USE_KEY, Action.RIGHT))

    def test_noncontiguous_key_ids_and_colliding_set_order(self):
        for ids in ((1015, 1086), (1005, 1037)):
            s = layout()
            for identity in ids:
                key = KeyEntity()
                key.entity_id = identity
                s.add((0, 1), key)
            door = LockedDoorEntity()
            door.entity_id = 5000
            s.add((1, 1), door)
            self.compare(s, (Action.PICK_UP, Action.WAIT, Action.WAIT, Action.USE_KEY))

    def test_key_can_unlock_current_cell_before_neighbors(self):
        s = layout()
        s.add((0, 1), KeyEntity())
        s.add((0, 1), LockedDoorEntity())
        s.add((1, 1), LockedDoorEntity())
        self.compare(s, (Action.PICK_UP, Action.USE_KEY))

    def test_sparse_reference_ids_across_single_pickup_traces(self):
        from tests.compact_p2_fixtures import identity_case

        for index in range(16):
            s, actions = identity_case(index)
            self.compare(s, actions)
