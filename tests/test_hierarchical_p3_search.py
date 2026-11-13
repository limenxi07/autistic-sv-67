"""Behavioral regressions for the six previously cap-limited searches."""

import unittest
from time import perf_counter

from grid_adventure.grid import step

from agents import hierarchical_p3
from tests.boss_p2_next_fixtures import boss_like


class SearchTests(unittest.TestCase):
    def test_resource_event_is_pickup_and_retains_complete_endpoint(self):
        from grid_adventure.entities import GemEntity
        from grid_adventure.step import Action

        from tests.planner_p1_fixtures import corridor

        state = corridor()
        state.add((1, 0), GemEntity())
        b, c, v = hierarchical_p3.encode(state)
        goal = next(i for i, s in enumerate(b.slots) if s.kind == "gem")
        endpoints, _, _, _ = hierarchical_p3.event_query(
            b, c, v, ("item", goal), deadline=perf_counter() + 1
        )
        self.assertTrue(endpoints)
        for endpoint, score, actions in endpoints:
            actual = state
            self.assertIn(Action.PICK_UP, actions)
            self.assertFalse(endpoint.items & (1 << goal))
            for action in actions:
                actual = step(actual, action)
            self.assertEqual(
                hierarchical_p3.read_state(b, actual), (endpoint, score)
            )

    def test_event_query_obeys_expired_original_deadline(self):
        state = boss_like(5, split="heldout")[0]
        b, c, v = hierarchical_p3.encode(state)
        goal = next(i for i, s in enumerate(b.slots) if s.kind == "gem")
        endpoints, transitions, records, peak = hierarchical_p3.event_query(
            b, c, v, ("item", goal), deadline=0
        )
        self.assertEqual((endpoints, transitions, records, peak), ((), 0, 0, 0))

    def test_effect_expiry_endpoints_match_public_with_all_actions(self):
        from grid_adventure.step import Action

        from tests.boss_p2_next_fixtures import semantic_projection
        from tools.differential_compact_p2 import project, project_fast

        state, witness, _ = boss_like(5, split="heldout")
        snapshots = []
        for action in witness:
            state = step(state, action)
            b, c, v = hierarchical_p3.encode(state)
            if c.effects and any(t == 1 for _, t, _ in c.effects):
                snapshots.append(state)
        self.assertTrue(snapshots)
        compared = 0
        for state in snapshots[:4]:
            b, c, v = hierarchical_p3.encode(state)
            goals = [
                ("item", i) for i, slot in enumerate(b.slots) if c.items & (1 << i)
            ]
            for goal in goals[:3]:
                endpoints, _, _, _ = hierarchical_p3.event_query(
                    b, c, v, goal, deadline=perf_counter() + 1
                )
                for endpoint, value, actions in endpoints:
                    actual = state
                    for action in actions:
                        actual = step(actual, action)
                    self.assertEqual(
                        hierarchical_p3.read_state(b, actual), (endpoint, value)
                    )
                    for action in Action:
                        successor, delta = hierarchical_p3.fast_step(
                            b, endpoint, action
                        )
                        self.assertEqual(
                            semantic_projection(
                                project(step(actual, action)), b.keys.bit_count()
                            ),
                            semantic_projection(
                                project_fast(b, successor, value + delta),
                                b.keys.bit_count(),
                            ),
                        )
                    compared += 1
        self.assertGreater(compared, 0)

    def test_invalid_event_win_is_never_exposed(self):
        state = boss_like(5, split="heldout")[0]
        b, c, v = hierarchical_p3.encode(state)
        calls = []

        def reject(actions, keys):
            calls.append(actions)
            return hierarchical_p3.Verification("invalid", None, ())

        r = hierarchical_p3.event_search(
            b, c, v, deadline=perf_counter() + 5, mode="p2a", verify_candidate=reject
        )
        self.assertTrue(calls)
        self.assertFalse(r.winning)
        self.assertFalse(r.actions)
        self.assertEqual(r.stop_reason, "invalid")

    def test_long_resource_route_finds_public_verified_win(self):
        state = boss_like(5, split="heldout")[0]
        state.seed = 42
        board, current, score = hierarchical_p3.encode(state)
        deadline = perf_counter() + 11
        result = hierarchical_p3.search(
            board,
            current,
            score,
            deadline=deadline,
            mode="p2a",
            verify_candidate=lambda a, k: hierarchical_p3.verify_plan(
                board, state, a, k, deadline=deadline
            ),
        )
        self.assertTrue(result.winning, result.stop_reason)
        for action in result.actions:
            state = step(state, action)
        self.assertTrue(state.win)
        self.assertFalse(state.lose)
        self.assertEqual(state.score, result.final_score)


class IncumbentTests(unittest.TestCase):
    def test_existing_verified_primitive_winner_is_preserved_exactly(self):
        from tests.planner_p1_fixtures import corridor

        state = corridor()
        b, c, v = hierarchical_p3.encode(state)
        deadline = perf_counter() + 1
        verify = lambda a, k: hierarchical_p3.verify_plan(
            b, state, a, k, deadline=deadline
        )
        old = hierarchical_p3.primitive_search(
            b, c, v, deadline=deadline, mode="p2a", verify_candidate=verify
        )
        new = hierarchical_p3.search(
            b, c, v, deadline=deadline, mode="p2a", verify_candidate=verify
        )
        self.assertTrue(old.winning)
        self.assertEqual(
            (new.actions, new.expected_keys, new.final_score, new.stop_reason),
            (old.actions, old.expected_keys, old.final_score, old.stop_reason),
        )

    def test_door_and_box_events_have_public_complete_endpoints(self):
        from grid_adventure.entities import BoxEntity, KeyEntity, LockedDoorEntity
        from grid_adventure.step import Action

        from tests.planner_p1_fixtures import layout

        for box in (False, True):
            state = layout(width=8)
            if box:
                state.add((1, 1), BoxEntity())
                goal = ("box", 1 + state.width)
            else:
                state.add((0, 1), KeyEntity())
                state.add((1, 1), LockedDoorEntity())
                state = step(state, Action.PICK_UP)
            b, c, v = hierarchical_p3.encode(state)
            if not box:
                goal = (
                    "door",
                    next(i for i, s in enumerate(b.slots) if s.kind == "door"),
                )
            endpoints, _, _, _ = hierarchical_p3.event_query(
                b, c, v, goal, deadline=perf_counter() + 1
            )
            self.assertTrue(endpoints)
            for endpoint, value, actions in endpoints:
                actual = state
                for action in actions:
                    actual = step(actual, action)
                self.assertEqual(
                    hierarchical_p3.read_state(b, actual), (endpoint, value)
                )
                self.assertTrue(
                    endpoint.boxes != c.boxes if box else endpoint.doors != c.doors
                )

    def test_verification_crossing_cutoff_exposes_no_actions(self):
        from tests.planner_p1_fixtures import corridor

        state = corridor()
        b, c, v = hierarchical_p3.encode(state)
        clock = [0.0]

        def verify(actions, keys):
            clock[0] = 2.0
            return hierarchical_p3.verify_plan(
                b, state, actions, keys, deadline=1.0, clock=lambda: clock[0]
            )

        r = hierarchical_p3.event_search(
            b,
            c,
            v,
            deadline=1.0,
            mode="p2a",
            verify_candidate=verify,
            clock=lambda: clock[0],
        )
        self.assertFalse(r.winning)
        self.assertEqual(r.stop_reason, "deadline")
        self.assertFalse(r.actions)


if __name__ == "__main__":
    unittest.main()
