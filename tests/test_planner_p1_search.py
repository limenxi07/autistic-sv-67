import unittest
from collections import deque
from time import perf_counter
from unittest.mock import patch

from grid_adventure.grid import GridState, step
from grid_adventure.step import Action

from agents.planner_p1 import search, state_key
from tests.planner_p1_fixtures import corridor


def oracle(start):
    frontier: deque[tuple[GridState, tuple[Action, ...]]] = deque([(start, ())])
    seen = {state_key(start)}
    while frontier:
        s, actions = frontier.popleft()
        if s.win:
            return actions
        for action in Action:
            n = step(s, action)
            if n.lose:
                continue
            key = state_key(n)
            if key not in seen:
                seen.add(key)
                frontier.append((n, actions + (action,)))
    return None


class SearchTests(unittest.TestCase):
    def replay(self, start, result):
        self.assertTrue(result.winning)
        self.assertEqual(len(result.expected_keys), len(result.actions) + 1)
        self.assertEqual(result.expected_keys[0], state_key(start))
        state = start
        for action, expected in zip(result.actions, result.expected_keys[1:]):
            state = step(state, action)
            self.assertEqual(state_key(state), expected)
        self.assertTrue(state.win)
        self.assertEqual(state.score, result.final_score)

    def test_shortest_and_parent_order(self):
        s = corridor()
        s.turn_limit = 6
        r = search(s, deadline=perf_counter() + 3)
        self.replay(s, r)
        self.assertEqual(r.actions, oracle(s))
        self.assertEqual(r.actions, (Action.RIGHT, Action.RIGHT))

    def test_unsolvable_and_turn_horizon(self):
        from grid_adventure.entities import WallEntity

        s = corridor()
        s.turn_limit = 3
        s.add((1, 0), WallEntity())
        r = search(s, deadline=perf_counter() + 3)
        self.assertFalse(r.winning)
        self.assertEqual(r.actions, ())
        self.assertEqual(r.expected_keys, ())
        s = corridor()
        s.turn = 149
        self.assertFalse(search(s, deadline=perf_counter() + 3).winning)
        s.turn = 148
        self.replay(s, search(s, deadline=perf_counter() + 3))

    def test_deadline_before_expansion(self):
        r = search(corridor(), deadline=1.0, clock=lambda: 2.0)
        self.assertFalse(r.winning)
        self.assertEqual(r.expanded, 0)
        self.assertEqual(r.stop_reason, "deadline")

    def test_deadline_after_winning_transition_preserves_win(self):
        now = [0.0]

        def transition(s, a):
            n = step(s, a)
            if n.win:
                now[0] = 2.0
            return n

        s = corridor()
        with patch("agents.planner_p1.grid_step", transition):
            r = search(s, deadline=1.0, clock=lambda: now[0])
        self.replay(s, r)
        self.assertEqual(r.stop_reason, "deadline")

    def test_resource_caps(self):
        for field, expected in [
            ("STATE_CAP", "state_cap"),
            ("RECORD_CAP", "record_cap"),
        ]:
            with (
                self.subTest(field=field),
                patch("agents.planner_p1." + field, 3),
            ):
                r = search(corridor(), deadline=perf_counter() + 3)
            self.assertFalse(r.winning)
            self.assertEqual(r.stop_reason, expected)
            if field == "STATE_CAP":
                self.assertLessEqual(r.peak_states, 3)
            else:
                self.assertLessEqual(r.peak_records, 3)

    def test_terminal_input(self):
        s = corridor()
        s.win = True
        self.replay(s, search(s, deadline=perf_counter() + 3))
        s.win = False
        s.lose = True
        self.assertFalse(search(s, deadline=perf_counter() + 3).winning)


class BestFirstTests(SearchTests):
    def coin_case(self):
        from grid_adventure.entities import CoinEntity

        s = corridor()
        s.turn_limit = 5
        s.add((1, 0), CoinEntity())
        return s

    def exhaustive_rewards(self, start):
        queue = [start]
        seen = {state_key(start)}
        rewards = []
        while queue:
            s = queue.pop()
            for action in Action:
                n = step(s, action)
                if n.win:
                    rewards.append(n.score)
                elif not n.lose and state_key(n) not in seen:
                    seen.add(state_key(n))
                    queue.append(n)
        return rewards

    def test_longer_coin_route_improves_reward(self):
        s = self.coin_case()
        bfs = search(s, deadline=perf_counter() + 3)
        best = search(s, deadline=perf_counter() + 3, mode="best_first")
        self.replay(s, best)
        assert best.final_score is not None and bfs.final_score is not None
        self.assertGreater(best.final_score, bfs.final_score)
        self.assertGreater(len(best.actions), len(bfs.actions))
        self.assertEqual(best.final_score, max(self.exhaustive_rewards(s)))

    def test_best_first_deadline_preserves_incumbent(self):
        now = [0.0]
        wins = []

        def transition(s, a):
            n = step(s, a)
            if n.win:
                wins.append(n.score)
                if len(wins) == 2:
                    now[0] = 2.0
            return n

        s = self.coin_case()
        with patch("agents.planner_p1.grid_step", transition):
            r = search(s, deadline=1.0, mode="best_first", clock=lambda: now[0])
        self.replay(s, r)
        self.assertEqual(r.final_score, max(wins))
        self.assertEqual(r.stop_reason, "deadline")

    def test_best_first_caps_preserve_win(self):
        s = self.coin_case()
        for name, value, reason in [
            ("RECORD_CAP", 12, "record_cap"),
            ("STATE_CAP", 8, "state_cap"),
        ]:
            with patch("agents.planner_p1." + name, value):
                result = search(s, deadline=perf_counter() + 3, mode="best_first")
            self.replay(s, result)
            self.assertEqual(result.stop_reason, reason)

    def test_heuristic_observed_targets(self):
        from grid_adventure.entities import GemEntity

        from agents.planner_p1 import relaxed_route_length

        s = corridor()
        self.assertEqual(relaxed_route_length(s), 2)
        s.add((1, 0), GemEntity())
        self.assertEqual(relaxed_route_length(s), 2)


class GuidedSearchTests(SearchTests):
    def test_wall_aware_route_distance(self):
        from grid_adventure.entities import WallEntity

        from agents.planner_p1 import RouteOrdering
        from tests.planner_p1_fixtures import layout

        s = layout(width=5)
        s.add((1, 1), WallEntity())
        self.assertEqual(RouteOrdering(s).estimate(s), 6)

    def test_required_multiple_with_restricted_simulation_budget(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["A3 Multiple Required Gems"]()
        s.turn_limit = 150
        calls = [0]

        def transition(state, action):
            calls[0] += 1
            return step(state, action)

        with patch("agents.planner_p1.grid_step", transition):
            result = search(
                s,
                deadline=100,
                mode="guided",
                clock=lambda counter=calls: float(counter[0]),
            )
        self.replay(s, result)
        self.assertLessEqual(calls[0], 100)

    def test_guided_caps_and_deadline(self):
        for field, reason in [("STATE_CAP", "state_cap"), ("RECORD_CAP", "record_cap")]:
            with patch("agents.planner_p1." + field, 3):
                result = search(corridor(), deadline=perf_counter() + 3, mode="guided")
            self.assertEqual(result.stop_reason, reason)
            if result.winning:
                self.replay(corridor(), result)
            self.assertLessEqual(
                result.peak_states if field == "STATE_CAP" else result.peak_records, 3
            )
        result = search(corridor(), deadline=0, mode="guided", clock=lambda: 1.0)
        self.assertEqual(result.expanded, 0)
        self.assertEqual(result.stop_reason, "deadline")

    def test_guided_hazard_and_key_routes_under_restricted_budget(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        for label in (
            "A5 Hazard Detour",
            "A7 Shield Powerup",
            "A10 Combined Mechanics",
        ):
            with self.subTest(label=label):
                s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label]()
                s.turn_limit = 150
                calls = [0]

                def transition(state, action, counter=calls):
                    counter[0] += 1
                    return step(state, action)

                with patch("agents.planner_p1.grid_step", transition):
                    result = search(
                        s,
                        deadline=600,
                        mode="guided",
                        clock=lambda counter=calls: float(counter[0]),
                    )
                self.replay(s, result)

    def test_guided_eventually_enumerates_every_action(self):
        calls = []
        s = corridor()
        s.turn_limit = 1

        def transition(state, action):
            calls.append(action)
            return step(state, action)

        with patch("agents.planner_p1.grid_step", transition):
            result = search(s, deadline=perf_counter() + 3, mode="guided")
        self.assertFalse(result.winning)
        self.assertEqual(set(calls), set(Action))
        self.assertEqual(len(calls), len(Action))

    def test_guided_refines_optional_coin_reward(self):
        from public_levels.catalog import (
            PUBLIC_GAMEPLAY_LEVEL_BUILDERS,
            PUBLIC_LEVEL_MAX_REWARDS,
        )

        s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["A2 Optional Coin Path"]()
        s.turn_limit = 150
        calls = [0]

        def transition(state, action):
            calls[0] += 1
            return step(state, action)

        with patch("agents.planner_p1.grid_step", transition):
            result = search(
                s, deadline=2000, mode="guided", clock=lambda: float(calls[0])
            )
        self.replay(s, result)
        self.assertEqual(
            result.final_score, PUBLIC_LEVEL_MAX_REWARDS["A2 Optional Coin Path"]
        )

    def test_guided_collects_keys_needed_beyond_the_next_door(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
        s.turn_limit = 150
        calls = [0]

        def transition(state, action):
            calls[0] += 1
            return step(state, action)

        with patch("agents.planner_p1.grid_step", transition):
            result = search(
                s, deadline=800, mode="guided", clock=lambda: float(calls[0])
            )
        self.replay(s, result)

    def test_guided_routes_around_pushed_boxes(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B6 Warehouse"]()
        s.turn_limit = 150
        calls = [0]

        def transition(state, action):
            calls[0] += 1
            return step(state, action)

        with patch("agents.planner_p1.grid_step", transition):
            result = search(
                s, deadline=250, mode="guided", clock=lambda: float(calls[0])
            )
        self.replay(s, result)

    def test_guided_finds_hazard_detours_before_spending_search_budget(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B5 Ow Oof Owie"]()
        s.turn_limit = 150
        calls = [0]

        def transition(state, action):
            calls[0] += 1
            return step(state, action)

        with patch("agents.planner_p1.grid_step", transition):
            result = search(
                s, deadline=150, mode="guided", clock=lambda: float(calls[0])
            )
        self.replay(s, result)

    def test_guided_routes_phasing_to_a_reachable_landing(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B8 Choose Wisely"]()
        s.turn_limit = 150
        calls = [0]

        def transition(state, action):
            calls[0] += 1
            return step(state, action)

        with patch("agents.planner_p1.grid_step", transition):
            result = search(
                s, deadline=1500, mode="guided", clock=lambda: float(calls[0])
            )
        self.replay(s, result)

    def test_route_estimate_can_leave_wall_after_phasing_expires(self):
        from grid_adventure.entities import PhasingPowerUpEntity, WallEntity

        from agents.planner_p1 import RouteOrdering
        from tests.planner_p1_fixtures import layout

        s = layout(width=10)
        s.add((1, 1), PhasingPowerUpEntity())
        s.add((6, 1), WallEntity())
        for action in (Action.RIGHT, Action.PICK_UP) + (Action.RIGHT,) * 5:
            s = step(s, action)
        self.assertFalse(s.lose)
        self.assertLess(RouteOrdering(s).estimate(s), 10000)
        after = step(s, Action.RIGHT)
        self.assertFalse(after.lose)

    def test_ordering_stops_expensive_work_at_search_deadline(self):
        from agents.planner_p1 import RouteOrdering

        ordering = RouteOrdering(corridor(), deadline=0.0, clock=lambda: 1.0)
        self.assertEqual(ordering.estimate(corridor()), 0.0)
        self.assertFalse(ordering.maps)
        self.assertFalse(ordering.tours)

    def test_guided_saves_last_phasing_pickup_for_the_final_barrier(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B10 Khoa Was Here"]()
        s.turn_limit = 150
        calls = [0]

        def transition(state, action):
            calls[0] += 1
            return step(state, action)

        with patch("agents.planner_p1.grid_step", transition):
            result = search(
                s, deadline=1200, mode="guided", clock=lambda: float(calls[0])
            )
        self.replay(s, result)

    def test_guided_refinement_improves_coin_and_shield_routes(self):
        from public_levels.catalog import (
            PUBLIC_GAMEPLAY_LEVEL_BUILDERS,
            PUBLIC_LEVEL_MAX_REWARDS,
        )

        for label, budget in (
            ("A6 Pushable Box", 3000),
            ("B2 The Vault", 2000),
            ("B7 Fire Resistance", 2000),
        ):
            with self.subTest(label=label):
                s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label]()
                s.turn_limit = 150
                calls = [0]

                def transition(state, action, counter=calls):
                    counter[0] += 1
                    return step(state, action)

                with patch("agents.planner_p1.grid_step", transition):
                    result = search(
                        s,
                        deadline=budget,
                        mode="guided",
                        clock=lambda counter=calls: float(counter[0]),
                    )
                self.replay(s, result)
                self.assertEqual(result.final_score, PUBLIC_LEVEL_MAX_REWARDS[label])
