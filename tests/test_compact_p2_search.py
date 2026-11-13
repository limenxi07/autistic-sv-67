import unittest
from time import perf_counter
from unittest.mock import patch

from grid_adventure.grid import step
from grid_adventure.step import Action

from agents import compact_p2
from tests.planner_p1_fixtures import corridor, fixture_cases


def solve(s, mode="p2a", clock=perf_counter, deadline=None):
    board, state, score = compact_p2.encode(s)
    end = perf_counter() + 2 if deadline is None else deadline
    return compact_p2.search(
        board,
        state,
        score,
        deadline=end,
        mode=mode,
        clock=clock,
        verify_candidate=lambda actions, keys: compact_p2.verify_plan(
            board, s, actions, keys, deadline=end, clock=clock
        ),
    )


class SearchTests(unittest.TestCase):
    def test_reference_verified_wins_and_finite_horizon(self):
        for builder, _ in fixture_cases("mechanics").values():
            s = builder()
            r = solve(s)
            self.assertTrue(r.winning, r)
            for a in r.actions:
                s = step(s, a)
            self.assertTrue(s.win)
            self.assertEqual(r.final_score, s.score)
        s = corridor()
        s.turn_limit = 1
        self.assertFalse(solve(s).winning)

    def test_deadline_and_resource_caps(self):
        self.assertEqual(
            solve(corridor(), clock=lambda: 2, deadline=1).stop_reason, "deadline"
        )
        for name, reason in [("STATE_CAP", "state_cap"), ("RECORD_CAP", "record_cap")]:
            with patch.object(compact_p2, name, 2):
                r = solve(corridor())
            self.assertEqual(r.stop_reason, reason)
            self.assertFalse(r.winning)

    def test_failed_upgrade_preserves_verified_incumbent(self):
        s = corridor()
        s.turn_limit = 6
        board, state, score = compact_p2.encode(s)
        count = [0]

        def verify(actions, keys):
            count[0] += 1
            if count[0] > 1:
                return compact_p2.Verification("invalid", None, ())
            return compact_p2.verify_plan(
                board, s, actions, keys, deadline=perf_counter() + 1
            )

        r = compact_p2.search(
            board,
            state,
            score,
            deadline=perf_counter() + 1,
            mode="p2a",
            verify_candidate=verify,
        )
        self.assertTrue(r.winning)
        self.assertEqual(r.actions, (Action.RIGHT, Action.RIGHT))

    def test_same_policy_under_equal_transition_budget(self):
        import sys

        from tools.benchmark_planner_p1 import load_agent

        ref = sys.modules[load_agent("artifacts/p2/baseline/task1_agent.py").__module__]
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        for label in ("A3 Multiple Required Gems", "B5 Ow Oof Owie", "B6 Warehouse"):
            s = PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label]()
            s.turn_limit = 150
            calls = [0]
            original = ref.grid_step

            def ref_step(state, action, calls=calls, original=original):
                calls[0] += 1
                return original(state, action)

            with patch.object(ref, "grid_step", ref_step):
                old = ref.search(
                    s, mode="guided", deadline=600, clock=lambda calls=calls: calls[0]
                )
            calls[0] = 0
            original_fast = compact_p2.fast_step

            def fast(board, state, action, calls=calls, original_fast=original_fast):
                calls[0] += 1
                return original_fast(board, state, action)

            with patch.object(compact_p2, "fast_step", fast):
                new = solve(s, clock=lambda calls=calls: calls[0], deadline=600)
            self.assertEqual(
                (new.winning, new.final_score, new.actions),
                (old.winning, old.final_score, old.actions),
                label,
            )


class RewardBoundTests(unittest.TestCase):
    def tiny_states(self):
        from collections import deque

        from grid_adventure.entities import CoinEntity, GemEntity

        from agents.planner_p1 import state_key

        for coin, gem in ((True, False), (True, True), (False, True)):
            s = corridor()
            s.turn_limit = 5
            if coin:
                s.add((0 if gem else 1, 0), CoinEntity())
            if gem:
                s.add((1, 0), GemEntity())
            queue = deque([s])
            seen = {state_key(s)}
            states = []
            while queue:
                state = queue.popleft()
                states.append(state)
                if state.win or state.lose:
                    continue
                for a in Action:
                    n = step(state, a)
                    k = state_key(n)
                    if k not in seen:
                        seen.add(k)
                        queue.append(n)
            yield s, states

    def oracle(self, start):
        from agents.planner_p1 import state_key

        queue = [start]
        seen = set()
        best = None
        while queue:
            s = queue.pop()
            key = state_key(s)
            if key in seen:
                continue
            seen.add(key)
            if s.win and not s.lose:
                best = s.score if best is None else max(best, s.score)
            elif not s.lose:
                queue.extend(step(s, a) for a in Action)
        return best

    def test_bounds_sound_on_all_reachable_tiny_states(self):
        for _, states in self.tiny_states():
            for s in states:
                if s.lose:
                    continue
                best = self.oracle(s)
                if best is None:
                    continue
                b, c, value = compact_p2.encode(s)
                for mst in (False, True):
                    self.assertGreaterEqual(
                        compact_p2.reward_upper_bound(
                            b, c, value, mst=mst
                        ),
                        best,
                    )

    def test_p2b_matches_exhaustive_reward_and_prefers_longer_coin_route(self):
        for s, _ in self.tiny_states():
            result = solve(s, mode="p2b")
            self.assertTrue(result.winning)
            self.assertEqual(result.final_score, self.oracle(s))
        from grid_adventure.entities import CoinEntity

        s = corridor()
        s.turn_limit = 5
        s.add((1, 0), CoinEntity())
        r = solve(s, mode="p2b")
        self.assertIn(Action.PICK_UP, r.actions)
        self.assertEqual(r.final_score, -1)

    def test_invalid_and_expired_upgrades_retain_actual_incumbent(self):
        from grid_adventure.entities import CoinEntity

        for failure in ("invalid", "deadline"):
            s = corridor()
            s.turn_limit = 6
            s.add((1, 0), CoinEntity())
            b, c, value = compact_p2.encode(s)
            calls = [0]

            def verify(actions, keys, calls=calls, failure=failure, b=b, s=s):
                calls[0] += 1
                if calls[0] > 1:
                    return compact_p2.Verification(failure, None, ())
                return compact_p2.verify_plan(
                    b, s, actions, keys, deadline=perf_counter() + 1
                )

            result = compact_p2.search(
                b,
                c,
                value,
                mode="p2b",
                deadline=perf_counter() + 1,
                verify_candidate=verify,
            )
            self.assertEqual(calls[0], 2)
            self.assertTrue(result.winning)
            self.assertEqual(result.final_score, -3)
            self.assertEqual(result.stop_reason, failure)

    def test_bounds_with_speed_phasing_and_shield_timers(self):
        from grid_adventure.entities import (
            LavaEntity,
            PhasingPowerUpEntity,
            ShieldPowerUpEntity,
            SpeedPowerUpEntity,
            WallEntity,
        )

        from tests.test_planner_p1_state import player

        for cls in (SpeedPowerUpEntity, PhasingPowerUpEntity, ShieldPowerUpEntity):
            s = corridor()
            s.turn_limit = 5
            s.add((0, 0), cls())
            if cls == PhasingPowerUpEntity:
                s.add((1, 0), WallEntity())
            if cls == ShieldPowerUpEntity:
                player(s).set_health(1)
                s.add((1, 0), LavaEntity())
            optimum = self.oracle(s)
            self.assertIsNotNone(optimum)
            b, c, value = compact_p2.encode(s)
            self.assertGreaterEqual(
                compact_p2.reward_upper_bound(b, c, value), optimum
            )
            self.assertEqual(solve(s, mode="p2b").final_score, optimum)

    def test_caps_after_win_keep_reconstructible_verified_plan(self):
        from grid_adventure.entities import CoinEntity

        s = corridor()
        s.turn_limit = 8
        s.add((0, 0), CoinEntity())
        for mode in ("p2a", "p2b"):
            with patch.object(compact_p2, "RECORD_CAP", 8):
                result = solve(s, mode=mode)
            self.assertTrue(result.winning, result)
            current = s
            for action in result.actions:
                current = step(current, action)
            self.assertTrue(current.win)
            self.assertEqual(current.score, result.final_score)
