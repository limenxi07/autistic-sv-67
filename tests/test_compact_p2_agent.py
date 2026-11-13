import unittest
from time import perf_counter
from unittest.mock import patch

from grid_adventure.grid import step
from grid_adventure.step import Action

from agents import compact_p2
from tests.planner_p1_fixtures import corridor


class AgentTests(unittest.TestCase):
    def test_cache_and_unexpected_observation_obey_original_cutoff(self):
        s = corridor()
        agent = compact_p2.Agent()
        a = agent.step(s)
        self.assertIsInstance(a, Action)
        s = step(s, a)
        agent.step(s)
        self.assertEqual(agent._stats["cache_hits"], 1)
        s.turn += 2
        agent._cutoff = 0
        self.assertEqual(agent.step(s), Action.WAIT)

    def test_wrong_simulator_never_dispatches_unverified_plan(self):
        s = corridor()
        real = compact_p2.fast_step

        def wrong(board, state, action):
            successor, delta = real(board, state, action)
            return successor, delta + 10

        with (
            patch.object(compact_p2, "fast_step", wrong),
            patch.object(
                compact_p2.ReferenceAgent,
                "step_until",
                return_value=Action.WAIT,
            ),
        ):
            agent = compact_p2.Agent()
            self.assertEqual(agent.step(s), Action.WAIT)
            self.assertFalse(agent._actions)
            self.assertEqual(agent._stats["stop_reason"], "invalid")

    def test_verification_rejects_score_drift_and_deadline(self):
        s = corridor()
        board, state, score = compact_p2.encode(s)
        actions = (Action.RIGHT, Action.RIGHT)
        keys = [compact_p2.predicted_key(board, state, score)]
        for a in actions:
            state, delta = compact_p2.fast_step(board, state, a)
            score += delta
            keys.append(compact_p2.predicted_key(board, state, score))
        self.assertEqual(
            compact_p2.verify_plan(
                board, s, actions, tuple(keys), deadline=perf_counter() + 1
            ).status,
            "verified",
        )
        bad = keys[:-1] + [(state, score + 1)]
        self.assertEqual(
            compact_p2.verify_plan(
                board, s, actions, tuple(bad), deadline=perf_counter() + 1
            ).status,
            "invalid",
        )
        self.assertEqual(
            compact_p2.verify_plan(
                board, s, actions, tuple(keys), deadline=0
            ).status,
            "deadline",
        )

    def test_unsupported_input_uses_bounded_reference(self):
        s = corridor()
        s.mystery = 1
        with patch.object(
            compact_p2.ReferenceAgent,
            "step_until",
            return_value=Action.RIGHT,
        ) as fallback:
            agent = compact_p2.Agent()
            self.assertEqual(agent.step(s), Action.RIGHT)
            self.assertLessEqual(fallback.call_args.args[1], agent._cutoff)

    def test_ambiguous_pickup_board_uses_bounded_reference_path(self):
        from grid_adventure.entities import CoinEntity, GemEntity

        s = corridor()
        s.add((0, 0), CoinEntity())
        s.add((0, 0), GemEntity())
        with patch.object(
            compact_p2.ReferenceAgent,
            "step_until",
            return_value=Action.RIGHT,
        ) as fallback:
            agent = compact_p2.Agent()
            self.assertEqual(agent.step(s), Action.RIGHT)
            self.assertEqual(agent._stats["stop_reason"], "unsupported")
            self.assertEqual(agent._stats["backend"], "reference")
            self.assertLessEqual(fallback.call_args.args[1], agent._cutoff)
