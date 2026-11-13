import unittest
from typing import cast
from unittest.mock import patch

from grid_adventure.env import ImageObservation
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents.planner_p1 import Agent, SearchResult, search
from tests.planner_p1_fixtures import corridor


class AgentTests(unittest.TestCase):
    def test_one_search_for_matching_episode(self):
        with patch("agents.planner_p1.search", wraps=search) as planner:
            a = Agent()
            state = corridor()
            while not state.win:
                action = a.step(state)
                self.assertIsInstance(action, Action)
                state = step(state, action)
            self.assertEqual(planner.call_count, 1)
            self.assertEqual(int(a.info()["cache_hits"]), 1)

    def test_mismatch_discards_suffix(self):
        with patch("agents.planner_p1.search", wraps=search) as planner:
            a = Agent()
            state = corridor()
            a.step(state)
            # Same board, changed resources: stale RIGHT is not consumed.
            state.score = -4
            a.step(state)
            self.assertEqual(planner.call_count, 2)
            self.assertEqual(a.info()["cache_hits"], "0")

    def test_repair_does_not_reset_budget(self):
        now = [10.0]
        fail = SearchResult((), (), False, None, "deadline", 0)
        with (
            patch(
                "agents.planner_p1.perf_counter",
                side_effect=lambda: now[0],
            ),
            patch(
                "agents.planner_p1.search", return_value=fail
            ) as planner,
        ):
            a = Agent()
            state = corridor()
            self.assertIs(a.step(state), Action.WAIT)
            self.assertEqual(planner.call_args.kwargs["deadline"], 13.0)
            now[0] = 13.1
            a.step(state)
            self.assertEqual(planner.call_args.kwargs["deadline"], 13.5)
            now[0] = 13.6
            a.step(state)
            self.assertEqual(planner.call_count, 2)

    def test_terminal_returns_action(self):
        for flag in ("win", "lose"):
            s = corridor()
            setattr(s, flag, True)
            self.assertIs(Agent().step(s), Action.WAIT)
        self.assertIs(
            Agent().step(cast(ImageObservation, {"image": None})), Action.WAIT
        )

    def test_no_plan_returns_action(self):
        s = corridor()
        s.turn = 149
        self.assertIs(Agent().step(s), Action.WAIT)
