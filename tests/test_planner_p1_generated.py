import unittest

from grid_adventure.grid import step

from agents.planner_p1 import state_key
from tests.planner_p1_fixtures import fixture_cases


class GeneratedTests(unittest.TestCase):
    def test_fixed_disjoint_splits_and_verified_witnesses(self):
        tuning, heldout = fixture_cases("tuning"), fixture_cases("heldout")
        self.assertEqual(len(tuning), 50)
        self.assertEqual(len(heldout), 50)
        self.assertFalse(tuning.keys() & heldout.keys())
        signatures = set()
        widths = set()
        heights = set()
        healths = set()
        for label, (builder, actions) in {**tuning, **heldout}.items():
            s = builder()
            signatures.add(state_key(s))
            widths.add(s.width)
            heights.add(s.height)
            healths.update(
                e.health.current_health
                for c in s.grid
                for cell in c
                for e in cell
                if getattr(e, "agent", None)
            )
            for action in actions:
                s = step(s, action)
            self.assertTrue(s.win, label)
        self.assertEqual(len(signatures), 100)
        self.assertGreater(len(widths), 3)
        self.assertGreater(len(heights), 2)
        self.assertGreater(len(healths), 1)

    def test_unverified_fixtures_excluded(self):
        from unittest.mock import patch

        from grid_adventure.step import Action

        from tests.planner_p1_fixtures import corridor
        from tools.benchmark_planner_p1 import verified_fixture_cases

        cases = {
            "good": (corridor, (Action.RIGHT, Action.RIGHT)),
            "bad": (corridor, (Action.WAIT,)),
        }
        with patch(
            "tests.planner_p1_fixtures.fixture_cases",
            return_value=cases,
        ):
            verified, excluded = verified_fixture_cases("tuning")
        self.assertEqual(verified, ["good"])
        self.assertEqual(excluded[0]["level"], "bad")
