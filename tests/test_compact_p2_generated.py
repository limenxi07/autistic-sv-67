import itertools
import unittest
from collections import Counter

from grid_adventure.grid import step
from grid_adventure.step import Action

from tests.compact_p2_fixtures import FAMILIES, heldout_case


class HeldoutTopologyTests(unittest.TestCase):
    def test_fresh_family_layouts_have_winding_witnesses_and_replay(self):
        families = []
        for i in range(200):
            s, actions, family, _ = heldout_case(i)
            families.append(family)
            moves = [
                a
                for a in actions
                if a in (Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN)
            ]
            self.assertGreaterEqual(
                sum(a != b for a, b in itertools.pairwise(moves)), 4
            )
            for a in actions:
                s = step(s, a)
            self.assertTrue(s.win, (i, family, s.turn))
        self.assertEqual(Counter(families), Counter({f: 20 for f in FAMILIES}))
