import unittest
from unittest.mock import patch

import numpy as np

from agents.vision_v4_bridge import CLASSES
from agents.vision_v4_perception import classify_evidence
from agents.vision_v4_tracking import initial_hypotheses
from agents.vision_v4_types import (
    CellScores,
    PerceptionEvidence,
    PerceptionPolicy,
)
from tests.vision_v3_fixtures import vision_observation
from tests.planner_p1_fixtures import corridor


def scores(kinds):
    p = np.zeros((1, len(kinds), 14), np.float32)
    for x, kind in enumerate(kinds):
        p[0, x, CLASSES.index(kind)] = 1
    return CellScores(p, len(kinds), 1)


class EvidenceTests(unittest.TestCase):
    def evidence(self, full, inner, view="inset_005"):
        with (
            patch(
                "agents.vision_v4_perception._perception_runtime.policy",
                PerceptionPolicy(1, 0.85, 0.2, view),
            ),
            patch(
                "agents.vision_v4_perception.classify", return_value=full
            ),
            patch(
                "agents.vision_v4_perception.classify_subcrops",
                return_value=inner,
            ) as inset,
        ):
            result = classify_evidence({})
        return result, inset

    def test_confident_coin_still_triggers_rival_view(self):
        full = scores(["coin"])
        inner = scores(["gem"])
        result, inset = self.evidence(full, inner)
        inset.assert_called_once()
        self.assertEqual(inset.call_args.args[1], ((0, 0),))
        self.assertEqual(result.disagreements, ((0, 0, "coin", "gem"),))
        self.assertAlmostEqual(
            result.scores.probabilities[0, 0, CLASSES.index("gem")], 0.5
        )

    def test_full_policy_does_not_call_inset(self):
        full = scores(["coin"])
        result, inset = self.evidence(full, scores(["gem"]), "full")
        inset.assert_not_called()
        self.assertEqual(result.disagreements, ())
        np.testing.assert_array_equal(result.scores.probabilities, full.probabilities)

    def test_structural_cells_keep_full_scale(self):
        protected = ("floor", "wall", "box", "exit", "locked", "opened", "lava")
        full = scores(protected)
        full.probabilities[:] *= 0.6
        full.probabilities[:, :, CLASSES.index("gem")] = 0.4
        original = full.probabilities.copy()
        result, inset = self.evidence(full, scores(["gem"] * len(protected)))
        inset.assert_not_called()
        np.testing.assert_array_equal(result.scores.probabilities, original)

    def test_partial_batch_nan_cells_are_not_reclassified(self):
        full = scores(["human", "coin", "exit"])
        full.probabilities[0, 0] = np.nan
        full.probabilities[0, 2] = np.nan
        inner = scores(["human", "gem", "exit"])
        result, inset = self.evidence(full, inner)
        self.assertEqual(inset.call_args.args[1], ((1, 0),))
        self.assertTrue(np.isnan(result.scores.probabilities[0, 0]).all())
        self.assertTrue(np.isnan(result.scores.probabilities[0, 2]).all())

    def test_view_winners_survive_fused_ranking(self):
        base = scores(["human", "coin", "exit"])
        values = base.probabilities.copy()
        # Ghost ranks second in the fused vector, while the actual inset winner
        # gem ranks third. Both measured view winners must survive.
        values[0, 1] = 0
        values[0, 1, CLASSES.index("coin")] = 0.45
        values[0, 1, CLASSES.index("ghost")] = 0.35
        values[0, 1, CLASSES.index("gem")] = 0.20
        evidence = PerceptionEvidence(
            CellScores(values, 3, 1), ((1, 0, "coin", "gem"),), "inset_005"
        )
        hypotheses = initial_hypotheses(
            evidence, vision_observation(corridor())["info"]
        )
        kinds = {
            tuple(e.appearance.name for e in h.state.grid[1][0]) for h in hypotheses
        }
        self.assertEqual(kinds, {("coin",), ("gem",)})

    def test_three_disagreements_remain_unresolved(self):
        base = scores(["human", "coin", "coin", "coin", "exit"])
        evidence = PerceptionEvidence(
            base, tuple((x, 0, "coin", "gem") for x in (1, 2, 3)), "inset_005"
        )
        # Use a one-row public state matching this geometry.
        from grid_adventure.entities import AgentEntity, ExitEntity
        from grid_adventure.grid import GridState
        from grid_adventure.movements import MOVEMENTS
        from grid_adventure.objectives import OBJECTIVES

        state = GridState(
            width=5,
            height=1,
            movement=MOVEMENTS["cardinal"],
            objective=OBJECTIVES["collect_gems_and_exit"],
        )
        state.add((0, 0), AgentEntity())
        state.add((4, 0), ExitEntity())
        hypotheses = initial_hypotheses(evidence, vision_observation(state)["info"])
        self.assertTrue(hypotheses)
        self.assertTrue(all(h.unresolved for h in hypotheses))
