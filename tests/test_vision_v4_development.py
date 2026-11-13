import tempfile
import unittest
from pathlib import Path

from tools.vision_v4_development import matrix_sources


class MatrixTests(unittest.TestCase):
    def test_packaged_cross_corners_keep_perception_and_recovery_separate(self):
        export = Path("artifacts/v3/models/selected-export").resolve()
        policy = {
            "temperature": 0.25,
            "min_confidence": 0.85,
            "min_margin": 0.20,
            "view": "full",
        }
        with tempfile.TemporaryDirectory() as d:
            sources = matrix_sources(
                Path("fixture-candidate.py"),
                Path(d),
                production_policy=policy,
                new_export=export,
                old_export=export,
            )
            old_new = sources["old_perception_new_tracking"].read_text()
            new_old = sources["new_perception_old_tracking"].read_text()
            self.assertIn("def _emit_action", old_new)
            self.assertIn('"v3_gem_shield"', old_new)
            self.assertNotIn("def _emit_action", new_old)
            self.assertIn(
                "return classify_evidence(observation, cells).scores", new_old
            )
            from grid_adventure.step import Action

            from tests.planner_p1_fixtures import corridor
            from tools.benchmark_planner_p1 import load_agent

            for name in ("old_perception_new_tracking", "new_perception_old_tracking"):
                self.assertEqual(
                    load_agent(str(sources[name]))().step(corridor()), Action.RIGHT
                )
