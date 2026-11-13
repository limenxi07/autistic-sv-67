import unittest

import numpy as np

from agents.vision_v4_bridge import CLASSES
from tools.vision_v4_crop_regressions import compare_crops


class CropRegressionTests(unittest.TestCase):
    def test_reports_errors_without_replacing_predictions_with_truth(self):
        old = np.zeros((1, 3, len(CLASSES)), np.float32)
        old[0, :, CLASSES.index("coin")] = 1
        new = old.copy()
        new[0, 0] = 0
        new[0, 0, CLASSES.index("gem")] = 1
        new[0, 2] = np.nan
        before = new.copy()
        rows = compare_crops([["gem", "coin", "gem"]], old, new)
        self.assertEqual([(r["x"], r["new"]) for r in rows], [(0, "gem"), (2, None)])
        self.assertEqual([r["corrected"] for r in rows], [True, False])
        np.testing.assert_array_equal(new, before)

    def test_geometry_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_crops([["gem"]], np.zeros((1, 2, 14)), np.zeros((1, 2, 14)))
