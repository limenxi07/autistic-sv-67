import unittest

from tools.vision_v3_wandb import epoch_metrics


class WandbMetricsTests(unittest.TestCase):
    def test_class_axis_and_resource_metrics_keep_their_meaning(self):
        row = {
            "epoch": 2,
            "loss": 0.2,
            "learning_rate": 0.001,
            "samples": 896,
            "elapsed_seconds": 4.0,
            "samples_per_second_including_validation": 224.0,
            "peak_vram_bytes": 1048576,
            "cell_accuracy": 0.85,
            "macro_recall": 0.85,
            "validation_nll": 0.3,
            "confusion": [[9, 1], [2, 8]],
            "axes": {
                "unseen-art": {
                    "boards": 40,
                    "exact_boards": 37,
                    "exact_board_accuracy": 0.925,
                    "cell_accuracy": 0.999,
                }
            },
        }
        metrics = epoch_metrics(row, ("coin", "gem"))
        self.assertEqual(metrics["train/loss"], 0.2)
        self.assertEqual(metrics["resource/gpu_peak_tensor_bytes"], 1048576)
        self.assertEqual(metrics["perf/samples_per_second_including_validation"], 224)
        self.assertAlmostEqual(metrics["validation/recall/coin"], 0.9)
        self.assertAlmostEqual(metrics["validation/precision/coin"], 9 / 11)
        self.assertEqual(metrics["validation/unseen-art/exact_boards"], 37)
        self.assertEqual(metrics["validation/unseen-art/exact_board_accuracy"], 0.925)

    def test_missing_classes_have_visible_zero_support(self):
        row = {"epoch": 1, "confusion": [[0, 0], [0, 3]], "axes": {}}
        metrics = epoch_metrics(row, ("coin", "gem"))
        self.assertEqual(metrics["validation/support/coin"], 0)
        self.assertEqual(metrics["validation/support/gem"], 3)
