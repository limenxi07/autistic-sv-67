import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agents import vision_v3_perception
from tools.train_vision_v3 import (
    CLASSES,
    CellCNN,
    augment_training_batch,
    calibrate,
    load_crops,
    train_step,
)


class ModelTests(unittest.TestCase):
    def test_initial_shape_view_separates_gem_from_shield(self):
        from PIL import Image

        from agents.vision_v3 import Agent
        from agents.vision_v3_perception import classify_initial

        Agent()
        image = np.asarray(Image.open("artifacts/v3/qualification/b10-initial.png"))
        observation = {"image": image, "info": {"config": {"width": 15, "height": 15}}}
        scores = classify_initial(observation)
        self.assertEqual(CLASSES[int(scores.probabilities[10, 10].argmax())], "gem")

    def test_shape_augmentation_preserves_foreground_and_can_remove_color_shortcuts(
        self,
    ):
        torch.manual_seed(20261109)
        images = torch.zeros(7, 3, 48, 48)
        images[:, 0, 12:36, 12:36] = 0.4
        images[:, 1, 12:36, 12:36] = 0.7
        images[:, 2, 12:36, 12:36] = 0.9
        result = augment_training_batch(images, grayscale_probability=1.0)
        self.assertEqual(tuple(result.shape), (7, 3, 48, 48))
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)
        torch.testing.assert_close(result[:, 0], result[:, 1])
        torch.testing.assert_close(result[:, 1], result[:, 2])
        self.assertTrue((result[:, :, 24, 24] > result[:, :, 0, 0] + 0.2).all())

    def test_calibration_reduces_overconfident_validation_nll(self):
        logits = torch.tensor([[10.0, 0.0]] * 4)
        labels = torch.tensor([0, 0, 1, 1])
        result = calibrate(logits, labels)
        self.assertGreater(result["temperature"], 1)
        self.assertLess(result["nll_after"], result["nll_before"])
        # Positive scalar calibration preserves the class ranking.
        self.assertTrue(
            torch.equal((logits / result["temperature"]).argmax(1), logits.argmax(1))
        )

    def test_training_rejects_unaudited_or_mutated_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "crops-0000.npz"
            np.savez(
                path,
                images=np.zeros((2, 48, 48, 3), np.uint8),
                labels=np.array([0, 1], np.uint8),
            )
            report = {
                "split": "train",
                "crops": 2,
                "manifest_sha256": "fixture",
                "shards": [
                    {
                        "path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "samples": 2,
                    }
                ],
            }
            (root / "report.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                load_crops(root, expected_split="train")
            (root / "audit.json").write_text(
                json.dumps(
                    {
                        "verified_crops": 2,
                        "provenance_violations": 0,
                        "data_report_sha256": hashlib.sha256(
                            (root / "report.json").read_bytes()
                        ).hexdigest(),
                    }
                )
            )
            images, labels, _ = load_crops(root, expected_split="train")
            self.assertEqual(images.shape, (2, 48, 48, 3))
            np.testing.assert_array_equal(labels, [0, 1])
            with self.assertRaises(ValueError):
                load_crops(root, expected_split="final")
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_crops(root, expected_split="train")

    def test_training_reduces_synthetic_loss(self):
        torch.manual_seed(20261109)
        torch.set_num_threads(1)
        model = CellCNN()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        images = torch.stack([torch.zeros(3, 48, 48), torch.ones(3, 48, 48)])
        labels = torch.tensor([0, 1])
        initial = train_step(model, optimizer, images, labels)
        for _ in range(19):
            final = train_step(model, optimizer, images, labels)
        self.assertLess(final, initial * 0.5)

    def test_classifier_shapes_and_class_order(self):
        self.assertEqual(
            CLASSES,
            (
                "boots",
                "box",
                "coin",
                "exit",
                "floor",
                "gem",
                "ghost",
                "human",
                "key",
                "lava",
                "locked",
                "opened",
                "shield",
                "wall",
            ),
        )
        model = CellCNN().eval()
        self.assertLess(sum(p.numel() for p in model.parameters()), 200000)
        self.assertEqual(tuple(model(torch.zeros(7, 3, 48, 48)).shape), (7, 14))
        vision_v3_perception.configure(lambda: model)
        image = np.zeros((91, 137, 4), dtype=np.uint8)
        obs = {"image": image, "info": {"config": {"width": 3, "height": 2}}}
        scores = vision_v3_perception.classify(obs)
        self.assertEqual(scores.probabilities.shape, (2, 3, 14))
        np.testing.assert_allclose(scores.probabilities.sum(axis=2), 1, atol=1e-6)
        partial = vision_v3_perception.classify(obs, cells=((2, 1),))
        self.assertTrue(np.isnan(partial.probabilities[0, 0]).all())
        self.assertTrue(np.isfinite(partial.probabilities[1, 2]).all())
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)

    def test_importing_inference_adapter_does_not_import_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import agents.vision_v3_perception; assert 'torch' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
