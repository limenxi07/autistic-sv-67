import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.export_vision_v3 import export_model, validate_loader
from tools.train_vision_v3 import CellCNN


class ExportTests(unittest.TestCase):
    def test_export_matches_eager_batches_1_7_225(self):
        torch.manual_seed(20261109)
        model = CellCNN().eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": 20261109,
                    "temperature": 1.0,
                },
                checkpoint,
            )
            report = export_model(checkpoint, root / "export")
            namespace = {}
            exec((root / "export/model_loader.py").read_text(), namespace)  # noqa: S102
            restored = namespace["get_model"]()
            for n in (1, 7, 225):
                x = torch.rand(n, 3, 48, 48)
                with torch.inference_mode():
                    expected, actual = model(x), restored(x)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                self.assertTrue(torch.equal(expected.argmax(1), actual.argmax(1)))
            self.assertTrue(report["parity_passed"])
            self.assertLess(report["export_bytes"], 700000)
            self.assertEqual(report["batches"], [1, 7, 225])

    def test_export_rejects_training_module_dependency(self):
        with self.assertRaises(ValueError):
            validate_loader(
                "from tools.train_vision_v3 import CellCNN\ndef get_model(): return CellCNN()"
            )
        with self.assertRaises(ValueError):
            validate_loader(
                "import torch\ndef get_model(): return torch.load('external.pt', weights_only=False)"
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            torch.save({"state_dict": CellCNN().state_dict()}, checkpoint)
            with (
                patch(
                    "utils.generate_torch_loader_snippet",
                    return_value="import pickle\ndef get_model(): return pickle.loads(b'')",
                ),
                self.assertRaises(ValueError),
            ):
                export_model(checkpoint, root / "bad")
