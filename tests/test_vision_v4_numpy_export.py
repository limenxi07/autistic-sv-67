import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch


class NumpyExportTests(unittest.TestCase):
    def test_same_trained_cnn_runs_with_numpy_logits(self):
        try:
            from tools.export_vision_v4_numpy import export_numpy
        except ImportError:
            self.fail("NumPy export of the existing trained CNN is missing")
        source = Path("artifacts/v4/models/selected-export/model.pt")
        torch.set_num_threads(1)
        original = torch.jit.load(source, map_location="cpu").eval()
        with tempfile.TemporaryDirectory() as folder:
            export_numpy(source, Path(folder), temperature=1.8340080864093418)
            ns: dict[str, Any] = {"__name__": "numpy_export_test"}
            exec((Path(folder) / "model_loader.py").read_text(), ns)  # noqa: S102
            model = ns["get_model"]()
            for batch in (1, 7, 225):
                values = np.random.default_rng(12).random(
                    (batch, 3, 48, 48), dtype=np.float32
                )
                with torch.inference_mode():
                    expected = original(torch.from_numpy(values)).numpy()
                actual = model(values)
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
                np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            report = json.loads((Path(folder) / "report.json").read_text())
            self.assertEqual(report["parameters"], 104526)
            self.assertTrue(report["tensor_bytes_equal"])

    def test_identical_pixels_are_inferred_once_and_cache_is_bounded(self):
        from tools.export_vision_v4_numpy import RUNTIME

        ns = {}
        exec(RUNTIME, ns)  # noqa: S102
        model = ns["_NumpyCNN"].__new__(ns["_NumpyCNN"])
        model._cache = {}
        calls = []

        def infer(images):
            calls.append(len(images))
            return np.repeat(images[:, 0, 0, 0, None], 14, axis=1)

        model._infer = infer
        a = np.zeros((1, 3, 48, 48), dtype=np.float32)
        b = np.ones_like(a)
        out = model(np.concatenate([a, b, a]))
        np.testing.assert_array_equal(out[:, 0], [0, 1, 0])
        out[:] = 99
        np.testing.assert_array_equal(model(np.concatenate([b, a]))[:, 0], [1, 0])
        self.assertEqual(calls, [2])
        for i in range(1030):
            model(a + i)
        self.assertLessEqual(len(model._cache), 1024)
