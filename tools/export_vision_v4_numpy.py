"""Export the existing V4 model for NumPy inference without importing Torch."""

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

RUNTIME = """class _NumpyCNN:
    numpy_inference = True

    def __init__(self, weights):
        self._cache = {}
        self.weights = weights
        self.kernels = [
            weights[f"features.{i}.weight"].transpose(0, 2, 3, 1)
            .reshape(weights[f"features.{i}.weight"].shape[0], -1).T.copy()
            for i in (0, 2, 4, 6)
        ]

    def eval(self):
        return self

    def __call__(self, images):
        import numpy as np

        if images.ndim != 4 or images.shape[1:] != (3, 48, 48):
            raise ValueError("Expected NCHW 48px RGB crops")
        if not len(images):
            return np.empty((0, 14), np.float32)
        # Exact FP32 input bytes only; never reuse inferred labels by position.
        keys = [row.tobytes() for row in images]
        missing = {}
        for i, key in enumerate(keys):
            if key not in self._cache and key not in missing:
                missing[key] = i
        fresh = {}
        if missing:
            logits = self._infer(images[list(missing.values())])
            fresh = {key: row.copy() for key, row in zip(missing, logits)}
        # Assemble before eviction, including boards larger than the cache.
        result = np.stack([fresh[key] if key in fresh else self._cache[key] for key in keys])
        for key, row in fresh.items():
            if len(self._cache) >= 1024:
                del self._cache[next(iter(self._cache))]
            self._cache[key] = row
        return result

    def _infer(self, images):
        import numpy as np

        if images.ndim != 4 or images.shape[1:] != (3, 48, 48):
            raise ValueError("Expected NCHW 48px RGB crops")
        results = []
        # Bound intermediate im2col memory independently of board size.
        for offset in range(0, len(images), 32):
            x = images[offset:offset + 32].transpose(0, 2, 3, 1)
            for i, stride, kernel in zip((0, 2, 4, 6), (1, 2, 2, 2), self.kernels):
                x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
                windows = np.lib.stride_tricks.sliding_window_view(
                    x, (3, 3), axis=(1, 2)
                )[:, ::stride, ::stride]
                n, h, w, c, _, _ = windows.shape
                rows = windows.transpose(0, 1, 2, 4, 5, 3).reshape(n * h * w, 9 * c)
                x = (rows @ kernel + self.weights[f"features.{i}.bias"]).reshape(n, h, w, -1)
                np.maximum(x, 0, out=x)
            x = x.reshape(len(x), 2, 3, 2, 3, 96).mean(axis=(2, 4))
            x = x.transpose(0, 3, 1, 2).reshape(len(x), 384)
            x = x @ self.weights["head.1.weight"].T + self.weights["head.1.bias"]
            np.maximum(x, 0, out=x)
            results.append(x @ self.weights["head.3.weight"].T + self.weights["head.3.bias"])
        return np.concatenate(results) if results else np.empty((0, 14), np.float32)


def get_model(device="cpu", dtype=None):
    import base64
    import io
    import numpy as np

    if device != "cpu" or dtype not in (None, "float32", np.float32):
        raise ValueError("Only FP32 CPU inference is supported")
    data = base64.b64decode(_NUMPY_MODEL_B64)
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    return _NumpyCNN(weights)
"""


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_numpy(source, output, *, temperature):
    import numpy as np
    import torch

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError(output)
    original = torch.jit.load(source, map_location="cpu").eval()
    weights = {
        key: value.numpy().copy() for key, value in original.state_dict().items()
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **weights)
    model = output / "model.npz"
    model.write_bytes(buffer.getvalue())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    loader = output / "model_loader.py"
    loader.write_text("_NUMPY_MODEL_B64 = " + repr(encoded) + "\n\n" + RUNTIME)
    namespace: dict[str, Any] = {"__name__": "numpy_export_parity"}
    exec(loader.read_text(), namespace)  # noqa: S102
    restored = namespace["get_model"]()
    assert all(restored.weights[k].tobytes() == v.tobytes() for k, v in weights.items())
    errors = []
    for batch in (1, 7, 225):
        inputs = np.random.default_rng(20261112).random(
            (batch, 3, 48, 48), dtype=np.float32
        )
        with torch.inference_mode():
            expected = original(torch.from_numpy(inputs)).numpy()
        actual = restored(inputs)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        errors.append(float(np.abs(actual - expected).max()))
    report = {
        "format": "numpy-fp32",
        "model_filename": "model.npz",
        "original_torchscript_sha256": digest(source),
        "checkpoint_sha256": json.loads((source.parent / "report.json").read_text())[
            "checkpoint_sha256"
        ],
        "export_sha256": digest(model),
        "loader_sha256": digest(loader),
        "temperature": temperature,
        "parameters": sum(v.size for v in weights.values()),
        "tensor_bytes_equal": True,
        "parity_passed": True,
        "batches": [1, 7, 225],
        "max_logit_errors": errors,
        "tensor_sha256": {
            k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in weights.items()
        },
        "export_bytes": model.stat().st_size,
        "loader_bytes": loader.stat().st_size,
        "new_training_fits": 0,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
