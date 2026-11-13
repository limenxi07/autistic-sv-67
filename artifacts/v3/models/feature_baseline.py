"""Validation-only feature baseline; never evaluates the frozen final corpus."""
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agents.v3_bridge import CLASSES
from tools.v3_data import fit_linear_baseline, linear_predict


def load(directory):
    report = json.loads((directory / "report.json").read_text())
    if report["split"] not in ("train", "validation"):
        raise ValueError("Final data is excluded")
    audit = json.loads((directory / "audit.json").read_text())
    assert audit["data_report_sha256"] == hashlib.sha256((directory / "report.json").read_bytes()).hexdigest()
    images, labels = [], []
    for shard in report["shards"]:
        path = directory / shard["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"]
        with np.load(path) as arrays:
            images.append(arrays["images"])
            labels.append(arrays["labels"])
    return np.concatenate(images), np.concatenate(labels)


def main():
    root = ROOT / "artifacts/v3/data"
    output = ROOT / "artifacts/v3/models/feature-linear"
    output.mkdir(exist_ok=False)
    started = perf_counter()
    images, labels = load(root / "train-200k-v2")
    model = fit_linear_baseline(images, labels)
    fit_seconds = perf_counter() - started
    del images, labels
    images, labels = load(root / "validation-14k-v2")
    predicted = linear_predict(model, images)
    confusion = np.zeros((14, 14), np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    report = {"fit_and_load_seconds": fit_seconds, "validation_cell_accuracy": float((labels == predicted).mean()),
              "class_recall": dict(zip(CLASSES, (np.diag(confusion) / confusion.sum(1)).tolist())),
              "confusion": confusion.tolist(), "axes": {}, "evidence_boundary": "Validation-only ridge fallback."}
    for axis in ("known-layout-unseen-art-v2", "new-layout-known-art-v2", "new-layout-unseen-art-v2"):
        directory = root / axis
        exact = 0
        for line in (directory / "boards.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["appearance_split"] != "final" and row["geometry_split"] != "final"
            with Image.open(directory / row["image"]) as image:
                pixels = np.asarray(image.convert("RGB"))
            ih, iw = pixels.shape[:2]
            h, w = row["height"], row["width"]
            crops, truth = [], []
            for y in range(h):
                for x in range(w):
                    crop = pixels[y * ih // h:(y + 1) * ih // h, x * iw // w:(x + 1) * iw // w]
                    crops.append(np.asarray(Image.fromarray(crop).resize((48, 48), Image.Resampling.BILINEAR)))
                    truth.append(CLASSES.index(row["labels"][y][x][0]))
            exact += int(np.array_equal(linear_predict(model, np.stack(crops)), truth))
        report["axes"][axis] = {"boards": 40, "exact_boards": exact, "exact_board_accuracy": exact / 40}
    np.savez_compressed(output / "weights.npz", **model)
    report["weights_bytes"] = (output / "weights.npz").stat().st_size
    report["weights_sha256"] = hashlib.sha256((output / "weights.npz").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
